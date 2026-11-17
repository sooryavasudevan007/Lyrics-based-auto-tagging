"""Run stylistic, error-analysis, and matched-subset audio experiments.

All model fitting uses train; all exploratory evaluation uses validation.
There is deliberately no test-scoring command here.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import joblib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler, Normalizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.data import load_split, load_categories, _songs
from src.models import make_vectorizer
from src.stylistic import FEATURE_NAMES, extract_features
from src.experiment_utils import (ensure_train_val, fit_lr, write_json, sha256, report_model,
                               coefficient_importance, load_scores, subset_split)
from src.error_analysis import (sample_errors, confused_pairs, song_quality, quality_correlations,
                                length_repetition_groups, explain_tfidf, tag_error_rates)
from src.audio_features import fixed_subset, find_clip, decode_audio, summarize_waveform, AUDIO_FEATURE_NAMES


def load_data(args):
    train = load_split("train", processed_dir=args.processed_dir, splits_dir=args.splits_dir)
    val = load_split("val", processed_dir=args.processed_dir, splits_dir=args.splits_dir)
    ensure_train_val(train, val)
    return train, val, load_categories(args.splits_dir)


def metadata(args, train, val):
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "split": "val", "test_evaluated": False,
            "n_train": len(train.ids), "n_val": len(val.ids), "K": len(train.tags),
            "seed": C.SEED, "splits_sha256": sha256(args.splits_dir / "splits.csv"),
            "vocab_sha256": sha256(args.splits_dir / "vocab.json"),
            "data_sha256": sha256(args.processed_dir / "songs.jsonl.gz"),
            "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "cmudict"]},
            "annotation_note": "tuned_on_val thresholds and evaluation share validation; optimistic annotation estimates",
            "fit_note": "scalers, vocabulary, and model parameters fitted on train only"}


def style(args):
    train, val, cats = load_data(args)
    out = args.output_dir / "stylistic"
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    records = _songs(args.processed_dir)
    features = extract_features([r["lyrics"] for r in records], [r["id"] for r in records])
    features.to_csv(out / "features_all.csv.gz", index=False, compression="gzip")
    feature_time = time.perf_counter() - t
    by_id = features.set_index("id")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(by_id.loc[train.ids, FEATURE_NAMES])
    X_val = scaler.transform(by_id.loc[val.ids, FEATURE_NAMES])
    write_json(out / "train_scaler.json", {"names": FEATURE_NAMES, "mean": scaler.mean_, "scale": scaler.scale_})
    print(f"Extracted {len(features):,} songs, {len(FEATURE_NAMES)} features in {feature_time:.1f}s", flush=True)
    rows = []
    t = time.perf_counter()
    model = fit_lr(X_train, train.Y)
    joblib.dump({"model": model, "vectorizer": None, "scaler": scaler}, out / "style_lr.joblib")
    rows_part, _ = report_model(out, "style_lr", val, model.predict_proba(X_val), cats, time.perf_counter() - t)
    rows.extend(rows_part)
    importance = coefficient_importance(model, FEATURE_NAMES, val.tags, cats)
    importance.to_csv(out / "importance_per_tag.csv", index=False)
    importance.groupby(["category", "feature"], as_index=False).abs_coefficient.mean().sort_values(
        ["category", "abs_coefficient"], ascending=[True, False]).to_csv(out / "importance_per_category.csv", index=False)
    print("Stylistic-only model evaluated", flush=True)
    t = time.perf_counter()
    vectorizer = make_vectorizer()
    tf_train = vectorizer.fit_transform(train.texts)
    tf_val = vectorizer.transform(val.texts)
    vectorize_seconds = time.perf_counter() - t
    # Each representation has unit row norm before concatenation. This keeps
    # the 15 dense statistics from dominating the TF-IDF block geometrically.
    normalized_train = Normalizer().transform(X_train)
    normalized_val = Normalizer().transform(X_val)
    for name, A, B in [("tfidf_lr", tf_train, tf_val),
                        ("tfidf_style_lr", hstack([tf_train, csr_matrix(normalized_train)], format="csr"),
                         hstack([tf_val, csr_matrix(normalized_val)], format="csr"))]:
        t = time.perf_counter()
        model = fit_lr(A, train.Y)
        fitted = time.perf_counter() - t
        joblib.dump({"model": model, "vectorizer": vectorizer, "scaler": scaler,
                     "style_block_l2": name == "tfidf_style_lr"}, out / f"{name}.joblib")
        rows_part, _ = report_model(out, name, val, model.predict_proba(B), cats, fitted)
        rows.extend(rows_part)
        print(f"{name}: fitted in {fitted:.1f}s", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary_val.csv", index=False)
    base = pd.read_csv(out / "tfidf_lr_tuned_on_val_per_category.csv").set_index("category")
    combined = pd.read_csv(out / "tfidf_style_lr_tuned_on_val_per_category.csv").set_index("category")
    delta = combined[["macro_F1", "MAP", "ROC-AUC", "P@10"]] - base[["macro_F1", "MAP", "ROC-AUC", "P@10"]]
    delta.add_prefix("delta_").to_csv(out / "combined_minus_tfidf_per_category.csv")
    write_json(out / "run_info.json", {**metadata(args, train, val), "feature_seconds": feature_time,
                                       "tfidf_seconds": vectorize_seconds, "n_all_features": len(features),
                                       "n_tfidf_features": tf_train.shape[1], "rhyme": "CMUdict phonetic, local two-line window",
                                       "combined_preprocessing": "train-standardized stylistic block, then per-song L2 normalization; TF-IDF already L2 normalized",
                                       "chorus": "repeated contiguous two-line block coverage; proxy only"})
    print(summary[["model", "thresholds", "macro_F1", "MAP", "ROC-AUC"]].to_string(index=False), flush=True)


def errors(args):
    train, val, cats = load_data(args)
    out = args.output_dir / "errors"
    out.mkdir(parents=True, exist_ok=True)
    if args.scores:
        score_path = args.scores
    else:
        summary = pd.read_csv(args.output_dir / "stylistic" / "summary_val.csv")
        summary = summary[summary.thresholds == "tuned_on_val"].sort_values(
            ["macro_F1", "MAP", "model"], ascending=[False, False, True])
        score_path = args.output_dir / "stylistic" / f"{summary.iloc[0].model}_val_scores.npz"
    scores, thresholds = load_scores(score_path, val)
    selected_model = score_path.name.replace("_val_scores.npz", "")
    sample = sample_errors(val, scores, thresholds, cats, args.per_category)
    prior_info = out / "run_info.json"
    prior_sample = out / "review_sample.csv"
    if prior_info.exists() and prior_sample.exists():
        info = json.loads(prior_info.read_text())
        if info.get("scores_sha256") == sha256(score_path):
            reviewed = pd.read_csv(prior_sample, dtype={"id": str}, keep_default_na=False)
            previous = reviewed.set_index(["id", "category"])
            for i, row in sample.iterrows():
                key = (row.id, row.category)
                if key in previous.index:
                    for field in ["pattern", "review_notes", "reviewed"]:
                        sample.at[i, field] = previous.loc[key, field]
    sample.to_csv(out / "review_sample.csv", index=False)
    pred = scores >= thresholds
    pairs = confused_pairs(val.Y, pred, val.tags)
    pairs.to_csv(out / "confused_pairs.csv", index=False)
    tag_error_rates(val.Y, pred, val.tags, thresholds, cats).to_csv(out / "tag_error_rates.csv", index=False)
    quality = song_quality(val.Y, pred)
    features = extract_features(val.texts, val.ids)
    pd.concat([features, quality], axis=1).to_csv(out / "song_quality.csv.gz", index=False, compression="gzip")
    quality_correlations(features, quality, val.artists, args.bootstrap).to_csv(out / "quality_correlations.csv", index=False)
    length_repetition_groups(features, quality).to_csv(out / "quality_by_group.csv", index=False)
    # Qualitative review material stays ignored, rather than publishing lyrics.
    private = args.output_dir / "private_review"
    private.mkdir(parents=True, exist_ok=True)
    lookup = dict(zip(val.ids, val.texts))
    with (private / "sample_lyrics.jsonl").open("w") as f:
        for sid in sorted(set(sample.id)):
            f.write(json.dumps({"id": sid, "lyrics": lookup[sid]}, ensure_ascii=False) + "\n")
    # For combined models the TF-IDF and stylistic contributions are explained separately.
    t = time.perf_counter()
    artifact_path = args.output_dir / "stylistic" / f"{selected_model}.joblib"
    artifact = joblib.load(artifact_path) if artifact_path.exists() else None
    vec = artifact["vectorizer"] if artifact and artifact["vectorizer"] is not None else make_vectorizer()
    A = None if artifact else vec.fit_transform(train.texts)
    B = vec.transform(val.texts) if selected_model != "style_lr" else None
    if selected_model == "style_lr":
        scaler = artifact["scaler"] if artifact else StandardScaler()
        if artifact:
            model = artifact["model"]
        else:
            A = scaler.fit_transform(extract_features(train.texts)[FEATURE_NAMES])
            model = fit_lr(A, train.Y)
        expected = model.predict_proba(scaler.transform(features[FEATURE_NAMES]))
        coefficient_importance(model, FEATURE_NAMES, val.tags, cats).to_csv(out / "selected_model_feature_coefficients.csv", index=False)
        explanation_note = "Stylistic model has no word coefficients; signed standardized feature coefficients saved."
    elif selected_model == "tfidf_style_lr":
        scaler = artifact["scaler"] if artifact else StandardScaler()
        if artifact:
            model = artifact["model"]
        else:
            style_train = Normalizer().transform(scaler.fit_transform(extract_features(train.texts)[FEATURE_NAMES]))
            model = fit_lr(hstack([A, csr_matrix(style_train)], format="csr"), train.Y)
        style_val = Normalizer().transform(scaler.transform(features[FEATURE_NAMES]))
        expected = model.predict_proba(hstack([B, csr_matrix(style_val)], format="csr"))
        # explain_tfidf indexes only the text columns of the combined coefficient vector.
        explain_tfidf(val, sample, vec, model).to_csv(out / "selected_tag_terms.csv", index=False)
        explanation_note = "Exact text-column contributions of the combined model; intercept and style contributions must also be considered."
    elif selected_model == "tfidf_lr":
        model = artifact["model"] if artifact else fit_lr(A, train.Y)
        expected = model.predict_proba(B)
        explain_tfidf(val, sample, vec, model).to_csv(out / "selected_tag_terms.csv", index=False)
        explanation_note = "Exact signed TF-IDF contributions for the selected best lyrics model."
    else:
        expected = None
        explanation_note = "External score artifact: no matching estimator supplied, so word explanations are not inferred."
    if expected is not None and not np.allclose(expected, scores, atol=1e-4, rtol=1e-4):
        raise ValueError("Refitted explanation model does not reproduce the recorded scores")
    write_json(out / "run_info.json", {**metadata(args, train, val), "model": selected_model,
                                       "scores_sha256": sha256(score_path), "per_category": args.per_category,
                                       "sample_rows": len(sample), "unique_sample_songs": len(set(sample.id)),
                                       "qualitative_review_complete": bool(len(sample) and sample.reviewed.astype(str).str.lower().eq("true").all()
                                                                      and sample.pattern.str.strip().ne("").all()),
                                       "bootstrap": args.bootstrap,
                                       "explanations": explanation_note,
                                       "confusion_note": "co-occurring missed/extra labels; missing tags may be annotation omissions"})
    print(f"{selected_model}: sampled {len(sample)} category/song rows; {len(pairs)} error pairs. See review_sample.csv for review status.", flush=True)


def audio_manifest(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "audio_subset.csv"
    frozen = pd.read_csv(args.splits_dir / "splits.csv", dtype=str, keep_default_na=False)
    manifest = fixed_subset(frozen, args.subset_size)
    if destination.exists():
        existing = pd.read_csv(destination, dtype=str, keep_default_na=False)
        if existing.to_csv(index=False) != manifest.to_csv(index=False):
            raise ValueError("Refusing to replace an existing fixed subset with different song IDs")
    else:
        manifest.to_csv(destination, index=False)
    print(manifest.split.value_counts().to_dict(), flush=True)
    return manifest


def audio(args):
    manifest = audio_manifest(args)
    out = args.output_dir / "audio"
    out.mkdir(parents=True, exist_ok=True)
    availability = manifest.copy()
    availability["path"] = availability.id.map(lambda sid: str(p) if (p := find_clip(args.audio_dir, sid)) else "")
    availability["available"] = availability.path != ""
    availability.drop(columns="path").to_csv(out / "availability.csv", index=False)
    needed = availability[availability.split.isin(["train", "val"])]
    status = {"fixed_subset_songs": len(manifest), "available_all": int(availability.available.sum()),
              "required_train_val": len(needed), "available_train_val": int(needed.available.sum()),
              "subset_sha256": sha256(args.output_dir / "audio_subset.csv"), "test_evaluated": False}
    if not needed.available.all() and not args.audio_features:
        write_json(out / "access_status.json", {**status, "status": "blocked_missing_clips"})
        raise SystemExit(f"Audio blocked: {status['available_train_val']}/{len(needed)} train/validation clips; see access_status.json")
    # Extract label-independent features for all available fixed IDs, including
    # test clips, while model fitting and scoring still use only train/val.
    available = availability if args.audio_features else availability[availability.available]
    cache = out / "features_available.npz"
    expected_ids = available.id.tolist()
    if args.audio_features:
        precomputed_sha256 = sha256(args.audio_features)
        frame = pd.read_csv(args.audio_features, dtype={"id": str})
        if (list(frame.columns) != ["id", *AUDIO_FEATURE_NAMES] or frame.id.duplicated().any()
                or set(frame.id) != set(manifest.id)):
            raise ValueError("Precomputed features must cover the exact fixed IDs and audio feature schema")
        values = frame.set_index("id").loc[expected_ids, AUDIO_FEATURE_NAMES].to_numpy(dtype=np.float32)
    elif cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if z["ids"].tolist() != expected_ids or z["feature_names"].tolist() != AUDIO_FEATURE_NAMES:
                raise ValueError("Audio feature cache does not match fixed IDs or extraction schema")
            values = z["values"]
    else:
        t = time.perf_counter()
        def extract(item):
            sid, path = item
            return summarize_waveform(decode_audio(path))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = []
            for i, values_row in enumerate(pool.map(extract, zip(available.id, available.path))):
                rows.append(values_row)
                if (i + 1) % 500 == 0:
                    print(f"Extracted {i + 1}/{len(available)} audio clips", flush=True)
        values = np.stack(rows)
        np.savez_compressed(cache, ids=np.asarray(expected_ids, dtype=str), feature_names=np.asarray(AUDIO_FEATURE_NAMES), values=values)
        print(f"Audio extraction: {time.perf_counter() - t:.1f}s", flush=True)
    if values.shape != (len(available), len(AUDIO_FEATURE_NAMES)) or not np.isfinite(values).all():
        raise ValueError("Invalid cached audio features")
    public_features = pd.DataFrame(values, columns=AUDIO_FEATURE_NAMES)
    public_features.insert(0, "id", expected_ids)
    feature_output = out / "features_all.csv.gz"
    if not args.audio_features or args.audio_features.resolve() != feature_output.resolve():
        public_features.to_csv(feature_output, index=False, compression="gzip")
    train, val, cats = load_data(args)
    train = subset_split(train, manifest.loc[manifest.split == "train", "id"].tolist())
    val = subset_split(val, manifest.loc[manifest.split == "val", "id"].tolist())
    by_id = pd.DataFrame(values, index=expected_ids)
    scaler = StandardScaler()
    A = scaler.fit_transform(by_id.loc[train.ids])
    B = scaler.transform(by_id.loc[val.ids])
    rows = []
    t = time.perf_counter()
    model = fit_lr(A, train.Y)
    rr, _ = report_model(out, "audio_mfcc_lr", val, model.predict_proba(B), cats, time.perf_counter() - t)
    rows.extend(rr)
    # Retrain lyrics on the exact same subset: no extra lyrics-only training songs.
    vectorizer = make_vectorizer()
    A = vectorizer.fit_transform(train.texts)
    B = vectorizer.transform(val.texts)
    style_scaler = StandardScaler()
    style_train = style_scaler.fit_transform(extract_features(train.texts)[FEATURE_NAMES])
    style_val = style_scaler.transform(extract_features(val.texts)[FEATURE_NAMES])
    for name, aa, bb in [("matched_tfidf_lr", A, B),
                         ("matched_style_lr", style_train, style_val),
                         ("matched_tfidf_style_lr", hstack([A, csr_matrix(Normalizer().transform(style_train))], format="csr"),
                          hstack([B, csr_matrix(Normalizer().transform(style_val))], format="csr"))]:
        t = time.perf_counter()
        model = fit_lr(aa, train.Y)
        rr, _ = report_model(out, name, val, model.predict_proba(bb), cats, time.perf_counter() - t)
        rows.extend(rr)
    pd.DataFrame(rows).to_csv(out / "summary_val.csv", index=False)
    candidates = pd.DataFrame(rows)
    candidates = candidates[(candidates.thresholds == "tuned_on_val") & candidates.model.str.startswith("matched_")]
    lyrics_name = candidates.sort_values(["macro_F1", "MAP", "model"], ascending=[False, False, True]).iloc[0].model
    ca = pd.read_csv(out / "audio_mfcc_lr_tuned_on_val_per_category.csv").set_index("category")
    cl = pd.read_csv(out / f"{lyrics_name}_tuned_on_val_per_category.csv").set_index("category")
    comparison = ca.add_prefix("audio_").join(cl.add_prefix("lyrics_"))
    for metric in ["macro_F1", "MAP", "ROC-AUC", "P@10"]:
        comparison[f"audio_minus_lyrics_{metric}"] = ca[metric] - cl[metric]
    comparison.to_csv(out / "per_category_comparison.csv")
    Sa, tha = load_scores(out / "audio_mfcc_lr_val_scores.npz", val)
    Sl, thl = load_scores(out / f"{lyrics_name}_val_scores.npz", val)
    disagreements = []
    for j, tag in enumerate(val.tags):
        diff = (Sa[:, j] >= tha[j]) != (Sl[:, j] >= thl[j])
        example_idx = np.flatnonzero(diff)
        examples = [{"id": val.ids[i], "artist": val.artists[i], "observed": int(val.Y[i, j]),
                     "audio_score": float(Sa[i, j]), "lyrics_score": float(Sl[i, j])} for i in example_idx[:5]]
        disagreements.append({"tag": tag, "category": cats.get(tag, "other"),
                              "n_disagree": int(diff.sum()), "disagreement_rate": float(diff.mean()),
                              "examples": json.dumps(examples)})
    pd.DataFrame(disagreements).sort_values(["n_disagree", "tag"], ascending=[False, True]).to_csv(out / "modality_disagreements.csv", index=False)
    write_json(out / "access_status.json", {**status, "status": "evaluated_on_validation"})
    write_json(out / "run_info.json", {**metadata(args, train, val), **status,
                                       "audio_features": AUDIO_FEATURE_NAMES,
                                       "features_extracted": len(available), "lyrics_model": lyrics_name,
                                       "analysis_status": {"category_comparison": "in progress",
                                                           "modality_disagreements": "in progress"},
                                       "analysis_note": 'Category comparisons and modality-disagreement examples are preliminary; interpretation and example-song review are in progress (Objective 3 PI3 and PI4).',
                                       "features_source": "precomputed" if args.audio_features else "local_waveforms",
                                       "precomputed_sha256": precomputed_sha256 if args.audio_features else None,
                                       "comparison": "Both models trained and scored on identical fixed subset IDs",
                                       "limitation": "MFCC summaries are a limited audio baseline, not an upper bound on audio performance",
                                       "pretrained_embedding_complete": False})
    print(pd.DataFrame(rows)[["model", "thresholds", "macro_F1", "MAP", "ROC-AUC"]].to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["style", "errors", "audio-manifest", "audio"])
    ap.add_argument("--processed-dir", type=Path, default=C.PROCESSED_DIR)
    ap.add_argument("--splits-dir", type=Path, default=C.SPLITS_DIR)
    ap.add_argument("--output-dir", type=Path, default=C.ROOT / "results" / "Harish")
    ap.add_argument("--audio-dir", type=Path, default=C.M4A_DIR / "audios")
    ap.add_argument("--audio-features", type=Path, help="Reuse the checked numeric feature table instead of decoding clips")
    ap.add_argument("--subset-size", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-category", type=int, default=12)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--scores", type=Path)
    args = ap.parse_args()
    if args.workers < 1 or args.per_category < 1 or args.bootstrap < 0:
        ap.error("workers/per-category must be positive; bootstrap must be nonnegative")
    {"style": style, "errors": errors, "audio-manifest": audio_manifest, "audio": audio}[args.command](args)


if __name__ == "__main__":
    main()
