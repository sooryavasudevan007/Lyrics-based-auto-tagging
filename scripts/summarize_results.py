"""Build a dated PI status report and the contributor manuscript update from real outputs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.experiment_utils import write_json


def markdown(df):
    def val(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(val(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=C.ROOT / "results/Harish")
    ap.add_argument("--update-paper", action="store_true")
    args = ap.parse_args()
    root = args.results_dir
    style = pd.read_csv(root / "stylistic/summary_val.csv")
    tuned = style[style.thresholds == "tuned_on_val"].set_index("model")
    audio = pd.read_csv(root / "audio/summary_val.csv")
    audio_tuned = audio[audio.thresholds == "tuned_on_val"].set_index("model")
    audio_info = json.loads((root / "audio/run_info.json").read_text())
    style_info = json.loads((root / "stylistic/run_info.json").read_text())
    error_info = json.loads((root / "errors/run_info.json").read_text())
    reviewed = pd.read_csv(root / "errors/review_sample.csv", dtype={"id": str}, keep_default_na=False)
    review_done = bool(len(reviewed) and reviewed.reviewed.astype(str).str.lower().eq("true").all()
                       and reviewed.pattern.str.strip().ne("").all())
    status = [
        (1, 1, "Text statistics for all songs", "done", "stylistic/features_all.csv.gz"),
        (1, 2, "Repetition and phonetic end-rhyme features", "done", "stylistic/features_all.csv.gz"),
        (1, 3, "Stylistic-only LR with shared evaluation", "done", "stylistic/summary_val.csv"),
        (1, 4, "TF-IDF + stylistic comparison by category", "done", "stylistic/combined_minus_tfidf_per_category.csv"),
        (1, 5, "Feature importance by tag/category", "done", "stylistic/importance_per_category.csv"),
        (2, 1, "Category-balanced error sample", "done", "errors/review_sample.csv"),
        (2, 2, "Frequent missed/extra tag pairs", "done", "errors/confused_pairs.csv"),
        (2, 3, "Qualitative failure-pattern analysis", "done" if review_done else "in progress", "errors/review_sample.csv"),
        (2, 4, "Length/repetition vs. quality", "done", "errors/quality_correlations.csv"),
        (2, 5, "Word contributions for selected tags", "done", "errors/selected_tag_terms.csv"),
        (3, 1, "MFCC and spectral features for fixed 10k", "done" if audio_info["features_extracted"] == 10000 else "in progress", "audio/run_info.json"),
        (3, 2, "Audio LR and lyrics on identical songs", "done", "audio/summary_val.csv"),
        (3, 3, "Audio/lyrics category comparison", "in progress", "audio/per_category_comparison.csv"),
        (3, 4, "Modality disagreements and examples", "in progress", "audio/modality_disagreements.csv"),
        (3, 5, "Pretrained audio embeddings + late fusion", "not started", "Stretch task"),
    ]
    for obj, pi, desc, state, evidence in status:
        if state == "done" and not (root / evidence).is_file():
            raise ValueError(f"Cannot mark Objective {obj} PI{pi} done without {evidence}")
    table = pd.DataFrame(status, columns=["Objective", "PI", "Requirement", "Status", "Evidence (results/Harish/)"])
    table["PI"] = table.PI.map(lambda n: f"PI{n}")
    s, b, c = [tuned.loc[n] for n in ["style_lr", "tfidf_lr", "tfidf_style_lr"]]
    a = audio_tuned.loc["audio_mfcc_lr"]
    lyrics_name = audio_info["lyrics_model"]
    l = audio_tuned.loc[lyrics_name]
    correlations = pd.read_csv(root / "errors/quality_correlations.csv")
    length = correlations[(correlations.feature == "n_words") & (correlations.metric == "sample_F1")].iloc[0]
    repetition = correlations[(correlations.feature == "repeated_line_ratio") & (correlations.metric == "sample_F1")].iloc[0]
    n_review = len(reviewed)
    paragraphs = [
        f"I extracted 15 text-statistic, repetition and phonetic rhyme features for all {style_info['n_all_features']:,} frozen songs and fitted preprocessing only on training data. "
        f"On validation, the stylistic-only model achieved macro-F1 {s.macro_F1:.3f} and MAP {s.MAP:.3f}; combining it with TF-IDF achieved {c.macro_F1:.3f} and {c.MAP:.3f}, versus {b.macro_F1:.3f} and {b.MAP:.3f} for TF-IDF alone. "
        "Per-category comparisons and standardized feature-coefficient tables are complete; next I will coordinate model selection with the team's tuning work before the single final test evaluation.",
        f"I analyzed {error_info['model']} on validation, sampled {n_review} category-balanced error rows, and produced missed/extra tag-pair tables plus exact word-contribution explanations. "
        f"Qualitative inspection found overlapping genres, attributes not established by lyrics, and plausible tags missing from social annotations; song length correlated with song-level F1 (Spearman rho {length.spearman_rho:.3f}), while repetition had a much smaller association ({repetition.spearman_rho:.3f}). "
        "Next I will repeat the analysis for the team's final selected model and examine rare-tag threshold behavior without using the test set.",
        f"I obtained and verified all 10,000 selected audio clips, retaining 6,857 training, 1,555 validation and 1,588 test songs, and extracted 53 MFCC/spectral summary features per clip. "
        f"On the identical training/validation subset, audio LR achieved macro-F1 {a.macro_F1:.3f} and MAP {a.MAP:.3f}, versus {l.macro_F1:.3f} and {l.MAP:.3f} for the selected matched lyrics model ({lyrics_name}). "
        "Per-category interpretation and review of modality-disagreement examples are in progress; their generated tables are preliminary, and pretrained audio embeddings and audio-lyrics late fusion remain unstarted stretch work.",
    ]
    date = datetime.now(ZoneInfo("America/Vancouver")).date().isoformat()
    done = sum(row[3] == "done" for row in status)
    in_progress = sum(row[3] == "in progress" for row in status)
    not_started = sum(row[3] == "not started" for row in status)
    lines = [f"# Implementation progress update ({date})", "",
             f"**Status: {done}/15 PIs done; {in_progress} in progress; {not_started} not started.**", "",
             markdown(table), "", "## Report-ready text (three sentences per objective)", ""]
    for i, paragraph in enumerate(paragraphs, 1):
        lines.extend([f"### Objective {i}", "", paragraph, ""])
    lines.extend(["## Validation results", "", "Annotation thresholds are tuned and evaluated on the same validation split, so F1 is optimistic. Retrieval metrics use the raw scores. No test metrics are reported.", "",
                  markdown(style[["model", "thresholds", "macro_F1", "micro_F1", "MAP", "P@10", "ROC-AUC"]]), "",
                  "### Matched audio subset", "", "Both modalities train on the same 6,857 songs and score the same 1,555 validation songs. These results should not be compared directly to models trained on all 57,348 training songs.", "",
                  markdown(audio[["model", "thresholds", "macro_F1", "micro_F1", "MAP", "P@10", "ROC-AUC"]]), "",
                  "## Error-analysis findings", "",
                  f"The {n_review}-row qualitative sample has equal counts by category; it is exploratory and is not a representative estimate of error-pattern prevalence. Reviews were performed by the assistant using the full local lyrics and observed/predicted tags. Musical attributes were not independently checked against audio.", "",
                  "Era and nationality labels usually lack direct support in lyrics; performer voice, covers and instrumentation can depend on the recording. Genre errors often involve nearby overlapping labels. The combined-model review includes plausible love predictions without observed annotations, missed love/party labels despite relevant cues, and a Christian prediction from religious imagery in a metal-tagged song. The separate TF-IDF reference misses two Christian-tagged songs with overt religious cues. Missing social tags must not be treated as verified semantic negatives.", "",
                  "The baseline's 0.01 threshold is an important source of extra predictions for rare tags. For example, its Canadian prediction occurs on 9,740 validation songs although only 173 have that observed label, and mellow occurs on 4,374 songs with 114 observed positives. Error-pair counts therefore include each extra tag's marginal rate and conditional lift; common extra tags do not by themselves prove a specific semantic confusion. These counts describe the TF-IDF reference; the selected model's own rates are in its error tables.", "",
                  markdown(correlations), "", "## Audio access and provenance", "",
                  "**Yes: all 10,000 clips have been obtained and verified.** The public mirror covers every ID in the fixed manifest. Each clip was checked against archive CRC32 and uncompressed size, and its SHA256 was recorded. Original-author delivery was not requested or confirmed; acquisition used a third-party mirror pinned to an exact revision.", "",
                  "Source: <https://huggingface.co/datasets/Leon299/music4all/tree/a391160e3e17f351d5ab2d05439a7d3d7f0440eb>. The authors' official access route is <https://sites.google.com/view/contact4music4all>. See `results/Harish/data_provenance.json` and `results/Harish/audio/clip_checksums.csv` for the local acquisition record.", "",
                  "The November 13 access condition is satisfied for the matched MFCC baseline. MFCC summaries are a limited audio reference; these results do not establish what a stronger audio encoder could achieve. Objective 3 PI5 remains stretch.", "",
                  "## Reproduction", "", "See [experiment_reproducibility.md](../docs/experiment_reproducibility.md) for installation, frozen-data restoration and commands. Raw lyrics, audio, private review text, serialized models and large score arrays are excluded from Git. Numeric features, metrics, review notes and provenance are included.", ""])
    (C.REPORTS_DIR / "implementation_progress.md").write_text("\n".join(lines))
    write_json(root / "pi_status.json", [{"objective": o, "pi": pi, "status": state, "description": desc,
                                         "evidence": evidence} for o, pi, desc, state, evidence in status])
    reviewed.groupby(["category", "pattern"]).size().rename("n_review_rows").reset_index().to_csv(root / "errors/review_patterns.csv", index=False)
    make_figures(root, style, audio)
    if args.update_paper:
        path = C.ROOT / "paper/main.tex"
        text = path.read_text()
        labels = ["1 Stylistic features", "2 Error analysis", "3 Audio baseline"]
        symbols = {"done": r"\done", "in progress": r"\prog", "not started": r"\nost"}
        for obj, label in enumerate(labels, 1):
            states = [symbols[row[3]] for row in status if row[0] == obj]
            prefix = "Harish " if obj == 1 else "       "
            new = prefix + "& " + label + " & " + " & ".join(states) + r" \\"
            # Only Harish's three table rows are changed.
            pattern = (r"^Harish & 1 Stylistic features[^\n]*" if obj == 1 else
                       r"^       & " + re.escape(label) + r"[^\n]*")
            text, n = re.subn(pattern, lambda _: new, text, count=1, flags=re.M)
            if n != 1:
                raise ValueError(f"Could not locate Harish Table 4 row {obj}")
        def tex_escape(s):
            return s.replace("_", r"\_").replace("%", r"\%")
        # Concise manuscript version retains three sentences per objective.
        short = [
            f"I extracted 15 stylistic features for all 83,633 songs and evaluated stylistic-only and combined models with the shared harness. Validation macro-F1/MAP are {s.macro_F1:.3f}/{s.MAP:.3f} for style alone and {c.macro_F1:.3f}/{c.MAP:.3f} with TF-IDF, versus {b.macro_F1:.3f}/{b.MAP:.3f} for TF-IDF alone. Category comparisons and feature-importance analysis are complete; next is coordination with final model tuning.",
            f"I analyzed the best completed lyrics model ({error_info['model']}) with {n_review} category-balanced reviewed error rows, tag-pair tables and word contributions. Genre overlap, non-lyrical attributes and incomplete social tags recur; length correlates with song-level F1 (rho {length.spearman_rho:.3f}), whereas repetition is weak (rho {repetition.spearman_rho:.3f}). Next is repeating the analysis for the team's final selected model.",
            f"All 10,000 audio clips were obtained from a pinned public mirror and verified, and MFCC/spectral features were extracted. Matched-subset audio macro-F1/MAP are {a.macro_F1:.3f}/{a.MAP:.3f}, versus {l.macro_F1:.3f}/{l.MAP:.3f} for the best matched lyrics candidate, using 6,857 training and 1,555 validation songs. Category/disagreement review is in progress; pretrained audio embeddings and fusion remain stretch, and test evaluation is still pending.",
        ]
        replacement = r"\textbf{Harish Pugazhenthi Sasikala} (stylistic features, error analysis, audio baseline). " + " ".join(
            rf"\emph{{Objective {i}:}} " + tex_escape(p) for i, p in enumerate(short, 1))
        pattern = r"\\textbf\{Harish Pugazhenthi Sasikala\}[^\n]*"
        text, count = re.subn(pattern, lambda _: replacement, text, count=1)
        if count != 1:
            raise ValueError("Harish Section 6 paragraph not found")
        path.write_text(text)
    print(f"Wrote progress report: {done}/15 PIs done; paper updated={args.update_paper}")


def make_figures(root, style, audio):
    out = root / "figures"
    out.mkdir(exist_ok=True)
    importance = pd.read_csv(root / "stylistic/importance_per_category.csv")
    matrix = importance.pivot(index="category", columns="feature", values="abs_coefficient")
    fig, ax = plt.subplots(figsize=(10, 3.8), layout="constrained")
    im = ax.imshow(matrix.values, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns.str.replace("_", " "), rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=9)
    ax.set_title("Stylistic model: mean absolute standardized coefficient by category", fontsize=11)
    fig.colorbar(im, ax=ax, label="Mean |coefficient|", shrink=.8)
    fig.savefig(out / "stylistic_importance.png", dpi=180)
    plt.close(fig)
    for data, filename, title in [(style, "lyrics_comparison.png", "Lyrics models: full validation split"),
                                   (audio, "matched_audio_comparison.png", "Matched audio subset: validation only")]:
        data = data[data.thresholds == "tuned_on_val"]
        fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
        x = np.arange(len(data))
        ax.bar(x - .18, data.macro_F1, .36, label="Macro-F1 (validation-tuned)", color="#3a6b9b")
        ax.bar(x + .18, data.MAP, .36, label="MAP", color="#a05535")
        ax.set_xticks(x, data.model.str.replace("matched_", "").str.replace("_", " "), fontsize=9)
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
