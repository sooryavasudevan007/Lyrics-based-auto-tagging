"""Scientific invariants and isolated end-to-end tests for the experiment pipelines."""
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
import importlib.util
import struct
import zlib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.audio_features import fixed_subset, summarize_waveform, SAMPLE_RATE, AUDIO_FEATURE_NAMES
from src.data import Split
from src.error_analysis import confused_pairs, sample_errors, song_quality
from src.experiment_utils import ensure_train_val, fit_lr, load_scores, save_scores
from src.stylistic import extract_one, extract_features, FEATURE_NAMES


class FeatureTests(unittest.TestCase):
    def test_repetition_and_rhyme_are_distinct(self):
        f = dict(zip(FEATURE_NAMES, extract_one("Into the night\nFollow the light\nInto the night\nFollow the light")))
        self.assertEqual(f["n_lines"], 4)
        self.assertEqual(f["repeated_line_ratio"], .5)
        self.assertEqual(f["chorus_repetition"], 1)
        self.assertEqual(f["end_rhyme_rate"], 1)
        same = dict(zip(FEATURE_NAMES, extract_one("night\nnight\nnight")))
        self.assertEqual(same["end_rhyme_rate"], 0)

    def test_empty_and_unknown_words_are_finite(self):
        for text in ["", "[Chorus]", "qzxzqv\nxxvzkq", "!!!"]:
            self.assertTrue(np.isfinite(extract_one(text)).all())
        with self.assertRaises(ValueError):
            extract_features(["a", "b"], ["same", "same"])

    def test_waveform_features_measure_frequency(self):
        t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
        low = summarize_waveform(.2 * np.sin(2 * np.pi * 220 * t))
        high = summarize_waveform(.2 * np.sin(2 * np.pi * 2200 * t))
        self.assertEqual(len(low), len(AUDIO_FEATURE_NAMES))
        i = AUDIO_FEATURE_NAMES.index("centroid_mean")
        self.assertLess(low[i], high[i])
        self.assertTrue(np.isfinite(summarize_waveform(np.zeros(SAMPLE_RATE))).all())


class ProtocolTests(unittest.TestCase):
    def split(self, ids=("a", "b"), artists=("one", "two")):
        return Split(list(ids), list(artists), ["text"] * len(ids),
                     np.array([[1, 0], [0, 1]]), ["rock", "sad"])

    def test_scores_align_by_id_and_reject_tag_mismatch(self):
        split = self.split()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "scores.npz"
            save_scores(p, split, [[.8, .1], [.2, .9]], [.4, .5])
            reverse = Split(["b", "a"], ["two", "one"], ["text", "text"], split.Y[::-1], split.tags)
            score, th = load_scores(p, reverse)
            np.testing.assert_allclose(score, [[.2, .9], [.8, .1]])
            reverse.tags = ["sad", "rock"]
            with self.assertRaises(ValueError):
                load_scores(p, reverse)

    def test_artist_and_id_leakage_rejected(self):
        train = self.split()
        with self.assertRaises(ValueError):
            ensure_train_val(train, self.split(("c", "d"), (" ONE ", "three")))
        with self.assertRaises(ValueError):
            ensure_train_val(train, self.split(("b", "d"), ("three", "four")))

    def test_fixed_subset_is_order_independent_and_proportional(self):
        data = pd.DataFrame({"id": [f"{i:016d}" for i in range(100)],
                             "artist": [f"artist{i}" for i in range(100)],
                             "split": ["train"] * 70 + ["val"] * 15 + ["test"] * 15})
        a = fixed_subset(data, 20)
        b = fixed_subset(data.sample(frac=1, random_state=2), 20)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.split.value_counts().to_dict(), {"train": 14, "test": 3, "val": 3})

    def test_multilabel_confusions_are_fn_fp_events(self):
        true = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
        pred = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 1]])
        pairs = confused_pairs(true, pred, ["a", "b", "c"])
        self.assertEqual({(r.missed_tag, r.extra_tag): r.count for r in pairs.itertuples()},
                         {("a", "b"): 1, ("b", "c"): 1})
        q = song_quality(np.zeros((1, 2)), np.zeros((1, 2)))
        self.assertTrue(np.isnan(q.sample_F1.iloc[0]))

    def test_sparse_model_fits_noncanonical_matrix(self):
        X = csr_matrix(np.array([[1., 2.], [0., 3.], [1., 0.], [0., 0.]]))
        X.has_sorted_indices = False
        fitted = fit_lr(X, np.array([[1, 0], [0, 1], [1, 0], [0, 1]]))
        self.assertEqual(fitted.predict_proba(X).shape, (4, 2))

    def test_archive_entry_spanning_volumes_checks_integrity(self):
        spec = importlib.util.spec_from_file_location("fetch_audio", ROOT / "scripts/fetch_audio_subset.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = b"small audio fixture" * 100
        comp = zlib.compressobj(wbits=-15)
        packed = comp.compress(payload) + comp.flush()
        name = b"audios/test.mp3"
        header = struct.pack("<4s5H3L2H", b"PK\x03\x04", 20, 0, 8, 0, 0,
                             zlib.crc32(payload), len(packed), len(payload), len(name), 0) + name
        blob = header + packed + b"X" * 600
        obj = object.__new__(module.RemoteArchive)
        obj.volumes, obj.sizes = ["a", "b"], {"a": 20, "b": len(blob) - 20}
        pieces = {"a": blob[:20], "b": blob[20:]}
        obj.range = lambda volume, start, length: pieces[volume][start:start + length]
        entry = {"flags": 0, "method": 8, "crc32": zlib.crc32(payload),
                 "compressed": len(packed), "size": len(payload), "disk": 0, "offset": 0}
        self.assertEqual(obj.extract(entry), payload)
        with self.assertRaises(ValueError):
            obj.extract({**entry, "crc32": 0})


class EndToEndTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg needed for audio integration test")
    def test_all_commands_use_isolated_fixture_and_leave_test_unscored(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            processed, splits, out, audio = [d / s for s in ["processed", "splits", "output", "audios"]]
            for p in [processed, splits, audio]: p.mkdir()
            rows = []
            for i in range(60):
                split = "train" if i < 36 else "val" if i < 48 else "test"
                happy = i % 2 == 0
                words = "guitar electric metal thunder" if happy else "tears sorrow sadness lonely"
                text = f"{words} night\n{words} light\n" * (2 + i % 3)
                rows.append({"id": f"{i:016d}", "artist": f"artist{i}", "split": split,
                             "split_song_level": split, "lyrics": text, "tags": ["rock" if happy else "sad"]})
                if split != "test":
                    t = np.arange(int(SAMPLE_RATE * 1.1)) / SAMPLE_RATE
                    samples = (.2 * np.sin(2 * np.pi * (220 if happy else 1100) * t) * 32767).astype("<i2")
                    with wave.open(str(audio / f"{i:016d}.wav"), "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(samples.tobytes())
            with gzip.open(processed / "songs.jsonl.gz", "wt") as f:
                for row in rows: f.write(json.dumps(row) + "\n")
            pd.DataFrame(rows)[["id", "artist", "split", "split_song_level"]].to_csv(splits / "splits.csv", index=False)
            (splits / "vocab.json").write_text(json.dumps(["rock", "sad"]))
            pd.DataFrame({"tag": ["rock", "sad"], "category": ["genre", "mood/emotion"]}).to_csv(splits / "tag_categories.csv", index=False)
            base = [sys.executable, str(ROOT / "scripts/run_experiments.py")]
            flags = ["--processed-dir", str(processed), "--splits-dir", str(splits),
                     "--output-dir", str(out), "--subset-size", "60", "--audio-dir", str(audio),
                     "--bootstrap", "5", "--per-category", "2", "--workers", "2"]
            for command in ["style", "errors", "audio"]:
                r = subprocess.run(base + [command] + flags, capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            for path in out.rglob("run_info.json"):
                self.assertFalse(json.loads(path.read_text())["test_evaluated"])
            self.assertTrue((out / "errors/review_sample.csv").exists())
            self.assertTrue((out / "audio/modality_disagreements.csv").exists())
            access = json.loads((out / "audio/access_status.json").read_text())
            self.assertEqual(access["available_all"], 48)  # test audio is unnecessary for validation
            sample = pd.read_csv(out / "errors/review_sample.csv", dtype={"id": str})
            self.assertFalse(set(sample.id) & {r["id"] for r in rows if r["split"] == "test"})
            # Incomplete feature tables cannot silently shrink the fixed sample.
            bad_features = out / "audio/features_all.csv.gz"
            r = subprocess.run(base + ["audio"] + flags + ["--audio-features", str(bad_features)],
                               capture_output=True, text=True, timeout=60)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("exact fixed IDs", r.stderr)
            # Complete numeric tables can reproduce models with no audio files.
            feature_table = pd.read_csv(bad_features, dtype={"id": str})
            extra = []
            for row in rows:
                if row["split"] == "test":
                    x = feature_table.iloc[0].copy()
                    x["id"] = row["id"]
                    extra.append(x)
            full_features = d / "complete_fixture_features.csv.gz"
            pd.concat([feature_table, pd.DataFrame(extra)], ignore_index=True).to_csv(full_features, index=False)
            shutil.rmtree(audio)
            reused_out = d / "reused"
            r = subprocess.run(base + ["audio"] + flags + ["--audio-features", str(full_features),
                                                         "--output-dir", str(reused_out)],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            info = json.loads((reused_out / "audio/run_info.json").read_text())
            self.assertEqual(info["features_source"], "precomputed")
            self.assertEqual(info["available_all"], 0)
            self.assertFalse(info["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
