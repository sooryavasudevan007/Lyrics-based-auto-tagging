"""Quick look at your Music4All folder: file names, columns, and sample rows.

    python scripts/inspect_m4a.py --m4a-dir /path/to/music4all
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--m4a-dir", required=True)
d = Path(ap.parse_args().m4a_dir)

print(f"Folder: {d}")
for p in sorted(d.iterdir()):
    if p.is_dir():
        files = list(p.iterdir())
        print(f"  [dir] {p.name}/  ({len(files):,} files, e.g. {files[0].name if files else '-'})")
    else:
        print(f"  {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
        if p.suffix in (".csv", ".tsv", ".txt"):
            with open(p, encoding="utf-8", errors="replace") as f:
                for _ in range(3):
                    line = f.readline().rstrip("\n")
                    print(f"      {line[:150]!r}")
