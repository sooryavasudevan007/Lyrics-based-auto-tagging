"""Fetch only the fixed subset from a public, multipart Music4All mirror.

This is a third-party mirror, not the original authors' distribution endpoint.
Pin the revision and check the archive CRC32 and uncompressed size of every clip.
HTTP ranges avoid downloading the full ~52 GB audio archive. No audio is committed.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import struct
import sys
import threading
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment_utils import sha256, write_json

REPO = "Leon299/music4all"
REVISION = "a391160e3e17f351d5ab2d05439a7d3d7f0440eb"


class RemoteArchive:
    def __init__(self, repo=REPO, revision=REVISION):
        self.repo, self.revision = repo, revision
        self.base = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}", timeout=30) as r:
            files = json.load(r)
        self.sizes = {x["path"]: x["size"] for x in files if x["type"] == "file"}
        self.volumes = sorted(n for n in self.sizes if n.startswith("audios.z") and n != "audios.zip") + ["audios.zip"]
        if not self.volumes or self.volumes[-1] not in self.sizes:
            raise ValueError("Audio archive volumes are missing")
        self._urls, self._lock = {}, threading.Lock()
        self._client = None
        try:
            import httpx
            self._client = httpx.Client(follow_redirects=True, timeout=60,
                                        limits=httpx.Limits(max_connections=32, max_keepalive_connections=32))
            self._http_errors = (httpx.HTTPError,)
        except ImportError:
            self._http_errors = ()

    def range(self, volume, start, length):
        if length <= 0 or start < 0 or start + length > self.sizes[volume]:
            raise ValueError("Invalid archive range")
        end = start + length - 1
        last_error = None
        for attempt in range(4):
            try:
                # A resolved CDN URL can serve many range requests without a new HF redirect.
                with self._lock:
                    url = self._urls.get(volume)
                url = url or self.base + volume + "?download=true"
                expected = f"bytes {start}-{end}/{self.sizes[volume]}"
                if self._client is not None:
                    with self._client.stream("GET", url, headers={"Range": f"bytes={start}-{end}"}) as r:
                        if r.status_code != 206 or r.headers.get("Content-Range") != expected:
                            raise ValueError(f"Server did not honor exact range for {volume}")
                        data = r.read()
                        resolved = str(r.url)
                else:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        if r.status != 206 or r.headers.get("Content-Range") != expected:
                            raise ValueError(f"Server did not honor exact range for {volume}")
                        data, resolved = r.read(length + 1), r.geturl()
                if len(data) != length:
                    raise ValueError("Truncated or oversized range response")
                with self._lock:
                    self._urls[volume] = resolved
                return data
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, *self._http_errors) as exc:
                last_error = exc
                with self._lock:
                    self._urls.pop(volume, None)
                if attempt < 3:
                    time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(f"Failed range download for {volume}: {last_error}")

    def span(self, disk, offset, length):
        chunks = []
        while length:
            if disk >= len(self.volumes):
                raise ValueError("Archive entry extends beyond its volumes")
            name = self.volumes[disk]
            take = min(length, self.sizes[name] - offset)
            if take <= 0:
                raise ValueError("Invalid multipart archive offset")
            chunks.append(self.range(name, offset, take))
            length -= take
            disk += 1
            offset = 0
        return b"".join(chunks)

    def index(self):
        size = self.sizes["audios.zip"]
        tail = self.range("audios.zip", size - min(size, 65536), min(size, 65536))
        at = tail.rfind(b"PK\x05\x06")
        if at < 0:
            raise ValueError("ZIP end record missing")
        _, disk, cd_disk, _, _, cd_size, cd_offset, _ = struct.unpack_from("<4s4H2LH", tail, at)
        if cd_disk != disk or cd_disk != len(self.volumes) - 1:
            raise ValueError("Unsupported central-directory volume layout")
        data = self.range("audios.zip", cd_offset, cd_size)
        entries = {}
        offset = 0
        while data[offset:offset + 4] == b"PK\x01\x02":
            h = struct.unpack_from("<4s6H3L5H2L", data, offset)
            flags, method, crc, compressed, size = h[3], h[4], h[7], h[8], h[9]
            nlen, xlen, clen, disk = h[10:14]
            name = data[offset + 46:offset + 46 + nlen].decode("utf-8" if flags & 0x800 else "cp437")
            if name.endswith(".mp3"):
                sid = Path(name).stem
                if sid in entries:
                    raise ValueError("Duplicate audio IDs in archive")
                entries[sid] = {"name": name, "flags": flags, "method": method, "crc32": crc,
                                "compressed": compressed, "size": size, "disk": disk, "offset": h[-1]}
            offset += 46 + nlen + xlen + clen
        if not entries:
            raise ValueError("No audio entries found")
        return entries

    def extract(self, entry):
        if entry["flags"] & 1:
            raise ValueError("Archive is encrypted; obtain access from the dataset authors")
        # Local headers in this release are 85 bytes; allow variable extra fields.
        disk, offset = entry["disk"], entry["offset"]
        remaining = sum(self.sizes[n] for n in self.volumes[disk:]) - offset
        data = self.span(disk, offset, min(entry["compressed"] + 512, remaining))
        if data[:4] != b"PK\x03\x04":
            raise ValueError("Local ZIP header signature mismatch")
        header = struct.unpack_from("<4s5H3L2H", data)
        start = 30 + header[-2] + header[-1]
        if header[2] & 1 or header[3] != entry["method"]:
            raise ValueError("Local ZIP header disagrees with archive index")
        if start + entry["compressed"] > len(data):
            data = self.span(disk, offset, start + entry["compressed"])
        payload = data[start:start + entry["compressed"]]
        if entry["method"] == 8:
            decoded = zlib.decompress(payload, -15)
        elif entry["method"] == 0:
            decoded = payload
        else:
            raise ValueError("Unsupported ZIP compression method")
        if len(decoded) != entry["size"] or zlib.crc32(decoded) & 0xffffffff != entry["crc32"]:
            raise ValueError("Audio size/CRC verification failed")
        return decoded


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("results/Harish/audio_subset.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/music4all/audios"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--record-prefix", default="audio")
    args = ap.parse_args()
    if not 1 <= args.workers <= 32:
        ap.error("Use 1-32 download workers")
    if not re.fullmatch(r"[a-zA-Z0-9_]+", args.record_prefix):
        ap.error("Record prefix must contain only letters, digits, or underscores")
    manifest = pd.read_csv(args.manifest, dtype=str)
    if manifest.id.duplicated().any() or not manifest.id.str.fullmatch(r"[A-Za-z0-9]{16}").all():
        raise ValueError("Expected unique 16-character Music4All IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = RemoteArchive(revision=args.revision)
    index = archive.index()
    missing = set(manifest.id) - set(index)
    if missing:
        raise ValueError(f"Archive does not contain {len(missing)} required subset IDs")
    print(f"Archive covers all {len(manifest)} fixed IDs; starting checked downloads", flush=True)
    def download(sid):
        entry, target = index[sid], args.output_dir / f"{sid}.mp3"
        if target.exists():
            data = target.read_bytes()
            valid = len(data) == entry["size"] and zlib.crc32(data) & 0xffffffff == entry["crc32"]
        else:
            valid = False
        if not valid:
            data = archive.extract(entry)
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as f:
                temp = Path(f.name)
                f.write(data)
            try:
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
        return {"id": sid, "bytes": len(data), "crc32": entry["crc32"], "sha256": __import__("hashlib").sha256(data).hexdigest()}
    rows, failed = [], []
    t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, sid): sid for sid in manifest.id}
        for f in concurrent.futures.as_completed(futures):
            try:
                rows.append(f.result())
            except Exception as exc:
                failed.append({"id": futures[f], "error": str(exc)})
            if (len(rows) + len(failed)) % 250 == 0:
                print(f"Verified {len(rows)}/{len(manifest)}; failures {len(failed)}; {time.perf_counter() - t:.0f}s", flush=True)
    pd.DataFrame(rows).sort_values("id").to_csv(args.output_dir.parent / f"{args.record_prefix}_checksums.csv", index=False)
    write_json(args.output_dir.parent / f"{args.record_prefix}_download.json", {
        "source_repository": f"https://huggingface.co/datasets/{REPO}", "source_revision": args.revision,
        "source_type": "third-party mirror", "subset_sha256": sha256(args.manifest),
        "required": len(manifest), "verified": len(rows), "failed": failed,
        "seconds": time.perf_counter() - t, "verification": "ZIP CRC32 + uncompressed size; per-file SHA256 recorded"})
    if failed:
        raise SystemExit(f"{len(failed)} clips failed. Rerun to resume; verified clips are retained.")
    print(f"All {len(rows)} fixed-subset clips verified", flush=True)


if __name__ == "__main__":
    main()
