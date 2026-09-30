"""Write the download manifest, optionally download, and verify raw resources."""

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

import pandas as pd

from .common import sha256_file
from .config import CATALOG_CONFIG, CatalogConfig


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec - integrity check required by upstream deposit
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path, timeout: int, chunk_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "JumpNicheRewire-evidence-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
        while chunk := response.read(chunk_bytes):
            output.write(chunk)
    temporary.replace(path)


def run_catalog(config: CatalogConfig = CATALOG_CONFIG) -> Path:
    config.manifest_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    selected = set(config.selected_resource_ids)
    for item in config.resources:
        destination = config.raw_root / item.evidence_track / item.filename
        requested = item.resource_id in selected
        if config.download_enabled and requested and not destination.exists():
            print(f"[download] {item.resource_id} -> {destination}")
            _download(item.url, destination, config.timeout_seconds, config.chunk_bytes)
        state = "not_selected"
        observed_bytes: int | None = None
        observed_md5 = ""
        observed_sha256 = ""
        if requested:
            state = "missing"
            if destination.exists():
                observed_bytes = destination.stat().st_size
                observed_md5 = _md5(destination) if item.md5 else ""
                observed_sha256 = sha256_file(destination)
                size_ok = item.expected_bytes is None or observed_bytes == item.expected_bytes
                md5_ok = not item.md5 or observed_md5.lower() == item.md5.lower()
                state = "verified" if size_ok and md5_ok else "verification_failed"
        rows.append({
            "resource_id": item.resource_id, "evidence_track": item.evidence_track,
            "selected": requested, "required": item.required, "filename": item.filename,
            "url": item.url, "source_page": item.source_page, "destination": str(destination),
            "expected_bytes": item.expected_bytes, "observed_bytes": observed_bytes,
            "expected_md5": item.md5, "observed_md5": observed_md5, "state": state,
            "observed_sha256": observed_sha256,
            "purpose": item.purpose,
        })
    frame = pd.DataFrame(rows)
    catalog_path = config.manifest_root / "public_resource_catalog.csv"
    status_path = config.manifest_root / "public_resource_status.csv"
    frame.drop(columns=["observed_bytes", "observed_md5", "observed_sha256", "state"]).to_csv(catalog_path, index=False)
    frame.to_csv(status_path, index=False)
    lines = [
        "# 独立证据 benchmark：手动下载清单", "",
        "默认 `download_enabled=False`，请用浏览器打开链接，保持文件名并保存到 Destination。",
        "若要由 Python 下载，把 `evidence_benchmark/config.py` 中 `download_enabled=True`。", "",
    ]
    for track, group in frame.groupby("evidence_track", sort=False):
        lines.extend([f"## {track}", ""])
        for row in group.itertuples(index=False):
            lines.extend([
                f"### {row.resource_id}", "",
                f"- 状态：`{row.state}`", f"- 用途：{row.purpose}",
                f"- 下载：{row.url}", f"- 来源页：{row.source_page}",
                f"- 保存到：`{row.destination}`", "",
            ])
    manual_path = config.manifest_root / "MANUAL_DOWNLOAD.md"
    manual_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[catalog] {manual_path}")
    return status_path



