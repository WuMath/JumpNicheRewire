"""Install official simulator source without replacing it by local reimplementation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import zipfile

from .common import sha256_file, write_json
from .config import INSTALL_CONFIG, InstallConfig


def _r_package_version(executable: str, package: str) -> str:
    """Return an installed R package version without using Windows' locale codec."""

    completed = subprocess.run(
        [executable, "-e", f"cat(as.character(packageVersion('{package}')))"],
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def run_install(config: InstallConfig = INSTALL_CONFIG) -> Path:
    config.external_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"SERGIO": None, "sCCIgen": None, "spatialperturbseq": None}
    if config.install_sergio_source:
        if not config.sergio_archive.exists():
            raise FileNotFoundError(
                f"Missing {config.sergio_archive}. Run run_evidence_download.py first or download it manually."
            )
        destination = config.external_root / "SERGIO"
        archive_sha256 = sha256_file(config.sergio_archive)
        marker = config.external_root / "SERGIO_source_sha256.txt"
        if destination.exists() and not marker.exists():
            raise FileExistsError(
                f"{destination} already exists without a source marker. Rename it or verify it manually before install."
            )
        if marker.exists() and marker.read_text(encoding="utf-8").strip() != archive_sha256:
            raise FileExistsError(
                "Downloaded SERGIO archive changed but an older installed source exists; preserve the old run and use a new external_root."
            )
        if not destination.exists():
            with tempfile.TemporaryDirectory(prefix="SERGIO_extract_", dir=config.external_root) as temporary_name:
                temporary = Path(temporary_name)
                with zipfile.ZipFile(config.sergio_archive) as archive:
                    archive.extractall(temporary)
                roots = [item for item in temporary.iterdir() if item.is_dir()]
                if len(roots) != 1:
                    raise ValueError(f"Unexpected SERGIO archive layout: {roots}")
                roots[0].replace(destination)
            marker.write_text(archive_sha256 + "\n", encoding="utf-8")
        print(f"[installed source] SERGIO -> {destination}")
        manifest["SERGIO"] = {
            "source_archive": str(config.sergio_archive.resolve()),
            "archive_sha256": archive_sha256,
            "installed_to": str(destination.resolve()),
        }
    if config.install_sccigen_with_r:
        expression = (
            "if (!requireNamespace('Giotto', quietly=TRUE)) "
            "stop('sCCIgen imports the optional Giotto package. Install Giotto first, or set "
            "InstallConfig.install_sccigen_with_r=False when using the official pre-simulated example.');"
            "if (!requireNamespace('remotes', quietly=TRUE)) install.packages('remotes', repos='https://cloud.r-project.org');"
            f"remotes::install_github('{config.sccigen_repository}', upgrade='never');"
            "if (!requireNamespace('sCCIgen', quietly=TRUE)) stop('sCCIgen installation did not complete')"
        )
        subprocess.run([config.rscript_executable, "-e", expression], check=True)
        version = _r_package_version(config.rscript_executable, "sCCIgen")
        manifest["sCCIgen"] = {"repository": config.sccigen_repository, "package_version": version}
        print(f"[installed R package] sCCIgen {version}")
    if config.install_spatialperturbseq_with_r:
        expression = (
            "if (!requireNamespace('remotes', quietly=TRUE)) install.packages('remotes', repos='https://cloud.r-project.org');"
            f"remotes::install_github('{config.spatialperturbseq_repository}', upgrade='never')"
        )
        subprocess.run([config.rscript_executable, "-e", expression], check=True)
        version = _r_package_version(config.rscript_executable, "spatialperturbseq")
        manifest["spatialperturbseq"] = {
            "repository": config.spatialperturbseq_repository, "package_version": version,
        }
        print("[installed R package] spatialperturbseq")
    write_json(config.external_root / "simulator_install_manifest.json", manifest)
    return config.external_root



