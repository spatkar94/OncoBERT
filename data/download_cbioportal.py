#!/usr/bin/env python3
"""Download public cBioPortal studies listed in ``public_studies.txt``.

The downloader retrieves archives but does not harmonize response labels; the
benchmark READMEs document the manuscript label definitions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
from urllib.request import urlretrieve


BASE_URL = "https://datahub.assets.cbioportal.org/{study}.tar.gz"


def safe_extract(handle: tarfile.TarFile, destination: Path) -> None:
    """Extract an archive after rejecting path traversal entries."""
    root = destination.resolve()
    members = []
    for member in handle.getmembers():
        target = (destination / member.name).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe archive member: {member.name}")
        members.append(member)
    handle.extractall(destination, members=members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path,
                        default=Path(__file__).with_name("public_studies.txt"))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "raw" / "cbioportal")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    studies = [line.strip() for line in args.studies.read_text().splitlines()
               if line.strip() and not line.startswith("#")]
    for study in studies:
        archive = args.output / f"{study}.tar.gz"
        if not archive.exists():
            print(f"Downloading {study}", flush=True)
            urlretrieve(BASE_URL.format(study=study), archive)
        if args.extract:
            destination = args.output / study
            destination.mkdir(exist_ok=True)
            with tarfile.open(archive, "r:gz") as handle:
                safe_extract(handle, destination)


if __name__ == "__main__":
    main()
