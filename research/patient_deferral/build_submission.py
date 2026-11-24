"""Build the arXiv source archive from a fixed, data-free file list."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "manuscript"
FILES = ("main.tex", "results.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission" / "arxiv-source.zip")
    args = parser.parse_args()
    entries = {name: (MANUSCRIPT / name).read_bytes() for name in FILES}
    with ZipFile(args.output, "w", compression=ZIP_STORED) as archive:
        for name, content in entries.items():
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
