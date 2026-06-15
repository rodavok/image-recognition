"""Download and prepare CUB-200-2011 dataset."""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path


CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
DATA_DIR = Path(__file__).parent.parent / "data"


def download_file(url: str, dest: Path):
    """Download with progress using curl (handles servers that block urllib)."""
    import subprocess
    print(f"Downloading {url}")
    subprocess.run(
        ["curl", "-L", "-A", "Mozilla/5.0", "--progress-bar", "-o", str(dest), url],
        check=True,
    )


def extract_tarball(path: Path, dest: Path):
    """Extract tar.gz file."""
    print(f"Extracting {path}")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(dest)


def organize_dataset(raw_dir: Path, output_dir: Path, train_ratio: float = 0.8):
    """Organize into train/val splits by class folder."""
    images_dir = raw_dir / "CUB_200_2011" / "images"

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("Organizing into train/val splits...")

    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)

        images = sorted(class_dir.glob("*.jpg"))
        split_idx = int(len(images) * train_ratio)

        for img in images[:split_idx]:
            shutil.copy2(img, train_dir / class_name / img.name)
        for img in images[split_idx:]:
            shutil.copy2(img, val_dir / class_name / img.name)

        print(f"  {class_name}: {split_idx} train, {len(images) - split_idx} val")

    print(f"\nDataset ready at {output_dir}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tarball_path = DATA_DIR / "CUB_200_2011.tgz"
    raw_dir = DATA_DIR / "raw"

    # Download
    if not tarball_path.exists():
        download_file(CUB_URL, tarball_path)
    else:
        print(f"Tarball already exists: {tarball_path}")

    # Extract
    if not (raw_dir / "CUB_200_2011").exists():
        raw_dir.mkdir(exist_ok=True)
        extract_tarball(tarball_path, raw_dir)
    else:
        print("Already extracted")

    # Organize
    if not (DATA_DIR / "train").exists():
        organize_dataset(raw_dir, DATA_DIR)
    else:
        print("Dataset already organized")

    # Cleanup prompt
    print(f"\nOptional: remove tarball and raw files to save ~2.5GB:")
    print(f"  rm {tarball_path}")
    print(f"  rm -rf {raw_dir}")


if __name__ == "__main__":
    main()
