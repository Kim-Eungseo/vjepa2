"""Resize 512x512 PNG frames to 256x256 in background DR directories.

Scans ep000~ep999 under the given dataset directory for bg_* subdirectories,
finds any 512x512 images, and resizes them to 256x256 in-place using
cv2.INTER_AREA (best quality for downsampling).

Usage:
    python resize_frame.py \
        --dataset-dir /data1/maniskill/datasets/franka/LiftPegUpright-v1 \
        --workers 16

    # Dry-run (just count, don't modify):
    python resize_frame.py \
        --dataset-dir /data1/maniskill/datasets/franka/LiftPegUpright-v1 \
        --dry-run
"""
import os
import argparse
import cv2
from multiprocessing import Pool
from tqdm import tqdm


def find_512_images(dataset_dir):
    """Find all 512x512 PNG files inside bg_* directories."""
    targets = []
    for ep_name in sorted(os.listdir(dataset_dir)):
        ep_path = os.path.join(dataset_dir, ep_name)
        if not os.path.isdir(ep_path) or not ep_name.startswith("ep"):
            continue

        for dr_name in sorted(os.listdir(ep_path)):
            if not dr_name.startswith("bg_"):
                continue
            obs_dir = os.path.join(ep_path, dr_name, "obs")
            if not os.path.isdir(obs_dir):
                continue

            for fname in sorted(os.listdir(obs_dir)):
                if not fname.endswith(".png"):
                    continue
                fpath = os.path.join(obs_dir, fname)
                img = cv2.imread(fpath)
                if img is not None and img.shape[0] == 512 and img.shape[1] == 512:
                    targets.append(fpath)

    return targets


def resize_single(fpath):
    """Resize a single 512x512 image to 256x256 in-place."""
    img = cv2.imread(fpath)
    if img is None:
        return fpath, False
    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imwrite(fpath, resized)
    return fpath, True


def main():
    parser = argparse.ArgumentParser(
        description="Resize 512x512 background DR frames to 256x256"
    )
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Root dataset directory containing epNNN folders")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers (default: 16)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only scan and count, don't resize")
    args = parser.parse_args()

    print("Scanning for 512x512 images in bg_* directories...")
    targets = find_512_images(args.dataset_dir)
    print(f"Found {len(targets)} images to resize")

    if len(targets) == 0:
        print("Nothing to do!")
        return

    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")
        # Show a few examples
        for p in targets[:10]:
            print(f"  Would resize: {p}")
        if len(targets) > 10:
            print(f"  ... and {len(targets) - 10} more")
        return

    print(f"Resizing with {args.workers} workers...")
    if args.workers <= 1:
        results = []
        for fpath in tqdm(targets, desc="Resizing"):
            results.append(resize_single(fpath))
    else:
        with Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(resize_single, targets),
                total=len(targets),
                desc="Resizing",
            ))

    success = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"\nDone! Resized {success} images, {failed} failures")


if __name__ == "__main__":
    main()
