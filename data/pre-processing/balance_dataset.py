import os
import pandas as pd
import numpy as np
from pathlib import Path
import random

def flip_keypoints_horizontally_csv(path: str) -> str:
    """
    Read a CSV of MediaPipe keypoints (one row per frame), flip horizontally,
    swap left/right landmark blocks, and save to a new CSV file.

    Args:
        path: Path to the input CSV file.

    Returns:
        Path to the saved flipped CSV file.
    """
    df = pd.read_csv(path)
    # Flip x coordinates
    x_cols = [c for c in df.columns if c.startswith('x_')]
    for col in x_cols:
        df[col] = 1 - df[col]
    # Define mirror pairs
    mirror_pairs = [
        (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
        (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
        (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
    ]
    cols = df.columns.tolist()
    for left, right in mirror_pairs:
        l_start, r_start = left * 4, right * 4
        left_block = cols[l_start:l_start+4]
        right_block = cols[r_start:r_start+4]
        temp = df[left_block].copy()
        df[left_block] = df[right_block]
        df[right_block] = temp
    out_path = Path(path).with_name(Path(path).stem + '_flipped' + Path(path).suffix)
    df.to_csv(out_path, index=False)
    print(f'Saved flipped skeletons to: {out_path}')
    return str(out_path)


if __name__ == '__main__':
    combined_DS_path = Path("data/processed/combined_DS/30_frame_segments")
    classes = ["pull_up", "push_up", "split_squat"]

    # Count files per class
    num_files_dict = {cls: len(list((combined_DS_path/cls).glob('*.csv'))) for cls in classes}
    for cls, count in num_files_dict.items():
        print(f"{cls}: {count} files")

    split_count = num_files_dict["split_squat"]
    pull_count = num_files_dict["pull_up"]
    push_count = num_files_dict["push_up"]

    # Balance logic
    if split_count > pull_count and split_count > push_count:
        # Add flipped files to pull_up and push_up
        for cls in ["pull_up", "push_up"]:
            cls_path = combined_DS_path / cls
            current = len(list(cls_path.glob('*.csv')))
            needed = split_count - current
            print(f"\nClass {cls}: current {current}, target {split_count}, need {needed} flips")

            # 1) Flip from original ('_o_') files (exclude already-flipped outputs)
            if needed > 0:
                orig_candidates = [
                    p for p in cls_path.glob('*.csv')
                    if '_o_' in p.name 
                    and '_flipped' not in p.name
                    and not (cls_path / f"{p.stem}_flipped{p.suffix}").exists()
                ]
                to_flip = min(needed, len(orig_candidates))
                if to_flip > 0:
                    for p in random.sample(orig_candidates, to_flip):
                        flip_keypoints_horizontally_csv(p)
                    needed -= to_flip
                    print(f"Flipped {to_flip} original (_o_) files, still need {needed}")

            # 2) Flip from 'k' files if still needed (exclude already-flipped outputs)
            if needed > 0:
                k_candidates = [
                    p for p in cls_path.glob('*.csv')
                    if '_k_' in p.name
                    and '_flipped' not in p.name
                    and not (cls_path / f"{p.stem}_flipped{p.suffix}").exists()
                ]
                to_flip_k = min(needed, len(k_candidates))
                if to_flip_k > 0:
                    for p in random.sample(k_candidates, to_flip_k):
                        flip_keypoints_horizontally_csv(p)
                    needed -= to_flip_k
                    print(f"Flipped {to_flip_k} k files, still need {needed}")

            if needed > 0:
                print(f"Warning: only satisfied part of the deficit for {cls}, {needed} more flips needed (no more candidates).")

    elif split_count < pull_count and split_count < push_count:
        # Delete excess flipped-k files from pull_up and push_up
        target = split_count
        print(f"\nBalancing dataset to {target} files per class...")
        for cls in ["pull_up", "push_up"]:
            cls_path = combined_DS_path / cls
            all_csvs = list(cls_path.glob('*.csv'))
            total = len(all_csvs)
            excess = total - target
            if excess <= 0:
                print(f"{cls}: {total} ≤ {target}, no deletion needed.")
                continue

            # Deletion candidates: files containing both '_k_' and '_flipped'
            candidates = [p for p in all_csvs if '_k_' in p.name and '_flipped' in p.name]
            if not candidates:
                print(f"{cls}: No flipped-k files to delete, but need to remove {excess} files.")
                continue

            to_remove = min(excess, len(candidates))
            print(f"{cls}: total {total}, target {target}, deleting {to_remove} of {len(candidates)} candidates.")
            for p in random.sample(candidates, to_remove):
                try:
                    p.unlink()
                    print(f"Deleted: {p.name}")
                except Exception as e:
                    print(f"Error deleting {p.name}: {e}")

            new_total = len(list(cls_path.glob('*.csv')))
            print(f"New total for {cls}: {new_total}")

    else:
        if split_count == pull_count == push_count:
            print("\nDataset is already perfectly balanced!")
            print(f"Each class has {split_count} files")
        else:
            print("\nUnexpected imbalance:")
            for cls, count in num_files_dict.items():
                print(f"{cls}: {count} files")
