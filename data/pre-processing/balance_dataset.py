import random
import pandas as pd
from pathlib import Path

def flip_keypoints_horizontally_csv(path: Path) -> Path:
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

    out_path = path.with_name(path.stem + '_flipped' + path.suffix)
    df.to_csv(out_path, index=False)
    print(f'Saved flipped skeletons to: {out_path}')
    return out_path

def get_total_count(folder: Path) -> int:
    """Count all CSV files (including flipped) in the folder."""
    return len(list(folder.glob('*.csv')))

def balance_class(folder: Path, max_files: int) -> int:
    """
    Balance one class folder to exactly max_files total CSVs by
    - augmenting via horizontal flips of '_o_' then '_k_' files, and
    - if in excess, deleting random '_k_' originals.
    Returns the new total count.
    """
    current = get_total_count(folder)
    delta = max_files - current

    # --- AUGMENT if we're below the target ---
    if delta > 0:
        # candidates for flipping (_o_ first)
        origs = [
            p for p in folder.glob('*.csv')
            if '_o_' in p.stem and '_flipped' not in p.stem
        ]
        ks = [
            p for p in folder.glob('*.csv')
            if '_k_' in p.stem and '_flipped' not in p.stem
        ]

        to_flip_o = min(delta, len(origs))
        for p in random.sample(origs, to_flip_o):
            flip_keypoints_horizontally_csv(p)
        delta -= to_flip_o
        if to_flip_o:
            print(f"  Augmented {to_flip_o} '_o_' files")

        to_flip_k = min(delta, len(ks))
        for p in random.sample(ks, to_flip_k):
            flip_keypoints_horizontally_csv(p)
        delta -= to_flip_k
        if to_flip_k:
            print(f"  Augmented {to_flip_k} '_k_' files")

        if delta > 0:
            print(f"  Warning: shortage of candidates, {delta} more needed")

    # --- PRUNE if we're above the target ---
    elif delta < 0:
        # only delete from original '_k_' files (not flips)
        k_files = [
            p for p in folder.glob('*.csv')
            if '_k_' in p.stem and '_flipped' not in p.stem
        ]
        n_delete = min(len(k_files), abs(delta))
        for p in random.sample(k_files, n_delete):
            p.unlink()
            print(f"  Deleted {p.name}")
        delta += n_delete
        if delta < 0:
            print(f"  Warning: still {abs(delta)} excess files (no more '_k_' originals)")

    return get_total_count(folder)

if __name__ == '__main__':
    combined_DS_path = Path("data/processed/combined_DS/v3/30_frame_segments")

    # discover classes
    classes = [d for d in combined_DS_path.iterdir() if d.is_dir()]
    print("Found classes:", [d.name for d in classes])

    # initial counts
    initial_counts = {cls.name: get_total_count(cls) for cls in classes}
    for name, cnt in initial_counts.items():
        print(f"  {name}: {cnt} files")

    # compute the maximum
    max_files = max(initial_counts.values())
    max_files = 800
    print("\nBalancing all classes to", max_files, "files each...\n")

    # balance each
    final_counts = {}
    for cls in classes:
        print(f"Class {cls.name}:")
        new_cnt = balance_class(cls, max_files)
        final_counts[cls.name] = new_cnt
        print(f"  New total: {new_cnt}\n")

    # summary
    if all(cnt == max_files for cnt in final_counts.values()):
        print("Perfectly balanced dataset!")
    else:
        print("Dataset still unbalanced:")
        for name, cnt in final_counts.items():
            if cnt != max_files:
                print(f"  {name}: {cnt} files (target {max_files})")
