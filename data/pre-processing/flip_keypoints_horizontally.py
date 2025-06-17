import pandas as pd
import numpy as np


def flip_keypoints_horizontally_csv(path: str) -> str:
    """
    Read a CSV of MediaPipe keypoints (one row per frame), flip horizontally,
    swap left/right landmark blocks, and save to a new CSV file.

    Args:
        path: Path to the input CSV file.

    Returns:
        Path to the saved flipped CSV file.
    """
    # Load data
    df = pd.read_csv(path)

    # 1) Flip x-coordinates (every column starting with 'x_')
    x_cols = [c for c in df.columns if c.startswith('x_')]
    for col in x_cols:
        df[col] = 1 - df[col]

    # 2) Define left/right landmark index pairs (0-based landmark IDs)
    mirror_pairs = [
        (1, 4),   # inner eye
        (2, 5),   # eye
        (3, 6),   # outer eye
        (7, 8),   # ear
        (9, 10),  # mouth corners
        (11, 12), # shoulder
        (13, 14), # elbow
        (15, 16), # wrist
        (17, 18), # pinky
        (19, 20), # index finger
        (21, 22), # thumb
        (23, 24), # hip
        (25, 26), # knee
        (27, 28), # ankle
        (29, 30), # heel
        (31, 32)  # foot index
    ]

    # 3) Swap blocks of 4 columns for each left/right pair
    # Each landmark i has columns: [x_i, y_i, z_i, vis_i] in that order
    cols = df.columns.tolist()
    for left, right in mirror_pairs:
        # Calculate column slice indices
        l_start = left * 4
        r_start = right * 4
        left_block = cols[l_start:l_start+4]
        right_block = cols[r_start:r_start+4]

        # Perform swap by copying
        temp = df[left_block].copy()
        df[left_block] = df[right_block]
        df[right_block] = temp

    # 4) Save to new file
    out_path = path.replace('.csv', '_flipped.csv')
    df.to_csv(out_path, index=False)
    print(f'Saved flipped skeletons to: {out_path}')
    return out_path