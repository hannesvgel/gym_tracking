import numpy as np
import pandas as pd
import tensorflow as tf


def flip_keypoints_horizontally(keypoints: tf.Tensor) -> tf.Tensor:
    """
    keypoints: tf.Tensor of shape (frames, 132)
    Assumes x coordinates are every 4th column starting at 0,
    and that mirror_pairs is a list of (left_block_idx, right_block_idx).
    """
    frames = tf.shape(keypoints)[0]
    dim = tf.shape(keypoints)[1]

    # 1) Flip all x-coords (every 4th column starting at 0):
    x_cols = tf.range(0, dim, 4)                       # [0, 4, 8, …]
    rows = tf.range(frames)
    rows_mat, cols_mat = tf.meshgrid(rows, x_cols, indexing='ij')
    idxs = tf.stack([rows_mat, cols_mat], axis=-1)     # shape [frames, num_x, 2]
    idxs_flat = tf.reshape(idxs, [-1, 2])              # shape [frames*num_x, 2]
    orig_x = tf.gather_nd(keypoints, idxs_flat)
    flipped_x = 1.0 - orig_x
    keypoints = tf.tensor_scatter_nd_update(keypoints, idxs_flat, flipped_x)

    # 2) Swap each left/right block of 4 columns:
    mirror_pairs = [
        (1, 4), (2, 5), (3, 6),
        (7, 8), (9, 10), (11, 12),
        (13, 14), (15, 16), (17, 18),
        (19, 20), (21, 22), (23, 24),
        (25, 26), (27, 28), (29, 30),
        (31, 32)
    ]
    for l_blk, r_blk in mirror_pairs:
        for offset in range(4):
            l_col = l_blk * 4 + offset
            r_col = r_blk * 4 + offset

            # build 2-D indices for this single column
            cols_l = tf.fill([frames], l_col)
            cols_r = tf.fill([frames], r_col)
            rows = tf.range(frames)
            l_idxs = tf.stack([rows, cols_l], axis=1)  # [frames, 2]
            r_idxs = tf.stack([rows, cols_r], axis=1)

            # gather & swap
            l_vals = tf.gather_nd(keypoints, l_idxs)
            r_vals = tf.gather_nd(keypoints, r_idxs)
            keypoints = tf.tensor_scatter_nd_update(keypoints, l_idxs, r_vals)
            keypoints = tf.tensor_scatter_nd_update(keypoints, r_idxs, l_vals)

    return keypoints


def rotate_keypoints(keypoints: list) -> np.array:
    # Convert input to numpy array and reshape to (30, 132)
    keypoints_array = np.array(keypoints).reshape(30, 132)
    
    # Generate random rotation angle between -35 and 35 degrees
    angle = np.random.uniform(-35, 35)
    angle_rad = np.radians(angle)
    
    # Create rotation matrix for 2D rotation
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Initialize array for rotated keypoints
    rotated_keypoints = np.zeros_like(keypoints_array)
    
    # Rotate each frame
    for frame_idx in range(30):
        # Process each keypoint (x,y,z,visibility)
        for i in range(0, 132, 4):
            # Get x,y coordinates
            x, y = keypoints_array[frame_idx, i], keypoints_array[frame_idx, i+1]
            
            # Apply rotation to x,y coordinates
            rotated_coords = rotation_matrix @ np.array([x, y])
            
            # Store rotated coordinates and keep z, visibility unchanged
            rotated_keypoints[frame_idx, i] = rotated_coords[0]
            rotated_keypoints[frame_idx, i+1] = rotated_coords[1]
            rotated_keypoints[frame_idx, i+2] = keypoints_array[frame_idx, i+2]  # z
            rotated_keypoints[frame_idx, i+3] = keypoints_array[frame_idx, i+3]  # visibility
    
    # Reshape back to expected format (1, 30, 132)
    return rotated_keypoints.reshape(1, 30, 132)
