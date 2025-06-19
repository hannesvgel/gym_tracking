import numpy as np
import pandas as pd


def flip_keypoints_horizontally(keypoints: list) -> np.array:

    sequence_flipped = []

    # Create flipped keypoints by negating x coordinates
    keypoints_flipped = keypoints.copy()
    for i in range(0, len(keypoints_flipped), 4):  # Step by 4 because each point has x,y,z,visibility
        keypoints_flipped[i] = 1-keypoints_flipped[i]  # Flip x coordinate
    
    sequence_flipped.append(keypoints_flipped)

    # change indicies of certain landmarks after flipping (e.g. left elbow vs right elbow)
    # list of (left, right) index pairs
    '''
    0 - nose
    1 - left eye (inner)
    2 - left eye
    3 - left eye (outer)
    4 - right eye (inner)
    5 - right eye
    6 - right eye (outer)
    7 - left ear
    8 - right ear
    9 - mouth (left)
    10 - mouth (right)
    11 - left shoulder
    12 - right shoulder
    13 - left elbow
    14 - right elbow
    15 - left wrist
    16 - right wrist
    17 - left pinky
    18 - right pinky
    19 - left index
    20 - right index
    21 - left thumb
    22 - right thumb
    23 - left hip
    24 - right hip
    25 - left knee
    26 - right knee
    27 - left ankle
    28 - right ankle
    29 - left heel
    30 - right heel
    31 - left foot index
    32 - right foot index
    '''
    mirror_pairs = [
        ( 1,  4), ( 2,  5), ( 3,  6),
        ( 7,  8), ( 9, 10), (11, 12),
        (13, 14), (15, 16), (17, 18),
        (19, 20), (21, 22), (23, 24),
        (25, 26), (27, 28), (29, 30),
        (31, 32)
    ]

    for left_idx, right_idx in mirror_pairs:
        # compute the start positions in the flat list
        l0, r0 = left_idx*4, right_idx*4
        # swap the 4 values for each landmark
        for offset in range(4):
            keypoints_flipped[l0+offset], keypoints_flipped[r0+offset] = (
                keypoints_flipped[r0+offset],
                keypoints_flipped[l0+offset]
            )

    array_flipped = np.array(sequence_flipped).reshape(1, 30, 132)

    return array_flipped

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
