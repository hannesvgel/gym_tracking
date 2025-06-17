import numpy as np

def flip_keypoints_horizontally(keypoints: list) -> np.array:

    sequence_flipped = []

    # Create flipped keypoints by negating x coordinates
    keypoints_flipped = keypoints.copy()
    for i in range(0, len(keypoints_flipped), 4):  # Step by 4 because each point has x,y,z,visibility
        keypoints_flipped[i] = 1-keypoints_flipped[i]  # Flip x coordinate
    
    sequence_flipped.append(keypoints_flipped)

    # change indicies of certain landmarks after flipping (e.g. left elbow vs right elbow)
    # Todo

    array_flipped = np.array(sequence_flipped).reshape(1, 30, 132)

    return array_flipped

def rotate_keypoints(keypoints: list) -> np.array:
    # Todo: rotate keypoints by a random angle: -35deg < alpha < 35deg
    return 0
