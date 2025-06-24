# Datasest Structure

The data directory is organized for model training as follows:  
Single-frame segments (`1_frame_segments`) were initially used to train a CNN model, while multi-frame segments (`30_frame_segments`) were later utilized to train an LSTM model.

```css
gym_tracking/       
├── ...
├── data/
│   ├── ...
│   ├── processed                       # final datasets processed to CSVs
│   │   ├── combined_DS                 # combined dataset from the outher subdirectories
│   │   │   ├── ...
│   │   │   ├── v3                      # v3: final version of the combined dataset
│   │   │   │   ├── 1_frame_segments    # CSV with mediapipe skeleton landmarks from 1 video frame
│   │   │   │   ├── 30_frame_segments   # CSV with mediapipe skeleton landmarks from 30 consequetive video frames
│   │   ├── fit3D_DS
│   │   │   ├── 30_frame_segments
│   │   ├── kaggle_DS
│   │   │   ├── 30_frame_segments
│   │   ├── own_DS
│   │   │   ├── 1_frame_segments
│   │   │   ├── 30_frame_segments
├── ...
```

As the data is sensitive and not for commercial use, it is not tracked in git. For more information or to request access, please send a message via GitHub.