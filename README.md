# AI Gym Tracker

**AI Gym Tracker** is a real-time exercise analysis system designed to enhance workout feedback and performance monitoring. It leverages **MediaPipe** for accurate pose detection, a **Bidirectional LSTM** model for exercise classification, and **rule-based logic** using thresholds and local minima to assess the quality of each repetition. To ensure accessibility and ease of use, the entire system is integrated into a **user-friendly web application**.

<img src="assets/web_app_1.png" alt="Gym Tracker Skeleton" width="70%">
<img src="assets/web_app_2.png" alt="Gym Tracker Classification & Evaluation" width="70%">


## 1. Prerequisites
- Python 3.8+ (tested with Python 3.11.4)
- Dependencies listed in ```requirements.txt```
- Modern browser with WebRTC support (Chrome recommended for optimal functioning with Mediapipe)
- ngrok (optional, for public WebApp exposure)
- Machine with multi-core processor (Intel i5/AMD Ryzen 5 or higher) with minimum 4GB RAM
- Working webcam: integrated or external with minimum recommended resolution of 640x480 or higher


## 2. Installation
### Clone the project from Github
```bash
git clone https://github.com/hannesvgel/gym_tracking.git
```

### Install Dependencies
Create a virtual environment
```bash
cd gym_tracking
python -m venv .venv
```

Activate the venv
- Windows 
    ```bash
    .venv\Scripts\activate
    ```
- macOS/Linux
    ```bash
    source .venv/bin/activate
    ```

Install the needed Dependencies
```bash
pip install -r requirements.txt
```


## 3. Usage
To start the web app, navigate to the project root and run:
```bash
python app.py
```
The server will start on http://localhost:8000 and automatically open the browser;
If it doesn't open automatically, click on the link http://localhost:8000 shown in the terminal


## 4. Project Structure & Purpose
**Project goal:** Development of a system to descriminate different gym exercises and the correctness of execution. Two models were created, one able to discriminate between 3 different exercises and the other one between 6 exercises. This was done in order to make the classification task more difficult for the model: in fact, the last three exercises were paired to the first three because of a similar execution. The paired exercises are: squat and split squat, push-up and bench press, pull-up and lat pull-down. In this way the model created is better in discriminate between different movements.
Then, the evaluation of the performance correctness is done using some parameters like, for example, the joint angle of a certain articulation.
A web app has been finally created as interface between our models and the user.

Below is a visualization of the project structure, along with explanations of its key components:
```css
gym_tracking/           # root
├── data/               # different dataset, functions for pre-processing & validation
├── models/             # versions of the trained models (lstm & cnn) for exercise classification
├── notebooks/          # simple notebooks to quickly test models & validate extracted skeletons
├── src/                # code for training models, evaluating exercises & realime inference
├── static/             # web application frontend
├── app.py              # main flask server
├── config.yaml         # config containing models, classes and paths to the datasests
├── README.md       
├── requirements.txt    # required dependencies
```

The following subsections describe the main building blocks of the code and pipeline, from data acquisition to the final web app.


### 4.1 Datasets & Pre-Processing
```css
gym_tracking/
├── data/               # different dataset, functions for pre-processing & validation
│   ├── aquisition      # script to download datasets from kaggle
|   ├── pre-processing  # functions to extract frames from videos, crop videos, balance dataset, ...
|   ├── processed       # final datasets with mediapipe extraxted keypoints stored as CSVs
├── ...
```

All datasets and data processing are managed within the `data` folder, which contains the following subdirectories:
- ``aquisition`` holds a simple script used to download the [workout videos](https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video) and [workout images](https://www.kaggle.com/datasets/hasyimabdillah/workoutexercises-images) datasets from Kaggle.
- `pre-processing` contains various helper functions used for: splitting videos into CSVs of 30 frames with MediaPipe skeleton landmarks for LSTM training, converting 30-frame CSVs into single-frame CSVs for CNN training, combining and balancing different datasets, and more.
- `processed` contains the finalized datasets, including both 30-frame and single-frame CSVs, as well as a combined dataset used for training the final LSTM model.

Our raw data consisted of exercise videos. We used [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=de) for pose estimation, extracting skeletons from each frame. Each skeleton is represented by 33 landmarks, each with x, y, z coordinates and a visibility value. For training our LSTM model, we used sequences of 30 consecutive frames of these landmarks to capture temporal information.

The following subsections explain how each individual dataset — our [own](https://drive.google.com/drive/folders/1H9mTOb1FsfSzcLXhqJkY6SCRLRxz_LK-), the [Kaggle dataset](https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video), and the [Fit3D dataset](https://fit3d.imar.ro/) — were acquired, processed, and ultimately combined into the final dataset.

#### 4.1.1 Our Dataset
**Data Acquisition:**
1) We used three phone webcams in a fixed reference system
2) We considered different subjects
3) Every subject executed different possible exercises (split squat, bulgarian squat, push-up, pull-up, lat pull-down, bench press)
4) Every subject executed multiple series (correct and wrong execution) of the same exercise

**Description:** At the end of the acquisition phase, we had collected videos for each subject and each exercise performed. Each video contained multiple repetitions of the same movement. The final Exercises were split squat, bulgarian squat, push-up, pull-up, lat pull-down & bench press.


#### 4.1.2 Kaggle Workout Videos Dataset
**Data Acquisition:** simply download the Dataset from Kaggle by running ``data\aquisition\download_kaggle_ds.py``.

**Description:** This dataset consists of workout videos organized by exercise type, with each folder named after the corresponding workout. The videos are in `.mp4` format, sourced primarily from YouTube and supplemented with original recordings. Video resolution and duration vary, but each video contains at least one repetition of the exercise.

#### 4.1.3 Fit3D Dataset
**Data Acquisition:** Download the dataset from [https://fit3d.imar.ro/download](https://fit3d.imar.ro/download). Access must be requested, but if one uses a university email for academic purposes, approval is typically granted almost immediately.

**Description:**  
The Fit3D dataset is a large-scale collection designed for 3D human pose, shape, and motion analysis in fitness training. It contains over 3 million images and corresponding 3D ground-truth data, covering 47 exercises performed by both instructors and trainees. The dataset features recordings from multiple subjects using a marker-based motion capture system, captured from four camera angles at 900x900 resolution and 50 fps. Each recording includes segmented exercise repetitions, 3D skeletons with 25 joints, and aligned mesh models (GHUM and SMPLX). For our application we utilized the raw video recordings.

#### 4.1.4 Pre-Processing
The pre-processing steps involved were similar for all datasets and included:
1) Manually trimming the videos to include only repetitions of a single exercise, removing any introductory or concluding sequences from each video.
2) Processing the videos into CSV files, where each file contains segments of 30 consecutive frames of MediaPipe skeleton landmarks. This is handled by ``data\pre-processing\video_seg_processing.py``.

#### 4.1.5 Combining the Datasets
Each of the three datasets covered different exercises, with some overlap between them. To create a balanced final dataset, we selected samples from all three sources. We excluded the Bulgarian squat from our dataset and replaced it with the standard squat from the other datasets, as the Bulgarian squat proved problematic during training and classification—primarily due to its sensitivity to camera angle, which was a result of being consistently recorded from the same viewpoint during data acquisition. As a result, our final dataset focuses on the following exercises:
- bench press
- lat machine
- pull up
- push up
- split squat
- squat

The illustration below shows how the final dataset is constructed from the different sources:

<img src="assets/datasets.drawio.svg" alt="Datasets combined" width="100%">





### 4.2 Exercise Classification
```css
gym_tracking/       
├── ...
├── models/                 # versions of the trained models (lstm & cnn) for exercise classification
├── notebooks/              # simple notebooks to quickly test models & validate extracted skeletons
├── src/                    # code for training models, evaluating exercises & realime inference
│   ├── ...
│   ├── models              # train functions for DL models
│   ├── realtime inference  # scripts for realtime exercise classification
├── config.yaml             # config containing models, classes and paths to the datasests
├── ...
```

**TODO Hannes:** Section about the exercise classification & realtime inference


### 4.3 Exercise Evaluation & Web App
```css
gym_tracking/
├── ...
├── models/             # versions of the trained models (lstm & cnn) for exercise classification
├── static/             # web application frontend
├── app.py              # main flask server
├── config.yaml         # config containing models, classes and paths to the datasests
├── ...
```


#### 4.3.1 ngrok installation
Download ngrok from https://ngrok.com/ and register in order to obtain an authtoken, which has to be configured:
```css
ngrok config add-authtoken YOUR_AUTHTOKEN
```
Run then app.py. Open a terminal in the section where you extracted ngrok and enter the commands:
```css
.\ngrok.exe authtoken YOUR_TOKEN # To activate the public tunnel
.\ngrok.exe http http://localhost:8000 # To generate the public link
```
ngrok will provide a random public URL (e.g. https://e0b2-5-90-143-228.ngrok-free.app) accessible from any device.


#### 4.3.2 Webb App functioning
After launching the WebApp and opening the browser at the reference link, a series of elements will appear:
1) Camera selection: The user chooses a camera from the dropdown menu (if multiple cameras are available);
2) Video display: The video feed from the selected camera appears in the main section;
3) Start Session: resets counters, variables and starts the real-time analysis session;
4) Pose detection: The system starts tracking the user's posture, showing a green and red colored "skeleton" displayed in a box below the video
5) Exercise recognition: After collecting 30 consecutive frames of the user's posture, the system automatically identifies the exercise with the highest confidence, displaying its name in the "Current Exercise" section
6) Real-Time Feedback:
    - Form analysis: The system continuously evaluates by analyzing each single frame the execution correctness and shows in the "Form Analysis" section:
        - Green: "Correct Execution!" when the exercise is executed correctly;
        - Orange: Specific warnings to correct posture;
        - Gray: Neutral state;
    - Audio feedback: Vocal warnings, reporting the Orange state message, are played when necessary, to allow the user to correct themselves during training;
    - Confirmation sounds: A high-pitched sound confirms each correct repetition;
7) Progress Monitoring ReportExercise History:
    - The system counts correct repetitions for each exercise and saves them in a list that updates automatically;
    - Warning History: All warnings are recorded with timestamps in the "Warning History" section;
    - Session Status: Display of the number of processed frames and session status;
9) Recording Functions
    - Start Recording: The user can start recording poses during training;
    - Stop recording: stops recording;
    - Play Recording: Possibility to review recorded movements. In the same box where real-time pose detection is produced, now a blue and purple skeleton is produced that reproduces all recorded movements to have visual feedback of the quality of executions performed (before pressing "Play Recording" you need to press "stop recording" and "end session");
    - Stop playback: Stop playback of saved poses;
    - End Session: ends the analysis.


#### 4.3.3 Common problem resolution
1) "Model not available"
    - ErrorVerify that lstm_bidir_6cl.h5 is present in the main directory;
    - Check that TensorFlow is installed correctly.
3) Webcam not detected
    - Grant camera permissions to the browser if requested;
    - Try a different browser (Chrome recommended for optimal Mediapipe functioning);
    - Verify that no other application is using the webcam;
    - Try selecting another camera from the list of available ones.
5) Slow Performance
    - Close other applications that intensively use CPU/GPU because MediaPipe + TensorFlow can consume a lot of CPU;
    - Use Chrome for optimal performance with MediaPipe;
    - Modify the modelComplexity parameter in the Mediapipe configuration:
        - modelComplexity: 0: computationally lighter but less accurate;
        - modelComplexity: 1: balanced;
        - modelComplexity: 2: very accurate but computationally heavy.
7) Installation and ngrok Connection Issues
    - Allow Windows Defender Firewall to install the ngrok.exe file in case it gets blocked due to antivirus conflicts;
    - Verify internet connection;
    - Restart ngrok if the tunnel becomes unstable;
    - Ngrok with free account has traffic limitations. Sessions expire after 2 hours, after this time it's necessary to reload the site page.
9) Classification and movement quality analysis
The quality of analysis depends on lighting and framing:
    - For optimal functioning of classification and rule-based algorithm, it is recommended to frame the left profile of the subject during execution of Squat, Split Squat, Push Up, Lat Pull-down and frame the right profile of the subject during execution of Pull Up and Bench Press;
    - To increase the quality of Mediapipe detection, it is advisable to position yourself in a well-lit environment, wear tight-fitting clothes, have the most uniform background possible and with a strong contrast compared to the subject.


## 5. Limitations
**TODO**
- Video acquisition quality
- Joints extraction accuracy
- Limited dataset
- We had to change some exercises along the way and then used online dataset/make data augmentations
- Model misclassifications 
- Model Limited to predefined classes


## 6. Future Developments
**TODO**
- A larger dataset is needed to make the model more robust - more repetitions and more subjects
- 3D reconstruction from multiple point of view
- A more accurate model for the skeleton extraction
- Enhance the number of physical exercises that the model can recognize and discriminate
- Increase the number of parameters for a more complete quality evaluation
- ...


## 5. Authors
- Coluccino Edoardo (10630619)
- Freddo Sara (10740747)
- Girolami Francesca (10689084)
- Vogel Hannes (11109115)
