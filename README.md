# AI Gym Tracker

**AI Gym Tracker** is a real-time exercise analysis system designed to enhance workout feedback and performance monitoring. It leverages **MediaPipe** for accurate pose detection, a **Bidirectional LSTM** model for exercise classification, and **rule-based logic** using thresholds and local minima to assess the quality of each repetition. To ensure accessibility and ease of use, the entire system is integrated into a **user-friendly web application**.

**Todo:** Insert Screenshot of Web App with Sekelton & Classification here


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

The following sections describe the main building blocks of the code and pipeline, from data acquisition to the final web app.

### 4.1 Datasets & Pre-Processing
```css
gym_tracking/
├── data/               # different dataset, functions for pre-processing & validation
│   ├── aquisition      # script to download datasets from kaggle
|   ├── pre-processing  # functions to extract frames from videos, crop videos, balance dataset, ...
|   ├── processed       # final datasets with mediapipe extraxted keypoints stored as CSVs
├── ...
```

**TODO Hannes:** Section about the different Datasets, how we processed the Data & how we got to our final DS.

#### 4.1.1 Our Dataset (from video acquisitions)
Data acquisition:
1) We used three phone webcams in a fixed reference system
2) We considered different subjects
3) Every subject executed different possible exercises (split squat, squat, push-up, pull-up, lat pull-down, bench press)
4) Every subject executed multiple series (correct and wrong execution) of the same exercise

At the end of the acquisition part, we obtained videos for each subject and for each exercise executed. Inside each video, a lot of repetitions of the same movement were present.

Pre-processing:
1) manual cutting of the videos in single repetitions 
2) splitting of the videos of the single ripetitions in segments of 30 consequetive frames, in order to reduce the computational load.

These 30 consecutive frames for all the repetitions done will be the input for the classification and evaluation model.


### 4.2 Exercise Classification
```css
gym_tracking/       
├── ...
├── models/             # versions of the trained models (lstm & cnn) for exercise classification
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

**TODO:** Transfer stuff from edoardos docx file here, but less detailes


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
- Coluccino Edoardo ()
- Freddo Sara (10740747)
- Girolami Francesca ()
- Vogel Hannes (11109115)
