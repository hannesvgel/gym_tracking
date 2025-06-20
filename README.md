# AI Gym Tracker
Coluccino Edoardo, Freddo Sara, Girolami Francesca, Vogel Hannes
# Project goal:
Development of a system to descriminate different gym exercises and the correct level of execution.
For each exercise we created a model which recognize the type of exercize executed by the subject in front of the camera and evaluate the performance correctness. A web app has been created as interface between our models and the user.
6 movements are recognizable by the model: squat, bulgarian squat, push-up, pull-up, lat machine and bench press.
For each movement, some parameters has been used in order to evaluate the correctness of the execution, like, for example, the joint angle of a certain articulation.
# Pipeline:
1. Data acquisition
    a. 3 phone webcams in a fixed reference system
    b. Different subjects
    c. 6 possible exercises (bulgarian squat, squat, push-up, pull-up, lat machine, bench press)
    d. Multiple series (correct and wrong execution)
2. Pre-processing:
    a.cutting of the videos in single repetitions
    b. from the acquired videos, we want to extract the joints coordinates over time taken from the best point of view (best camera), which will be given as input to the first AI model. This model will classify which exercise the subject is performing.
4. Quality evaluation: another model will evaluate the performance of the execution
5. Future goal:
    a. reconstruct the joint coordinates using all the three cameras instead of just one for a better reconstruction.
    b. Other parameters evaluation as output

# Requirements
- readme report
- final github project
- exam:
    - 10 min presentation (keep it simple, real-problem/need -> approach -> results with numbers -> main challenges -> next steps)
    - live demo (5 min)
    - questions (5min)
- project:
    - data aquisition
    - data analysis/ pre-processing
    - ML
    - GUI
