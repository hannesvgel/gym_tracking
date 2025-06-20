# AI Gym Tracker

# Project goal:
Development of a system to descriminate different gym exercises and the correct level of execution. Two models were created, one able to discriminate between 3 different exercises and the other one between 6 exercises. This was done in order to make the classification task more difficult for the model: in fact, the last three exercises were paired to the first three because of a similar execution. The paired exercises are: squat and bulgarian squat, push-up and bench press, pull-up and lat pull-down. In this way the model created is better in discriminate between different movements.
Then, the evaluattion of the performance correctness is done using some paramteres like, for example, the joint angle of a certain articulation.
A web app has been finally created as interface between our models and the user.
# Pipeline:
Data acquisition
    a. 3 phone webcams in a fixed reference system
    b. Different subjects
    c. 6 possible exercises (bulgarian squat, squat, push-up, pull-up, lat machine, bench press)
    d. Multiple series (correct and wrong execution)
Pre-processing:
    a. cutting of the videos in single repetitions
    b. from the acquired videos, we want to extract the joints coordinates over time taken from the best point of view (best camera), which will be given as input to the first AI model. This model will classify which exercise the subject is performing.
Quality evaluation: another model will evaluate the performance of the execution
Future goal:
    a. reconstruct the joint coordinates using all the three cameras instead of just one for a better reconstruction.
    b. Other parameters evaluation as output
    c. more exercises under evaluation
# Repository structure

# Requirements

# Installation

# Execution

# Authors
Coluccino Edoardo, Freddo Sara, Girolami Francesca, Vogel Hannes
