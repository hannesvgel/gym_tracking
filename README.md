# AI Gym Tracker

# Project goal:
Development of a system to descriminate different gym exercises and the correctness of execution. Two models were created, one able to discriminate between 3 different exercises and the other one between 6 exercises. This was done in order to make the classification task more difficult for the model: in fact, the last three exercises were paired to the first three because of a similar execution. The paired exercises are: squat and split squat, push-up and bench press, pull-up and lat pull-down. In this way the model created is better in discriminate between different movements.
Then, the evaluation of the performance correctness is done using some paramteres like, for example, the joint angle of a certain articulation.
A web app has been finally created as interface between our models and the user.
# Pipeline:
Data acquisition
1) Three phone webcams in a fixed reference system
2) Different subjects
3) Different possible exercises (split squat, squat, push-up, pull-up, lat pull-down, bench press)
4) Multiple series (correct and wrong execution)

Pre-processing:
1) cutting of the videos in single repetitions
2) from the acquired videos, we want to extract the joints coordinates over time taken from the best point of view (best camera), which will be given as input to the first AI model. This model will classify which exercise the subject is performing.

Quality evaluation

Future goal:
1) reconstruct the joint coordinates using all the three cameras instead of just one for a better reconstruction.
2) Other parameters evaluation as output
3) more exercises under evaluation
# Repository structure

# Requirements

# Installation

# Execution

# Authors
Coluccino Edoardo, Freddo Sara, Girolami Francesca, Vogel Hannes
