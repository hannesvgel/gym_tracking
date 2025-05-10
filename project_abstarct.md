# GYM TRACKER PROJECT

## Team Members
- Coluccino Edoardo
- Freddo Sara
- Girolami Francesca
- Vogel Hannes

## Project Goal
Develop a system to descriminate different gym exercises and the correct level of execution.

## Sensors Required
- Camera

## Health Data we plan to measure and analyse
For each exercise we want to evaluate one common error of execution in order to classify the performance correctness. 
For example, if we analyze the push-up movement, we would like the head to move above the hands line; so: 
-	above the line = correct execution
-	under the line = wrong execution

## Pipeline
1. Data acquisition  
    - 3 phone webcams in a fixed reference system (can be reduced to one)  
    - Different subjects  
    - 3 possible exercises (bulgarian squat, squat, push-up)  
    - Multiple series (correct and wrong execution)

2. Pre-processing: from the acquired videos, we want to extract the joints coordinates over time taken from the best point of view (best camera), which will be given as input to the first model.

3. Exercise Classification: A model will classify the exercise.

4. Quality evaluation: A model will evaluate the performance of the execution.

5. Future goal:  
    - Reconstruct the joint coordinates using all the three cameras instead of just one for a better reconstruction.  
    - Other parameters evaluation as output.



