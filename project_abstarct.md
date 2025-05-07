## GYM TRACKER PROJECT

## Name of the involved students;
Coluccino Edoardo, Freddo Sara, Girolami Francesca, Vogel Hannes

## Project goal(s);
Develop a system to descriminate different gym exercises and the correct level of execution

## Sensor(s) need.
Phone webacams 

## Health data you plan to measure and analyse;
For each exercise we want to evaluate one common error of execution in order to classify the performance correctness. For example, if we analyze the push-up movement, we would like the head to move above the hands line; so: 
-	above the line = correct execution
-	under the line = wrong execution

## Pipeline:
1.	Data acquisition
       a.	3 phone webcams in a fixed reference system
       b.	Different subjects
       c.	5 possible exercises (bulgarian squat, squat, push-up, pull-up, deadlift)
       d.	Multiple series (correct and wrong execution)
2.	Pre-processing: from the aquired videos, we want to extract the joints coordinates over time taken from the best point of view (best camera), which will be given as input to the first AI model. This model will classify which exercise the subject is performing.
3.	Quality evaluation: another model will evaluate the performance of the execution
4.	Future goal:
       a.	reconstruct the joint coordinates using all the three cameras instead of just one for a better reconstruction.
       b.	Other parameters evaluation as output


