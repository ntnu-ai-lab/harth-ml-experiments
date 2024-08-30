# Notes for HARChildren Experiments
The training (grid search and leave-one-subject-out cross-validation) is done using the 12 activities: walking, running, shuffling, stairs (ascending), stairs (descending), standing, sitting, lying, bending, cycling (sit), cycling (stand), and jumping.
Thus, samples with the transition class are removed from training.
After the leave-one-subject-out cross-validation, the shuffling and bending classes are combined with the standing class, both stair walking classes are combined with the walking class, and both cycling classes are combined to a single cycling class, in the confusion matrices for the final evaluation.
This results in 7 activities in total: walking, running, standing, sitting, lying, cycling, and jumping.

Thus, for reproducibility of the paper results, these class merging should be applied to the confusion matrices before calculating the performance metrics.
