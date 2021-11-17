# HARTH Dataset and Machine Learning experiments
Baseline Machine Learning models for the Human Activity Recognition Trondheim (HARTH) dataset

## HARTH Dataset
The folder [harth](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/harth) contains the Human Activity Recognition Trondheim Dataset (HARTH). It consists of acceleration data of 22 subjects, which wore two three-axial [Axivity AX3 (Axivity Ltd., Newcastle, UK)](https://axivity.com/product/ax3) accelerometers on the thigh and lower back.

#### Setup
- Acceleration signals
- 2 three-axial [Axivity AX3](https://axivity.com/product/ax3) accelerometers
- Attached to: thigh and lower back

[comment]: <> (#### Recordings)
[comment]: <> (- TODO)

#### Activity Annotations

| Label| Activity                  |  Notes 				    |
|------|:--------------------------|:--------------------------------------:|
| 1    | walking                   | 			                    |
| 2    | running                   | 			                    |
| 3    | shuffling 		   | standing with leg movement             |
| 4    | stairs (ascending)        | 			                    |
| 5    | stairs (descending)       | 			                    |
| 6    | standing                  | 			                    |
| 7    | sitting                   | 			                    |
| 8    | lying                     | 			                    |
| 13   | cycling (sit)             | 			                    |
| 14   | cycling (stand)           | 			                    |
| 18   | transport (sit)  	   |e.g., in a car 			    |
| 19   | transport (stand)	   |e.g., in a bus or train 		    |

## Machine Learning Experiments
The folder [experiments](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments) contains all our experiments. It is possible to train a K-Nearest Neighbors, a Support Vector Machine, a Random Forest, an Extreme Gradient Boost, a Convolutional Neural Network, a Bidirectional Long Short-term Memory, and a CNN with multi-resolution blocks.
### Requirements
- Python 3.9.7+
```bash
cd experiments
pip install -r requirements.txt
```
### Usage
Start a model training using HARTH
```bash
cd experiments
./run_training.sh -a <model_name> -d <path/to/dataset>
# Example: ./run_training.sh -a xgb -d ../harth/
```
Each model can be configured using the corresponding config.yml file: [xgb](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments/traditional_machine_learning/params/xgb_50hz/), [svm](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments/traditional_machine_learning/params/svm_50hz/), [rf](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments/traditional_machine_learning/params/rf_50hz/), [knn](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments/traditional_machine_learning/params/knn_50hz/)
