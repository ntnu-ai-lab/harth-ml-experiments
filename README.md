# HAR/SWR Datasets and Machine Learning experiments
Baseline Machine Learning models for Human Activity Recognition (HAR) and Sleep Wakefulness Recognition (SWR) using the Human Activity Recognition Trondheim (HARTH), the Human Activity Recognition 70+ (HAR70+), and the DualSleep datasets, proposed and used in our papers: [HARTH: A Human Activity Recognition Dataset for Machine Learning](https://doi.org/10.3390/s21237853), [A Machine Learning Classifier for Detection of Physical Activity Types and Postures During Free-Living](https://doi.org/10.1123/jmpb.2021-0015), [Validation of an Activity Type Recognition Model Classifying Daily Physical Behavior in Older Adults: The HAR70+ Model](https://doi.org/10.3390/s23052368), and [A Machine Learning Model for Predicting Sleep and Wakefulness Based on Accelerometry, Skin Temperature and Contextual Information](https://doi.org/10.2147/NSS.S452799).


## HARTH Dataset
The folder [harth](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/harth) contains the Human Activity Recognition Trondheim Dataset (HARTH). It consists of acceleration data of 22 subjects, which wore two three-axial [Axivity AX3 (Axivity Ltd., Newcastle, UK)](https://axivity.com/product/ax3) accelerometers on the thigh and lower back. The dataset is also uploaded to the [UC Irvine Machine Learning Repository](https://doi.org/10.24432/C5NC90).

#### Setup
- Acceleration signals
- 2 three-axial [Axivity AX3](https://axivity.com/product/ax3) accelerometers
- Attached to: thigh and lower back

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
| 130  | cycling (sit, inactive)   | cycling (sit) without leg movement     |
| 140  | cycling (stand, inactive) | cycling (stand) without leg movement   |

## HAR70+ Dataset
The folder [har70plus](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/har70plus) contains the Human Activity Recognition 70+ (HAR70+) dataset. It consists of acceleration data of 18 older-adult subjects, which wore two three-axial [Axivity AX3 (Axivity Ltd., Newcastle, UK)](https://axivity.com/product/ax3) accelerometers on the thigh and lower back. The dataset is also uploaded to the [UC Irvine Machine Learning Repository](https://doi.org/10.24432/C5CW3D).

#### Setup
- Acceleration signals
- 2 three-axial [Axivity AX3](https://axivity.com/product/ax3) accelerometers
- Attached to: thigh and lower back

#### Activity Annotations

| Label| Activity                  |  Notes 				    |
|------|:--------------------------|:--------------------------------------:|
| 1    | walking                   | 			                    |
| 3    | shuffling 		   | standing with leg movement             |
| 4    | stairs (ascending)        | 			                    |
| 5    | stairs (descending)       | 			                    |
| 6    | standing                  | 			                    |
| 7    | sitting                   | 			                    |
| 8    | lying                     | 			                    |

## DualSleep Dataset
The DualSleep dataset contains accelerometer recordings with sleep-stage annotations. It consists of acceleration data of 29 subjects, which wore two three-axial [Axivity AX3 (Axivity Ltd., Newcastle, UK)](https://axivity.com/product/ax3) accelerometers on the thigh and lower back. The dataset is available [here](https://doi.org/10.18710/UGNIFE). Download it for the sleep experiments.

#### Setup
- Acceleration signals
- 2 three-axial [Axivity AX3](https://axivity.com/product/ax3) accelerometers
- Attached to: thigh and lower back

#### Activity Annotations

| Label| Activity                  |  Notes 				    |
|------|:--------------------------|:--------------------------------------:|
| 81   | Wake	                   | 			                    |
| 82   | Non-REM1 		   |			                    |
| 83   | Non-REM2	           | 			                    |
| 84   | Non-REM3	           | 			                    |
| 85   | REM	                   | 			                    |
| 86   | Movement                  | Body movements during sleep            |

## Machine Learning Experiments
The folder [experiments](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments) contains all our experiments. It is possible to train a K-Nearest Neighbors, a Support Vector Machine, a Random Forest, an Extreme Gradient Boost, a Convolutional Neural Network, a Bidirectional Long Short-term Memory, and a CNN with multi-resolution blocks.
### Requirements
- Python 3.8.10+
```bash
cd experiments
pip install -r requirements.txt
```
### Usage
Start a model training using HARTH
```bash
cd experiments
./run_training.sh -c <path/to/model/config.yml> -d <path/to/dataset>
# Example: ./run_training.sh -c traditional_machine_learning/params/xgb_50hz/config.yml -d ../harth/
```
Each model can be configured using the corresponding config.yml file: [xgb](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments/traditional_machine_learning/params/xgb_50hz/), [svm](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments/traditional_machine_learning/params/svm_50hz/), [rf](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments/traditional_machine_learning/params/rf_50hz/), [knn](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments/traditional_machine_learning/params/knn_50hz/), [cnn](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments/deep_learning/params/cnn_50hz/), [multi_resolution_cnn](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments/deep_learning/params/inc_cnn_50hz/), [lstm](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/main/experiments/deep_learning/params/lstm_50hz/)

## Citation
If you use the HARTH dataset for your research, please cite the following papers:
```bibtex
@article{logacjovHARTHHumanActivity2021,
  title = {{{HARTH}}: {{A Human Activity Recognition Dataset}} for {{Machine Learning}}},
  shorttitle = {{{HARTH}}},
  author = {Logacjov, Aleksej and Bach, Kerstin and Kongsvold, Atle and B{\aa}rdstu, Hilde Bremseth and Mork, Paul Jarle},
  year = {2021},
  month = nov,
  journal = {Sensors},
  volume = {21},
  number = {23},
  pages = {7853},
  publisher = {{Multidisciplinary Digital Publishing Institute}},
  doi = {10.3390/s21237853}
}
```
```bibtex
@article{bachMachineLearningClassifier2021,
  title = {A {{Machine Learning Classifier}} for {{Detection}} of {{Physical Activity Types}} and {{Postures During Free-Living}}},
  author = {Bach, Kerstin and Kongsvold, Atle and B{\aa}rdstu, Hilde and Bardal, Ellen Marie and Kj{\ae}rnli, H{\aa}kon S. and Herland, Sverre and Logacjov, Aleksej and Mork, Paul Jarle},
  year = {2021},
  month = dec,
  journal = {Journal for the Measurement of Physical Behaviour},
  pages = {1--8},
  publisher = {{Human Kinetics}},
  doi = {10.1123/jmpb.2021-0015},
}
```

If you use the HAR70+ dataset for your research, please cite the following paper:
```bibtex
@article{ustadValidationActivityType2023,
  title = {Validation of an {{Activity Type Recognition Model Classifying Daily Physical Behavior}} in {{Older Adults}}: {{The HAR70}}+ {{Model}}},
  shorttitle = {Validation of an {{Activity Type Recognition Model Classifying Daily Physical Behavior}} in {{Older Adults}}},
  author = {Ustad, Astrid and Logacjov, Aleksej and Trolleb{\o}, Stine {\O}verengen and Thingstad, Pernille and Vereijken, Beatrix and Bach, Kerstin and Maroni, Nina Skj{\ae}ret},
  year = {2023},
  month = jan,
  journal = {Sensors},
  volume = {23},
  number = {5},
  pages = {2368},
  publisher = {{Multidisciplinary Digital Publishing Institute}},
  issn = {1424-8220},
  doi = {10.3390/s23052368},
  copyright = {http://creativecommons.org/licenses/by/3.0/}
}
```


If you use the DualSleep dataset for your research, please cite the following papers:
```bibtex
@article{logacjovMachineLearningModel2024,
  title = {A {{Machine Learning Model}} for {{Predicting Sleep}} and {{Wakefulness Based}} on {{Accelerometry}}, {{Skin Temperature}} and {{Contextual Information}}},
  author = {Logacjov, Aleksej and Skarpsno, Eivind Schjelderup and Kongsvold, Atle and Bach, Kerstin and Mork, Paul Jarle},
  date = {2024-06-06},
  journaltitle = {Nature and Science of Sleep},
  shortjournal = {NSS},
  volume = {16},
  pages = {699--710},
  publisher = {Dove Press},
  doi = {10.2147/NSS.S452799}
}
```

## Note
Our HARTH dataset is subject to changes in future releases. Therefore, consider version [v1.0](https://github.com/ntnu-ai-lab/harth-ml-experiments/tree/v1.0) for reproducibility purposes. It contains the dataset and experiments used in our article, [HARTH: A Human Activity Recognition Dataset for Machine Learning](https://doi.org/10.3390/s21237853)
