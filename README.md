# Cervical-Cancer-CNN-GA
This is the official implementation of the paper titled "Deep Features Selection through Genetic Algorithm for Cervical Pre-cancerous Cell Classification" under peer review in "Springer- Multimedia Tools and Applications".

Abstract: Cervical cancer affects more than 500,000 women in the world annually accounting for about 6-9\% of all cancer cases, but, its tedious detection procedure makes population-wide screening impossible. Classification of cervical pre-cancerous cells using computer-aided diagnosis tools is a challenging task and is posed an open problem for several decades. The most concerning issue is that only a small amount of data is available publicly. In this study, Deep Learning along with an evolutionary metaheuristic algorithm called the Genetic Algorithm is incorporated for cervical cell classification. Pre-trained Convolutional Neural Networks, namely GoogLeNet and ResNet-18 have been utilized to account for the fewer data available, for extracting deep features from the images. The extracted features are optimized by employing a Genetic Algorithm for feature selection which is coupled with the Support Vector Machines classifier for the final classification. The proposed method has been validated on two publicly available datasets which obtained promising results on 5-fold cross-validation justifying the framework to be reliable.

## Requirements

To install the dependencies, run the following using the command prompt:

`pip install -r requirements.txt`

## Running the code on the Cervical Cytology data



Required Directory Structure:
```

.
+-- data
|   +-- .
|   +-- train
|   +-- val
+-- CNN_Feature_Extractor.py
+-- mendeley_features.csv
+-- GA.py
+-- utils.py

```
Form the csv file for the feature set such that the labels of the corresponding samples are integer values present in the last column of the csv file. The feature set (GoogLeNet+ResNet-18) obtained from the Mendeley LBC dataset has been provided in this repository, for example. To run the Genetic Algorithm for the optimal feature subset selection run the following:

`python GA.py --csv_name mendeley_features.csv --csv_header yes`

If your csv file does not contain headers, just set `--csv_header` to `no`, or just remove the field (since `default='no'` has been set in the code).
