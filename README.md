# 12-Lead ECG Classification with Optimal Subset Selection
This repository is an implementation of the papers:  
  
- [Changxin Lai, Shijie Zhou, and Natalia Trayanova. 
**"Heart Rhythm Classification From an Optimal Lead Subset of the 12-lead Electrocardiogram by Deep Learning."** 
Circulation 142.Suppl_3 (2020): A15042-A15042.](https://www.ahajournals.org/doi/abs/10.1161/circ.142.suppl_3.15042)
(Presented at AHA Scientific Sessions 2020)  
- Changxin Lai, Shijie Zhou, and Natalia Trayanova. 
**"A Deep Learning Study on Heart Rhythm Classification: Using a Novel Approach to Increase Generalizability."**
(Under review)

We developed a multi-stage DL-based model to automatically detect heart rhythm types, which takes as input the 
raw 12-lead ECG data with variable length and outputs a heart rhythm interpretation for the whole signal. 
The model consists of three modules: 
1) a *feature extraction* module that automatically 
extracts features from each lead of the raw 12-lead ECG data, 
2) an optimal ECG-lead *subset selection* module that 
is used to find an optimal minimal lead subset, and 
3) a *decision-making* module that uses features extracted from 
the optimal ECG-lead subset to interpret heart rhythm types.  
  
## Code
### Environment
A ```Dockerfile``` with ```requirements.txt``` is provided to configure a [docker](https://www.docker.com/) 
environment for the project.

### Running
1) **Feature Extraction**: ```train.py``` trains neural networks with configurations ```train_config.json``` 
for processing single-lead ECG signals. ```single_lead_ECG_features.py```
 and ```single_lead_ECG_features_ext.py``` use trained models to extract features from ECG signals on the dataset.  
2) **Subset Selection**: ```subset_selection.py``` uses features extracted in the previous step and find an optimal
ECG lead subset.
3) **Decision Making**: ```train_decision_model.py``` trains decision making classifiers based on the optimal ECG-lead
subset.  

Finally, ```predict.py``` integrates all the steps with configuration ```predict_config.json``` and make predictions 
on new ECG data.
 
### Demo
We provided the models with weights in ```/save```, configurations in ```predict_config.json```, and example data in 
```/data/demo```. To run a demo of the model making predictions, execute:
```demo
python predict.py /data/demo predict_config.json
```
Results will be save in a file named ```predict_result.mat```.

