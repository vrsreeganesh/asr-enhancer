# Speech Enhancement for Robust Automatic Speech Recognition 

![Poster](poster.png)

## Overview
This project presents an audio-enhancement model that is trained to suppress background speakers and maintain the speech of the primary speaker. 


## Setup
First recreate the conda environment using 
```
conda env create -f environment.yml
```

## Training
Prior to training, change the training code in the following manner

- Update path to store Whisper models. 
- Update path where training and test wav-files are stored
- Update path where to store the trained models, stored data-objects, etc etc


## Inference
Prior to running inference, 
- update the path to the model, which needs to be loaded
- Update path to where whisper is stored or loaded. 
