# DENT-DDSP
Implementation of "data-efficient noisy speech generator using DDSP components" (DENT-DDSP). 

## Get started 
Training: python main_train_freq_with_valid.py\
Inference: python generate_noisy_from_ckpt.py\
Pre-trained model: checkpoints/03_07_2022_13_43_10_rats_small_train_thd0.8,1.0_sr8000_len10/ckpt-132\

## Novel DDSP components
Implementation of the proposed Computationally-efficient dynamic range compressor(DRC) and waveshaper can be found: model_freq_domain/core.py

## Simulation demo
https://guozixunnicolas.github.io/DENT-DDSP-demo/

##Tutorial
Want to get started in DENT-DDSP? Check out the google colab tutorial I have created: https://colab.research.google.com/drive/1Y1Qsz1PM71OuZ3i4vT7zJ4yLP6iwAKeH
