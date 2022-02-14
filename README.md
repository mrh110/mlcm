# MLCM creates a 2D Multi-Label Confusion Matrix
Please read the following paper for more information:\
M. Heydarian, T. Doyle, and R. Samavi, MLCM: Multi-Label Confusion Matrix, 
IEEE Access, Feb. 2022, DOI: 10.1109/ACCESS.2022.3151048\
For other projects please see https://biomedic.ai/


Please cite the paper if you are using the MLCM.\
This work is licensed under a Creative Commons Attribution 4.0 License. For more information,
see https://creativecommons.org/licenses/by/4.0/

# An example on how to use MLCM package:
% Importing libraries
>> from mlcm import mlcm\
>> import numpy as np

% Creating random input (multi-label data)
>> number_of_samples = 1000\
>> number_of_classes = 5\
>> label_true = np.random.randint(2, size=(number_of_samples, number_of_classes))\
>> label_pred = np.random.randint(2, size=(number_of_samples, number_of_classes))

% Calling mlcm and illustrating the results
>> conf_mat,normal_conf_mat = mlcm.cm(label_true,label_pred)\
>> print('\nRaw confusion Matrix:')\
>> print(conf_mat)\
>> print('\nNormalized confusion Matrix (%):')\
>> print(normal_conf_mat)

>> one_vs_rest = mlcm.stats(conf_mat)
