## Interpretable EEG seizure prediction using a multiobjective evolutionary algorithm


This is the code used for the paper "Interpretable EEG seizure prediction using a multiobjective evolutionary algorithm". It is a patient-specific Evolutionary Algorithm  for predicting epileptic seizures with the EEG signal, from data preprocessing to phenotype study.

## Code Organization Folders

- Data Processing
- Evolutionary Algorithm
- Phenotype Analysis
- Control method

## Data Folders

- Processed_data
- Results
- Trained_evol

## Preprocessing

You can not execute this code as it is necessary the raw data from EEG recordings. As the used dataset belongs to EPILEPSIAE, we can not make it publicly available online due to ethical concerns. We can only offer the extracted first-level features from non-overlapping windows of 5 seconds. In preprocessing code:
- [check_evts.py] - a code to check evts files, which are headlines for the raw binary data. It concerns patient selection: minimum number of 4 seizures separated by at least 4h30 of data.
- [check_gaps.py] - a code to check recording gaps, which concerns patient selection criteria: patients with gaps longer than 1 hour were discarded.
- [pre_processing.py] - a code to preprocess and extract first-level features, from raw data to chronologically extracted first-level features in non-overlapping windows of 5 seconds. output example: pat[patient_number]_seizure[seizure_number]_featureMatrix.npy

In preprocessed_data folder, we present this code output for patient 1321803.

## Evolutionary Algorithm

You can execute all the following scripts on patient 1321803 with the preprocessed files we present. You can also skip the execution and check the 30 performed runs from the paper, which are present in Trained_evol folder. The results are also present in Results folder. Here are the scripts you can run:

- [main_train.py]: to execute the EA.
- [main_test.py]: to get the EA results in new tested seizures.
- [get_training_info.py]: to get more information on the selected individuals of the executed EA.
- [repeat_random.py]: to have another statistical validation, using the random predictor instead of the surrogate analysis.

# Phenotype Analysis

You can execute all the following scripts on patient 1321803 with the data from Results and Trained_evol folders. These scripts perform the phenotype study , whose outputs are the graphs from the paper. Here are the scripts you can run:

- [bar_graphs.py]: this script provides the figures from individual gene presence, which are provided in the Supplementary Material paper.
 
Attention: to run the bar_graphs.py script, you need to have it in the Evolutionary Algorithm folder, along with barplot_annotate_brackets.py file.
- [chord_plot.py]: this script provides the chord plot from the paper, which shows gene individual presence and gene interaction.
- [connectivity_plot.py]: this script provides the brain connectivity plot from the paper, which shows gene interaction between electrodes .

# Control method

You can execute the control method for Patient 1321803, which uses the same data as the one from the evolutionary algorithm. You can execute the following scripts:

- [train_one_patient]: train one patient (find the best pre-ictal period and the best number of features using a grid-search).
- [get_patient_fitness]: to get the performance training value from the chosen set of pre-ictal and the best number of features.
- [test_one_patient]: test one patient, using the obtained best pre-ictal and number of features, with new tested seizures.


## More Information
More information, please just check the article at:


You are free to use any of this material. If you do, please just cite it as:
adasdjaksdasdakjdkasjdkajdkajdak 


