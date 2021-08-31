"""

A class for calculating the Random Predictor,
an alternative way of statistical validation, by:
    Schelter, Björn & Andrzejak, Ralph & Mormann, Florian. (2008)
    Can Your Prediction Algorithm Beat a Random Predictor?
    10.1002/9783527625192.ch18. 

"""

# RE-RUN RANDOM PREDICTOR WITH D = 1 (individual) AND D = NUMBER OF MODELS (global)

from Database import Database
from Patient import Patient
from Population import Population
from Classifier import Classifier
from SlidingWindow import SlidingWindow
from StatisticalValidation import StatisticalValidation

import numpy as np
import matplotlib.pyplot as plt

import pickle
import os

from scipy import stats

#%% Setup patient ID, algorithm, paths, etc.

#ID = "8902"
id_list = ["402","8902","11002","16202","23902","30802","32702","46702","50802"
,"53402","55202","56402","58602","59102","60002","64702","75202","80702","85202"
,"93402","93902","94402","95202","96002","98102","98202","101702","102202","104602"
,"109502","110602","112802","113902","114702","114902","123902"]
algorithm = "SMS-EMOA"

for ID in id_list:
    
    path = "D:\\TESE\\Dados\\Trained_evol\\pat_" + ID # retrieve trained evolutionary algorithms from here
    sep = os.path.sep
    if path[-1] != sep:
        path+=sep
    
    sliding_window_step = 0.5
    classifier_type = 'SVM_linear'
    fp_threshold = 0.7
    
    save_path_name = "D:\\TESE\\Resultados\\pat_" + ID # save test results in this directory
    filename = "pat" + ID + "_" + algorithm # patID_algorithm
    
    #%% Setup testing data
    
    db = Database("D:\\TESE\\Dados\\Evol\\")
    
    pat = Patient(ID)
    
    filenames = db.getFilenames()
    
    features, info = db.loadPatientData(pat.ID)
    legend = db.loadLegend()
    
    pat.seizure_data = features
    pat.seizure_metadata = info
    
    training_seizures = [0, 1, 2] # run evolutionary search on the first 3 seizures
    testing_seizures = np.arange(3, pat.getNumberOfSeizures()).tolist()
    
    print(pat)
    print('pattern | classification | vigilance | medicament | dosage')
    pat.printMetadata()
    print("\n")
    
    #%% Conduct statistical validation (random predictor + surrogate)
    
    alpha = 0.05 # for 95% statistical significance
    
    # # Load performance metrics and all the selected individuals 
    metrics = np.load(save_path_name + sep + filename +  "_performance.npy")
    selected_individuals = pickle.load(open(save_path_name + sep + filename + "_selected", "rb"))
    
    # RANDOM PREDICTOR
    
    print("\nValidating performance against a random predictor...")
    sops = [StatisticalValidation.computeSOP(ind) for ind in selected_individuals]
    number_test_seizures = len(testing_seizures)
    
    # Run the random predictor considering the mean SOP duration and mean FPR/h
    mean_sop = np.mean(sops)
    test_meansens = np.mean(metrics[:,0])
    test_stdsens = np.std(metrics[:,0])
    test_meanfpr = np.mean(metrics[:,1])
    test_stdfpr = np.std(metrics[:,1])
    
    # RANDOM GLOBAL -> d = number of models !!!
    random_predictor_global_sens = StatisticalValidation.getRandomPredictorSensitivity(number_test_seizures * len(selected_individuals), test_meanfpr, len(selected_individuals), mean_sop, alpha)
    random_global_t, random_global_p = StatisticalValidation.performOneSampleTTest(test_meansens, test_stdsens, random_predictor_global_sens, len(selected_individuals))
    
    if random_predictor_global_sens < test_meansens and random_global_p < alpha:
        beat_global_random = True
    else:
        beat_global_random = False
    
    # Save validation results (global random predictor)
    print(f"Random predictor global sensitivity: {random_predictor_global_sens} \nAverage sensitivity higher than the random predictor = {beat_global_random}", file=open(save_path_name + sep + filename + "_random_global_results.txt", 'w'))
    
    # Run the random predictor for each selected individual and compare their performances
    rand_sens = []
    beats_random = []
    for i in range(len(selected_individuals)):
        # RANDOM INDIVIDUAL -> d = 1 !!!
        random_predictor_sens = StatisticalValidation.getRandomPredictorSensitivity(number_test_seizures, metrics[i,1], 1, sops[i], alpha)
        rand_sens.append(random_predictor_sens)
        
        if random_predictor_sens < metrics[i,0]:
            beats_random.append(True)
        else:
            beats_random.append(False)
    
    rand_sens_mean = np.mean(rand_sens)
    rand_sens_std = np.std(rand_sens)
    beat_percentage_random = np.count_nonzero(np.array(beats_random)==True)/len(beats_random)
    
    # Save validation results (each individual)
    print(f"\nRandom predictor sensitivity: {rand_sens_mean} +- {rand_sens_std}\n{beat_percentage_random*100}% of individuals beat the random predictor!")
    print(f"Random predictor sensitivity: {rand_sens_mean} +- {rand_sens_std} \n{beat_percentage_random*100}% of individuals beat the random predictor!", file=open(save_path_name + sep + filename + "_random_results.txt", 'w'))
     
    np.save(save_path_name + sep + filename +  "_random_performance", rand_sens)
    np.save(save_path_name + sep + filename +  "_random_beats", beats_random)
    
    # Save validation results (average individual performance vs. average random predictor performance)
    random_average_t, random_average_p = stats.ttest_ind_from_stats(rand_sens_mean, rand_sens_std, len(rand_sens), test_meansens, test_stdsens, len(selected_individuals))
    
    if rand_sens_mean < test_meansens and random_average_p < alpha:
        beat_average_random = True
    else:
        beat_average_random = False
    
    print(f"\nAverage sensitivity higher than the random predictor = {beat_average_random}", file=open(save_path_name + sep + filename + "_random_average_results.txt", 'w'))
