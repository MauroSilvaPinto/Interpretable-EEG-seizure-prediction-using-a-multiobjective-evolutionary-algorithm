"""
Code to test the obtained EA output in the testing seizures.

Besides surrogate validation, it also calculates the Random Predictor,
an alternative way of statistical validation, by:
    Schelter, Björn & Andrzejak, Ralph & Mormann, Florian. (2008)
    Can Your Prediction Algorithm Beat a Random Predictor?
    10.1002/9783527625192.ch18. 

"""

from Database import Database
from Patient import Patient
from Population import Population
from Classifier import Classifier
from SlidingWindow import SlidingWindow
from StatisticalValidation import StatisticalValidation

import numpy as np

import pickle
import os

from scipy import stats

#%% Setup patient ID, algorithm, paths, etc.
algorithm = "NSGA2" # NSGA2, SMS-EMOA or MOGA (only NSGA2 will be used for the paper)

# where the data is
# go back to data folder
os.chdir("..")
os.chdir("..")
os.chdir("Data")
path=os.getcwd()
sep = os.path.sep
if path[-1] != sep:
    path+=sep
evol_path = path + "Processed_data" + sep #!!! mudar para Evol depois

ID = "1321803"
    
trained_evol_path = path + "Trained_evol" + sep + "pat_" + ID # retrieve trained evolutionary algorithms from here
if trained_evol_path[-1] != sep:
    trained_evol_path+=sep

sliding_window_step = 0.5
classifier_type = 'logreg'
fp_threshold = 0.7

save_path_name = path + "Results" + sep + "pat_" + ID # save test results here
filename = "pat" + ID + "_" + algorithm # patID_algorithm

if os.path.exists(save_path_name) == False:
        os.makedirs(save_path_name) # create patient-specific directory


#%% Setup testing data

db = Database(evol_path)

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

#%% Select solutions

# Set a minimum fitness threshold for the first two objectives (sensitivity and time under false alarm)
if algorithm == "MOGA":
    fitness_threshold = 0.6
else:
    fitness_threshold = 0.9

# Load list of saved individuals from a previously trained algorithm
solutions = sorted(os.listdir(trained_evol_path))
solutions = [s for s in solutions if filename in s and "lastgen" in s] # only files with "patID_algorithm"
solutions = [trained_evol_path + sep + s for s in solutions]

# Retrieve Pareto-optimal solutions with fitness > threshold (for the first two objectives)
# from every run
selected_individuals = []
each_run = []
for file in solutions:
    individuals = pickle.load(open(file, "rb"))
    count = 0
    
    sorted_ind, sorted_front = Population.applyNonDominatedSorting(individuals)
    front_individuals_idx = np.nonzero(np.array(sorted_front) == 1)[0]
    front_individuals = np.array([sorted_ind[f] for f in front_individuals_idx])
    
    # apply Decision Maker
    for ind in front_individuals:
        if ind.fitness[0] >= fitness_threshold and ind.fitness[1] >= fitness_threshold:
            selected_individuals.append(ind)
            count+=1
    
    if count == 0:
        # lower threshold in this case (rare, but it happens in some MOGA runs)
        for ind in front_individuals:
            if ind.fitness[0] >= fitness_threshold - 0.1 and ind.fitness[1] >= fitness_threshold - 0.1:
                selected_individuals.append(ind)
                count+=1
        
    each_run.append(count)

pickle.dump(selected_individuals, open(save_path_name + sep + filename + "_selected", "wb")) # save selected individuals

#%% Test solutions on new seizures and evaluate performance

print(f"Algorithm: {algorithm} | Number of selected solutions: {len(selected_individuals)}")

print('\nTesting Pareto-optimal individuals on new seizures...')
metrics = []
for i in range(len(selected_individuals)):
    selected_solution = selected_individuals[i]
    print(f"{i+1}/{len(selected_individuals)}")
    
    clf = Classifier(classifier_type)
    sw = SlidingWindow(selected_solution.features, sliding_window_step)
    
    SOP = sw.computePreictal(selected_solution.decodePreictalThreshold())
    SPH = 10
    
    # train iteratively: first with all the training seizures, test on the first 
    # testing seizure; then train with all the previous and test on the next one, etc.
    train = training_seizures[:]
    test = testing_seizures[:]
    
    metrics_sens = []
    false_alarms = 0
    total_interictal = 0
    lost_interictal = 0
    while len(test) != 0:
        # TRAINING (all available "train" seizures up until the first seizure in "test")
        for j in train:
            new_data, new_legend = sw.extractFeatures(pat.seizure_data[j], legend, selected_solution.decodePreictalThreshold())
            
            if j == train[0]:
                training_data = new_data
            else:
                training_data = np.hstack((training_data, new_data))
        
        clf.trainClassifier(training_data, new_legend)
    
        # TESTING (on the next seizure -> first in the "test" list)
        new_data, new_legend = sw.extractFeatures(pat.seizure_data[test[0]], legend, selected_solution.decodePreictalThreshold())
        testing_data = new_data
        
        clf_output, true_labels = clf.classifyData(testing_data, new_legend)
        clf_output_FP = Classifier.applyFiringPower(clf_output, true_labels, fp_threshold)
        clf_output_processed = Classifier.applyRefractoryBehavior(clf_output_FP, SOP, SPH, sliding_window_step)
    
        # COMPUTE METRICS 
        clf_seizuresens = Classifier.getSensitivity(clf_output_processed, true_labels)
        clf_falsealarms = Classifier.getNumberOfFalseAlarms(clf_output_processed, true_labels)
        clf_totalinterictal, clf_lostinterictal = Classifier.getInterictalTime(clf_output_processed, true_labels, SOP, SPH, sliding_window_step)
        
        # APPEND METRIC VALUES AND UPDATE TRAINING/TESTING GROUPS
        metrics_sens.append(clf_seizuresens)
        false_alarms += clf_falsealarms
        total_interictal += clf_totalinterictal
        lost_interictal += clf_lostinterictal
        train.append(test[0])
        test.pop(0)
    
    # COMPUTE OVERALL PERFORMANCE FOR THE SELECTED SOLUTION
    mean_sens = np.mean(metrics_sens) # average sensitivity
    mean_fprh = Classifier.getFPRh(false_alarms, true_labels, total_interictal, lost_interictal)
    
    # print("\nTEST PERFORMANCE RESULTS:")  
    # print(f"Sensitivity: {mean_sens} | FPR/h: {mean_fprh}")      
    
    metrics.append([mean_sens, mean_fprh])

# SAVE PERFORMANCE METRICS
metrics = np.array(metrics)
test_meansens = np.mean(metrics[:,0])
test_stdsens = np.std(metrics[:,0])
test_meanfpr = np.mean(metrics[:,1])
test_stdfpr = np.std(metrics[:,1])

print("\nTEST PERFORMANCE RESULTS:")  
print(f"Sensitivity: {test_meansens} +- {test_stdsens} | FPR/h: {test_meanfpr} +- {test_stdfpr}")    
print(f"Sensitivity: {test_meansens} +- {test_stdsens} | FPR/h: {test_meanfpr} +- {test_stdfpr}", file=open(save_path_name + sep + filename + "_results.txt", 'w'))    

np.save(save_path_name + sep + filename +  "_performance", metrics)

print("\a") # beep when done :) 

#%% Conduct statistical validation (random predictor + surrogate)

alpha = 0.05 # for 95% statistical significance

# # Load performance metrics and all the selected individuals 
metrics = np.load(save_path_name + sep + filename +  "_performance.npy")
selected_individuals = pickle.load(open(save_path_name + sep + filename + "_selected", "rb"))

#%% RANDOM PREDICTOR

print("\nValidating performance against a random predictor...")
sops = [StatisticalValidation.computeSOP(ind) for ind in selected_individuals]
number_test_seizures = len(testing_seizures)

# Run the random predictor considering the mean SOP duration and mean FPR/h
mean_sop = np.mean(sops)
test_meansens = np.mean(metrics[:,0])
test_stdsens = np.std(metrics[:,0])
test_meanfpr = np.mean(metrics[:,1])
test_stdfpr = np.std(metrics[:,1])

random_predictor_global_sens = StatisticalValidation.getRandomPredictorSensitivity(number_test_seizures * len(selected_individuals), test_meanfpr, 5, mean_sop, alpha)
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
    random_predictor_sens = StatisticalValidation.getRandomPredictorSensitivity(number_test_seizures, metrics[i,1], 5, sops[i], alpha)
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

#%% SURROGATE PREDICTOR

surrogate_runs = 30 # number of times the preictal labelling is shuffled

print('\nValidating performance against a surrogate predictor...')
surrogate_metrics = []
beats_surrogate = [] 
for i in range(len(selected_individuals)):
    selected_solution = selected_individuals[i]
    print(f"{i+1}/{len(selected_individuals)}")
    
    performance_metrics = []
    for i in range(surrogate_runs):
        clf = Classifier(classifier_type)
        sw = SlidingWindow(selected_solution.features, sliding_window_step)
        SOP = sw.computePreictal(selected_solution.decodePreictalThreshold())
        SPH = 10
        train = training_seizures[:]
        test = testing_seizures[:]
        
        metrics_sens = []
        false_alarms = 0
        total_interictal = 0
        lost_interictal = 0
        while len(test) != 0:
            # TRAINING (all available "train" seizures up until the first seizure in "test")
            for j in train:
                new_data, new_legend = sw.extractFeatures(pat.seizure_data[j], legend, selected_solution.decodePreictalThreshold())
                
                if j == train[0]:
                    training_data = new_data
                else:
                    training_data = np.hstack((training_data, new_data))
            
            clf.trainClassifier(training_data, new_legend)
        
            # TESTING -> SHUFFLE PREICTAL LABELS FOR SURROGATE ANALYSIS
            new_data, new_legend = sw.extractFeatures(pat.seizure_data[test[0]], legend, selected_solution.decodePreictalThreshold())
            testing_data = new_data
            
            clf_output, true_labels = clf.classifyData(testing_data, new_legend)
            shuffled_labels = StatisticalValidation.randomizePreictal(true_labels) # SHUFFLE HERE
            
            clf_output_FP = Classifier.applyFiringPower(clf_output, shuffled_labels, fp_threshold)
            clf_output_processed = Classifier.applyRefractoryBehavior(clf_output_FP, SOP, SPH, sliding_window_step)
        
            # COMPUTE METRICS
            clf_seizuresens = StatisticalValidation.getSensitivity(clf_output_processed, shuffled_labels)
            clf_falsealarms = StatisticalValidation.getNumberOfFalseAlarms(clf_output_processed, shuffled_labels)
            clf_totalinterictal, clf_lostinterictal = StatisticalValidation.getInterictalTime(clf_output_processed, shuffled_labels, SOP, SPH, sliding_window_step)
            
            # APPEND METRIC VALUES AND UPDATE TRAINING/TESTING GROUPS
            metrics_sens.append(clf_seizuresens)
            false_alarms += clf_falsealarms
            total_interictal += clf_totalinterictal
            lost_interictal += clf_lostinterictal
            train.append(test[0])
            test.pop(0)
        
        # COMPUTE OVERALL PERFORMANCE FOR THE SELECTED SOLUTION
        mean_sens = np.mean(metrics_sens)
        mean_fprh = StatisticalValidation.getFPRh(false_alarms, shuffled_labels, total_interictal, lost_interictal)
        
        performance_metrics.append([mean_sens, mean_fprh])
        
    performance_metrics = np.array(performance_metrics)  
    mean_sens_surrogate = np.mean(performance_metrics[:,0])
    std_sens_surrogate = np.std(performance_metrics[:,0])
    mean_fpr_surrogate = np.mean(performance_metrics[:,1])
    std_fpr_surrogate = np.std(performance_metrics[:,1])
    
    # save the average surrogate performance for each individual
    surrogate_metrics.append([mean_sens_surrogate, std_sens_surrogate, mean_fpr_surrogate, std_fpr_surrogate])
        
# SAVE SURROGATE PERFORMANCE METRICS
surrogate_metrics = np.array(surrogate_metrics)
surrogate_meansens = np.mean(surrogate_metrics[:,0])
surrogate_stdsens = np.std(surrogate_metrics[:,0])
surrogate_meanfpr = np.mean(surrogate_metrics[:,1])
surrogate_stdfpr = np.std(surrogate_metrics[:,1])

# check if the surrogate predictor is beaten
for i in range(len(selected_individuals)):
    surrogate_t, surrogate_p = StatisticalValidation.performOneSampleTTest(surrogate_metrics[i,0], surrogate_metrics[i,1], metrics[i,0], surrogate_runs)
    
    if surrogate_metrics[i,0] < metrics[i,0] and surrogate_p < alpha:
        beats_surrogate.append(True)
    else:
        beats_surrogate.append(False)

beat_percentage_surrogate = np.count_nonzero(np.array(beats_surrogate)==True)/len(beats_surrogate)

print("\nSURROGATE PERFORMANCE RESULTS:")  
print(f"Sensitivity: {surrogate_meansens} +- {surrogate_stdsens} | FPR/h: {surrogate_meanfpr} +- {surrogate_stdfpr}\n{beat_percentage_surrogate*100}% of individuals beat the surrogate predictor!")    
print(f"Sensitivity: {surrogate_meansens} +- {surrogate_stdsens} | FPR/h: {surrogate_meanfpr} +- {surrogate_stdfpr}\n{beat_percentage_surrogate*100}% of individuals beat the surrogate predictor!", file=open(save_path_name + sep + filename + "_surrogate_results.txt", 'w'))    

np.save(save_path_name + sep + filename +  "_surrogate_performance", surrogate_metrics)
np.save(save_path_name + sep + filename +  "_surrogate_beats", beats_surrogate)

# Save validation results (average individual performance vs. average surrogate predictor performance)
surrogate_average_t, surrogate_average_p = stats.ttest_ind_from_stats(surrogate_meansens, surrogate_stdsens, len(selected_individuals), test_meansens, test_stdsens, len(selected_individuals))

if surrogate_meansens < test_meansens and surrogate_average_p < alpha:
    beat_average_surrogate = True
else:
    beat_average_surrogate = False

print(f"\nAverage sensitivity higher than the surrogate predictor = {beat_average_surrogate}", file=open(save_path_name + sep + filename + "_surrogate_average_results.txt", 'w'))


print("\a") # beep when done :)
