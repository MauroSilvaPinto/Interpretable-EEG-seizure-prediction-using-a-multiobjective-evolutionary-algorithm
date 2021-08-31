# -*- coding: utf-8 -*-
"""
The code used to get some info about the EA execution,
for one patient


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

ID = "1330903"
algorithm = "NSGA2" # NSGA2, SMS-EMOA or MOGA (only NSGA2 will be used for the paper)

#path = "/Users/Tiago/Desktop/Research/Data"
path = "D:\\Paper\\Data"
sep = os.path.sep
if path[-1] != sep:
    path+=sep
evol_path = path + "Processed_data" + sep
    
    
trained_evol_path = path + "Trained_evol" + sep + "pat_" + ID # retrieve trained evolutionary algorithms from here
if trained_evol_path[-1] != sep:
    trained_evol_path+=sep

sliding_window_step = 0.5
classifier_type = 'logreg'
fp_threshold = 0.7

filename = "pat" + ID + "_" + algorithm # patID_algorithm

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
    
    
#%% get Information from solutions
pre_ictals=[]
sensitivities=[]
specificities=[]
electrodes=[]

for i in range(len(selected_individuals)):
    selected_solution = selected_individuals[i]
    
    pre_ictals.append(selected_solution.decodePreictalThreshold())
    sensitivities.append(selected_solution.fitness[0])
    specificities.append(selected_solution.fitness[1])
    electrodes.append(selected_solution.fitness[2])

#pat
print("Patient "+str(ID))    
#Pre-Ictal
print(str(round(np.mean(pre_ictals),2))+"$pm$"+str(round(np.std(pre_ictals),2)))
#Sensitivity
print(str(round(np.mean(sensitivities),2))+"$pm$"+str(round(np.std(sensitivities),2)))
#Specificity
print(str(round(np.mean(specificities),2))+"$pm$"+str(round(np.std(specificities),2)))
#Electrodes
print(str(round(np.mean(electrodes),2))+"$pm$"+str(round(np.std(electrodes),2)))
#Number individuals    
print(str(len(selected_individuals)))

    
    
   