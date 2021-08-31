"""
code to get our select patient phenotype
as a list of lists, where a transaction
is a individual (a set of 5 hyperfeatures and a pre-ictal period)


ps: in order to this script to work,
you need to put it in the Evolutionary Algorithm
folder, along with barplot_annotate_brackets.py

or

to simple paste in this folder the Evolutionary Algorithm classes.

"""


from Database import Database
from Patient import Patient
from Population import Population

import numpy as np

import pickle
import os


#%% Setup patient ID, algorithm, paths, etc.

ID = "1321803"
algorithm = "NSGA2" # NSGA2, SMS-EMOA or MOGA (only NSGA2 will be used for the paper)

# go back to data folder
os.chdir("..")
os.chdir("..")
os.chdir("Data")
path=os.getcwd()

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
    
#%% Transaction: one individual (5 features)
        
transactions=[]
for individual in selected_individuals:
    transaction=[]
    
    individual_preictal=individual.decodePreictalThreshold()
    for feature in individual.features:
        
        transaction.append(feature.decodeElectrode())
        transaction.append("preictal_"+str(feature.decodeDelay()+individual_preictal))
        transaction.append("window_length_"+str(feature.decodeWindowLength()))
        transaction.append(feature.checkActiveGenes())
    
    # eliminate repeated values
    transaction=np.unique(np.array(transaction)).tolist()    
    transactions.append(transaction)
        
        # order
        # electrode, delay, length, feature
    
        # podemos dividir ainda
        # electrodos em: hemisfério e lobo
        # features: grupos de features (hjorth parameters, relative power, wavelets, etc)

        
data_transaction_individual=transactions
np.save("apriori_transactions_patient_1321803.npy",data_transaction_individual)

# variable= np.load("data_transaction_individual",allow_pickle=True)
# variable=variable.tolist()