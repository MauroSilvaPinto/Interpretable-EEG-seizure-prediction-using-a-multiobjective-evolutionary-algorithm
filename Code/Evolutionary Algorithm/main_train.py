"""
The code for training the EA for a given patient. It will save the
EA executions (runs) in Trained_evol folder

Besides NSGA-II, this code is also ready to run SMS-EMOA algorithm
and the MOGA.

SMS-EMOA:
    Beume, Nicola, Boris Naujoks, and Michael Emmerich.
    "SMS-EMOA: Multiobjective selection based on dominated hypervolume."
    European Journal of Operational Research 181.3 (2007): 1653-1669.
    
MOGA:
    Fonseca, Carlos M., and Peter J. Fleming.
    "Genetic Algorithms for Multiobjective Optimization:
        Formulation, Discussion and Generalization.
        Icga. Vol. 93. No. July. 1993.
        
The Evolutionary Algorithm code was mostly developed by Tiago Coelho,
whose work can be found in:
Coelho, Tiago André Cruz.
EEG Epilepsy Seizure Prediction:
    A Multi-Objective Evolutionary Approach.
Diss. Universidade de Coimbra, 2020.

and was inspired by:
    Pinto, Mauro F., et al.
    "A personalized and evolutionary algorithm
    for interpretable EEG epilepsy seizure prediction." Scientific reports 11.1 (2021): 1-12.
"""

from Database import Database
from Patient import Patient
from Population import Population

import numpy as np
import os

import pickle

import warnings
warnings.filterwarnings("ignore")

#%% Setup hyperparameters

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

algorithm = "NSGA2" # NSGA2, SMS-EMOA or MOGA (only NSGA2 will be used for the paper)

number_objectives = 3

number_runs = 30

n_ind = 100 
n_feat = 5

sliding_window_step = 0.5

classifier_type = 'logreg'

fp_threshold = 0.7

if number_objectives == 2:
    objectives = ['sample_sensitivity', 't_under_false_alarm']
elif number_objectives == 3:
    objectives = ['sample_sensitivity', 't_under_false_alarm', 'electrodes'] 

if algorithm == "NSGA2":
    crossover_rate = 0.9
    mutation_rate = 1/(5*13 + 1) # 1/num_obj OR 1/l where l = string length for binary coded GAs (number of genes = 14)
elif algorithm == "SMS-EMOA":
    crossover_rate = 0.9
    mutation_rate = 1/(5*13 + 1) # 1/num_obj OR 1/l where l = string length for binary coded GAs (number of genes = 14)
elif algorithm == "MOGA":
    crossover_rate = 0.7
    mutation_rate = 1 - (0.9*2)**(-1/14) # 1 - (alpha*mu)^(-1/l)
    
selection = algorithm
replacement = algorithm

max_generations = 50

#%% Load patient data

db = Database(evol_path)

pat = Patient(ID)

filenames = db.getFilenames()

features, info = db.loadPatientData(pat.ID)
legend = db.loadLegend()

pat.seizure_data = features
pat.seizure_metadata = info

print(pat)
print('pattern | classification | vigilance | medicament | dosage')
pat.printMetadata()
print("\n")

#%% Split into training and testing datasets

training_seizures = [0, 1, 2] # run evolutionary search on the first 3 seizures
testing_seizures = np.arange(3, pat.getNumberOfSeizures()).tolist()

# pass arguments required for elitist replacement methods
args = [pat, training_seizures, legend, sliding_window_step, classifier_type, fp_threshold, objectives]

#%% Perform several runs of the algorithm

print(f'ALGORITHM SETTINGS: \n' +
          f'Individuals: {n_ind} | Features: {n_feat}\n' +
          f'Mutation rate: {mutation_rate} | Crossover rate: {crossover_rate}\n' +
          f'Algorithm: {algorithm} |  Runs: {number_runs} | Max generations: {max_generations}\n' +
          f'Objectives: {objectives}')

run_counter = 1
while(run_counter < number_runs + 1):
    
    # Generate random population and evaluate each individual
    path_name = path + "Trained_evol" + sep + "pat_" + ID + sep +  "pat" + ID + "_" + algorithm + "run" + str(run_counter)
    
    if os.path.exists(path + "Trained_evol" + sep + "pat_" + ID) == False:
        os.makedirs(path + "Trained_evol" + sep + "pat_" + ID) # create patient-specific directory
    
    print(f'\n--------- RUN #{run_counter} ---------')
    print('\nInitializing random population...\n')
    pop = Population(n_ind, n_feat)
    pop.evaluate(pat, training_seizures, legend, sliding_window_step, classifier_type, fp_threshold, objectives, False)
    
    print(pop)
    pop.plotFitness2D(['sample_sensitivity', 't_under_false_alarm'], path_name + "_plot" + str(pop.generation) + ".pdf")
    
    # Perform evolutionary search 
    while (pop.generation < max_generations): #and np.any(fitness_change > 0.05)):
        
        print('Evolving...')
        pop.evolve(selection, crossover_rate, mutation_rate, replacement, args)
        
        print('Evaluating...\n')
        pop.evaluate(pat, training_seizures, legend, sliding_window_step, classifier_type, fp_threshold, objectives, False)
    
        print(pop)
           
    # Save fitness history (of all generations), the last generation and the fitness plots for the 1st and last generations
    np.save(path_name + "_fitnesshistory", pop.fitness_history)
    pickle.dump(pop.individuals, open(path_name + "_lastgen", "wb"))
    pop.plotFitness2D(['sample_sensitivity', 't_under_false_alarm'], path_name + "_plot" + str(pop.generation) + ".pdf")
    
    run_counter += 1
    
# Beep when done :) 
print(f'\n\n\aFinished all {number_runs} runs!')