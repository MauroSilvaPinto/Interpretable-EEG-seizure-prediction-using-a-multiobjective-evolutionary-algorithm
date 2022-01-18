"""
FitnessFunction class calculates 
the fitness value for each objective:
    - sample sensitivity
    - sample specificity
    - electrode placement comfort
"""

import numpy as np

class FitnessFunction():
    
    def __init__(self, objectives_list):
        self.objectives = []
        self.values = np.zeros(len(objectives_list))
        
    def __repr__(self):  
        return "FitnessFunction()"
    
    def __str__(self):
        return f"Fitness function: {self.getNumberOfObjectives} objectives"
    
    # returns number of objectives in fitness function
    def getNumberOfObjectives(self):
        return len(self.objectives)
    
    # returns a normalized fitness value (between 0 and 1) for seizure sensitivity
    def evaluateSensitivity(classifier_sensitivity):
        return classifier_sensitivity 
    
    # returns a normalized fitness values (between 0 and 1) for FPR/h
    def evaluateFPRh(classifier_fprh):
        max_FPRh = 2 # set a reasonable limit for max FPRh?
        if classifier_fprh > max_FPRh: # will this make it too "greedy" for lower FPRh?
            return 0
        else:
            return 1 - classifier_fprh/max_FPRh # not sure how to set this one so that it's between 0 and 1...
    
    # returns a normalized fitness value (between 0 and 1) for sample sensitivity
    def evaluateSampleSensitivity(classifier_samplesensitivity):
        return classifier_samplesensitivity 
    
    # returns a normalized fitness value (between 0 and 1) for time under false alarm
    def evaluateTimeUnderFalseAlarm(classifier_timeunderfalsealarm):
        return 1 - classifier_timeunderfalsealarm
    
    # returns a normalized fitness value (between 0 and 1) considering the number of
    # different electrodes within the individual's features and the number of different
    # lobes that those electrodes belong to
    def evaluateElectrodes(individual_features):
        electrodes = []
        for f in individual_features:
            electrodes.append(f.decodeElectrode())
        
        unique_electrodes = np.unique(np.array(electrodes))
        number_electrodes = len(unique_electrodes)
        number_lobes = 0
        if any("F" in s for s in unique_electrodes): # FRONTAL
            number_lobes += 1
        if any("T" in s for s in unique_electrodes): # TEMPORAL
            number_lobes += 1
        if any("C" in s for s in unique_electrodes): # "CENTRAL"
            number_lobes += 1
        if any("P" in s for s in unique_electrodes): # PARIETAL
            number_lobes += 1
        if any("O" in s for s in unique_electrodes): # OCCIPITAL
            number_lobes += 1
            
        return 1.25*(1 - (number_electrodes * number_lobes)/(len(unique_electrodes)*5))
        
        
    # PLACEHOLDER METHOD: fitness = mean(sens, FPR) -> this was for the single-objective version of the algorithm...
    def evaluateFitness(values):
        return np.mean(np.array(values))

    
        
