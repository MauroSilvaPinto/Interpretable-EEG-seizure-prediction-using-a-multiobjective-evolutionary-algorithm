"""

A class for calculating statistical validation:
    in other words, surrogate analysis followed by a
    one-sample t-test

"""

import numpy as np
from scipy.special import comb
from scipy import stats
import random

class StatisticalValidation():
    
    def __init__(self):
        self.random_predictor = True
        self.surrogate_predictor = True
        
    def __repr__(self):  
        return "StatisticalValidation()"
    
    def __str__(self):
        return f"Statistical Validation: {self.getActiveMethods()}"
    
    # returns the activated validation methods
    def getActiveMethods(self):
        if self.random_predictor == True and self.surrogate_predictor == True:
            string = "Random predictor and Surrogate analysis"
        elif self.random_predictor == False and self.surrogate_predictor == True:
            string = "Surrogate analysis"
        elif self.random_predictor == True and self.surrogate_predictor == False:
            string = "Random predictor"
        else:
            string = "(none)"
            
        return string
    
    # computes the sensitivity of the random predictor, when given the total
    # number of seizures, an FPR/h, number of parameters (d), SOP and the 
    # statistical significance level (alpha = 0.05, for example)
    def getRandomPredictorSensitivity(number_seizures, FPR, d, SOP, alpha):
        v_PBinom = np.zeros(number_seizures)
        s_kmax = 0
        
        for seizure_i in range(0,number_seizures):
            v_Binom=comb(number_seizures,seizure_i+1)
            
            s_PPoi = FPR*SOP
            v_PBinom[seizure_i]=v_Binom*s_PPoi**(seizure_i+1)*((1-s_PPoi)**(number_seizures-seizure_i-1))
            
        v_SumSignif=1-(1-np.cumsum(np.flip(v_PBinom)))**d>alpha
        s_kmax=np.count_nonzero(v_SumSignif)/number_seizures
        
        return s_kmax
    
    # computes the Pre-Ictal Period for a given individual
    def computePreIctalPeriod(individual):
        delays = []
        for f in individual.features:
            delays.append(f.decodeDelay())
        
        return min(delays) + individual.decodePreictalThreshold()
    
        # computes SOP for a given individual (needed for the random predictor)
    def computeSOP(individual):
        delays = []
        for f in individual.features:
            delays.append(f.decodeDelay())
        
        return min(delays) + individual.decodePreictalThreshold()
    
    # randomizes pre-ictal label placement (used for the surrogate predictor)
    def randomizePreictal(labels):
        preictal_start_idx = np.where(np.diff(labels)==1)[0][0] + 1
        preictal_length = np.count_nonzero(labels == 1)
        
        # create new label with the "block" of 1's starting somewhere else
        new_labels = np.zeros(len(labels))
        random_start = random.randint(0, preictal_start_idx - 1) 
        new_labels[random_start:random_start+preictal_length] = 1
        
        return new_labels
    
    # performs a one-sample t-test for an independent mean; in this context, we
    # are comparing a sample_mean (single value) to a mean+-std from n observations
    def performOneSampleTTest(population_mean, population_std, sample_mean, number_samples):  
        t_value = abs(population_mean-sample_mean)/(population_std/np.sqrt(number_samples))
        
        p_value = stats.t.sf(np.abs(t_value), number_samples-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
        #print('t-statistic = %6.3f pvalue = %6.4f' % (t_value, p_value))
        
        return t_value, p_value
    
    # THE FUNCTIONS BELOW HAVE BEEN REWORKED FROM THE CLASSIFIER CLASS FOR SURROGATE ANALYSIS!
    # CONSIDERING THERE IS ONLY 1 SEIZURE IN EACH SHUFFLED LABEL VECTOR
    
    # computes the number of triggered alarms from a given classifier output
    def getNumberOfAlarms(predicted_class):
        return len(np.where(np.diff(predicted_class)==1)[0])
    
    # computes the number of existing seizures from a given shuffled label vector
    def getNumberOfSeizures(shuffled_class):
        if shuffled_class[0] == 1:
            return 1 + len(np.where(np.diff(shuffled_class)==1)[0])
        else:
            return len(np.where(np.diff(shuffled_class)==1)[0])
    
    # computes the number of false alarms for the surrogate predictor
    def getNumberOfFalseAlarms(predicted_class, shuffled_class):
        false_alarms = 0
        
        # find idx where inter-ictal and pre-ictal stages end
        if shuffled_class[0] != 1:
            interictal_end_idx = np.where(np.diff(shuffled_class)==1)[0] + 1
        else:
            interictal_end_idx = np.array([0])
        preictal_end_idx = np.where(np.diff(shuffled_class)==(-1))[0] + 1
        if preictal_end_idx.size == 0:
            preictal_end_idx = np.array([len(shuffled_class)])
        
        # verify if alarm triggered immediately (diff function has no effect here)
        if (predicted_class[0]==1):
            if interictal_end_idx != 0:
                false_alarms += 1
        
        # for every other raised alarm, check if it's within the seizure and outside the SPH (10 minutes = 20 samples)
        alarm_start_idx = np.where(np.diff(predicted_class)==1)[0] + 1
        for alarm in alarm_start_idx:
            if alarm < interictal_end_idx[0] or alarm > preictal_end_idx[0] - 20:
                false_alarms += 1
                
        return false_alarms
        
    # computes the number of true alarms for the surrogate predictor
    def getNumberOfTrueAlarms(predicted_class, shuffled_class):
        true_alarms = 0
        
        # find idx where inter-ictal and pre-ictal stages end
        if shuffled_class[0] != 1:
            interictal_end_idx = np.where(np.diff(shuffled_class)==1)[0] + 1
        else:
            interictal_end_idx = np.array([0])
        preictal_end_idx = np.where(np.diff(shuffled_class)==(-1))[0] + 1
        if preictal_end_idx.size == 0:
            preictal_end_idx = np.array([len(shuffled_class)])
        
        # for every other raised alarm, check if it's within the seizure and outside the SPH (10 minutes = 20 samples)
        alarm_start_idx = np.where(np.diff(predicted_class)==1)[0] + 1
        for alarm in alarm_start_idx:
            if alarm >= interictal_end_idx[0] and alarm <= preictal_end_idx[0] - 20:
                true_alarms += 1
        
        return true_alarms
    
    # computes the seizure sensitivity: predicted_seizures / total_seizures
    def getSensitivity(predicted_class, shuffled_class):
        predicted_seizures = StatisticalValidation.getNumberOfTrueAlarms(predicted_class, shuffled_class)
        total_seizures = StatisticalValidation.getNumberOfSeizures(shuffled_class)
        
        return predicted_seizures / total_seizures
    
    # computes FPR/h across several training "runs": number_false_alarms / (total_interictal - lost_interictal)
    def getFPRh(false_alarms, true_class, total_interictal, lost_interictal):
        #interictal_idx = np.where(np.array(true_class)==0)[-1]
        lost_interictal_hours = lost_interictal / 60
        total_interictal_hours = total_interictal / 60
        
        if total_interictal_hours - lost_interictal_hours == 0:
            return np.nan
        else:
            return false_alarms/(total_interictal_hours - lost_interictal_hours)
    
    # computes the total interictal time in minutes as well as the "lost" interictal
    # time caused by the algorithm's refractory behavior
    def getInterictalTime(predicted_class, shuffled_class, sop, sph, window_step):
        refractory_period = (sop + sph) / window_step
        alarm_start_idx = np.where(np.diff(predicted_class)==1)[0] + 1
        
        # find idx where inter-ictal and pre-ictal stages end
        if shuffled_class[0] != 1:
            interictal_end_idx = np.where(np.diff(shuffled_class)==1)[0] + 1
        else:
            interictal_end_idx = np.array([0])
        preictal_end_idx = np.where(np.diff(shuffled_class)==(-1))[0] + 1
        if preictal_end_idx.size == 0:
            preictal_end_idx = np.array([len(shuffled_class)])
        
        # label "lost" interictal time as 1's in the interictal vector
        interictal = np.zeros(len(shuffled_class))
        
        for alarm in alarm_start_idx:
            interictal[int(alarm):int(alarm + refractory_period)] = 1
        
        interictal[interictal_end_idx[0]:preictal_end_idx[0]] = 0 # the preictal period doesn't count for the "lost" time
        
        lost_interictal_minutes = np.count_nonzero(interictal == 1) * window_step
        
        interictal_idx = np.where(np.array(shuffled_class)==0)[-1]
        time_in_minutes = len(interictal_idx) * window_step
        return time_in_minutes, lost_interictal_minutes
        
            
