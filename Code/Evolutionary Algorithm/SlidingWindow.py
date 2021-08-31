"""
The code where we slide our window throughout all pre-processed features,
in order words: where we construct our hyper-features (second-level features)
by using the first-level features.

"""

import numpy as np
from Feature import Feature

class SlidingWindow:
    
    def __init__(self, features_list, time_step):
        self.features = features_list
        self.length = self.computeLength()
        self.step = time_step
    
    def __repr__(self):  
        return "SlidingWindow()"
    
    def __str__(self):
        number_features = self.getNumberOfFeatures()
        return f"Sliding window: {number_features} features, step = {self.step} mins"
    
    # retrieves the number of features in the sliding window
    def getNumberOfFeatures(self):
        return len(self.features)
    
    # retrieves the names of the decoded first-level features to be extracted
    def getFeatureLabels(self):
        feature_names = []
        for i in range(self.getNumberOfFeatures()):
            feature_names.append(self.features[i].decodeElectrode() + "_" + self.features[i].checkActiveGenes())
        
        return feature_names
    
    # retrieves each feature's delay and window length (needed in various methods)
    def getDelayAndWindowLength(self):
        delays = []
        window_lengths = []
        for i in range(self.getNumberOfFeatures()):
            delays.append(self.features[i].decodeDelay())
            window_lengths.append(self.features[i].decodeWindowLength())
        
        return delays, window_lengths
    
    # computes sliding window length from each feature's delays and window lenghts
    def computeLength(self):
        delays, window_lengths = self.getDelayAndWindowLength()
        
        min_delay = np.min(delays)
        max_time_away_from_onset = np.max(delays) + window_lengths[np.argmax(delays)]
        
        return max_time_away_from_onset - min_delay
    
    # returns the starting indexes of each feature on the timeline according 
    # to their delays and window lengths to prepare for extraction
    def initialize(self):
        delays, window_lengths = self.getDelayAndWindowLength()
        
        max_time_away_from_onset = np.max(np.add(delays, window_lengths))
        
        starts = []
        ends = []
        
        # conversion: 12 5-second windows = 1 minute
        for i in range(len(delays)):  
            ends.append(int((max_time_away_from_onset - delays[i]) * 12))
            starts.append(int((max_time_away_from_onset - delays[i] - window_lengths[i]) * 12))
        
        return starts, ends
    
    # applies a mathematical operator (average, variance or integral) to a given 
    # array of feature values
    def applyOperator(self, data, operator):
        if operator == "mean":
            return np.mean(data)
        elif operator == "variance":
            return np.var(data)
        elif operator == "integral":
            return np.trapz(data)
        elif operator == "max":
            return np.max(data)
        else:
            return []
    
    # returns the index of the feature matrix containing the respective feature value,
    # that is, with a given electrode and first-level feature (name)
    def getFeatureIndex(self, legend, name):
        return np.argwhere(legend == name)[0][0]
    
    # returns the preictal period (in minutes) as a sum of the shortest delay (in
    # the list of features) with the preictal threshold of the individual
    def computePreictal(self, preictal_threshold):
        delays, window_lengths = self.getDelayAndWindowLength()
        return np.min(delays) + preictal_threshold
    
    # extracts second-level features based on the given list; uses a feature matrix
    # and the corresponding legend with the following format: electrode_firstlevelfeature
    # labelling the pre-ictal class depends on the individual's pre-ictal threshold
    # as well as the shortest delay time of the individual's group of features
    def extractFeatures(self, feature_matrix, legend, preictal_threshold):
        new_feature_matrix = []
        new_legend = []
        new_label = [] # should be added to the new_feature_matrix
        
#        new_missingdata = []
#        new_flats = []
#        new_saturated = []
#        missingdata_idx = np.argwhere(legend == 'missingdata')[0][0]
#        missingdata = feature_matrix[missingdata_idx,:]
#        #flats_idx = np.argwhere(legend == 'flat')[0][0]
#        flats_idx = [i for i, s in enumerate(legend) if 'flat' in s]
#        flats = feature_matrix[flats_idx,:]
#        flats = feature_matrix[flats_idx,:]
#        saturated_idx = [i for i, s in enumerate(legend) if 'saturated' in s]
#        saturated = feature_matrix[saturated_idx,:]
        
        
        # get initial window boundaries for each feature in the timeline
        starts, ends = self.initialize()
        
        # # include CIRCADIAN RHYTHM feature:
        # delays, window_lengths = self.getDelayAndWindowLength()
        # # use the same windows as the feature with the shortest delay
        # circadian_idx = np.argmin(delays) 
        
        # get labels and onset (stopping criteria for extraction)
        labels_idx = np.argwhere(legend == 'class')[0][0]
        labels = feature_matrix[labels_idx,:]
        
        onset_idx = np.argmax(labels > 1)
        
        # define pre-ictal for labelling (pre-ictal = 1)
        preictal_start_idx = onset_idx - self.computePreictal(preictal_threshold) * 12
        labels[preictal_start_idx:onset_idx] = 1
        
        # extract second-level features until feature with the shortest delay
        # reaches the onset
        max_end = int(np.max(ends))
        while max_end < onset_idx:
#            # compute max missing data and flat percentage from the windows
#            max_missingdata = np.max(missingdata[int(np.min(starts)):int(np.max(ends))])
#            new_missingdata.append(max_missingdata)
#            
#            max_flats = np.max(flats[:,int(np.min(starts)):int(np.max(ends))], axis = 1)
#            new_flats.append(max_flats)
#            
#            max_saturated = np.max(saturated[:,int(np.min(starts)):int(np.max(ends))], axis = 1)
#            new_saturated.append(max_saturated)
            
            extracted_features = []
            for i in range(len(starts)): # +1 to include circadian rhythm
                if i == len(starts): # last one = circadian rhythm
                    feature_idx = self.getFeatureIndex(legend, "FP1_circadian")
                    # data = feature_matrix[feature_idx, starts[circadian_idx]:ends[circadian_idx]]
                    # secondlevel_feature = self.applyOperator(data, "max") # use mean operator
                else:
                    feature_idx = self.getFeatureIndex(legend, self.getFeatureLabels()[i])
                    data = feature_matrix[feature_idx,starts[i]:ends[i]]
                    secondlevel_feature = self.applyOperator(data, self.features[i].decodeMathematicalOperator())
                    starts[i] += int((self.step * 12))
                    ends[i] += int((self.step * 12))
                
                extracted_features.append(secondlevel_feature)
            
            new_label.append(int(labels[max_end]))
            max_end = int(np.max(ends))
            
            new_feature_matrix.append(extracted_features)
        
        # generate new legend vector containing names of second-level features
        for i in range(len(starts)): # +1 to include circadian rhythm
            if i == len(starts): # last one = circadian rhythm
                new_legend.append("circadian")
            else:
                new_legend.append(self.features[i].decodeMathematicalOperator() + "_" +
                              self.getFeatureLabels()[i] + "_delay" + str(self.features[i].decodeDelay()))
        
        new_legend.append("class")
#        new_legend.append("missingdata")
#        new_legend.extend(legend[flats_idx].tolist())
#        new_legend.extend(legend[saturated_idx].tolist())
        
        new_feature_matrix = np.vstack((np.array(new_feature_matrix).T, new_label))   
#        new_feature_matrix = np.vstack((np.array(new_feature_matrix), new_missingdata))
#        new_feature_matrix = np.vstack((np.array(new_feature_matrix), np.array(new_flats).T))
#        new_feature_matrix = np.vstack((np.array(new_feature_matrix), np.array(new_saturated).T))
        
        return new_feature_matrix, np.array(new_legend)
            