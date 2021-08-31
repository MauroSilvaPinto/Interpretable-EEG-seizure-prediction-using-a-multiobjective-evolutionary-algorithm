"""
Classifier class.

A class where we compute all stuf related with seizure prediciton in machine learning:
    calculate the inter-ictal time available, FPR/h, and sensitivity,
    and where we consider the SPH, remove NaN data, standardize the data, train a Classifier,
    apply the refractor period, etc

"""

import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
#from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy import signal
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
#from imblearn.over_sampling import RandomOverSampler
#from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression

class Classifier:
    
    def __init__(self, method):
        self.type = method
        self.trained = False
        self.model = None
        self.scaler = None
    
    def __repr__(self):  
        return "Classifier()"
    
    def __str__(self):
        return f"Classifier: {type}, trained = {self.trained}"
    
    # reshapes dataset so that each row is a sample and each column is a feature
    # to ensure compatibility with scikit-learn
    def reshapeData(dataset):
        return dataset.T
    
    # separates feature matrix into feature values and class label
    def splitFeatureMatrix(feature_matrix, legend):
        labels_idx = np.argwhere(legend == 'class')[0][0]
        labels = feature_matrix[labels_idx,:]
        
#        missingdata_idx = np.argwhere(legend == 'missingdata')[0][0]
#        missingdata = feature_matrix[missingdata_idx,:]
#        
#        #flats_idx = np.argwhere('flat' in legend)[0][0]
#        flats_idx = [i for i, s in enumerate(legend) if 'flat' in s]
#        flats = feature_matrix[flats_idx,:]
#        
#        saturated_idx = [i for i, s in enumerate(legend) if 'saturated' in s]
#        saturated = feature_matrix[saturated_idx,:]
        
        idx_to_remove = []
        idx_to_remove.append(labels_idx)
#        idx_to_remove.append(missingdata_idx)
#        idx_to_remove.extend(flats_idx)
#        idx_to_remove.extend(saturated_idx)
        features = np.delete(feature_matrix, idx_to_remove, 0)
        
        #return features, labels, missingdata, flats, saturated
        return features, labels
    
    # applies z-scoring to dataset; the scaler is specific to each classifier,
    # since the mean and std deviation are the ones from the training data
    def standardizeData(self, dataset, testing):
        # check if array is in correct format for sklearn: (samples, features)
        if dataset.shape[0] < dataset.shape[1]:
            dataset = Classifier.reshapeData(dataset)
            
        # fit scaler to initial training data and save mean + std for later
        if self.scaler == None:
            self.scaler = StandardScaler().fit(dataset)
        
        # only fit the scaler to data during training, never during testing!
        if testing == False:
            self.scaler = StandardScaler().fit(dataset)
            
        return self.scaler.transform(dataset)
        # return stats.zscore(dataset, axis = 1, ddof = 1)
    
    # replaces NaN values in the dataset with the mean value of each feature 
    # (e.g. Hjorth parameters may have NaN values when derivatives are zero)
    def replaceNaN(dataset):
        feature_means = np.nanmean(dataset, axis=1)
        
        for i in range(len(feature_means)):
            dataset[i, np.isnan(dataset)[i,:]] = feature_means[i]
            
        return dataset
    
    # computes weights for each of the classes
    def computeBalancedClassWeights(labels):
        class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
        
        return class_weights
    
    # computes weights for each sample, based on their class
    def computeSampleWeights(labels, class_weights):
        sample_weights = np.zeros(len(labels))
        
        sample_weights[np.where(labels==0)[0]] = class_weights[0]
        sample_weights[np.where(labels==1)[0]] = class_weights[1]
       
        return sample_weights
    
    # modifies sample weights to implement SPH, that is, the samples before each seizure
    # are given a null weight for training
    def applySPH(labels, sample_weights, sph, window_step):
        new_sample_weights = sample_weights[:]
        interictal_end_idx = np.where(np.diff(labels)==1)[0] + 1
        sph_windows = sph / window_step
        
        for idx in interictal_end_idx:
            new_sample_weights[int(idx-sph_windows):idx] = 0
        
        return new_sample_weights
    
    # modifies sample weights to reduce the influence of artifacts and missing data
    # missing data -> if any, weight = 0
    # flat data -> if more than 30%, weight = 0
    # saturated data -> if any, weight = 0
    # uses legend to check which electrodes each feature uses; weights are only applied
    # if the above conditions apply to any one of those electrodes
    def applyArtifactWeights(missingdata, flats, saturated, sample_weights, legend):
        final_sample_weights = np.array(sample_weights[:])
        
        # # find relevant row indexes (for the electrodes in the feature set)
        # number_features = np.argwhere(legend == 'class')[0][0] # class is always the next thing after the feature labels...
        # electrodes = []
        # for i in range(number_features):
        #     electrodes.append(legend[i].split('_')[1])
        # electrodes = np.unique(np.array(electrodes))
        
        # flats_legend = [i for i in legend.tolist() if 'flat' in i]
        # saturated_legend = [i for i in legend.tolist() if 'saturated' in i]
        # flats_rows = []
        # saturated_rows = []
        # for e in electrodes:
        #     flats_rows.append(flats_legend.index(e + '_flat'))
        #     saturated_rows.append(saturated_legend.index(e + '_saturated'))
            
        # # apply weights according to type of artifact found
        # missingdata_idx = np.nonzero(missingdata > 0)[0]
        # final_sample_weights[missingdata_idx] = 0
        
        # flats_relevant = flats[flats_rows,:]
        # flats_idx = np.unique(np.nonzero(flats_relevant > 30)[1])
        # final_sample_weights[flats_idx] = 0
        
        # saturated_relevant = saturated[saturated_rows,:]
        # saturated_idx = np.unique(np.nonzero(saturated_relevant > 30)[1])
        # final_sample_weights[saturated_idx] = 0
        
        return final_sample_weights
    
    # trains a classifier with a given dataset and corresponding legend (to find labels etc.)
    def trainClassifier(self, dataset, legend):
        #features, labels, missingdata, flats, saturated = Classifier.splitFeatureMatrix(dataset, legend)
        features, labels = Classifier.splitFeatureMatrix(dataset, legend)
        features = Classifier.replaceNaN(features)
        features_zscored = self.standardizeData(features, False)
        #features_zscored = features
        
        # balancing: sample weights (according to class prevalence, SPH and artifact presence)
        class_weights = Classifier.computeBalancedClassWeights(labels)
        sample_weights_class = Classifier.computeSampleWeights(labels, class_weights)
        sample_weights_sph = Classifier.applySPH(labels, sample_weights_class, 10, 0.5) # SPH = 10 MINS; STEP = 30 SECONDS
        #sample_weights = Classifier.applyArtifactWeights(missingdata, flats, saturated, sample_weights_sph, legend)
        sample_weights = sample_weights_sph
        
        if self.type == "SVM_linear":
            clf = svm.SVC(kernel='linear')
            self.model = clf.fit(features_zscored, labels, sample_weight = sample_weights)
            self.trained = True
        elif self.type == "SVM_RBF":
            clf = svm.SVC(gamma = 'scale', kernel = 'rbf')
            self.model = clf.fit(features_zscored, labels, sample_weight = sample_weights)
            self.trained = True
        elif self.type == 'KNN':
            clf = KNeighborsClassifier(n_neighbors = 5)
            self.model = clf.fit(features_zscored, labels)
            self.trained = True
        elif self.type == 'logreg':
            clf = LogisticRegression(solver = "lbfgs")
            self.model = clf.fit(features_zscored, labels, sample_weight = sample_weights)
            self.trained = True
    
    # returns the trained classifier's prediction output for new, unseen data
    # as well as the original labels (to evaluate performance afterwards)
    def classifyData(self, dataset, legend):
        #features, labels, missingdata, flats, saturated = Classifier.splitFeatureMatrix(dataset, legend)
        features, labels = Classifier.splitFeatureMatrix(dataset, legend)
        features = Classifier.replaceNaN(features)
        features_zscored = self.standardizeData(features, True)
        
        if self.trained == True:
            return self.model.predict(features_zscored), labels
        else:
            return [], []
        
    # applies the Firing Power post-processing method to the given classifier output
    def applyFiringPower(predicted_class, true_class, threshold):
        # FP = output / preictal_size
        try:
            preictal_size = np.where(np.diff(true_class)==-1)[0][0] - np.where(np.diff(true_class)==1)[0][0]
            
        except: # only one seizure, so the previous formula doesn't work...
            preictal_size = np.count_nonzero(true_class == 1)
        
        # design moving average filter with size = preictal_size
        b = np.ones(preictal_size)
        mov_avg = b / np.sum(b)
        
        # apply moving average filter to raw classifier output
        FP_output = signal.lfilter(mov_avg, 1, np.array(predicted_class))
        predicted_class_processed = np.where(np.array(FP_output) >= threshold, 1, 0)
        
        return predicted_class_processed
    
    # applies a refractory period behavior: after firing an alarm, no other alarm
    # can be raised until after SPH + SOP minutes
    def applyRefractoryBehavior(predicted_class, sop, sph, window_step):
        predicted_class_processed = np.array(predicted_class[:])
        refractory_period = (sop + sph) / window_step
        
        # detect when alarms start and end
        class_diff = np.diff(predicted_class)
        starts = np.nonzero(class_diff == 1)[0]
        starts = [i+1 for i in starts]
        ends = np.nonzero(class_diff == -1)[0]
        ends = [i+1 for i in ends]
        
        # in case an alarm is fired in the first sample
        if predicted_class[0] == 1:
            starts = [0] + starts
            
        # in case last predicted seizure lasts until the end
        if len(starts) != len(ends):
            ends.append(len(predicted_class))
            
        # if distance between current start and last start is >= ref. period,
        # make start[i] to ends[i] equal to 0
        for i in range(1, len(starts)):
            if starts[i] - starts[i-1] <= refractory_period: # MENOR OU IGUAL?
                predicted_class_processed[starts[i]:ends[i]] = 0
        
        return predicted_class_processed
    
    # computes the confusion matrix
    def getConfusionMatrix(predicted_class, true_class):
        return confusion_matrix(true_class, predicted_class, labels=[1,0])
    
    # computes sample sensitivity from a provided confusion matrix
    def getSampleSensitivity(predicted_class, true_class):
        confusion_matrix = Classifier.getConfusionMatrix(predicted_class, true_class)
        return (confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1]))
    
    # computes sample specificity from a provided confusion matrix
    def getSampleSpecificity(predicted_class, true_class):
        confusion_matrix = Classifier.getConfusionMatrix(predicted_class, true_class)
        return (confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1]))
    
    # computes the number of triggered alarms from a given classifier output
    def getNumberOfAlarms(predicted_class):
        return len(np.where(np.diff(predicted_class)==1)[0])
    
    # computes the number of existing seizures from a given label vector
    def getNumberOfSeizures(true_class):
        return len(np.where(np.diff(true_class)==1)[0])
    
    # computes the number of false alarms for the surrogate predictor
    def getNumberOfFalseAlarms(predicted_class, true_class):
        false_alarms = 0
        
        # find idx where inter-ictal and pre-ictal stages end
        if true_class[0] != 1:
            interictal_end_idx = np.where(np.diff(true_class)==1)[0] + 1
        else:
            interictal_end_idx = np.array([0])
        preictal_end_idx = np.where(np.diff(true_class)==(-1))[0] + 1
        if preictal_end_idx.size == 0:
            preictal_end_idx = np.array([len(true_class)])
        
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
    def getNumberOfTrueAlarms(predicted_class, true_class):
        true_alarms = 0
        
        # find idx where inter-ictal and pre-ictal stages end
        if true_class[0] != 1:
            interictal_end_idx = np.where(np.diff(true_class)==1)[0] + 1
        else:
            interictal_end_idx = np.array([0])
        preictal_end_idx = np.where(np.diff(true_class)==(-1))[0] + 1
        if preictal_end_idx.size == 0:
            preictal_end_idx = np.array([len(true_class)])
        
        # for every other raised alarm, check if it's within the seizure and outside the SPH (10 minutes = 20 samples)
        alarm_start_idx = np.where(np.diff(predicted_class)==1)[0] + 1
        for alarm in alarm_start_idx:
            if alarm >= interictal_end_idx[0] and alarm <= preictal_end_idx[0] - 20:
                true_alarms += 1
        
        return true_alarms
    
    # computes the seizure sensitivity: predicted_seizures / total_seizures
    def getSensitivity(predicted_class, true_class):
        predicted_seizures = Classifier.getNumberOfTrueAlarms(predicted_class, true_class)
        total_seizures = Classifier.getNumberOfSeizures(true_class)
        
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
    def getInterictalTime(predicted_class, true_class, sop, sph, window_step):
        refractory_period = (sop + sph) / window_step
        alarm_start_idx = np.where(np.diff(predicted_class)==1)[0] + 1
        
        # find idx where inter-ictal and pre-ictal stages end
        if true_class[0] != 1:
            interictal_end_idx = np.where(np.diff(true_class)==1)[0] + 1
        else:
            interictal_end_idx = np.array([0])
        preictal_end_idx = np.where(np.diff(true_class)==(-1))[0] + 1
        if preictal_end_idx.size == 0:
            preictal_end_idx = np.array([len(true_class)])
        
        # label "lost" interictal time as 1's in the interictal vector
        interictal = np.zeros(len(true_class))
        
        for alarm in alarm_start_idx:
            interictal[int(alarm):int(alarm + refractory_period)] = 1
        
        interictal[interictal_end_idx[0]:preictal_end_idx[0]] = 0 # the preictal period doesn't count for the "lost" time
        
        lost_interictal_minutes = np.count_nonzero(interictal == 1) * window_step
        
        interictal_idx = np.where(np.array(true_class)==0)[-1]
        time_in_minutes = len(interictal_idx) * window_step
        return time_in_minutes, lost_interictal_minutes
        
    # computes the time under false alarm
    def getTimeUnderFalseAlarm(predicted_class, true_class):
        interictal_idx = np.where(np.array(true_class) == 0)[-1]
        return np.sum(predicted_class[interictal_idx]) / len(interictal_idx)