"""
a script with methods used for the
control methodology

"""

import numpy as np
from sklearn import metrics
from scipy import stats


# by having a seizure data, calculate its label for a giving pre-ictal period
# output: vector with inter-ictal/ictal labels as 000000000.....0000001111111            
def getLabelsForSeizure(seizure_data, preictal_period):
    # create a vector of zeros
    labels=np.zeros([1,seizure_data.shape[1]])
    # convert the preictal period to number samples
    preictal_period_in_samples=preictal_period*12
    # put the pre-ictal period in the vector, put 1111111's
    labels[0,labels.size-preictal_period_in_samples:labels.size]=1;
    
    return labels

# a code to remove features with var<1e-9
def removeConstantFeatures(features):
    constant_features_index=[]
    
    #finding features with corr<1e-9
    for i in range(0,1,features.shape[1]):
        if np.var(features[:,i])<1e-9:
            constant_features_index.append(i)
      
    # deleting these features          
    features=np.delete(features,constant_features_index,axis=1)
    return [constant_features_index,features]

# a code to remove features with corr>0.95
def removeRedundantFeatures(features):
    redundant_features_index=[]
    
    #finding features with corr>0.95
    for i in range(0,1,features.shape[1]):
        for j in range(i,1,features.shape[1]):
            if abs(np.corrcoef(features[:,i],features[:,j])[0][1])>0.95:
                redundant_features_index.append(j)
    
    # deleting these features          
    features=np.delete(features,redundant_features_index,axis=1)
    return [redundant_features_index,features]

# a code to select the best n_features, by using correlation filter method
def filterFeatureSelectionCorr(features,labels,n_features):
    feature_corr_value=np.ones(features.shape[1])
    # get all features correlation with label
    for i in range(0,1,features.shape[1]):
        feature_corr_value[i]=abs(np.corrcoef(features[:,i],np.transpose(labels))[0][1])
        
    # sort features by descend order and get the best n_features
    # argsort sorts ascendetly, thus, we argsorted the negatives, to get it
    #   in descent order :)
    indexes_sorted_features=np.argsort(-feature_corr_value)
    
    # keep the best n_features
    indexes_sorted_features=indexes_sorted_features[0:n_features]
    features=features[:,indexes_sorted_features]
    
    return [indexes_sorted_features, features]


# a code to select the best n_features, by using AUC filter method
def filterFeatureSelectionAUC(features,labels,n_features):
    feature_auc_value=np.ones(features.shape[1])
    # get all features correlation with label
    for i in range(0,1,features.shape[1]):
        fpr, tpr, thresholds = metrics.roc_curve(labels, features[:,i], pos_label=1)
        feature_auc_value[i]=1-metrics.auc(fpr,tpr)
        
    # sort features by ascend order and get the best n_features
    # argsort sorts ascendently, which provides us the best auc since we made 1-auc
    # thus, we argsorted ascendetly to get in descent order :)
    indexes_sorted_features=np.argsort(feature_auc_value)
    
    # keep the best n_features
    indexes_sorted_features=indexes_sorted_features[0:n_features]
    features=features[:,indexes_sorted_features]
    
    return [indexes_sorted_features, features]

# a code that processes the classifier output with the firing power
# and refractory behavior
def FiringPowerAndRefractoryPeriod(predicted_labels,pre_ictal, window_length):
    predicted_labels=FiringPower(predicted_labels,pre_ictal,window_length)
    predicted_labels=RefractoryPeriod(predicted_labels,pre_ictal,window_length)
    
    return predicted_labels

#a code to implement the firing power
# which is a moving average filter (low-pass)
# with the size of the pre-ictal period
def FiringPower(predicted_labels,pre_ictal,window_length):
    kernel_size =int(pre_ictal*(60/window_length))
    kernel = np.ones(kernel_size) / kernel_size
    predicted_labels = np.convolve(predicted_labels, kernel, mode='same')
    
    threshold=0.7
    predicted_labels = [1 if predicted_labels_ > threshold else 0 for predicted_labels_ in predicted_labels]
    
    return predicted_labels

# a code to implement the refractory period
# the period in which you cannot let your classifier
# send an alarm, as it has already previously sent one
def RefractoryPeriod(predicted_labels,pre_ictal,window_length):
    refractory_bar_count=0
    refractory_on=False
    for i in range(0,len(predicted_labels)):
        if refractory_on==False:
            # when a new alarm is found, a refractory period begins
            if predicted_labels[i]==1:
                refractory_on=True
        else:
            # if we are on the refractory period, we set the labels to 0
            # and count the refractory period
            predicted_labels[i]=0
            refractory_bar_count=refractory_bar_count+1
            
            #when the refractory period reaches its period, it ends
            # and the refractory bar is set to 0
            if (refractory_bar_count >= pre_ictal*(60/window_length)):
                refractory_on=False
                refractory_bar_count=0
    
    return predicted_labels
                
# a code to verify if a model predicted the seizure
def didItPredictTheSeizure(predicted,labels):
    pre_ictal_length=len(np.argwhere(labels))
    pre_ictal_beginning=np.argwhere(labels)[0][0]
    did_it_predict_the_seizure=1 in predicted[pre_ictal_beginning:pre_ictal_beginning+pre_ictal_length]
    
    return did_it_predict_the_seizure

# a code to calculate the number of false alarms    
def calculateNumberFalseAlarms(predicted,labels):
     pre_ictal_beginning=np.argwhere(labels)[0][0]
     number_false_alarms=np.sum(predicted[0:pre_ictal_beginning])
    
     return number_false_alarms   
        
def specificity(tn,fp,fn,tp):
    return (tn/(tn+fp))

def sensitivity(tn,fp,fn,tp):
    return (tp/(tp+fn))

# code to retrieve the duration of time in which is possible to trigger an alarm
def calculateFPRdenominator(predicted,labels,pre_ictal,window_length):
    pre_ictal_beginning=np.argwhere(labels)[0][0]
    FPR_denominator=0;
    number_of_false_alarms=calculateNumberFalseAlarms(predicted,labels)
    false_alarm_indexes=np.argwhere(predicted[0:pre_ictal_beginning])
    
    for i in range(0,number_of_false_alarms):
        if abs(false_alarm_indexes[i][0]-pre_ictal_beginning)>(pre_ictal*(60/window_length)):
            FPR_denominator=FPR_denominator+pre_ictal/60
        else:
            FPR_denominator=FPR_denominator+abs(false_alarm_indexes[i][0]-pre_ictal_beginning)*window_length/3600
    
    inter_ictal_length=(len(labels)-sum(labels))*window_length/3600       
    return inter_ictal_length-FPR_denominator

# code to shuffle the pre-seizure labels for the surrogate
def shuffle_labels(labels, sop_length):
    firing_power_threshold=0.7
    surrogate_labels=np.zeros(len(labels))
    
    sop_size=np.sum(labels)
    sop_beginning_index=np.random.randint(sop_size*firing_power_threshold,len(labels)-sop_size)
    
    for i in range (0,int(sop_size)):
        surrogate_labels[sop_beginning_index+i]=1
        
    return surrogate_labels

#code that performs surrogate analysis and retrieves its sensitivity
    # in other words, how many times it predicted the surrogate seizure
    # in 30 chances
def surrogateSensitivity(predicted,labels):
    seizure_sensitivity=0
    sop_length=len(np.argwhere(labels))
    #lets do this 30 times
    surrogate_labels=shuffle_labels(labels,sop_length)
    seizure_sensitivity=seizure_sensitivity+didItPredictTheSeizure(predicted,surrogate_labels)
    
    return seizure_sensitivity

def t_test_one_independent_mean(population_mean, population_std, sample_mean, number_samples):   
    tt=abs(population_mean-sample_mean)/(population_std/np.sqrt(number_samples))
    
    pval = stats.t.sf(np.abs(tt), number_samples-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
    #print('t-statistic = %6.3f pvalue = %6.4f' % (tt, pval))
    
    return [tt,pval]