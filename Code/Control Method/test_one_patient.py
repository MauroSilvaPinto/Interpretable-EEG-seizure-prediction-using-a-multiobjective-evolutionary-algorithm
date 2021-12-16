"""
control method for one patient
    it returns seizure sensitivity, FPR/h
    and surrogate analysis

"""


# control method for one patient
#   to retrieve the fitness value from 
#   the k number of features and pre-ictal period 

# mauro pinto

import os
import numpy as np

from Utils import getLabelsForSeizure
from Utils import removeConstantFeatures
from Utils import removeRedundantFeatures
from Utils import filterFeatureSelectionCorr
from Utils import filterFeatureSelectionAUC
from Utils import FiringPowerAndRefractoryPeriod
from Utils import didItPredictTheSeizure
from Utils import calculateNumberFalseAlarms
from Utils import calculateFPRdenominator
from Utils import surrogateSensitivity
from Utils import t_test_one_independent_mean

from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier

patient_id=1321803
pre_ictal=35
k_features=10
total_seizures=4


def testOnePatient(patient_id,pre_ictal,k_features,total_seizures):

    # due to the np.delete function
    import warnings
    warnings.filterwarnings("ignore")
    
    # where the data is
    # go back to data folder
    os.chdir("..")
    os.chdir("..")
    os.chdir("Data")
    os.chdir("Processed_data")
    path=os.getcwd()
    # go to where the data is
    os.chdir(path)
        
    ########################### Loading Training Seizures ########################
    
    # load training seizures
    seizure_1_data=np.load("pat"+str(patient_id)+"_seizure1_featureMatrix.npy");
    seizure_2_data=np.load("pat"+str(patient_id)+"_seizure2_featureMatrix.npy");
    seizure_3_data=np.load("pat"+str(patient_id)+"_seizure3_featureMatrix.npy");
            
    #removing ictal data, that is, where - last line (class)
    #  ictal is equal to 2
    #  inter-ictal is equal to 0
    seizure_1_data=seizure_1_data[:,(np.where(seizure_1_data[-1,:]==0)[0])]
    seizure_2_data=seizure_2_data[:,(np.where(seizure_2_data[-1,:]==0)[0])]
    seizure_3_data=seizure_3_data[:,(np.where(seizure_3_data[-1,:]==0)[0])]
    
    ############## Seizure 1 ####################
    # removing the class label for the feature vector, for one seizure
    training_features_1=seizure_2_data[0:-1,:]
    # retrieving the testing labels for one seizure
    training_labels_1=getLabelsForSeizure(training_features_1,pre_ictal)
                 
    ############## Seizure 2 ####################       
    # removing the class label for the feature vector, for one seizure
    training_features_2=seizure_3_data[0:-1,:]
    # retrieving the testing labels for one seizure
    training_labels_2=getLabelsForSeizure(training_features_2,pre_ictal)
    
    ############## Seizure 3 ####################
    # removing the class label for the feature vector, for one seizure
    training_features_3=seizure_3_data[0:-1,:]
    # retrieving the testing labels for one seizure
    training_labels_3=getLabelsForSeizure(training_features_3,pre_ictal)
    ##############################################
                        
    # concatenate both testing_features and testing labels
    training_features=np.concatenate([training_features_1, training_features_2, training_features_3], axis=1)
    training_labels=np.concatenate([training_labels_1, training_labels_2, training_labels_3], axis=1)
                        
    del training_features_1
    del training_features_2
    del training_features_3
    del training_labels_1
    del training_labels_2
    del training_labels_3
    
    # we transpose the feature vector to have sample x feature    
    training_features=np.transpose(training_features)
                        
    # we fix the label vectors
    training_labels=np.transpose(training_labels)
    
    
    
    
    ################ Machine Learning pipeline for training seizures #############
                    
    ################### Missing value imputation ###############    
    # find missing values for training
    mising_values_indexes=np.unique(np.argwhere(np.isnan(training_features))[:,0])       
    training_features=np.delete(training_features,mising_values_indexes,axis=0)
    training_labels=np.delete(training_labels,mising_values_indexes,axis=0)
    
                    
    ################## Removing Constant and Redundant Values ###########
    # remove constant features from training features
    [constant_indexes,training_features]=removeConstantFeatures(training_features);
                    
    # remove redundant features from training (corr>0.95)
    #[redundant_indexes,training_features]=removeRedundantFeatures(training_features);
                    
    #################### Standardization #######################
    # training features
    scaler = preprocessing.StandardScaler().fit(training_features)
    training_features=scaler.transform(training_features)
                    
    #################### Feature Selection #######################               
    #Filter selection with corr, to get the best n_features
    # this has the intention of getting the process less heavy
    n_features=100
    # feature selection for training
    [indexes_corr_features,training_features]=filterFeatureSelectionCorr(training_features,training_labels,n_features)
                    
    #Filter selection with auc, to get the best n_features
    # this has the intention of getting the process less heavy
    n_features=25
    # feature selection for training
    [indexes_auc_features,training_features]=filterFeatureSelectionAUC(training_features,training_labels,n_features)
            
    # Recursive feature elimination selection
    n_features=k_features
            
    # training
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector = selector.fit(training_features, np.ravel(training_labels))
                    
    # eliminating non predictive features in training
    training_features=selector.transform(training_features)
                            
    ###################### Random Forest Training #######################
    clf = RandomForestClassifier(max_depth=2, random_state=0, class_weight='balanced_subsample')
    clf.fit(training_features, training_labels)
    
    
    
    
    
    
    
    ####################### Loading Testing Seizures #############################
    
    testing_features=[]
    testing_labels=[]
    for seizure_k in range(3,total_seizures):
        # load training seizures
        seizure_data=np.load("pat"+str(patient_id)+"_seizure"+str(seizure_k)+"_featureMatrix.npy");
    
        #removing ictal data, that is, where - last line (class)
        #  ictal is equal to 2
        #  inter-ictal is equal to 0
        seizure_data=seizure_data[:,(np.where(seizure_data[-1,:]==0)[0])]
        
        # removing the class label for the feature vector, for one seizure
        seizure_features=seizure_data[0:-1,:]
        # retrieving the labels for one seizure
        seizure_labels=getLabelsForSeizure(seizure_features,pre_ictal)
        
        # we transpose the feature vector to have sample x feature    
        seizure_features=np.transpose(seizure_features)
                        
        # we fix the label vectors
        seizure_labels=np.transpose(seizure_labels)
    
        testing_features.append(seizure_features)
        testing_labels.append(seizure_labels)
    
    ################ Machine Learning pipeline for testing seizures #############
    
    # iterate all seizures
    for i in range(0,len(testing_labels)):
        mising_values_indexes=np.unique(np.argwhere(np.isnan(testing_features[i]))[:,0])
        testing_features[i]=np.delete(testing_features[i],mising_values_indexes,axis=0)
        testing_labels[i]=np.delete(testing_labels[i],mising_values_indexes,axis=0)
        
                
    ################### Missing value imputation ###############  
    for i in range(0,len(testing_labels)):
        # find missing values for testing
        mising_values_indexes=np.unique(np.argwhere(np.isnan(testing_features[i]))[:,0])
        testing_features[i]=np.delete(testing_features[i],mising_values_indexes,axis=0)
        testing_labels[i]=np.delete(testing_labels[i],mising_values_indexes,axis=0)
                    
    ################## Removing Constant and Redundant Values ###########
    # remove the same features from testing features
    for i in range(0,len(testing_labels)):
        testing_features[i]=np.delete(testing_features[i],constant_indexes,axis=1)
     
    # for i in range(0,len(testing_labels)):               
        # remove the same features from testing features
        # testing_features[i]=np.delete(testing_features[i],redundant_indexes,axis=1)
                    
    #################### Standardization #######################                
    for i in range(0,len(testing_labels)): 
        # testing features
        testing_features[i]=scaler.transform(testing_features[i])
                    
    #################### Feature Selection #######################   
    for i in range(0,len(testing_labels)):             
        # keeping the selected features for testing
        testing_features[i]=testing_features[i][:,indexes_corr_features]
        testing_features[i]=testing_features[i][:,indexes_auc_features]
    
    for i in range(0,len(testing_labels)):         
        # Recursive feature elimination selection
        # eliminating non predictive features in testing
        testing_features[i]=testing_features[i][:,np.argwhere(selector.get_support()==True)[:,0]]
                            
    ###################### Classification #######################
    predicted_labels=[]
    for i in range(0,len(testing_labels)): 
        predicted_labels.append(clf.predict(testing_features[i]))
    ####################### delete SPH samples ######################
    # windows of 5 seconds and thus: 10 minutes of 10 sph
    # corresponds to 120 samples (12 per minute)
    window_length=5
    sph=10    
    for i in range(0,len(testing_labels)): 
        predicted_labels[i]=predicted_labels[i][0:len(predicted_labels)-1-int((60/window_length))*sph-1-1] 
        testing_labels[i]=testing_labels[i][0:len(testing_labels)-1-int((60/window_length))*sph-1-1] 
    
    ###################### Firing Power + Refractory #############
    for i in range(0,len(testing_labels)):
        predicted_labels[i]=FiringPowerAndRefractoryPeriod(predicted_labels[i],pre_ictal,window_length)
        
    ##################### Performance Metrics #####################
    seizure_sensitivity=0
    fpr_denominator=0
    number_false_alarms=0
        
    for i in range(0,len(testing_labels)):
        seizure_sensitivity=seizure_sensitivity+didItPredictTheSeizure(predicted_labels[i],testing_labels[i])
        number_false_alarms=number_false_alarms+calculateNumberFalseAlarms(predicted_labels[i],testing_labels[i])           
        fpr_denominator=fpr_denominator+calculateFPRdenominator(predicted_labels[i],testing_labels[i],pre_ictal,window_length) 
    
    FPR=number_false_alarms/fpr_denominator
    seizure_sensitivity=seizure_sensitivity/len(testing_labels)
    
    print("FPR/h: "+ str(FPR[0]))
    print("Sensitivity: "+ str(seizure_sensitivity))
    ###################### Statistical Validation ####################
    
    surrogate_sensitivity=[]
        
    for i in range(0,len(testing_labels)):
        for j in range(0,30):
            surrogate_sensitivity.append(surrogateSensitivity(predicted_labels[i],testing_labels[i]))
        
    surrogate_sensitivity=surrogate_sensitivity
    print("Surrogate Sensitivity "+str(np.mean(surrogate_sensitivity))+" +/- " + str(np.std(surrogate_sensitivity)))
    
    val=0
    pval=1
    print("Does it perform above chance?")
    if (np.mean(surrogate_sensitivity)<seizure_sensitivity):
        [tt,pval]=t_test_one_independent_mean(np.mean(surrogate_sensitivity), np.std(surrogate_sensitivity), seizure_sensitivity, 30)
        if pval<0.05:
            print("Yes")
            val=1
        else:
            print("No")
    else:
        print("No")
        
                    
    return [seizure_sensitivity, FPR, np.mean(surrogate_sensitivity), np.std(surrogate_sensitivity),pval,val]



testOnePatient(patient_id,pre_ictal,k_features,total_seizures)
