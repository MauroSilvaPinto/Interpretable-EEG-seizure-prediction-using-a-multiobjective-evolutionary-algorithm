"""
control method for one patient
  to retrieve the fitness value from 
  the k number of features and pre-ictal period 
"""

import os
import numpy as np

from Utils import getLabelsForSeizure
from Utils import removeConstantFeatures
from Utils import removeRedundantFeatures
from Utils import filterFeatureSelectionCorr
from Utils import filterFeatureSelectionAUC

from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix

from Utils import specificity
from Utils import sensitivity

import time

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

patient_id=1321803
pre_ictal=35
k_features=10


def calculateFitness(patient_id,k_features,pre_ictal):
    t = time.process_time()
    contador=0;
    
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
    
    performance_values=0
    # for each pre-ictal period value, we make a 3-fold cross validation
    for k in range(0,3):
        #seizure_1 for testing, seizure_2 and seizure_3 for training
        if k==0:
            # removing the class label for the feature vector
            testing_features=seizure_1_data[0:-1,:]
            # retrieving the training labels
            testing_labels=getLabelsForSeizure(testing_features,pre_ictal)
                    
            # removing the class label for the feature vector, for one seizure
            training_features_1=seizure_2_data[0:-1,:]
            # retrieving the testing labels for one seizure
            training_labels_1=getLabelsForSeizure(training_features_1,pre_ictal)
                    
            # removing the class label for the feature vector, for one seizure
            training_features_2=seizure_3_data[0:-1,:]
            # retrieving the testing labels for one seizure
            training_labels_2=getLabelsForSeizure(training_features_2,pre_ictal)
                    
            # concatenate both testing_features and testing labels
            training_features=np.concatenate([training_features_1, training_features_2], axis=1)
            training_labels=np.concatenate([training_labels_1, training_labels_2], axis=1)
                    
            del training_features_1
            del training_features_2
            del training_labels_1
            del training_labels_2
                    
        #seizure_2 for testing, seizure_1 and seizure_3 for training
        elif k==1:
            # removing the class label for the feature vector
            testing_features=seizure_2_data[0:-1,:]
            # retrieving the training labels
            testing_labels=getLabelsForSeizure(testing_features,pre_ictal)
                        
            # removing the class label for the feature vector, for one seizure
            training_features_1=seizure_1_data[0:-1,:]
            # retrieving the testing labels for one seizure
            training_labels_1=getLabelsForSeizure(training_features_1,pre_ictal)
                        
            # removing the class label for the feature vector, for one seizure
            training_features_2=seizure_3_data[0:-1,:]
            # retrieving the testing labels for one seizure
            training_labels_2=getLabelsForSeizure(training_features_2,pre_ictal)
                        
            # concatenate both testing_features and testing labels
            training_features=np.concatenate([training_features_1, training_features_2], axis=1)
            training_labels=np.concatenate([training_labels_1, training_labels_2], axis=1)
                        
            del training_features_1
            del training_features_2
            del training_labels_1
            del training_labels_2
                    
        #seizure_3 for testing, seizure_1 and seizure_2 for training
        elif k==2:
            # removing the class label for the feature vector
            testing_features=seizure_3_data[0:-1,:]
            # retrieving the training labels
            testing_labels=getLabelsForSeizure(testing_features,pre_ictal)
                        
            # removing the class label for the feature vector, for one seizure
            training_features_1=seizure_1_data[0:-1,:]
            # retrieving the testing labels for one seizure
            training_labels_1=getLabelsForSeizure(training_features_1,pre_ictal)
                        
            # removing the class label for the feature vector, for one seizure
            training_features_2=seizure_2_data[0:-1,:]
            # retrieving the testing labels for one seizure
            training_labels_2=getLabelsForSeizure(training_features_2,pre_ictal)
                        
            # concatenate both testing_features and testing labels
            training_features=np.concatenate([training_features_1, training_features_2], axis=1)
            training_labels=np.concatenate([training_labels_1, training_labels_2], axis=1)
                        
            del training_features_1
            del training_features_2
            del training_labels_1
            del training_labels_2
                    
        # we transpose the feature vector to have sample x feature    
        training_features=np.transpose(training_features)
        testing_features=np.transpose(testing_features)
                    
        # we fix the label vectors
        training_labels=np.transpose(training_labels)
        testing_labels=np.transpose(testing_labels)
                
        ################### Missing value imputation ###############    
        # find missing values for training
        mising_values_indexes=np.unique(np.argwhere(np.isnan(training_features))[:,0])       
        training_features=np.delete(training_features,mising_values_indexes,axis=0)
        training_labels=np.delete(training_labels,mising_values_indexes,axis=0)
                
        # find missing values for testing
        mising_values_indexes=np.unique(np.argwhere(np.isnan(testing_features))[:,0])
        testing_features=np.delete(testing_features,mising_values_indexes,axis=0)
        testing_labels=np.delete(testing_labels,mising_values_indexes,axis=0)
                
        ################## Removing Constant and Redundant Values ###########
        # remove constant features from training features
        [constant_indexes,training_features]=removeConstantFeatures(training_features);
        # remove the same features from testing features
        testing_features=np.delete(testing_features,constant_indexes,axis=1)
                
        # remove redundant features from training (corr>0.95)
        #[redundant_indexes,training_features]=removeRedundantFeatures(training_features);
        # remove the same features from testing features
        #testing_features=np.delete(testing_features,redundant_indexes,axis=1)
                
                
        #################### Standardization #######################
        # training features
        scaler = preprocessing.StandardScaler().fit(training_features)
        training_features=scaler.transform(training_features)
                
        # testing features
        testing_features=scaler.transform(testing_features)
                
        #################### Feature Selection #######################
                
        #Filter selection with corr, to get the best n_features
        # this has the intention of getting the process less heavy
        n_features=100
        # feature selection for training
        [indexes_corr_features,training_features]=filterFeatureSelectionCorr(training_features,training_labels,n_features)
        # keeping the selected features for testing
        testing_features=testing_features[:,indexes_corr_features]
                
        #Filter selection with auc, to get the best n_features
        # this has the intention of getting the process less heavy
        n_features=25
        # feature selection for training
        [indexes_auc_features,training_features]=filterFeatureSelectionAUC(training_features,training_labels,n_features)
        # keeping the selected features for testing
        testing_features=testing_features[:,indexes_auc_features]
        
        # Recursive feature elimination selection
        n_features=k_features
        
        # training
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector = selector.fit(training_features, np.ravel(training_labels))
                
        # eliminating non predictive features in training
        training_features=selector.transform(training_features)
        # eliminating non predictive features in testing
        testing_features=testing_features[:,np.argwhere(selector.get_support()==True)[:,0]]
                        
        ###################### Random Forest Training #######################
        clf = RandomForestClassifier(max_depth=2, random_state=0, class_weight='balanced_subsample')
        clf.fit(training_features, training_labels)
        
        ###################### Performance Evaluation #########################
        predicted_labels=clf.predict(testing_features)
        tn, fp, fn, tp = confusion_matrix(testing_labels, predicted_labels).ravel()    
                
        performance=np.sqrt(specificity(tn,fp,fn,tp)*sensitivity(tn,fp,fn,tp))
                
        # calcular performance pre_ictal com todos os k
        performance_values=performance_values+performance
                
        ############# to let me know how the loading bar is ##############
                
        contador=contador+1
        print(str(contador) +" of "+ str(3) + " iterations")
               
    elapsed_time = time.process_time() - t
    
    print("Patient "+str(patient_id)+": "+str(performance_values/3))
    print("Elapsed Time: "+str(elapsed_time))    
    
    return performance_values/3

calculateFitness(patient_id,k_features,pre_ictal)
