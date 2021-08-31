"""

Database class: where we keep all patients in a simple object.
This class should be discarded in the future

"""

import os
from Patient import Patient
import numpy as np

class Database:

    def __init__(self, path):
        self.directory = path
        
        self.patientIDs=[]
        self.getPatientIDs()
        
        self.patients=[]
    
    def __repr__(self):  
        return f"Database({self.directory})"
    
    def __str__(self):
        number_patients = len(self.patientIDs)
        return f"Database({self.directory}) | {number_patients} patients"
    
    # get names of all files in directory    
    def getFilenames(self):
        patient_list = sorted(os.listdir(self.directory))
        #patient_list = [s for s in patient_list if 'pat' in s] # only files with "pat"
        patient_list = [self.directory + s for s in patient_list]
        return patient_list
    
    
    def getIDfromFilename(name):
        return name.split(sep = "pat")[1].split(sep = "_")[0]
        
    # get list of all unique ID's present in the directory
    def getPatientIDs(self):
        filenames = self.getFilenames()
        
        IDs = []
        for i in range(len(filenames)):
            if "pat" in filenames[i]:
                IDs.append(Database.getIDfromFilename(filenames[i]))
        
        return np.unique(IDs)
    
    # add Patient object to the list
    def addPatient(self, pat):
        self.patients.append(pat)
    
    # loads all the feature matrices and seizure metadata for a given patient
    def loadPatientData(self, ID):
        filenames = self.getFilenames()
        
        feature_matrices = []
        seizure_info = None
        for i in range(len(filenames)):
            if "pat" + ID in filenames[i] and "feature" in filenames[i]:
                feature_matrices.append(np.load(filenames[i], allow_pickle = True))
            if "pat" + ID in filenames[i] and "Info" in filenames[i]:
                seizure_info = np.load(filenames[i], allow_pickle = True)
        
        return feature_matrices, seizure_info
    
    #load feature matrix legend
    def loadLegend(self):
        filenames = self.getFilenames()
        
        legend = None
        for i in range(len(filenames)):
            if "legend" in filenames[i]:
                legend = np.load(filenames[i], allow_pickle = True)
        
        return legend
        
#    # loads the patient list by iterating all files in the directory
#    def loadPatientList(self):
#        filenames=self.getFilenames()
#        
#        for i in range(len(filenames)):
#            if  Database.isFilenameSeizureData(filenames[i]):
#                patient_number=Database.getPatientNumberFromFilename(filenames[i])    
#                if (patient_number not in self.patient_list and 
#                    not "_pre" in patient_number and
#                    not patient_number == "pat"):
#                    self.addPatientToList(patient_number)  