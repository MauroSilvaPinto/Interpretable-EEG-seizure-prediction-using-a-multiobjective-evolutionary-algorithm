"""
Patient class: where each patient has an ID,
its recordings data and metadata.

As this algorithm is patient personalized, this class
should be removed in the future.

"""

class Patient:

    def __init__(self, number):
        self.ID = number
        self.seizure_data = []
        self.seizure_metadata = []
    
    def __repr__(self):  
        return f"Patient({self.ID})"
    
    def __str__(self):
        number_seizures = self.getNumberOfSeizures()
        return f"Patient {self.ID} | {number_seizures} seizures"
    
    # gets the number of seizures available for the patient
    def getNumberOfSeizures(self):
        return len(self.seizure_metadata)
    
    # retrieves the feature matrix from a given seizure
    def getSeizureData(self, number):
        if number > self.getNumberOfSeizures() or number <= 0:
            return None
        else:
            return self.seizure_data[number - 1]
        
    # retrieves the metadata from a given seizure
    def getSeizureInfo(self, number):
        if number > self.getNumberOfSeizures() or number <= 0:
            return None
        else:
            return self.seizure_metadata[number - 1,:]
        
    # print metadata from all seizures
    def printMetadata(self):
        for i in range(self.getNumberOfSeizures()):
            print(self.getSeizureInfo(i+1))
     