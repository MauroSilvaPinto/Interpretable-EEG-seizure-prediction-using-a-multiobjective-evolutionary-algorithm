"""
Feature class which represents an hyper-feature.
Each hyper-feature is constituted by several genes:
    - one mathematical operator gene
    - one electrode gene
    - one window length gene
    - one delay gene
    
    - one feature group gene (time or frequency)
    - one frequency feature gene (frequency band or spectral edge frequency)
    - one time feature gene (hjorth parameter or statistical moment)
    
    - one frequency band feature gene
    - one statistical moment feature gene
    - one frequency band feature gene
    - one wavelet band feature gene
    - one hjorth parameter gene
    - one spectral edge frequency feature gene
    

ps: electrodes are in the old nomenclature of 10-20 eeg system    
    T3 is now T7.
    T4 is now T8.
    T5 is now P7.
    T6 is now P8.

"""

import numpy as np
import random
import networkx as nx

class Feature:
    
    def __init__(self):
        self.mathematical_operator = Feature.generateRandomMathematicalOperator();
        self.electrode = Feature.generateRandomElectrode()
        self.window_length = Feature.generateRandomWindowLength()
        self.delay = Feature.generateRandomDelay()
        
        self.feature_group = Feature.generateRandomFeatureGroup()
        self.frequency_feature = Feature.generateRandomFrequencyFeature()
        self.time_feature = Feature.generateRandomTimeFeature()
        self.frequencyband_feature = Feature.generateRandomFrequencyBandFeature()
        
        self.statistical_moment = Feature.generateRandomStatisticalMoment()
        self.band = Feature.generateRandomFrequencyBand()
        self.wavelet_coef = Feature.generateRandomWaveletCoef()
        self.hjorth_parameter = Feature.generateRandomHjorthParameter()
        self.sef = Feature.generateRandomSEF()
    
    def __repr__(self):  
        return "Feature()"
    
    def __str__(self):
        return (f"Mathematical operator: {self.mathematical_operator}\n" +
                f"Electrode: {self.electrode}\n" +
                f"Window: {self.window_length} | Delay: {self.delay}\n" +
                f"Feature group: {self.feature_group} | Frequency feature type: {self.frequency_feature}\n" +
                f"Frequency band feature: {self.frequencyband_feature} | Time feature: {self.time_feature}\n" +
                f"Frequency band: {self.band} | SEF: {self.sef} | Wavelet coefficient: {self.wavelet_coef}\n" + 
                f"Statistical moment: {self.statistical_moment} | Hjorth parameter: {self.hjorth_parameter}")
    
    # returns the list of all possible operators in the genotype
    def getMathematicalOperatorList():
        return ["mean", "variance", "integral"]
        
    # returns the list of the 19 electrodes 
    def getElectrodesList():
        return ['C3','C4','Cz','F3','F4','F7', 'F8','FP1','FP2',
                 'Fz','O1','O2','P3','P4','Pz', 'T3','T4','T5','T6']
    
    # builds graph of the 19 electrodes based on the 10-20 system
    def getElectrodesGraph():
        G = nx.Graph()

        electrodes_list = Feature.getElectrodesList()

        for electrode in electrodes_list:
            G.add_node(electrode)
        
        G.add_edge("FP1","FP2"); G.add_edge("FP1","Fz"); G.add_edge("FP1","F3")
        G.add_edge("FP1","F7"); G.add_edge("FP2","Fz"); G.add_edge("FP2","F4")
        G.add_edge("FP2","F8"); G.add_edge("F7","F3"); G.add_edge("F7","C3")
        G.add_edge("F7","T3"); G.add_edge("F3","T3"); G.add_edge("F3","C3")
        G.add_edge("F3","Cz"); G.add_edge("F3","Fz"); G.add_edge("Fz","Cz")
        G.add_edge("Fz","F4"); G.add_edge("Fz","C4"); G.add_edge("F4","Cz")
        G.add_edge("F4","F8"); G.add_edge("F4","T4"); G.add_edge("F4","C4")     
        G.add_edge("F8","T4"); G.add_edge("F8","C4"); G.add_edge("T3","T5")
        G.add_edge("T3","C3"); G.add_edge("T3","P3"); G.add_edge("C3","T5")
        G.add_edge("C3","P3"); G.add_edge("C3","Cz"); G.add_edge("C3","Pz")
        G.add_edge("C3","Fz"); G.add_edge("Cz","P3"); G.add_edge("Cz","Pz")
        G.add_edge("Cz","C4"); G.add_edge("Cz","P4"); G.add_edge("C4","Pz")
        G.add_edge("C4","P4"); G.add_edge("C4","T6"); G.add_edge("C4","T4")       
        G.add_edge("T4","T6"); G.add_edge("T4","P4"); G.add_edge("T5","P3")
        G.add_edge("P3","Pz"); G.add_edge("P4","Pz"); G.add_edge("P4","T6")
        G.add_edge("O1","T5"); G.add_edge("O1","P3"); G.add_edge("O1","Pz")
        G.add_edge("O1","O2"); G.add_edge("O2","Pz"); G.add_edge("O2","P4")
        G.add_edge("O2","T6")
        
        return G
    
    # returns the list of all possible window lengths in the genotype
    def getWindowLengthList():
        return np.array([1, 5, 10, 15, 20])
    
    # returns the list of all possible delay durations in the genotype
    def getDelayList():
        return np.array([0, 5, 10, 15, 20, 25, 30])
    
    # returns the list of all possible feature groups (frequency or time
    # domain) in the genotype
    def getFeatureGroupList():
        return ["frequency", "time"]
    
    # returns the list of all possible frequency domain
    # feature types in the genotype
    def getFrequencyFeatureList():
        return ["sef", "band_division"]
    
    # returns the list of all possible frequency domain
    # features in the genotype
    def getTimeFeatureList():
        return ["statistical_moments", "hjorth_parameters"]
    
    # returns the list of all possible frequency domain
    # features in the genotype related to band division
    def getFrequencyBandFeatureList():
        return ["band_power", "wavelet_energy"]
    
    # returns the list of all possible statistical moments in the genotype
    # ordered by the moment order (1st -> 4th)
    def getStatisticalMomentList():
        return ["mean", "var", "skew", "kurt"]
    
    # returns the list of all possible frequency bands in the genotype
    # ordered by frequency 
    def getFrequencyBandList():
        return ["delta", "theta", "alpha", "beta", "lowgamma", "highgamma"]
    
    # returns the list of all possible hjorth parameters in the genotype
    # randomly ordered (no apparent order?)
    def getHjorthParameterList():
        return ["h_act", "h_mob", "h_com"]
    
    # returns the list of all possible SEF cut-offs in the genotype
    # ordered by cut-off frequency
    def getSEFList():
        return ["sef50", "sef75", "sef90"]
    
    # returns the list of all possible wavelet coefficients in the genotype
    def getWaveletCoefList():
        return ["a7", "d7", "d6", "d5", "d4", "d3", "d2", "d1"]
    
    # selects a random mathematical operator
    def generateRandomMathematicalOperator():
        return random.randint(0,len(Feature.getMathematicalOperatorList())-1)
    
    # selects a random electrode
    def generateRandomElectrode():
        return random.randint(0,len(Feature.getElectrodesList())-1)
    
    # selects a random window length
    def generateRandomWindowLength():
        return random.randint(0,len(Feature.getWindowLengthList())-1)
    
    # selects a random delay duration
    def generateRandomDelay():
        return random.randint(0,len(Feature.getDelayList())-1)
    
    # selects a random feature group (frequency/time)
    def generateRandomFeatureGroup():
        return random.randint(0,len(Feature.getFeatureGroupList())-1)
    
    # selects a random frequency feature type
    def generateRandomFrequencyFeature():
        return random.randint(0,len(Feature.getFrequencyFeatureList())-1)
    
    # selects a random time feature
    def generateRandomTimeFeature():
        return random.randint(0,len(Feature.getTimeFeatureList())-1)
    
    # selects a random frequency band feature
    def generateRandomFrequencyBandFeature():
        return random.randint(0,len(Feature.getFrequencyBandFeatureList())-1)
    
    # selects a random statistical moment
    def generateRandomStatisticalMoment():
        return random.randint(0,len(Feature.getStatisticalMomentList())-1)
    
    # selects a random frequency band
    def generateRandomFrequencyBand():
        return random.randint(0,len(Feature.getFrequencyBandList())-1)
    
    # selects a random Hjorth parameter
    def generateRandomHjorthParameter():
        return random.randint(0,len(Feature.getHjorthParameterList())-1)
    
    # selects a random SEF (different cut-off)
    def generateRandomSEF():
        return random.randint(0,len(Feature.getSEFList())-1)
    
    # selects a random wavelet coefficient
    def generateRandomWaveletCoef():
        return random.randint(0,len(Feature.getWaveletCoefList())-1)
    
    # returns the name of the mathematical operator as a string
    def decodeMathematicalOperator(self):
        return Feature.getMathematicalOperatorList()[self.mathematical_operator]
    
    # returns the name of the electrode as a string
    def decodeElectrode(self):
        return Feature.getElectrodesList()[self.electrode]
    
    # returns the window length as a string (in minutes)
    def decodeWindowLength(self):
        return Feature.getWindowLengthList()[self.window_length]
    
    # returns the delay duration as a string (in minutes)
    def decodeDelay(self):
        return Feature.getDelayList()[self.delay]
    
    # returns the name of the active feature group as a string
    def decodeFeatureGroup(self):
        return Feature.getFeatureGroupList()[self.feature_group]
    
    # returns the name of the active frequency feature type as a string
    def decodeFrequencyFeature(self):
        return Feature.getFrequencyFeatureList()[self.frequency_feature]
    
    # returns the name of the active time feature as a string
    def decodeTimeFeature(self):
        return Feature.getTimeFeatureList()[self.time_feature]
    
    # returns the name of the active frequency band feature as a string
    def decodeFrequencyBandFeature(self):
        return Feature.getFrequencyBandFeatureList()[self.frequencyband_feature]
    
    # returns the name of the statistical moment as a string
    def decodeStatisticalMoment(self):
        return Feature.getStatisticalMomentList()[self.statistical_moment]
    
    # returns the name of the frequency band as a string
    def decodeFrequencyBand(self):
        return Feature.getFrequencyBandList()[self.band]
    
    # returns the name of the Hjorth parameter as a string
    def decodeHjorthParameter(self):
        return Feature.getHjorthParameterList()[self.hjorth_parameter]
    
    # returns the name of the SEF cut-off as a string
    def decodeSEF(self):
        return Feature.getSEFList()[self.sef]
    
    # returns the name of the wavelet coefficient as a string
    def decodeWaveletCoef(self):
        return Feature.getWaveletCoefList()[self.wavelet_coef]
    
    # returns what feature is active depending on the feature group genes
    def checkActiveGenes(self):
        if self.decodeFeatureGroup() == "frequency":
            if self.decodeFrequencyFeature() == "band_division":
                if self.decodeFrequencyBandFeature() == "band_power":
                    return self.decodeFrequencyBand()
                elif self.decodeFrequencyBandFeature() == "wavelet_energy":
                    return self.decodeWaveletCoef()
            elif self.decodeFrequencyFeature() == "sef":
                return self.decodeSEF()
        elif self.decodeFeatureGroup() == "time":
            if self.decodeTimeFeature() == "statistical_moments":
                return self.decodeStatisticalMoment()
            elif self.decodeTimeFeature() == "hjorth_parameters":
                return self.decodeHjorthParameter()
        else:
            return None
    
    # mutates the mathematical operator
    def mutateMathematicalOperator(self):
        # generates a random operator (since they are all neighbors)
        new_mathematical_operator = Feature.generateRandomMathematicalOperator()
        while new_mathematical_operator == self.mathematical_operator:
            new_mathematical_operator = Feature.generateRandomMathematicalOperator()
            
        self.mathematical_operator=new_mathematical_operator
    
    # mutates the electrode (to one of its neighbors according to 10-20 system)
    def mutateElectrode(self):
        graph = Feature.getElectrodesGraph()
        current_electrode = self.decodeElectrode()
        electrode_neighbors = list(graph.neighbors(current_electrode))
        mutated_electrode = np.random.choice(electrode_neighbors)
        
        self.electrode = Feature.getElectrodesList().index(mutated_electrode)
    
    # mutates the window length
    def mutateWindowLength(self):
        # mutation step in either direction
        mutation_direction = random.randint(0,1) # 0 = up; 1 = down
        
        if mutation_direction == 0:
            if self.window_length == len(Feature.getWindowLengthList()) - 1:
                self.window_length -= 1
            else:
                self.window_length += 1
        elif mutation_direction == 1:
            if self.window_length == 0:
                self.window_length += 1
            else:
                self.window_length -= 1
    
    # mutates the delay duration
    def mutateDelay(self):
        # mutation step in either direction
        mutation_direction = random.randint(0,1) # 0 = up; 1 = down
        
        if mutation_direction == 0:
            if self.delay == len(Feature.getDelayList()) - 1:
                self.delay -= 1
            else:
                self.delay += 1
        elif mutation_direction == 1:
            if self.delay == 0:
                self.delay += 1
            else:
                self.delay -= 1
    
    # mutates the active feature group
    def mutateFeatureGroup(self):
        # binary switch
        if self.feature_group == 0:
            self.feature_group = 1
        else:
            self.feature_group = 0
    
    # mutates the active frequency feature type
    def mutateFrequencyFeature(self):
        # binary switch
        if self.frequency_feature == 0:
            self.frequency_feature = 1
        else:
            self.frequency_feature = 0
    
    # mutates the active time feature 
    def mutateTimeFeature(self):
        # binary switch
        if self.time_feature == 0:
            self.time_feature = 1
        else:
            self.time_feature = 0
    
    # mutates the active frequency band feature
    def mutateFrequencyBandFeature(self):
        # binary switch
        if self.frequencyband_feature == 0:
            self.frequencyband_feature = 1
        else:
            self.frequencyband_feature = 0
    
    # mutates the frequency band
    def mutateFrequencyBand(self):
        # mutation step in either direction
        mutation_direction = random.randint(0,1) # 0 = up; 1 = down
        
        if mutation_direction == 0:
            if self.band == len(Feature.getFrequencyBandList()) - 1:
                self.band -= 1
            else:
                self.band += 1
        elif mutation_direction == 1:
            if self.band == 0:
                self.band += 1
            else:
                self.band -= 1    
    
    # mutates the wavelet coefficient
    def mutateWaveletCoef(self):
        # mutation step in either direction
        mutation_direction = random.randint(0,1) # 0 = up; 1 = down
        
        if mutation_direction == 0:
            if self.wavelet_coef == len(Feature.getWaveletCoefList()) - 1:
                self.wavelet_coef -= 1
            else:
                self.wavelet_coef += 1
        elif mutation_direction == 1:
            if self.wavelet_coef == 0:
                self.wavelet_coef += 1
            else:
                self.wavelet_coef -= 1 
    
    # mutates the SEF cut-off
    def mutateSEF(self):
        # mutation step in either direction
        mutation_direction = random.randint(0,1) # 0 = up; 1 = down
        
        if mutation_direction == 0:
            if self.sef == len(Feature.getSEFList()) - 1:
                self.sef -= 1
            else:
                self.sef += 1
        elif mutation_direction == 1:
            if self.sef == 0:
                self.sef += 1
            else:
                self.sef -= 1 
    
    # mutates the statistical moment
    def mutateStatisticalMoment(self):
        # mutation step in either direction
        mutation_direction = random.randint(0,1) # 0 = up; 1 = down
        
        if mutation_direction == 0:
            if self.statistical_moment == len(Feature.getStatisticalMomentList()) - 1:
                self.statistical_moment -= 1
            else:
                self.statistical_moment += 1
        elif mutation_direction == 1:
            if self.statistical_moment == 0:
                self.statistical_moment += 1
            else:
                self.statistical_moment -= 1 
    
    # mutates the Hjorth parameter
    def mutateHjorthParameter(self):
        # mutation step in either direction
        mutation_direction = random.randint(0,1) # 0 = up; 1 = down
        
        if mutation_direction == 0:
            if self.hjorth_parameter == len(Feature.getHjorthParameterList()) - 1:
                self.hjorth_parameter -= 1
            else:
                self.hjorth_parameter += 1
        elif mutation_direction == 1:
            if self.hjorth_parameter == 0:
                self.hjorth_parameter += 1
            else:
                self.hjorth_parameter -= 1 
    
    # recombines the mathematical operators from two parents
    def recombineMathematicalOperators(feature1, feature2):
        value1 = feature1.mathematical_operator
        value2 = feature2.mathematical_operator
        
        # no values between those of the parents, so it becomes either one or the other
        return np.random.choice([value1, value2])
    
    # recombines the electrodes from two parents (based on the shortest paths in the graph)
    def recombineElectrodes(feature1, feature2):
        value1 = feature1.electrode
        value2 = feature2.electrode
        graph = Feature.getElectrodesGraph()
        
        # retrieve decoded names of the parents' electrodes
        electrode_parent1 = Feature.getElectrodesList()[value1]
        electrode_parent2 = Feature.getElectrodesList()[value2]
        
        if electrode_parent1 == electrode_parent2:
            return value1
        else:
            # compute all shortest paths in the graph between the two electrodes and choose one randomly
            shortest_paths = list(nx.all_shortest_paths(graph, source = electrode_parent1, target = electrode_parent2))
            path = shortest_paths[np.random.choice(np.arange(0, len(shortest_paths)))]
            
            if len(path)==2:
                return np.random.choice([value1, value2]) # no electrodes between the two parents  
            else:
                recombined_electrode = np.random.choice(np.array(path[1:-1])) # choose any electrode within the chosen shortest path
                return Feature.getElectrodesList().index(recombined_electrode)
    
    # recombines the window lengths from two parents
    def recombineWindowLengths(feature1, feature2):
        value1 = feature1.window_length
        value2 = feature2.window_length
        
        # choose any value between the ones of the parents (including one of the two)
        if value1 == value2:
            return value1
        elif value1 < value2:
            new_value = np.random.choice(np.arange(value1, value2 + 1))
            return new_value
        elif value1 > value2:
            new_value = np.random.choice(np.arange(value2, value1 + 1))
            return new_value
    
    # recombines the delay durations from two parents
    def recombineDelays(feature1, feature2):
        value1 = feature1.delay
        value2 = feature2.delay
        
        # choose any value between the ones of the parents (including one of the two)
        if value1 == value2:
            return value1
        elif value1 < value2:
            new_value = np.random.choice(np.arange(value1, value2 + 1))
            return new_value
        elif value1 > value2:
            new_value = np.random.choice(np.arange(value2, value1 + 1))
            return new_value
        
    # recombines the active feature groups from two parents   
    def recombineFeatureGroups(feature1, feature2):
        value1 = feature1.feature_group
        value2 = feature2.feature_group
        
        # no values between those of the parents, so it becomes either one or the other
        return np.random.choice([value1, value2])
    
    # recombines the active frequency features from two parents    
    def recombineFrequencyFeatures(feature1, feature2):
        value1 = feature1.frequency_feature
        value2 = feature2.frequency_feature
        
        # no values between those of the parents, so it becomes either one or the other
        return np.random.choice([value1, value2])
    
    # recombines the active time features from two parents
    def recombineTimeFeatures(feature1, feature2):
        value1 = feature1.time_feature
        value2 = feature2.time_feature
        
        # no values between those of the parents, so it becomes either one or the other
        return np.random.choice([value1, value2])
    
    # recombines the active frequency band features from two parents
    def recombineFrequencyBandFeatures(feature1, feature2):
        value1 = feature1.frequencyband_feature
        value2 = feature2.frequencyband_feature
        
        # no values between those of the parents, so it becomes either one or the other
        return np.random.choice([value1, value2])
    
    # recombines the bands from two parents
    def recombineFrequencyBands(feature1, feature2):
        value1 = feature1.band
        value2 = feature2.band
        
        # choose any value between the ones of the parents (including one of the two)
        if value1 == value2:
            return value1
        elif value1 < value2:
            new_value = np.random.choice(np.arange(value1, value2 + 1))
            return new_value
        elif value1 > value2:
            new_value = np.random.choice(np.arange(value2, value1 + 1))
            return new_value
    
    # recombines the wavelet coefficients from two parents
    def recombineWaveletCoefs(feature1, feature2):
        value1 = feature1.wavelet_coef
        value2 = feature2.wavelet_coef
        
        # choose any value between the ones of the parents (including one of the two)
        if value1 == value2:
            return value1
        elif value1 < value2:
            new_value = np.random.choice(np.arange(value1, value2 + 1))
            return new_value
        elif value1 > value2:
            new_value = np.random.choice(np.arange(value2, value1 + 1))
            return new_value
    
    # recombines the SEF cut-offs from two parents
    def recombineSEFs(feature1, feature2):
        value1 = feature1.sef
        value2 = feature2.sef
        
        # choose any value between the ones of the parents (including one of the two)
        if value1 == value2:
            return value1
        elif value1 < value2:
            new_value = np.random.choice(np.arange(value1, value2 + 1))
            return new_value
        elif value1 > value2:
            new_value = np.random.choice(np.arange(value2, value1 + 1))
            return new_value
    
    # recombines the statistical moments from two parents
    def recombineStatisticalMoments(feature1, feature2):
        value1 = feature1.statistical_moment
        value2 = feature2.statistical_moment
        
        # choose any value between the ones of the parents (including one of the two)
        if value1 == value2:
            return value1
        elif value1 < value2:
            new_value = np.random.choice(np.arange(value1, value2 + 1))
            return new_value
        elif value1 > value2:
            new_value = np.random.choice(np.arange(value2, value1 + 1))
            return new_value
        
    # recombines the Hjorth parameters from two parents
    def recombineHjorthParameters(feature1, feature2):
        value1 = feature1.hjorth_parameter
        value2 = feature2.hjorth_parameter
        
        # choose any value between the ones of the parents (including one of the two)
        if value1 == value2:
            return value1
        elif value1 < value2:
            new_value = np.random.choice(np.arange(value1, value2 + 1))
            return new_value
        elif value1 > value2:
            new_value = np.random.choice(np.arange(value2, value1 + 1))
            return new_value
    
    # # computes the distance/difference to another given feature
    # def computeDistance(self, other_feature):
    #     distance = 0
        
    #     # mathematical_operator
    #     if self.mathematical_operator != other_feature.mathematical_operator:
    #         distance += 1
    #     # electrode
    #     graph = Feature.getElectrodesGraph()
    #     distance += nx.shortest_path_length(graph, source = self.decodeElectrode(), target = other_feature.decodeElectrode())
    #     # window_length
    #     if self.window_length != other_feature.window_length:
    #         distance += 1
    #     # delay
    #     if self.delay != other_feature.delay:
    #         distance += 1
    #     # feature_group
    #     if self.feature_group != other_feature.feature_group:
    #         distance += 4 # bigger difference? 
    #     else:
    #         # FREQUENCY
    #         if self.decodeFeatureGroup == "frequency":
    #             # frequency_feature
    #             if self.frequency_feature != other_feature.frequency_feature:
    #                 distance += 3
    #             else:
    #                 # frequencyband_feature
    #                 if self.frequencyband_feature != other_feature.frequencyband_feature:
    #                     distance += 2
    #                 else:
    #                     # band
    #                     if self.band != other_feature.band:
    #                         distance += 1
    #                     # wavelet_coef
    #                     if self.wavelet_coef != other_feature.wavelet_coef:
    #                         distance += 1
    #                     # sef 
    #                     if self.sef != other_feature.sef:
    #                         distance += 1
                        
    #         # TIME
    #         if self.decodeFeatureGroup() == "time":
    #             # time_feature
    #             if self.time_feature != other_feature.time_feature:
    #                 distance += 2
    #             else:
    #                 # statistical_moment
    #                 if self.statistical_moment != other_feature.statistical_moment:
    #                     distance += 1
    #                 # hjorth_parameter
    #                 if self.hjorth_parameter != other_feature.hjorth_parameter:
    #                     distance += 1
    #     return distance
    
    # computes the distance/difference to another given feature
    def computeDistance(self, other_feature):
        distance = 0
        
        weights = np.ones(13) # different weights for each gene?
        # 1 = mathematical operator
        # 2 = electrode
        # 3 = window length
        # 4 = delay
        # 5 = active feature group
        # 6 = active frequency feature
        # 7 = active time feature
        # 8 = active frequency band feature
        # 9 = band power group
        # 10 = wavelet energy group
        # 11 = SEF group
        # 12 = statistical moment group
        # 13 = hjorth parameter group
        
        # mathematical_operator
        distance +=  (weights[0] * 
        abs(self.mathematical_operator - other_feature.mathematical_operator) / len(Feature.getMathematicalOperatorList()))
        # electrode
        graph = Feature.getElectrodesGraph()
        distance += (weights[1] * 
        nx.shortest_path_length(graph, source = self.decodeElectrode(), target = other_feature.decodeElectrode())/nx.diameter(graph))
        # window_length
        distance +=  (weights[2] * 
        abs(self.window_length - other_feature.window_length) / len(Feature.getWindowLengthList()))
        # delay
        distance +=  (weights[3] * 
        abs(self.delay - other_feature.delay) / len(Feature.getDelayList()))
        
        # feature_group
        distance +=  (weights[4] * 
        abs(self.feature_group - other_feature.feature_group) / len(Feature.getFeatureGroupList()))
        # frequency_feature
        distance +=  (weights[5] * 
        abs(self.frequency_feature - other_feature.frequency_feature) / len(Feature.getFrequencyFeatureList()))
        # time_feature
        distance +=  (weights[6] * 
        abs(self.time_feature - other_feature.time_feature) / len(Feature.getTimeFeatureList()))
        # frequencyband_feature
        distance +=  (weights[7] * 
        abs(self.frequencyband_feature - other_feature.frequencyband_feature) / len(Feature.getFrequencyBandFeatureList()))
        
        # band
        distance +=  (weights[8] * 
        abs(self.band - other_feature.band) / len(Feature.getFrequencyBandList()))
        # wavelet_coef
        distance +=  (weights[9] * 
        abs(self.wavelet_coef - other_feature.wavelet_coef) / len(Feature.getWaveletCoefList()))
        # sef 
        distance +=  (weights[10] * 
        abs(self.sef - other_feature.sef) / len(Feature.getSEFList()))
        # statistical_moment
        distance +=  (weights[11] * 
        abs(self.statistical_moment - other_feature.statistical_moment) / len(Feature.getStatisticalMomentList()))
        # hjorth_parameter
        distance +=  (weights[12] * 
        abs(self.hjorth_parameter - other_feature.hjorth_parameter) / len(Feature.getHjorthParameterList()))
        
        return distance
    
    # returns the number of all possible values for the entire genotype
    def getNumberOfPossibleValues():
        number_values = (len(Feature.getDelayList()) + len(Feature.getElectrodesList()) +
                  len(Feature.getFeatureGroupList()) + len(Feature.getFrequencyBandFeatureList()) +
                  len(Feature.getFrequencyBandList()) +  len(Feature.getFrequencyFeatureList()) +
                  len(Feature.getHjorthParameterList()) + len(Feature.getMathematicalOperatorList()) +
                  len(Feature.getSEFList()) + len(Feature.getStatisticalMomentList()) + 
                  len(Feature.getTimeFeatureList()) + len(Feature.getWaveletCoefList()) +
                  len(Feature.getWindowLengthList()))
        
        return number_values
        
        
        
