"""
Individual Class

each individual is a set of hyper_features,
and has a given set of fitness values (one for each objective)

"""
from Feature import Feature
import random
import numpy as np
from copy import deepcopy

class Individual:
    
    def __init__(self, number):
        self.features = []
        self.generateRandomFeatures(number)
        self.preictal_threshold = Individual.generateRandomPreictalThreshold()
        self.fitness = []
        self.fitness_labels = []
    
    def __repr__(self):  
        return "Individual()"
    
    def __str__(self):
        number_features = self.getNumberOfFeatures()
        return f"Individual: {number_features} features, fitness = {self.fitness}"
    
    # retrieves the number of features in genotype
    def getNumberOfFeatures(self):
        return len(self.features)
    
    # generates a feature randomly
    def generateFeature(self):
        return Feature()
            
    # adds a Feature to the Individual
    def addFeature(self, feature):
        self.features.append(feature)
        
    # generates a number of features for the Individual
    def generateRandomFeatures(self, number):
        if number > 0:
            for i in range(number):
                self.addFeature(self.generateFeature())
            
    # returns the list of all possible pre-ictal thresholds in the genotype
    def getPreictalThresholdList():
        return np.array([30, 35, 40, 45, 50, 55, 60])
    
    # selects a random pre-ictal threshold
    def generateRandomPreictalThreshold():
        return random.randint(0,len(Individual.getPreictalThresholdList())-1)
    
    # returns the pre-ictal threshold duration as a string (in minutes)
    def decodePreictalThreshold(self):
        return Individual.getPreictalThresholdList()[self.preictal_threshold]
    
    # prints each of the individual's features (genotype) as integers
    def printGenotype(self):
        print(f"Pre-ictal threshold: {self.preictal_threshold}")
        for i in range(self.getNumberOfFeatures()):
            print(f"FEATURE {i+1}")
            print(self.features[i])
            print("")
            
    # prints each of the individual's features (phenotype) as strings
    def printPhenotype(self):
        print(f"Pre-ictal threshold: {self.decodePreictalThreshold()}")
        for i in range(self.getNumberOfFeatures()):
            print(f"FEATURE {i+1}")
            print(f"Mathematical operator: {self.features[i].decodeMathematicalOperator()}\n" +
                f"Electrode: {self.features[i].decodeElectrode()}\n" +
                f"Window: {self.features[i].decodeWindowLength()} | Delay: {self.features[i].decodeDelay()}\n" +
                f"Feature group: {self.features[i].decodeFeatureGroup()} | Frequency feature type: {self.features[i].decodeFrequencyFeature()}\n" +
                f"Frequency band feature: {self.features[i].decodeFrequencyBandFeature()} | Time feature: {self.features[i].decodeTimeFeature()}\n" +
                f"Frequency band: {self.features[i].decodeFrequencyBand()} | SEF: {self.features[i].decodeSEF()} | Wavelet coeff.: {self.features[i].decodeWaveletCoef()}\n" +
                f"Statistical moment: {self.features[i].decodeStatisticalMoment()} | Hjorth parameter: {self.features[i].decodeHjorthParameter()}")
            print("")
            
    # updates fitness values and labels (with the corresponding name of each objective)
    def updateFitness(self, new_fitness, labels):
        self.fitness = new_fitness
        self.fitness_labels = labels
    
    # mutates one of the individual's features OR its pre-ictal threshold
    def mutate(self):
        # reset fitness values + labels
        self.fitness = []
        self.fitness_labels = []
        
        features_or_preictal = [0, 1]
        weights = self.getMutationProbabilityWeights()
        mutation_type = np.random.choice(features_or_preictal, p = weights)
        
        # pre-ictal threshold (at the Individual level; affects all features) -> weighted differently
        if mutation_type == 0:
            self.mutatePreictalThreshold()
            
        # apply mutation to ONE of the individual's features -> weighted by the number of gene values
        else:
            choice = list(range(1,14))
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
            
            # randomly choose which feature to mutate
            feature_to_mutate = random.randint(0, self.getNumberOfFeatures()-1)
            
            weights_features = Individual.getFeatureMutationProbabilityWeights()
            mutation = np.random.choice(choice, p = weights_features)
            
            if mutation == 1:
                self.features[feature_to_mutate].mutateMathematicalOperator()
            elif mutation == 2:
                self.features[feature_to_mutate].mutateElectrode()
            elif mutation == 3:
                self.features[feature_to_mutate].mutateWindowLength()
            elif mutation == 4:
                self.features[feature_to_mutate].mutateDelay()
            elif mutation == 5:
                self.features[feature_to_mutate].mutateFeatureGroup()
            elif mutation == 6:
                self.features[feature_to_mutate].mutateFrequencyFeature()
            elif mutation == 7:
                self.features[feature_to_mutate].mutateTimeFeature()
            elif mutation == 8:
                self.features[feature_to_mutate].mutateFrequencyBandFeature()
            elif mutation == 9:
                self.features[feature_to_mutate].mutateFrequencyBand()
            elif mutation == 10:
                self.features[feature_to_mutate].mutateWaveletCoef()
            elif mutation == 11:
                self.features[feature_to_mutate].mutateSEF()
            elif mutation == 12:
                self.features[feature_to_mutate].mutateStatisticalMoment()
            elif mutation == 13:
                self.features[feature_to_mutate].mutateHjorthParameter()
            
    
    # returns the mutation probability for the pre-ictal threshold OR one of the
    # features, according to how many different values each can take
    def getMutationProbabilityWeights(self):
        probs = np.zeros(2)
        probs[0] = len(Individual.getPreictalThresholdList())
        probs[1] = Feature.getNumberOfPossibleValues() * self.getNumberOfFeatures()
        
        return probs/np.sum(probs)
    
    # returns the mutation probability for each of the feature's genes, according
    # to the number of possible values it can take
    def getFeatureMutationProbabilityWeights():
        probs = np.zeros(13)
        probs[0] = len(Feature.getMathematicalOperatorList())
        probs[1] = Feature.getElectrodesGraph().number_of_nodes()
        probs[2] = len(Feature.getWindowLengthList())
        probs[3] = len(Feature.getDelayList())
        probs[4] = len(Feature.getFeatureGroupList())
        probs[5] = len(Feature.getFrequencyFeatureList())
        probs[6] = len(Feature.getTimeFeatureList())
        probs[7] = len(Feature.getFrequencyBandFeatureList())
        probs[8] = len(Feature.getFrequencyBandList())
        probs[9] = len(Feature.getWaveletCoefList())
        probs[10] = len(Feature.getSEFList())
        probs[11] = len(Feature.getStatisticalMomentList())
        probs[12] = len(Feature.getHjorthParameterList())
        #probs[13] = len(Individual.getPreictalThresholdList())
        
        return probs/np.sum(probs)
            
    # mutates the individual's pre-ictal threshold duration
    def mutatePreictalThreshold(self):
        mutation_direction = random.randint(0,1) # 0 = up; 1 = down
        
        if mutation_direction == 0:
            if self.preictal_threshold == len(Individual.getPreictalThresholdList()) - 1:
                self.preictal_threshold -= 1
            else:
                self.preictal_threshold += 1
        elif mutation_direction == 1:
            if self.preictal_threshold == 0:
                self.preictal_threshold += 1
            else:
                self.preictal_threshold -= 1
    
    # recombines the current individual with another one
    def recombine(self, other):
        offspring = Individual(0) # new individual, with no fitness values
        
        # recombine pre-ictal thresholds of the two parents
        offspring.preictal_threshold = Individual.recombinePreictalThresholds(self, other)
    
        # sort the current parent's features by similarity to the other individual's features
        sorted_features = self.sortFeaturesAccordingTo(other)
        
        # recombine features from each parent and add them to the offspring
        for i in range(self.getNumberOfFeatures()):
            new_feature = Feature()
            
            new_operator = Feature.recombineMathematicalOperators(sorted_features[i], other.features[i])
            new_electrode = Feature.recombineElectrodes(sorted_features[i], other.features[i])
            new_window = Feature.recombineWindowLengths(sorted_features[i], other.features[i])
            new_delay = Feature.recombineDelays(sorted_features[i], other.features[i])
            new_featuregroup = Feature.recombineFeatureGroups(sorted_features[i], other.features[i])
            new_freqfeature = Feature.recombineFrequencyFeatures(sorted_features[i], other.features[i])
            new_timefeature = Feature.recombineTimeFeatures(sorted_features[i], other.features[i])
            new_frequencybandfeature = Feature.recombineFrequencyBandFeatures(sorted_features[i], other.features[i])
            new_band = Feature.recombineFrequencyBands(sorted_features[i], other.features[i])
            new_wavelet = Feature.recombineWaveletCoefs(sorted_features[i], other.features[i])
            new_sef = Feature.recombineSEFs(sorted_features[i], other.features[i])
            new_stat = Feature.recombineStatisticalMoments(sorted_features[i], other.features[i])
            new_hjorth = Feature.recombineHjorthParameters(sorted_features[i], other.features[i])
            
            new_feature.mathematical_operator = new_operator
            new_feature.electrode = new_electrode
            new_feature.window_length = new_window
            new_feature.delay = new_delay
            new_feature.feature_group = new_featuregroup
            new_feature.frequency_feature = new_freqfeature
            new_feature.time_feature = new_timefeature
            new_feature.frequencyband_feature = new_frequencybandfeature
            new_feature.band = new_band
            new_feature.wavelet_coef = new_wavelet
            new_feature.sef = new_sef
            new_feature.statistical_moment = new_stat
            new_feature.hjorth_parameter = new_hjorth
            
            offspring.addFeature(new_feature)
        
        return offspring
    
    # sorts the individual's features according to their similarity
    def sortFeaturesAccordingTo(self, other):
        # order the other parent's features by similarity to the current individual's features
        order = []
        for i in range(self.getNumberOfFeatures()):
            distances = []
            for j in range(other.getNumberOfFeatures()):
                distances.append(self.features[i].computeDistance(other.features[j]))
            order.append(np.argmin(distances))
        
        # if two features are "equally similar", choose the next most similar feature
        order_unique = np.argsort(order).tolist()
        sorted_features = []
        for i in range(other.getNumberOfFeatures()):
            sorted_features.append(self.features[order_unique.index(i)]) # (?) works well enough...
            
        return sorted_features
    
    # recombines the pre-ictal thresholds from two parents
    def recombinePreictalThresholds(individual1, individual2):
        value1 = individual1.preictal_threshold
        value2 = individual2.preictal_threshold
        
        # choose any value between those of the parents (including one of the two)
        if value1 == value2:
            return value1
        elif value1 < value2:
            new_value = np.random.choice(np.arange(value1, value2 + 1))
            return new_value
        elif value1 > value2:
            new_value = np.random.choice(np.arange(value2, value1 + 1))
            return new_value
    
    # checks whether one individual dominates another, returns a boolean
    def dominates(self, other):
        domination = False
        
        number_objectives = len(self.fitness)
        higher_count = 0
        strictly_higher_count = 0
        
        for i in range(number_objectives):
            if self.fitness[i] >= other.fitness[i]:
                higher_count += 1
                if self.fitness[i] > other.fitness[i]:
                    strictly_higher_count += 1
        
        # must be at least as high or higher for ALL objectives and strictly higher
        # for AT LEAST ONE objective
        if higher_count == number_objectives and strictly_higher_count > 0:
            domination = True
        
        return domination
    
    # clones the individual's phenotype while resetting its fitness values + labels
    def clone(self):
        cloned = deepcopy(self)
        
        cloned.fitness = []
        cloned.fitness_labels = []
        
        return cloned
        
     
    