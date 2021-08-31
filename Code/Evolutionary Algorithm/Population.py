"""
Population Class

A population concerns the group of individuals,
and is also characterized by its generation number.

It is also in this code where we make parent selection
for the mating pool, and survival selection.


"""

from Individual import Individual
from SlidingWindow import SlidingWindow
from Classifier import Classifier
from FitnessFunction import FitnessFunction
import random
import numpy as np
#from pygmo import hypervolume -> couldn't get it to work on MacOS; only needed for SMS-EMOA
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.spatial import distance

class Population:
    
    def __init__(self, size, number_features):
        self.individuals = []
        self.generation = 0
        self.generateRandomIndividuals(size, number_features)
        self.fitness_history = []
    
    def __repr__(self):  
        return "Population()"
    
    def __str__(self):
        number_individuals = self.getNumberOfIndividuals()
        best_fitness = np.round(self.getBestFitnessValue(), decimals = 3)
        avg_fitness = np.round(self.getAverageFitnessValue(), decimals = 3)
        return f"Population: {number_individuals} individuals | Generation: {self.generation} | Best fitness = {best_fitness} | Average fitness = {avg_fitness}"
    
    # retrieves the number of individuals in the population
    def getNumberOfIndividuals(self):
        return len(self.individuals)
    
    # generates a random individual with a given number of features
    def generateIndividual(self, number_features):
        return Individual(number_features)
            
    # adds an Individual to the Population
    def addIndividual(self, individual):
        self.individuals.append(individual)
        
    # generates a number of individuals with a given number of features
    def generateRandomIndividuals(self, number_individuals, number_features):
        if number_individuals > 0:
            for i in range(number_individuals):
                self.addIndividual(self.generateIndividual(number_features))
    
    # retrieves the list of all fitness values
    def getFitnessList(self):
        fitness_values = []
        for ind in self.individuals:
            fitness_values.append(np.array(ind.fitness))
        
        return fitness_values
    
    # retrieves the highest fitness value found within the population
    def getBestFitnessValue(self):
        fitness_values = self.getFitnessList()
        # assume that the best fitness value = the one with the highest sum
        # for all objectives
        sums = []
        for i in fitness_values:
            sums.append(np.sum(i))
        
        return fitness_values[np.argmax(sums)].tolist()
    
    # retrieves the average fitness values (for each objective) found within the population
    def getAverageFitnessValue(self):
        fitness_values = self.getFitnessList()
        number_objectives = len(fitness_values[0])
        fitness_array = np.vstack(fitness_values)
        
        means = []
        for i in range(number_objectives):
            means.append(np.mean(fitness_array[:,i]))
        
        return means
    
    # retrieves the individual with the highest fitness value
    # if more than one individual with that fitness value is found, the first one is retrieved
    def getFittestIndividual(self):
        fitness_values = self.getFitnessList()
        
        return self.individuals[np.argmax(fitness_values)]
    
    # prints the details of all the individuals in the current generation
    def printIndividuals(self):
        for i in range(self.getNumberOfIndividuals()): 
            print(self.individuals[i])
    
    # retrieves the list of fitness values as coordinates/points (useful to
    # compute hypervolume)
    def getFitnessPoints(individuals):
        fitness_points = []
        for ind in individuals:
            fitness_points.append(ind.fitness[:])
        
        return fitness_points
    
    # computes the hypervolume metric for a given front of individuals
    def computeHypervolume(front):
        points = Population.getFitnessPoints(front)
        
        # convert to minimization (only here, everything else is assuming maximization as of 01/06/2020)
        for i in range(len(front)):
            for j in range(len(points[0])):
                points[i][j] = 1.00 - points[i][j]
                
        referencePoint = list(np.ones(len(points[0]))) # THIS IS ASSUMING MINIMIZATION!
        
        # !!! couldn't get pygmo to work on MacOS
        #hv = hypervolume(np.unique(points, axis = 0).tolist())
        #return hv.compute(referencePoint)
    
    # returns a sorted list according to the number of dominated individuals
    # for non-dominated sorting (used in MOGA)
    def applyMOGARanking(individuals):
        front_number = np.zeros(len(individuals))
        
        for i in range(len(individuals)):
            dominated_count = 0 # number of individuals the current individual is dominated by
            for j in range(len(individuals)):
                if i != j and individuals[j].dominates(individuals[i]):
                    dominated_count += 1
            # front number is the number of individuals dominating the current individual plus one
            front_number[i] = dominated_count + 1   
        
        # sort current population by its front number
        #zipped = zip(front_number, current_population)
        sorted_ind = [ind for _,ind in sorted(zip(front_number, individuals), key = lambda t: t[0], reverse = False)]
        sorted_front_number = [f for f,_ in sorted(zip(front_number, individuals), key = lambda t: t[0], reverse = False)]
        return sorted_ind, sorted_front_number
    
    # returns a sorted list with the new fitness values according tot he MOGA paper
    # first, it interpolates the fitness from 0 to 1 for each individual, and then
    # the average among each rank is assigned to all the individuals within that rank
    def applyMOGAScaling(sorted_individuals, sorted_fronts):
        fitness_interp = np.linspace(1,0,len(sorted_individuals))
        
        # average the fitness of all the individuals within each rank
        new_fitness = []
        for i in np.unique(sorted_fronts):
            front_individuals_idx = np.nonzero(np.array(sorted_fronts) == i)[0]
            front_individuals = [sorted_individuals[f] for f in front_individuals_idx]
            
            avg_fitness = np.mean(fitness_interp[front_individuals_idx])
            new_fitness.extend(np.ones(len(front_individuals)) * avg_fitness)
        
        return new_fitness    
    
    # computes the appropriate value for sigma_share from the non-dominated
    # individuals, as described in the MOGA paper
    # equation was simplified for 2 and 3 objectives! check sigma_share.png
    def computeSigmaShare(self, nondominated_individuals): 
        number_objectives = len(nondominated_individuals[0].fitness)
        
        N = self.getNumberOfIndividuals()
        
        fitnesses = np.array(Population.getFitnessPoints(nondominated_individuals))
        m = np.min(fitnesses, axis=0)
        M = np.max(fitnesses, axis=0)
        
        if number_objectives == 2:
            equation_solutions = np.roots([N-1, -(M[1]-m[1]+M[0]-m[0]), 0])
        elif number_objectives == 3:
            equation_solutions = np.roots([N-1, -(M[2]-m[2]+M[1]-m[1]+M[0]-m[0]),
                                           -(M[2]-m[2]+M[1]-m[1]+M[0]-m[0] + (M[1]-m[1])*(M[0]-m[0])), 0])
            
        sigma_share = np.max(equation_solutions)
        print(f'Sigma_share = {sigma_share}')
        return sigma_share
    
    # applies fitness sharing as in the MOGA paper; sigma_share is computed from
    # the minimum and maximum values of each objective found within the rank 1 individuals
    def applyFitnessSharing(self, sorted_individuals, sorted_fronts, sorted_fitness):
        new_sorted_fitness = []
        alpha = 1
        
        nondominated_idx = np.nonzero(np.array(sorted_fronts) == 1)[0]
        nondominated_individuals = [sorted_individuals[f] for f in nondominated_idx]
        sigma_share = self.computeSigmaShare(nondominated_individuals)
        #sigma_share = 0.15 
        
        for i in np.unique(sorted_fronts):
            front_individuals_idx = np.nonzero(np.array(sorted_fronts) == i)[0]
            front_individuals = [sorted_individuals[f] for f in front_individuals_idx]
            fitness = [sorted_fitness[f] for f in front_individuals_idx]
            
            niche_counts = np.zeros(len(front_individuals))
    
            for ind in front_individuals:
                for other in front_individuals:   
                    dist = distance.chebyshev(ind.fitness, other.fitness)
                    
                    # sharing function sh(d)
                    if dist < sigma_share:
                        niche_counts[front_individuals.index(ind)] += 1 - (dist/sigma_share)**alpha # niche_count = sum(sh(d))
            
            for f in range(len(fitness)):
                new_sorted_fitness.append(fitness[f]/niche_counts[f]) # F' = F / niche_count
                
        return new_sorted_fitness        
    
    # returns a sorted list of individuals according to their nondomination rank
    # as well as the corresponding front number; used in NSGA-II (and implemented
    # according to the original paper)
    def applyNonDominatedSorting(individuals):
        front_numbers = []
        sorted_inds = []
        
        Np = [] # for each individual, how many others dominate it
        Sp = [] # for each individual, the set of dominated solutions
        for i in range(len(individuals)):
            dominated_count = 0 # number of individuals the current individual is dominated by
            dominated_list = [] # list of individuals (index) dominated by the current individual
            for j in range(len(individuals)):
                if i != j and individuals[j].dominates(individuals[i]):
                    dominated_count += 1
                elif i !=j and individuals[i].dominates(individuals[j]):
                    dominated_list.append(j)
                    
            Np.append(dominated_count)
            Sp.append(dominated_list)
            
        # first front: np = 0; add it to the sorted list
        front_idx = np.nonzero(np.array(Np)==0)[0]
        sorted_inds.extend([individuals[i] for i in front_idx])
        front_numbers.extend(np.ones(len(front_idx)))
        
        front_counter = 2
        while len(sorted_inds) < len(individuals):
            Q = [] # new front
            for p in front_idx:
                for q in Sp[p]:
                    Np[q] -= 1 # subtract domination count of the dominated individuals
                    if Np[q] == 0: 
                        Q.append(q) 
            
            # add the newly determined front to the sorted list
            sorted_inds.extend([individuals[i] for i in Q])
            front_numbers.extend(front_counter * np.ones(len(Q)))
            
            front_counter += 1
            front_idx = Q[:] # repeat the process considering the newly determined front
        
        return sorted_inds, front_numbers
        
    
    # computes the crowding distance for a given list of individuals from the same
    # front (according to the NSGA-II paper)
    def computeCrowdingDistance(front):
        distances = np.zeros(len(front))
        number_objectives = len(front[0].fitness)
        
        for m in range(number_objectives):
            # sort by each objective's fitness, ascending
            sorted_front = front[:] # deep copy (sort of?)
            sorted_front.sort(key=lambda x: x.fitness[m], reverse=False)
            
            fitness_list = [x.fitness[m] for x in sorted_front] #fitness values in the same order
            fmax = np.max(fitness_list)
            fmin = np.min(fitness_list)
            
            # first and last in the sorted front = infinite (boundaries are always selected)
            distances[front.index(sorted_front[0])] = np.inf
            distances[front.index(sorted_front[-1])] = np.inf
            
            # remaining ones computed with formula:
            for i in range(1, len(sorted_front) - 1):
                if fmax-fmin != 0:
                    distances[front.index(sorted_front[i])] += (fitness_list[i+1] - fitness_list[i-1]) / (fmax-fmin)
                else:
                    distances[front.index(sorted_front[i])] += 0
                    
        return distances   
    
    # calculates the accumulated fitness and normalized so that the last element
    # is valued 1
    def calculateAccumulatedFitnessNormalized(fitness_list):
        individuals_fitness = fitness_list
        individuals_fitness=individuals_fitness+abs(np.min(individuals_fitness))
        
        return np.cumsum(individuals_fitness) / np.sum(individuals_fitness)
        
    # applies SUS to a previously sorted population (for MOGA) and returns the
    # selected individuals
    def applyStochasticUniversalSampling(sorted_pop, sorted_fitness, step_pointer):
        sorted_pop = np.array(sorted_pop)
        if np.sum(sorted_fitness)==0: # to avoid when all individuals have fitness 0
           indexes=[]
           for i in range (0,1):
               indexes.append(random.randint(0,len(sorted_pop)-1))
               selected = sorted_pop[indexes]
               return selected
        else:    
            pointers=Population.getEquallySpacedPointers(random.random(), step_pointer) 
            indexes = Population.selectIndividualsIndexByPointer(sorted_fitness, pointers)
            selected = sorted_pop[indexes]
            return selected
    
    # for the vector of pointers, selects the pointer
    def selectIndividualsIndexByPointer(fitness_list, pointers):
        indexes=[]
        for pointer in pointers:
            if pointer < Population.calculateAccumulatedFitnessNormalized(fitness_list)[0]:
                indexes.append(0)
            else:
                indexes.append(np.where(pointer>=Population.calculateAccumulatedFitnessNormalized(fitness_list))[0][-1]+1)
        
        return indexes
    
    # given a pointer, this method will return a series of equally spaced pointers 
    # with the step_size of the pointer    
    def getEquallySpacedPointers(pointer_father, step_pointer):
        upper_part=np.arange(pointer_father,1,step_pointer)
        down_part=np.flip(np.arange(pointer_father,0,-step_pointer))
        return np.concatenate((down_part,upper_part[1:]),axis=0)
    
    
    # selects parents based on binary tournaments
    # e.g. randomly select 2 individuals from a given list, which MUST be sorted
    # according to the individual's fitness (e.g. in NSGA-II, by nondominated sorting + crowding distance)
    def applyBinaryTournamentSelection(individuals, mating_pool_size):
        mating_pool = []
        while len(mating_pool) < mating_pool_size:
            participants_idx = np.random.choice(np.arange(0, len(individuals)), 2, replace = False)
            
            # since individuals are sorted by nondominated sorting + crowding distance
            # a lower index means a higher fitness
            winner = individuals[np.min(participants_idx)]
            mating_pool.append(winner)
        
        return mating_pool
    
    # performs parent selection as described in the NSGA-II paper:
    # first, the current population is sorted by non-domination; then, each front
    # is sorted by crowding distance; finally, binary tournament selection is applied
    def applyNSGA2Selection(self, mating_pool_size):
        mating_pool = []
        
        # apply non-dominated sorting to current population (higher front number = better)
        sorted_pop, front_numbers = Population.applyNonDominatedSorting(deepcopy(self.individuals))
        
        # sort each front based on the crowding distance
        sorted_individuals = []
        for i in np.unique(front_numbers):
            front_individuals_idx = np.nonzero(np.array(front_numbers) == i)[0]
            front_individuals = [sorted_pop[f] for f in front_individuals_idx]
            
            crwd_dist = Population.computeCrowdingDistance(front_individuals)
            
            sorted_ind = [ind for _,ind in sorted(zip(crwd_dist, front_individuals), key = lambda t: t[0], reverse = True)]
            sorted_distances = [f for f,_ in sorted(zip(crwd_dist, front_individuals), key = lambda t: t[0], reverse = True)]
            
            sorted_individuals.extend(sorted_ind)
        
        mating_pool = Population.applyBinaryTournamentSelection(sorted_individuals, mating_pool_size)
        return mating_pool
    
    # # performs parent selection as described in the SMS-EMOA paper: randomly select
    # # two parents to generate ONE individual
    # def applySMSEMOASelection(self):
    #     parents_idx = np.random.choice(np.arange(0, self.getNumberOfIndividuals()), 2, replace = False)
    #     return [deepcopy(self.individuals[i]) for i in parents_idx]
    
    # performs parent selection combining both NSGA-II and SMS-EMOA: individuals sorted by
    # non-dominance, and then by their hypervolume contribution; then, they are selected
    # using binary tournaments
    def applySMSEMOASelection(self, mating_pool_size):
        mating_pool = []
        
        # apply non-dominated sorting to current population (higher front number = better)
        sorted_pop, front_numbers = Population.applyNonDominatedSorting(deepcopy(self.individuals))
        
        # sort each front based on the hypervolume contribution
        sorted_individuals = []
        for i in np.unique(front_numbers):
            front_individuals_idx = np.nonzero(np.array(front_numbers) == i)[0]
            front_individuals = np.array([sorted_pop[f] for f in front_individuals_idx])
            
            if len(front_individuals) == 1:
                sorted_individuals.extend(front_individuals) # no need to sort in this case
            else:
                # compute contribution = hypervolume without accounting for the individual
                hv_contribution = []
                S = Population.computeHypervolume(front_individuals)
                for j in range(len(front_individuals)):
                    S_ = Population.computeHypervolume(front_individuals[np.arange(len(front_individuals))!=j]) # hypervolume without individual i
                    hv_contribution.append(S - S_)
                
                sorted_ind = [ind for _,ind in sorted(zip(hv_contribution, front_individuals), key = lambda t: t[0], reverse = True)]
                sorted_contributions = [f for f,_ in sorted(zip(hv_contribution, front_individuals), key = lambda t: t[0], reverse = True)]
                
                sorted_individuals.extend(sorted_ind)
        
        mating_pool = Population.applyBinaryTournamentSelection(sorted_individuals, mating_pool_size)
        return mating_pool
    
    # performs parent selection as described in the MOGA paper: after ranking
    # individuals based on domination, a fitness value is assigned by interpolating
    # a linear function and averaging within each rank; then, parents are selected
    # using SUS to avoid selection errors;
    def applyMOGASelection(self, mating_pool_size):
        mating_pool = []
        
        # apply ranking according to number of individuals dominating each one (+1)
        sorted_pop, ranks = Population.applyMOGARanking(deepcopy(self.individuals))
        
        # apply scaling (as stated in the MOGA paper) to get a list of descending fitness
        scaled_fitness = Population.applyMOGAScaling(sorted_pop, ranks)
        
        # apply fitness sharing
        sorted_fitness = self.applyFitnessSharing(sorted_pop, ranks, scaled_fitness) 
        
        # select parents using Stochastic Universal Sampling (SUS)
        mating_pool = Population.applyStochasticUniversalSampling(sorted_pop, sorted_fitness, 1/mating_pool_size)
        
        return mating_pool
    
    # selects parents based on tournaments
    # e.g. randomly select k individuals, choose the one with the highest fitness
    # and repeat until mating pool reaches a given size
    def applyTournamentSelection(self, tournament_size, mating_pool_size):
        #random.shuffle(self.individuals)
        mating_pool = []
        while len(mating_pool) < mating_pool_size:
            participants_idx = np.random.choice(np.arange(0, self.getNumberOfIndividuals()), tournament_size, replace = False)
            participants = [self.individuals[i] for i in participants_idx]
            
            winner = participants[np.argmax([[i.fitness for i in participants]])]
            mating_pool.append(winner)
        
        return mating_pool
    
    # selects parents based on their relative fitness
    # e.g. each individual has a probability of fitness/sum_fitness to be selected
    def applyRouletteWheelSelection(self, mating_pool_size):
        mating_pool = []
        
        probability_sum = np.sum([ind.fitness for ind in self.individuals])
        selection_probabilities = [ind.fitness / probability_sum for ind in self.individuals]
        
        while len(mating_pool) < mating_pool_size:
            selected = np.random.choice(self.individuals, p = selection_probabilities)
            mating_pool.append(selected)
        
        return mating_pool
    
    # ranks the individuals by sorting them in descending order of fitness and
    # assigns a probability of rank/cumsum(ranks) of being chosen
    def applyRankingSelection(self, mating_pool_size):
        mating_pool = []
        current_population = self.individuals
        current_population.sort(key=lambda x: x.fitness, reverse=True) # sort by fitness, descending
        
        # linear ranking selection probability
        probabilities = np.arange(self.getNumberOfIndividuals(), 0, -1)/np.sum(np.arange(self.getNumberOfIndividuals(), 0, -1))
        
        while len(mating_pool) < mating_pool_size:
            selected = np.random.choice(self.individuals, p = probabilities)
            mating_pool.append(selected)
        
        return mating_pool
        
    # starts by ranking the individuals (sorted in descending order by fitness) and then selects the
    # parents based on tournaments, which are done in sequential order
    def applyRankedTournamentSelection(self, tournament_size, mating_pool_size):
        mating_pool = []
        
        current_population = self.individuals
        current_population.sort(key=lambda x: x.fitness, reverse=True) # sort by fitness, descending
        
        participants_idx = np.arange(0, tournament_size)
        while len(mating_pool) < mating_pool_size:
            participants = [self.individuals[i] for i in participants_idx]
            
            winner = participants[np.argmax([[i.fitness for i in participants]])]
            mating_pool.append(winner)
            
            participants_idx += tournament_size # next k individuals
            
            # start from the individuals with the highest fitness again after reaching the end
            if np.count_nonzero(participants_idx >= self.getNumberOfIndividuals()):
               participants_idx = np.arange(0, tournament_size) 
        
        return mating_pool
    
    # performs survivor selection as described in the NSGA-II paper:
    # first, the current population (twice its normal size) is sorted by non-domination; 
    # entire fronts are added to the new population until the last one has too many individuals
    # this last one is then sorted by crowding distance, and the better individuals (with the
    # highest distances) fill out the rest of the new generation
    def applyNSGA2Replacement(self, offspring, args):
        # add offspring to current population
        self.individuals.extend(offspring)
        new_generation = []
        
        # evaluate the offspring
        self.evaluate(args[0], args[1], args[2], args[3], args[4], args[5], args[6], True)
        
        # apply non-dominated sorting to the new merged population (higher front number = better)
        sorted_pop, front_numbers = Population.applyNonDominatedSorting(deepcopy(self.individuals))
        
        # add entire fronts until adding the next one results in too many individuals
        for i in np.unique(front_numbers):
            front_individuals_idx = np.nonzero(np.array(front_numbers) == i)[0]
            front_individuals = [sorted_pop[f] for f in front_individuals_idx]
            
            if len(front_individuals) <= self.getNumberOfIndividuals()/2 - len(new_generation):
                new_generation.extend(front_individuals)
            else:
                # sort the last front based on the crowding distance
                crwd_dist = Population.computeCrowdingDistance(front_individuals)
                
                sorted_ind = [ind for _,ind in sorted(zip(crwd_dist, front_individuals), key = lambda t: t[0], reverse = True)]
                sorted_distances = [f for f,_ in sorted(zip(crwd_dist, front_individuals), key = lambda t: t[0], reverse = True)]
                
                remaining_individuals = sorted_ind[0:int(self.getNumberOfIndividuals()/2) - len(new_generation)]
                new_generation.extend(remaining_individuals)
                
                break
        
        self.individuals = new_generation
        
    # # performs survivor selection as described in the SMS-EMOA paper:
    # # apply non-dominated sorting to the population (plus the new individual)
    # # and rank them in fronts; compute, for the worst ranked front, the contribution
    # # of each individual for the hypervolume; discard the one with the lowest contribution
    # def applySMSEMOAReplacement(self, offspring, args):
    #     # add offspring to current population
    #     self.individuals.extend(offspring) # only adding the first generated one, to follow the paper...
    #     new_generation = []
        
    #     # evaluate the offspring
    #     self.evaluate(args[0], args[1], args[2], args[3], args[4], args[5], args[6], True)
        
    #     # apply non-dominated sorting to the new merged population (higher front number = better)
    #     sorted_pop, front_numbers = Population.applyNonDominatedSorting(deepcopy(self.individuals)[0:int(self.getNumberOfIndividuals()/2) + 1])
        
    #     # compute hypervolume contribution for each individual in the worst front
    #     worst_front_idx = np.nonzero(np.array(front_numbers) == np.max(front_numbers))[0]
    #     worst_front = np.array([sorted_pop[i] for i in worst_front_idx])
        
    #     if len(worst_front) == 1:
    #         sorted_pop.pop(-1) # if only one individual in the worst front, discard it
    #         new_generation = sorted_pop
            
    #     else:
    #         hv_contribution = []
            
    #         S = Population.computeHypervolume(worst_front)
    #         for i in range(len(worst_front)):
    #             S_ = Population.computeHypervolume(worst_front[np.arange(len(worst_front))!=i]) # hypervolume without individual i
    #             hv_contribution.append(S - S_)
            
    #         discard_idx = worst_front_idx[np.argmin(hv_contribution)]
        
    #         # discard the worst individual of the worst front
    #         new_generation = sorted_pop[:discard_idx] + sorted_pop[discard_idx:]
        
    #     self.individuals = new_generation
    
    # performs survivor selection combining both NSGA-II and SMS-EMOA:
    # apply non-dominated sorting to the population (current generation + offspring)
    # and rank them in fronts; add fronts until the last one is too large for the new
    # generation; rank this front by the hypervolume contribution and keep the best ones
    def applySMSEMOAReplacement(self, offspring, args):
        # add offspring to current population
        self.individuals.extend(offspring)
        new_generation = []
        
        # evaluate the offspring
        self.evaluate(args[0], args[1], args[2], args[3], args[4], args[5], args[6], True)
        
        # apply non-dominated sorting to the new merged population (higher front number = better)
        sorted_pop, front_numbers = Population.applyNonDominatedSorting(deepcopy(self.individuals))
        
        # add entire fronts until adding the next one results in too many individuals
        for i in np.unique(front_numbers):
            front_individuals_idx = np.nonzero(np.array(front_numbers) == i)[0]
            front_individuals = np.array([sorted_pop[f] for f in front_individuals_idx])
            
            if len(front_individuals) <= self.getNumberOfIndividuals()/2 - len(new_generation):
                new_generation.extend(front_individuals)
            else:
                # sort the last front based on the hypervolume contribution
                if len(front_individuals) == 1:
                    new_generation.extend(front_individuals) # no need to sort in this case
                else:
                    # compute contribution = hypervolume without accounting for the individual
                    hv_contribution = []
                    S = Population.computeHypervolume(front_individuals)
                    for j in range(len(front_individuals)):
                        S_ = Population.computeHypervolume(front_individuals[np.arange(len(front_individuals))!=j]) # hypervolume without individual i
                        hv_contribution.append(S - S_)
                    
                    sorted_ind = [ind for _,ind in sorted(zip(hv_contribution, front_individuals), key = lambda t: t[0], reverse = True)]
                    sorted_contributions = [f for f,_ in sorted(zip(hv_contribution, front_individuals), key = lambda t: t[0], reverse = True)]
                    
                    remaining_individuals = sorted_ind[0:int(self.getNumberOfIndividuals()/2) - len(new_generation)]
                    new_generation.extend(remaining_individuals)
                
                break
        
        self.individuals = new_generation
    
    # performs survivor selection as found in papers/PhD thesis about MOGA: generational
    # but a small percentage (recommended = 10%) of random immigrants are added
    def applyMOGAReplacement(self, offspring, immigrant_percentage):
        new_generation = offspring
        
        number_immigrants = int(round(len(self.individuals)*immigrant_percentage, 0))
        random_immigrants = []
        for i in range(number_immigrants):
            random_immigrants.append(Individual(offspring[0].getNumberOfFeatures()))
        
        #new_generation = offspring[0:len(offspring) - number_immigrants]
        new_generation.extend(random_immigrants) # fill out the rest of the generation with immigrants
        
        self.individuals = new_generation
        
        
    # replaces all the individuals of the previous generation with the new offspring
    def applyGenerationalReplacement(self, offspring):
        self.individuals = offspring
        
    # merges both parents and offspring, ranks them by fitness, and keeps the top
    # ones (with equal size to the population)
    def applyMuPlusLambdaReplacement(self, parents, offspring):
        merged = parents + offspring
        merged.sort(key=lambda x: x.fitness, reverse=True) # sort by fitness, descending
        
        self.individuals = merged[0:self.getNumberOfIndividuals()]
    
    # discards parents, and keeps the top offspring (with equal size to the population)
    # typically used when there are more individuals in the offspring than population size
    def applyMuCommaLambdaReplacement(self, offspring):
        offspring.sort(key=lambda x: x.fitness, reverse=True) # sort by fitness, descending
        
        self.individuals = offspring[0:self.getNumberOfIndividuals()]
    
    # evaluates the entire population, that is, for each individual, it performs 2nd-level feature extraction
    # and trains a classifier with given data; then, this classifier is tested on new data, its output is regularized
    # with the Firing Power method and its performance is evaluated; the fitness function evaluates each individual
    # for each of the given objectives      
    def evaluate(self, patient, training_seizures, legend, sliding_window_step,
                 classifier_type, FP_threshold, objectives, offspring_flag):
        # patient = Patient() object with seizure data loaded
        # training_seizures, testing_seizures = lists with idx of seizure_data
        # legend = list with 1st-level feature labels (loaded from Database)
        # sliding_window_step = step for 2nd-level feature extraction in minutes
        # classifier type = string specifying which kind of classifier to use
        # FP_threshold = threshold (from 0 to 1) for the Firing Power method
        # objectives = list of strings containing the objectives to be evaluated
        # offspring_flag = boolean indicating whether we are evaluating the entire 
        #                  population or just the offspring (in the case of NSGA-II, we have twice as many individuals)
        
        if offspring_flag == False:
            pop_range = [0, self.getNumberOfIndividuals()]
        else:
            pop_range = [int(self.getNumberOfIndividuals()/2), self.getNumberOfIndividuals()] #offspring is the other half of the population (NSGA-II)
            
        for i in range(pop_range[0], pop_range[1]):
            if self.individuals[i].fitness == []: # only evaluate if fitness hasn't been evaluated yet (obviously...)
                #print(f"Evaluating individual {i+1}/{pop_range[1]}...")
                #print("Creating classifier and sliding window...")
                clf = Classifier(classifier_type)
                sw = SlidingWindow(deepcopy(self.individuals[i].features), sliding_window_step)
                
                #print("Computing SOP and SPH...")
                SOP = sw.computePreictal(self.individuals[i].decodePreictalThreshold())
                SPH = 10
                
                # train iteratively: train with the 1st seizure, test on the 2nd one
                # train with the first 2 seizures, test on the 3rd one, etc...
                train = [0]
                test = training_seizures[1:]
                
                metrics_sens = []
                false_alarms = 0
                total_interictal = 0
                lost_interictal = 0
                metrics_samplesens = []
                metrics_timefalsealarm = []
                while len(test) != 0:
                    #print(f"Training with seizures: {train}")
                    # TRAINING (all available "train" seizures up until the first seizure in "test")
                    for j in train:
                        #print(f"Extracting features from seizure #{j}...")
                        new_data, new_legend = sw.extractFeatures(patient.seizure_data[j], legend, self.individuals[i].decodePreictalThreshold())
                        
                        if j == train[0]:
                            training_data = new_data
                        else:
                            training_data = np.hstack((training_data, new_data))
                    
                    #print("Training classifier...")
                    clf.trainClassifier(training_data, new_legend)
                
                    # TESTING (on the next seizure -> first in the "test" list)
                    #print(f"Extracting features from seizure #{test[0]}...")
                    #print(patient.seizure_data[test[0]])
                    new_data, new_legend = sw.extractFeatures(patient.seizure_data[test[0]], legend, self.individuals[i].decodePreictalThreshold())
                    testing_data = new_data
                    
                    #print("Classifying data...")
                    #print(new_legend)
                    #print(testing_data)
                    clf_output, true_labels = clf.classifyData(testing_data, new_legend)
                    #print("Applying Firing Power...")
                    clf_output_processed = Classifier.applyFiringPower(clf_output, true_labels, FP_threshold)
                    
                    # not using the refractory behavior here... only for testing outside the evolutionary search
                    
                    # COMPUTE METRICS
                    if 'sensitivity' in objectives:
                        #print("Computing sensitivity...")
                        clf_seizuresens = Classifier.getSensitivity(clf_output_processed, true_labels)
                        metrics_sens.append(clf_seizuresens)
                    if 'fprh' in objectives:
                        #print("Computing FPR/h...")
                        clf_falsealarms = Classifier.getNumberOfFalseAlarms(clf_output_processed, true_labels)
                        clf_totalinterictal, clf_lostinterictal = Classifier.getInterictalTime(clf_output_processed, true_labels, SOP, SPH, sliding_window_step)
                        false_alarms += clf_falsealarms
                        total_interictal += clf_totalinterictal
                    if 'sample_sensitivity' in objectives:
                        #print("Computing sample sensitivity...")
                        clf_samplesens = Classifier.getSampleSensitivity(clf_output_processed, true_labels)
                        metrics_samplesens.append(clf_samplesens)
                    if 't_under_false_alarm' in objectives:
                        #print("Computing time under false alarm...")
                        clf_timefalsealarm = Classifier.getTimeUnderFalseAlarm(clf_output_processed, true_labels)
                        metrics_timefalsealarm.append(clf_timefalsealarm)
                    
                    # UPDATE TRAINING/TESTING GROUPS
                    train.append(test[0])
                    test.pop(0)
                
                # EVALUATE AND UPDATE FITNESS
                fitness_scores = []
                names = []
                if 'sensitivity' in objectives:
                    #print("Evaluating sensitivity objective...")
                    mean_sens = np.mean(metrics_sens) # average sensitivity
                    ind_fitness_sens = FitnessFunction.evaluateSensitivity(mean_sens)
                    fitness_scores.append(ind_fitness_sens)
                    names.append('sensitivity')
                if 'fprh' in objectives:
                   # print("Evaluating FPR/h objective...")
                    mean_fprh = Classifier.getFPRh(false_alarms, true_labels, total_interictal, lost_interictal)
                    ind_fitness_fprh = FitnessFunction.evaluateFPRh(mean_fprh)
                    fitness_scores.append(ind_fitness_fprh)
                    names.append('fprh')
                if 'sample_sensitivity' in objectives:
                    #print("Evaluating sample sensitivity objective...")
                    mean_samplesens = np.mean(metrics_samplesens) # average sample sensitivity
                    ind_fitness_samplesens = FitnessFunction.evaluateSampleSensitivity(mean_samplesens)
                    fitness_scores.append(ind_fitness_samplesens)
                    names.append('sample_sensitivity')
                if 't_under_false_alarm' in objectives:
                    #print("Evaluating time under false alarm objective...")
                    mean_timefalsealarm = np.mean(metrics_timefalsealarm) # average time under false alarm
                    ind_fitness_timefalsealarm = FitnessFunction.evaluateTimeUnderFalseAlarm(mean_timefalsealarm)
                    fitness_scores.append(ind_fitness_timefalsealarm)
                    names.append('t_under_false_alarm')
                if 'electrodes' in objectives:
                    #print("Evaluating electrode objective...")
                    ind_fitness_electrodes = FitnessFunction.evaluateElectrodes(self.individuals[i].features)
                    fitness_scores.append(ind_fitness_electrodes)
                    names.append('electrodes')
                #ind_total_fitness = FitnessFunction.evaluateFitness([ind_fitness_sens, ind_fitness_fprh])
                
                self.individuals[i].updateFitness(fitness_scores, names)
        
        # UPDATE GENERATION NUMBER (unless we're evaluating the offspring for an elitist replacement method)
        if offspring_flag == False:
            self.generation += 1
        
        # SAVE GENERATION FITNESS FOR EVERY INDIVIDUAL (unless we're only evaluating the offspring)
        if offspring_flag == False:
            self.fitness_history.append(self.getFitnessList()) # each generation's fitness list is a list of numpy arrays!
        
    
    # applies parent selection, variation operators (recombination + mutation) and replacement
    # to the current generation, with given methods and mutation/crossover rates
    def evolve(self, selection_method, crossover_rate, mutation_rate, replacement_method, args):
        # args = list of arguments for the evaluate() method, needed for elitist replacement methods
        
        # SELECTION
        if selection_method == "tournament":
            selected = self.applyTournamentSelection(3, self.getNumberOfIndividuals()/2) # k = 3 at the moment!
        elif selection_method == "roulette":
            selected = self.applyRouletteWheelSelection(self.getNumberOfIndividuals()/2)
        elif selection_method == "ranking":
            selected = self.applyRankingSelection(self.getNumberOfIndividuals()/2) 
        elif selection_method == "ranked_tournament":
            selected = self.applyRankedTournamentSelection(3, self.getNumberOfIndividuals()/2) # k = 3 at the moment!
            
        elif selection_method == "NSGA2":
            print("Performing NSGA-II selection...")
            selected = self.applyNSGA2Selection(self.getNumberOfIndividuals()/2)
        elif selection_method == "SMS-EMOA":
            selected = self.applySMSEMOASelection(self.getNumberOfIndividuals()/2)
        elif selection_method == "MOGA":
            selected = self.applyMOGASelection(self.getNumberOfIndividuals()/2)
        
        # apply DECISION MAKER to guarantee that the mating pool has a high fitness (as well as variability)
        print("Applying Decision Maker...")
        selected = Population.applyDecisionMaker(selected, 0.9, 0.9)
            
        # VARIATION OPERATORS
        offspring = []
        
        random.shuffle(selected) # randomly shuffle selected individuals
        
        # apply RECOMBINATION to selected parents
        if selection_method == "MOGA":
            number_offspring = int(self.getNumberOfIndividuals() - 0.1*self.getNumberOfIndividuals()) # random immigrants...
        else:
            number_offspring = self.getNumberOfIndividuals()
        
        print("Performing recombination...")
        while len(offspring) < number_offspring:
            for j in range(len(selected)):
                roll = random.randint(1, 100) / 100
                
                # recombine the two individuals
                if roll <= crossover_rate:
                    if j == len(selected) - 1:
                        offspring.append(selected[j].recombine(selected[0]))
                    else:
                        offspring.append(selected[j].recombine(selected[j+1]))
                # "clone" one of the parents
                else:
                    if j == len(selected) - 1:
                        offspring.append(selected[random.choice([0,j])].clone())
                    else:
                        offspring.append(selected[random.choice([j,j+1])].clone())
                
                if len(offspring) == number_offspring:
                    break
        
        # apply MUTATION to resulting offspring
        print("Performing mutation...")
        for j in range(len(offspring)):
            roll = random.randint(1, 100) / 100
            if roll <= mutation_rate:
                offspring[j].mutate()
        
        # REPLACEMENT
        if replacement_method == "generational":
            self.applyGenerationalReplacement(offspring)
        elif replacement_method == "mupluslambda":
            self.applyMuPlusLambdaReplacement(selected, offspring) # missing offspring evaluation (args)
        elif replacement_method == "mucommalambda":
            self.applyMuCommaLambdaReplacement(offspring) # missing offspring evaluation (args)
            
        elif replacement_method == "NSGA2":
            print("Performing NSGA-II replacement...")
            self.applyNSGA2Replacement(offspring, args)
        elif replacement_method == "SMS-EMOA":
            self.applySMSEMOAReplacement(offspring, args)
        elif replacement_method == "MOGA":
            self.applyMOGAReplacement(offspring, 0.1) # second argument = immigrant percentage (10%, recommended in the literature)
    
    # applies non-dominated sorting and plots the current generation's individuals' fitness values
    # in a 2D plot; the Pareto front is represented as a red line in the graph
    def plotFitness2D(self, objectives, save_path = ""):
        # apply non-dominated sorting to the new merged population (higher front number = better)
        sorted_ind, sorted_front = Population.applyNonDominatedSorting(deepcopy(self.individuals))
        
        # find the corresponding indices of the two objectives
        labels = sorted_ind[0].fitness_labels
        obj1_idx = labels.index(objectives[0])
        obj2_idx = labels.index(objectives[1])
        
        if sorted_ind[0].fitness_labels[obj1_idx] == "sample_sensitivity":
            label1 = "Sample sensitivity"
        elif sorted_ind[0].fitness_labels[obj1_idx] == "t_under_false_alarm":
            label1 = "Time under false alarm"
        else:
            label1 = sorted_ind[0].fitness_labels[obj1_idx]
        
        if sorted_ind[0].fitness_labels[obj2_idx] == "sample_sensitivity":
            label2 = "Sample sensitivity"
        elif sorted_ind[0].fitness_labels[obj2_idx] == "t_under_false_alarm":
            label2 = "Time under false alarm"
        else:
            label2 = sorted_ind[0].fitness_labels[obj2_idx]
        
        # plot everything
        fig = plt.figure()

        points = np.array(Population.getFitnessPoints(sorted_ind))
        plt.scatter(points[:,obj1_idx], points[:,obj2_idx])
        
        pareto_front_idx = np.nonzero(np.array(sorted_front) == 1)[0]
        pareto_points = np.array(Population.getFitnessPoints([sorted_ind[i] for i in pareto_front_idx]))
        #pareto_points_sorted = np.sort(pareto_points, axis=0)
        
        if len(pareto_front_idx) == 1:
            plt.scatter(pareto_points[:,obj1_idx], pareto_points[:,obj2_idx], c = 'red', zorder=1)
        else:
            plt.scatter(pareto_points[:,obj1_idx], pareto_points[:,obj2_idx], c = 'red', zorder=1)
            #plt.plot(pareto_points_sorted[:,obj1_idx], pareto_points_sorted[:,obj2_idx], 'red', zorder=1)
            
        plt.xlim(0,1.1)
        plt.ylim(0,1.1)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title('Generation ' + str(self.generation))
        
        plt.show()
        plt.tight_layout()
        
        if save_path != "":
            fig.savefig(save_path)
    
    # returns the average fitness values found within the Pareto optimal set
    # for each of the objectives (used as stopping criteria) -> deprecated 
    def getMeanParetoFitness(self):
        sorted_ind, sorted_front = Population.applyNonDominatedSorting(deepcopy(self.individuals))
        
        nondominated_idx = np.nonzero(np.array(sorted_front) == 1)[0]
        nondominated_individuals = [sorted_ind[f] for f in nondominated_idx]
        
        fitness_array = np.array(Population.getFitnessPoints(nondominated_individuals))
        
        return np.mean(fitness_array, axis=0)
    
    # applies a Decision Maker to restrict the mating pool to individuals with
    # a high fitness in the first two objectives (sample sensitivity and time under false alarm) 
    def applyDecisionMaker(mating_pool, obj1_threshold, obj2_threshold):
        selected_parents = []
        count = 0
        for ind in mating_pool:
            if ind.fitness[0] >= obj1_threshold and ind.fitness[1] >= obj2_threshold:
                selected_parents.append(ind)
                count+=1
                #print(count)
        
        while count < len(mating_pool)/2:
            # lower threshold , in this case, until at least half of the mating pool is kept
            # to guarantee some variety within the parents
            selected_parents = []
            count = 0
            obj1_threshold -= 0.05
            obj2_threshold -= 0.05
            #print(obj1_threshold)
            for ind in mating_pool:
                if ind.fitness[0] >= obj1_threshold and ind.fitness[1] >= obj2_threshold:
                    selected_parents.append(ind)
                    count+=1
                    #print(count)
        
        return selected_parents
        
        
            
    
    