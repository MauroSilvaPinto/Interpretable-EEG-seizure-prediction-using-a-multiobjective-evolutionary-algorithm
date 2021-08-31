"""
study the phenotype of the obtained individuals
by studying gene presence, and making bar plots.

ps: in order to this script to work,
you need to put it in the Evolutionary Algorithm
folder, along with barplot_annotate_brackets.py

or

to simple paste in this folder the Evolutionary Algorithm classes.

"""

from Database import Database
from Patient import Patient
from Population import Population
from StatisticalValidation import StatisticalValidation
from barplot_annotate_brackets import barplot_annotate_brackets

import numpy as np
import matplotlib.pyplot as plt

import pickle
import os


#%% Setup patient ID, algorithm, paths, etc.

ID = "1321803"
algorithm = "NSGA2" # NSGA2, SMS-EMOA or MOGA (only NSGA2 will be used for the paper)

# go back to data folder
os.chdir("..")
os.chdir("..")
os.chdir("Data")
path=os.getcwd()

sep = os.path.sep
if path[-1] != sep:
    path+=sep
evol_path = path + "Processed_data" + sep
    
    
trained_evol_path = path + "Trained_evol" + sep + "pat_" + ID # retrieve trained evolutionary algorithms from here
if trained_evol_path[-1] != sep:
    trained_evol_path+=sep

sliding_window_step = 0.5
classifier_type = 'logreg'
fp_threshold = 0.7

filename = "pat" + ID + "_" + algorithm # patID_algorithm

#%% Setup testing data

db = Database(evol_path)

pat = Patient(ID)

filenames = db.getFilenames()

features, info = db.loadPatientData(pat.ID)
legend = db.loadLegend()

pat.seizure_data = features
pat.seizure_metadata = info

training_seizures = [0, 1, 2] # run evolutionary search on the first 3 seizures
testing_seizures = np.arange(3, pat.getNumberOfSeizures()).tolist()

#%% Select solutions

# Set a minimum fitness threshold for the first two objectives (sensitivity and time under false alarm)
if algorithm == "MOGA":
    fitness_threshold = 0.6
else:
    fitness_threshold = 0.9

# Load list of saved individuals from a previously trained algorithm
solutions = sorted(os.listdir(trained_evol_path))
solutions = [s for s in solutions if filename in s and "lastgen" in s] # only files with "patID_algorithm"
solutions = [trained_evol_path + sep + s for s in solutions]

# Retrieve Pareto-optimal solutions with fitness > threshold (for the first two objectives)
# from every run
selected_individuals = []
each_run = []
for file in solutions:
    individuals = pickle.load(open(file, "rb"))
    count = 0
    
    sorted_ind, sorted_front = Population.applyNonDominatedSorting(individuals)
    front_individuals_idx = np.nonzero(np.array(sorted_front) == 1)[0]
    front_individuals = np.array([sorted_ind[f] for f in front_individuals_idx])
    
    # apply Decision Maker
    for ind in front_individuals:
        if ind.fitness[0] >= fitness_threshold and ind.fitness[1] >= fitness_threshold:
            selected_individuals.append(ind)
            count+=1
    
    if count == 0:
        # lower threshold in this case (rare, but it happens in some MOGA runs)
        for ind in front_individuals:
            if ind.fitness[0] >= fitness_threshold - 0.1 and ind.fitness[1] >= fitness_threshold - 0.1:
                selected_individuals.append(ind)
                count+=1
        
    each_run.append(count)
    
#%% Study the SOP
  
pre_ictals=[]
for individual in selected_individuals:
    pre_ictals.append(StatisticalValidation.computeSOP(individual))
    

count_pre_ictals=[]
count_pre_ictals.append(pre_ictals.count(30))
count_pre_ictals.append(pre_ictals.count(35))
count_pre_ictals.append(pre_ictals.count(40))
count_pre_ictals.append(pre_ictals.count(45))
count_pre_ictals.append(pre_ictals.count(50))
count_pre_ictals.append(pre_ictals.count(55))
count_pre_ictals.append(pre_ictals.count(60))
count_pre_ictals.append(pre_ictals.count(65))
count_pre_ictals.append(pre_ictals.count(70))
count_pre_ictals.append(pre_ictals.count(75))   

ind = np.arange(len(count_pre_ictals))  # the x locations for the groups
width = 0.70  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(ind , count_pre_ictals/np.sum(count_pre_ictals), width,
                alpha=0.70)


labels=["30", "35", "40", "45", "50", "55", "60", "65", "70", "75"]

ax.set_ylabel('Occurrence Ratio (0-1)')
ax.set_xlabel('Minutes')

ax.set_title('Pre-ictals for Patient '+ ID)
ax.set_xticks(ind)
ax.set_xticklabels(labels,fontsize='small',
        rotation=0)
plt.grid(color='k', alpha=0.10, linestyle='-', linewidth=1)

#%% ################ Window Length, Instants, and Electrode Histogram ##################

electrodes_list=['C3','C4','Cz','F3','F4','F7', 'F8','FP1','FP2',
                 'Fz','O1','O2','P3','P4','Pz', 'T3','T4','T5','T6']

lobes_list=["Central","Frontal","Occipital", "Parietal","Temporal"]

hemispheres_list=["Left","Central","Right"]    

window_lengths = [1, 5, 10, 15, 20]
delays = [0, 5, 10, 15, 20, 25, 30]

electrode_count = np.zeros(19)
lobe_count = np.zeros(5)
hemisphere_count=np.zeros(3)
delays_count=np.zeros(len(delays))
windows_count=np.zeros(len(window_lengths))

unique_electrodes = np.zeros(5)
unique_lobes = np.zeros(5)
unique_hemispheres=np.zeros(3)
unique_windows=np.zeros(5)
unique_delays=np.zeros(5)

for ind in selected_individuals:
    elecs = []
    lobes = []
    hemispheres= []
    winds = []
    dls = []
    for f in ind.features:
        
        electrode_count[f.electrode] += 1
        windows_count[window_lengths.index(f.decodeWindowLength())] += 1
        delays_count[delays.index(f.decodeDelay())] += 1
    
        # Lobes
        if "C" == f.decodeElectrode()[0]:
            lobe_count[0] += 1
        if "F" == f.decodeElectrode()[0]: 
            lobe_count[1] += 1
        if "O" == f.decodeElectrode()[0]: 
            lobe_count[2] += 1
        if "P" == f.decodeElectrode()[0]: 
            lobe_count[3] += 1
        if "T" == f.decodeElectrode()[0]: 
            lobe_count[4] += 1
        
        # Hemispheres
        if "1" == f.decodeElectrode()[-1]:
            hemisphere_count[0] += 1
        if "3" == f.decodeElectrode()[-1]:
            hemisphere_count[0] += 1
        if "5" == f.decodeElectrode()[-1]:
            hemisphere_count[0] += 1
        if "7" == f.decodeElectrode()[-1]:
            hemisphere_count[0] += 1
            
        if "z" == f.decodeElectrode()[-1]:
            hemisphere_count[1] += 1
            
        if "2" == f.decodeElectrode()[-1]:
            hemisphere_count[2] += 1
        if "4" == f.decodeElectrode()[-1]:
            hemisphere_count[2] += 1
        if "6" == f.decodeElectrode()[-1]:
            hemisphere_count[2] += 1
 
        winds.append(f.decodeWindowLength())
        dls.append(f.decodeDelay())
        elecs.append(f.decodeElectrode())
        lobes.append(f.decodeElectrode()[0])
        
        #Hemispheres
        if "1" == f.decodeElectrode()[-1]:
            hemispheres.append("left")
        if "3" == f.decodeElectrode()[-1]:
            hemispheres.append("left")
        if "5" == f.decodeElectrode()[-1]:
            hemispheres.append("left")
        if "7" == f.decodeElectrode()[-1]:
            hemispheres.append("left")
            
        if "z" == f.decodeElectrode()[-1]:
            hemispheres.append("central")
            
        if "2" == f.decodeElectrode()[-1]:
            hemispheres.append("right")
        if "4" == f.decodeElectrode()[-1]:
            hemispheres.append("right")
        if "6" == f.decodeElectrode()[-1]:
            hemispheres.append("right")
            
    unique_electrodes[len(np.unique(elecs))-1] += 1
    unique_lobes[len(np.unique(lobes))-1] += 1
    unique_hemispheres[len(np.unique(hemispheres))-1] += 1
    unique_windows[len(np.unique(winds))-1] += 1
    unique_delays[len(np.unique(dls))-1] += 1
    
    
presence=(windows_count[0:5]/np.sum(windows_count)).tolist()
presence.append(0)
presence=presence+np.zeros(5).tolist()
presence.append(0)

presence_number=np.zeros(5).tolist()
presence_number.append(0)
presence_number=presence_number+(unique_windows[0:5]/np.sum(unique_windows)).tolist()
presence_number.append(0)

presence=presence+(delays_count[0:7]/np.sum(delays_count)).tolist()
presence.append(0)
presence=presence+np.zeros(5).tolist()
presence.append(0)

presence_number=presence_number+np.zeros(7).tolist()
presence_number.append(0)
presence_number=presence_number+(unique_delays[0:5]/np.sum(unique_delays)).tolist()
presence_number.append(0)

presence=presence+(electrode_count[0:19]/np.sum(electrode_count)).tolist()
presence.append(0)
presence=presence+np.zeros(5).tolist()
presence.append(0)
 
presence_number=presence_number+np.zeros(19).tolist()
presence_number.append(0)
presence_number=presence_number+(unique_electrodes[0:5]/np.sum(unique_electrodes)).tolist()
presence_number.append(0)

presence=presence+(lobe_count[0:5]/np.sum(lobe_count)).tolist()
presence.append(0)
presence=presence+np.zeros(5).tolist()
presence.append(0)
 
presence_number=presence_number+np.zeros(5).tolist()
presence_number.append(0)
presence_number=presence_number+(unique_lobes[0:5]/np.sum(unique_lobes)).tolist()
presence_number.append(0)

presence=presence+(hemisphere_count[0:3]/np.sum(hemisphere_count)).tolist()
presence.append(0)
presence=presence+np.zeros(3).tolist()
presence.append(0)
 
presence_number=presence_number+np.zeros(3).tolist()
presence_number.append(0)
presence_number=presence_number+(unique_hemispheres[0:3]/np.sum(unique_hemispheres)).tolist()
presence_number.append(0)

window_lengths_list = ["1", "5", "10", "15", "20"]
different_windows_list=["1","2","3","4","5"]

labels=window_lengths_list[0:5]
labels.append("")
labels=labels+different_windows_list[0:5]
labels.append("")

delays_list = ["0", "5", "10", "15", "20", "25", "30"]
different_delays_list=["1","2","3","4","5"]

labels=labels+delays_list[0:7]
labels.append("")
labels=labels+different_delays_list[0:5]
labels.append("")

different_electrodes_list=["1","2","3","4","5"]

labels=labels+electrodes_list[0:19]
labels.append("")
labels=labels+different_electrodes_list[0:5]
labels.append("")

different_lobes_list=["1","2","3","4","5"]

labels=labels+lobes_list[0:5]
labels.append("")
labels=labels+different_lobes_list[0:5]
labels.append("")

different_hemispheres_list=["1","2","3"]

labels=labels+hemispheres_list[0:3]
labels.append("")
labels=labels+different_hemispheres_list[0:5]
labels.append("")


ind = np.arange(len(presence))  # the x locations for the groups
width = 0.60  # the width of the bars


fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(ind , presence, width,
                label='Presence (0-1)',alpha=0.80)

rects3 = ax.bar(ind, presence_number, width,
                label='Different Elements Presence (0-1)',alpha=0.80)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('$Presence(gene_{value})$ (0-1)')
#ax.set_title('Electrode')
ax.set_xticks(ind)
ax.set_xticklabels(labels,fontsize='x-small',
        rotation=70)
ax.legend(bbox_to_anchor=(0.50, 0.95))
ax.set_title('Spatio-Temporal study for Patient '+ ID)
plt.grid(color='k', alpha=0.10, linestyle='-', linewidth=1)

fig.tight_layout()

bars=np.arange(len(presence))
heights=presence

barplot_annotate_brackets(0, 4, "Window \n Length", bars, heights, dh=0.30)

barplot_annotate_brackets(6, 10, "#Window \n Lengths", bars, heights, dh=0.65)

barplot_annotate_brackets(12, 18, "Instant", bars, heights, dh=0.20)

barplot_annotate_brackets(20, 24, "#Instants", bars, heights, dh=0.75)
                          
barplot_annotate_brackets(26, 44, "Electrode", bars, heights, dh=0.15)

barplot_annotate_brackets(46, 50, "#Electrodes", bars, heights, dh=0.52)
                          
barplot_annotate_brackets(52, 56, "Lobe", bars, heights, dh=0.03)

barplot_annotate_brackets(58, 62, "#Lobes", bars, heights, dh=0.70)
                          
barplot_annotate_brackets(64, 66, "Hemisphere", bars, heights, dh=0.01)

barplot_annotate_brackets(68, 70, "#Hemispheres", bars, heights, dh=0.95)

#plt.savefig("temporal_study.pdf", dpi=None, facecolor='w', edgecolor='w',
        #orientation='portrait', papertype=None, format=None,
        #transparent=False, bbox_inches=None, pad_inches=0.1,
        #frameon=None, metadata=None)
plt.show()


#%% ################ Features ##################

feature_names = ["Mean","Variance","Skewness","Kurtosis","Hjorth activity","Hjorth mobility","Hjorth complexity",
                 "Delta rel. pwr.","Theta rel. pwr.","Beta rel. pwr.","Alpha rel. pwr.","Low gamma rel. pwr.","High gamma rel. pwr.",
                 "SEF50","SEF75","SEF90","A7 energy","D7 energy","D6 energy","D5 energy","D4 energy","D3 energy","D2 energy","D1 energy"]

feat_labels = ["mean", "var", "skew", "kurt", "h_act", "h_mob", "h_com", 
               "delta", "theta", "beta", "alpha", "lowgamma", "highgamma",
               "sef50", "sef75", "sef90", "a7", "d7", "d6", "d5", "d4", "d3", "d2", "d1"]

feature_types_list = ["Time domain", "Frequency domain"]

mathematical_operators = ["Mean", "Variance", "Integral"]

features_group_list=["Stat. Moments","Hjorth Param","Rel. Spectral Power","Spectral Edge Freq.","Wavelet Energy"]

feature_group_count=np.zeros(5)
feature_count = np.zeros(len(feat_labels))
unique_features = np.zeros(5)
operators_count = np.zeros(len(mathematical_operators))
unique_operators = np.zeros(5)
unique_feature_types = np.zeros(2)
unique_feature_group=np.zeros(5)
for ind in selected_individuals:
    feats = []
    feats_idx = []
    opers = []
    feats_groups=[]
    for f in ind.features:
        feats.append(f.checkActiveGenes())
        feats_idx.append(feat_labels.index(f.checkActiveGenes()))
        opers.append(f.decodeMathematicalOperator())
        
        feature_count[feat_labels.index(f.checkActiveGenes())] += 1
        operators_count[mathematical_operators.index(f.decodeMathematicalOperator().capitalize())] += 1
    
        unique_features[len(np.unique(feats))-1] += 1
        unique_operators[len(np.unique(opers))-1] += 1
    
        if feat_labels.index(f.checkActiveGenes())>=0 and feat_labels.index(f.checkActiveGenes())<4:
            feats_groups.append(0)
        elif feat_labels.index(f.checkActiveGenes())>=4 and feat_labels.index(f.checkActiveGenes())<7:
            feats_groups.append(1)
        elif feat_labels.index(f.checkActiveGenes())>=7 and feat_labels.index(f.checkActiveGenes())<13:
            feats_groups.append(2)
        elif feat_labels.index(f.checkActiveGenes())>=13 and feat_labels.index(f.checkActiveGenes())<16:
            feats_groups.append(3)
        elif feat_labels.index(f.checkActiveGenes())>=16 and feat_labels.index(f.checkActiveGenes())<23:
            feats_groups.append(4)
    
        if any(x < 7 for x in feats_idx) and any(x >= 7 for x in feats_idx):
            unique_feature_types[1] += 1
        else:
            unique_feature_types[0] += 1
        
    unique_feature_group[len(np.unique(feats_groups))-1] += 1

feature_types = [np.sum(feature_count[:7]), np.sum(feature_count[7:])]
feature_group_count=[np.sum(feature_count[0:4]), np.sum(feature_count[4:7]),
                     np.sum(feature_count[7:13]),np.sum(feature_count[13:16]),
                     np.sum(feature_count[16:23])]

presence=(feature_count[0:24]/np.sum(feature_count)).tolist()
presence.append(0)
presence=presence+np.zeros(5).tolist()
presence.append(0)

presence_number=np.zeros(24).tolist()
presence_number.append(0)
presence_number=presence_number+(unique_features[0:5]/np.sum(unique_features)).tolist()
presence_number.append(0)

presence=presence+(feature_group_count[0:5]/np.sum(feature_group_count)).tolist()
presence.append(0)
presence=presence+np.zeros(5).tolist()
presence.append(0)

presence_number=presence_number+np.zeros(5).tolist()
presence_number.append(0)
presence_number=presence_number+(unique_feature_group[0:5]/np.sum(unique_feature_group)).tolist()
presence_number.append(0)

presence=presence+(feature_types[0:2]/np.sum(feature_types)).tolist()
presence.append(0)
presence=presence+np.zeros(2).tolist()
presence.append(0)

presence_number=presence_number+np.zeros(2).tolist()
presence_number.append(0)
presence_number=presence_number+(unique_feature_types[0:5]/np.sum(unique_feature_types)).tolist()
presence_number.append(0)

presence=presence+(operators_count[0:3]/np.sum(operators_count)).tolist()
presence.append(0)
presence=presence+np.zeros(3).tolist()
presence.append(0)

presence_number=presence_number+np.zeros(3).tolist()
presence_number.append(0)
presence_number=presence_number+(unique_operators[0:3]/np.sum(unique_operators)).tolist()
presence_number.append(0)


different_features_list=["1","2","3","4","5"]
different_features_types_list=["1","2"]

labels=feat_labels[0:24]
labels.append("")
labels=labels+different_features_list[0:5]
labels.append("")

labels=labels+features_group_list[0:5]
labels.append("")
labels=labels+different_features_list[0:5]
labels.append("")

labels=labels+feature_types_list[0:2]
labels.append("")
labels=labels+different_features_types_list[0:2]
labels.append("")

labels=labels+mathematical_operators[0:3]
labels.append("")
labels=labels+different_features_list[0:3]
labels.append("")

ind = np.arange(len(presence))  # the x locations for the groups
width = 0.60  # the width of the bars


fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(ind , presence, width,
                label='Presence (0-1)',alpha=0.80)

rects3 = ax.bar(ind, presence_number, width,
                label='Different Elements Presence (0-1)',alpha=0.80)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('$Presence(gene_{value})$ (0-1)')
#ax.set_title('Electrode')
ax.set_xticks(ind)
ax.set_xticklabels(labels,fontsize='x-small',
        rotation=90)
ax.legend(bbox_to_anchor=(0.45, 0.75),loc='best')
ax.set_title('Characteristics study for Patient '+ ID)
plt.grid(color='k', alpha=0.10, linestyle='-', linewidth=1)

fig.tight_layout()

bars=np.arange(len(presence))
heights=presence

barplot_annotate_brackets(0, 23, "Feature", bars, heights, dh=0.15)

barplot_annotate_brackets(25, 29, "#Features", bars, heights, dh=0.35)
                          
barplot_annotate_brackets(31, 35, "Group", bars, heights, dh=0.10)

barplot_annotate_brackets(37, 41, "#Groups", bars, heights, dh=0.63)

barplot_annotate_brackets(43, 44, "Domain", bars, heights, dh=0.10) 

barplot_annotate_brackets(46, 47, "    #Domains", bars, heights, dh=0.85)          
                          
barplot_annotate_brackets(49, 51, "Math. \n Operator", bars, heights, dh=0.10) 

barplot_annotate_brackets(53, 55, "#Math. \n Operators", bars, heights, dh=0.72) 
                          
#plt.savefig("temporal_study.pdf", dpi=None, facecolor='w', edgecolor='w',
        #orientation='portrait', papertype=None, format=None,
        #transparent=False, bbox_inches=None, pad_inches=0.1,
        #frameon=None, metadata=None)
plt.show()
