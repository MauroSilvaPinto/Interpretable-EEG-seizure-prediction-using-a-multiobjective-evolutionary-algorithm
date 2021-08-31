"""
a code to make a chrd plot,
by plotting the all genes interaction

"""

import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import matplotlib
import nxviz as nv

from chord import Chord
from apyori import apriori


#%% loading the features, as lists of lists
# of the selected patient for interpretability study
transactions= np.load("apriori_transactions_patient_1321803.npy",allow_pickle=True)
transactions=transactions.tolist()

#%% passing from window_length_time to: win_time
for i in range (0,len(transactions)):
    for j in range(0,len(transactions[i])):
        if "window_length" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("window_length_","")
        if "preictal" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("preictal_","")
        if "mean" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("mean","Intensity Mean")
        if "var" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("var","Intensity Variance")
        if "skew" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("skew","Skewness")
        if "kurt" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("kurt","Kurtosis")
        if "h_act" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("h_act","Hjorth Activity")
        if "h_mob" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("h_mob","Hjorth Mobility")
        if "h_com" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("h_com","Hjorth Complexity")
        if "delta" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("delta","Band Power Delta")
        if "theta" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("theta","Band Power Theta")
        if "beta" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("beta","Band Power Beta")
        if "alpha" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("alpha","Band Power Alpha")
        if "lowgamma" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("lowgamma","Band Power Low Gamma")
        if "highgamma" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("highgamma","Band Power High Gamma")   
        if "sef50" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("sef50","SEF 50")
        if "sef75" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("sef75","SEF 75")
        if "sef90" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("sef90","SEF 90")
        if "a7" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("a7","Wavelet A7")
        if "d7" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("d7","Wavelet D7")
        if "d6" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("d6","Wavelet D6")
        if "d5" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("d5","Wavelet D5")
        if "d4" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("d4","Wavelet D4")
        if "d3" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("d3","Wavelet D3")
        if "d2" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("d2","Wavelet D2")
        if "d1" in transactions[i][j]:
            transactions[i][j]=transactions[i][j].replace("d1","Wavelet D1")
        
#%% making the apriori algorithm
association_rules = apriori(transactions, min_support=0.07, min_confidence=0.1, min_lift=1, max_length=15)
association_results = list(association_rules)   

#%% associations with more than one item, since i already have individual gene presence
relevant_associations=[]
for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    
    # only interested in len>1 because i already have individual gene presence
    if len(items)>1:
        relevant_associations.append(item)
        
#%% sort associations by decreasing lift

# A function that returns the length of the value:
def sortByLift(item):
  return item[2][0][3]

relevant_associations.sort(reverse=True, key=sortByLift)        
        

#%% print the 20 most relevant associations by reverse (the last printed element is the highest one)

for i in range (20,0,-1):

    item=relevant_associations[i]
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
        
    items_string="Rule: "
    for item_i in items:
        items_string=items_string+str(item_i)+", "
    print(items_string)
    
    #second index of the inner list
    print("Support: " + str(item[1]))
    
    #third index of the list located at 0th
    #of the third index of the inner list
    
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

#%%
  
import numpy as np
from numpy.testing import assert_raises
 
from mne.viz import plot_connectivity_circle, circular_layout
 

feat_labels = ["Intensity Mean", "Intensity Variance", "Skewness", "Kurtosis", "Hjorth Activity", 
               "Hjorth Mobility", "Hjorth Complexity", 
                "Band Power Delta", "Band Power Theta", "Band Power Beta", "Band Power Alpha", 
                "Band Power Low Gamma", "Band Power High Gamma",
                "SEF 50", "SEF 75", "SEF 90", "Wavelet A7", "Wavelet D7", "Wavelet D6", "Wavelet D5",
                "Wavelet D4", "Wavelet D3", "Wavelet D2", "Wavelet D1", "Features"]

electrodes_list=['C3','C4','Cz','F3','F4','F7', 'F8','FP1','FP2',
                  'Fz','O1','O2','P3','P4','Pz', 'T3','T4','T5','T6',
                  "Electrodes"]

window_lengths = ["1", "5", "10", "15", "20", "Window (minutes)"]

preictal_periods = ["30", "35", "40", "45",
                    "50", "55", "60", "65",
                    "70", "75", "80", "85", "90", "Pre-Ictal (minutes)"]

node_order=[feat_labels + electrodes_list + window_lengths + preictal_periods][0]    

import matplotlib.pyplot as plt

label_names = node_order
 
group_boundaries = [0, len(label_names) / 2]
group_boundaries=[0, 25, 45, 51]
node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)

#create connections
con=np.zeros([len(node_order),len(node_order)])
#con = np.random.RandomState(0).randn(len(node_order), len(node_order))

n_relevant_associations=len(relevant_associations)
# getting the nodes and connections
number_of_connections=np.zeros(len(node_order))
for i in range (n_relevant_associations-1,-1,-1):

    item=relevant_associations[i]    

    pair = item[0] 
    items = [x for x in pair]
    
    combinations=list(it.combinations(items,2))
    for combination in combinations:
        a_index=node_order.index(combination[0])
        b_index=node_order.index(combination[1])
        connectivity_value=(int(item[2][0][3])*500)
        
        con[a_index,b_index]=con[a_index,b_index]+connectivity_value
        con[b_index,a_index]=con[b_index,a_index]+connectivity_value
        
        number_of_connections[a_index]=number_of_connections[a_index]+1
        number_of_connections[b_index]=number_of_connections[b_index]+1


#%% count individual gene presence
count_individual_gene_presence=np.zeros(len(node_order))
for gene in node_order:
    for transaction in transactions:
        if gene in transaction:
            item_index=node_order.index(gene)
            count_individual_gene_presence[item_index]=count_individual_gene_presence[item_index]+1
        
       
cmap_reversed = matplotlib.cm.get_cmap('bone_r')

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

viridis = cm.get_cmap('Reds', np.max(count_individual_gene_presence))
colors=viridis(range(int(np.max(count_individual_gene_presence)+1)))

node_colors=np.zeros([len(node_order),4])
for i in range(0,len(node_order)):
    node_colors[i]=colors[int(count_individual_gene_presence[i])]
    
#%%  
    
fig=plot_connectivity_circle(con, label_names, n_lines=300,
                             node_angles=node_angles,
                             show=True, facecolor="white", textcolor="black",
                             colormap=cmap_reversed,
                             node_colors=node_colors,
                             colorbar_pos=(-0.5, 0.3),
                             colorbar=False,
                             subplot=(1,1,1))

cbar=fig[0].colorbar(cm.ScalarMappable(cmap=viridis),shrink=0.25,location="bottom",
                     pad=-0.15, anchor=(0.25,0.0))
cbar.ax.set_title('Gene Presence',size=8)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(8)
     

cbar=fig[0].colorbar(cm.ScalarMappable(cmap=cmap_reversed),shrink=0.25, location="bottom",
                     pad=-0.10, anchor=(0.75,0.0))
cbar.ax.set_title('Connection Strength', size=8)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(8)
 

#fig[0].savefig("chord_diagram.pdf", facecolor='white')

    
