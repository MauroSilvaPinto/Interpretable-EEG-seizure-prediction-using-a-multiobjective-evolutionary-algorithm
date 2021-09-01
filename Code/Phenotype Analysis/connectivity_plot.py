"""
a code to make a connectivity plot,
by plotting the electrode genes interaction

this image is interactive:
you click on each gene to isolate its interactions

"""

import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import matplotlib

from matplotlib import cm
from apyori import apriori
from nilearn import plotting

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



electrodes_list_conversion=["FP1","FP2","Fz","F3","F4",
                            "F7","F8","Cz","C3","C4",
                            "T3","T4","Pz","P3","P4",
                            "T5","T6","O1","O2"]

#create connections
con=np.zeros([len(electrodes_list_conversion),len(electrodes_list_conversion)])

number_of_connections=np.zeros(len(electrodes_list_conversion))
# getting the nodes and connections
for i in range (0,len(relevant_associations)):
    item=relevant_associations[i]    

    pair = item[0] 
    items = [x for x in pair]
    
    combinations=list(it.combinations(items,2))
    for combination in combinations:
        if (combination[0] in electrodes_list_conversion) and (combination[1] in electrodes_list_conversion):
            a_index=electrodes_list_conversion.index(combination[0])
            b_index=electrodes_list_conversion.index(combination[1])
            connectivity_value=(int(item[2][0][3])*500)
        
            con[a_index,b_index]=con[a_index,b_index]+connectivity_value
            con[b_index,a_index]=con[b_index,a_index]+connectivity_value
            
            number_of_connections[a_index]=number_of_connections[a_index]+1
            number_of_connections[b_index]=number_of_connections[b_index]+1
        
        
# coordinate conversion, from EEG 10-20 system electrodes to NMIST coordinates
# From: Okamoto et al., 2004, NeuroImage					
# https://www.sciencedirect.com/science/article/pii/S1053811903005366?via%3Dihub					
electrodes_mnist_coordinates=np.array([
    [-21.5, 70.2, -0.1],
    [28.4, 69.1, -0.4],
    [0.6, 40.9, 53.9],
    [-35.5, 49.4, 32.4],
    [40.2, 47.6, 32.1],
    [-54.8, 33.9, -3.5],
    [56.6, 30.8, -4.1],
    [0.8, -14.7, 73.9],
    [-52.2, -16.4, 57.8],
    [54.1, -18.0, 57.5],
    [-70.2, -21.3, -10.7],
    [71.9, -25.2, -8.2],
    [0.2, -62.1, 64.5],
    [-39.5, -76.3, 47.4],
    [36.8, -74.9, 49.2],
    [-61.5, -65.3, 1.1],
    [59.3, -67.6, 3.8],
    [-26.8, -100.2, 12.8],
    [24.1, -100.5, 14.1],
    ])

node_colors=[]
for electrode in electrodes_list_conversion:
    if "C"==electrode[0]:
        node_colors.append("orange")
    elif "F"==electrode[0]:
        node_colors.append("blue")
    elif "O"==electrode[0]:
        node_colors.append("black")
    elif "P"==electrode[0]:
        node_colors.append("green")
    elif "T"==electrode[0]:
        node_colors.append("red")

test_adj=np.random.RandomState(0).randn(len(electrodes_list_conversion), len(electrodes_list_conversion))


test_colors = ['black']*19
plotting.plot_connectome(con, electrodes_mnist_coordinates,colorbar=False, edge_cmap=cm.Reds,
                         display_mode="ortho",node_size=number_of_connections/2+3,
                         node_color=node_colors)

