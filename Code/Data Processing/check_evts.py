"""
a code to check evts, which are headlines for the binary data
concerning the patients eeg recordings (original data)
as this concerns a patient criteria selection

with this code, we can see how many patients have the minimum of
available data to work it: minimum of 4 seizures, where each seizure
has at least 4h30 of temporal distance between other seizures

this code can not be executed
as the original data from Epilepsiae can not be available online for public use
due to ethical concers
"""

import numpy as np
import os
import datetime as dt

#%% Path setup and interseizure interval criteria

path = "D:\\Research\\Data"
sep = os.path.sep
if path[-1] != sep:
    path+=sep
    
h_before_onset = dt.timedelta(hours = 4) # how many hours before onset?
h_between_onsets = dt.timedelta(hours = 4.5) # how many hours between seizures (cluster assumption)?
#%% Retrieve all the original EVTS files and convert them to Numpy arrays

original_evts_list = sorted(os.listdir(path + 'GeneratedEVTS' + sep))
original_evts_list = [s for s in original_evts_list if '.evts' in s] # only .evts files
original_evts_list = [path + 'GeneratedEVTS' + sep + s for s in original_evts_list]

for evts in original_evts_list:
    file = open(evts, 'r')
    file_lines = file.readlines()[1:]
    
    evts_array = []
    for line in file_lines:
        info_vector = line.split("\t")
        
        try:
            info_vector[1] = dt.datetime.strptime(info_vector[1], '%Y-%m-%d %H:%M:%S.%f')
        except:
            pass
        try:
            info_vector[3] = dt.datetime.strptime(info_vector[3], '%Y-%m-%d %H:%M:%S.%f')
        except:
            pass
        try:
            info_vector[7] = dt.datetime.strptime(info_vector[7], '%Y-%m-%d %H:%M:%S.%f')
        except:
            pass
        try:
            info_vector[9] = dt.datetime.strptime(info_vector[9], '%Y-%m-%d %H:%M:%S.%f')
        except:
            pass
        
        evts_array.append(info_vector)
    
    np.save(path + 'EVTS' + sep + evts.split(sep)[-1].split(".")[0] + "_dataEvts.npy", np.array(evts_array))


#%% Check how many seizures meet the interseizure interval criteria

evts_list = sorted(os.listdir(path + 'EVTS' + sep))
evts_list = [s for s in evts_list if 'dataEvts' in s] # only files with "dataEvts"
evts_list = [path + 'EVTS' + sep + s for s in evts_list]

number_of_seizures = []

for EVTS in evts_list:
    all_onsets = np.load(EVTS, allow_pickle = True)[:,1]
    all_offsets = np.load(EVTS, allow_pickle = True)[:,7]
    exogenous = np.load(EVTS, allow_pickle = True)[:, 11:] # pattern, classification, vigilance, medicament, dosage
    
    # find any onsets / offsets that are invalid (offset before onset, rare...)
    annotation_errors = []
    for i in range(len(all_onsets)):
        if all_onsets[i] > all_offsets[i]:
            annotation_errors.append(i)
            
    # discard seizures that are too close together  
    clusters = []
    for i in range(1,len(all_onsets)):
        if all_onsets[i] - all_offsets[i-1] < h_between_onsets:
            clusters.append(i)

# Can't check if the first seizure has enough data without checking the file header... 
# leave it for pre-processing         

#    # check if the first seizure has enough data before the onset; otherwise, discard it
#    not_enough_data = []
#    for pat in patient_list:
#        if "pat_" + ID in pat:
#            time_list = sorted(os.listdir(pat))
#            time_list = [s for s in time_list if 'timeVector' in s] # only files with "timeVector"
#            time_list = [pat + s for s in time_list]
#            
#            rec_start = np.load(time_list[0], allow_pickle=True)[0]
#            
#            if (all_onsets[0] - rec_start) < h_before_onset:
#                not_enough_data.append(0)
    
    discard = np.unique(annotation_errors + clusters)
    onsets = np.delete(all_onsets, discard)
    offsets = np.delete(all_offsets, discard)
    exogenous = np.delete(exogenous, discard, 0)
    
    number_of_seizures.append([int(EVTS.split("/")[-1].split("_")[0]), int(len(onsets))])
    print(f'{EVTS.split("/")[-1]} contains {len(onsets)}/{len(onsets)+len(discard)} seizures which meet the requirements')

number_of_seizures = np.array(number_of_seizures)

#%% Select patients with at least 4 seizures which meet the previous criteria

# previously selected patients according to sampling frequency (>= 256Hz)
selected_patients_fs = [102,202,402,7302,8902,11002,11502,12702,16202,19202,21602
,21902,22602,23902,26102,30002,30802,32202,32502,32702,45402,46702,50802,51002,52302
,53402,55202,56402,58602,59102,60002,64702,75102,75202,79502,80602,80702,81102,81402
,85202,92102,93402,93902,94402,95202,96002,98102,98202,100002,100302,101702,102202
,103002,103802,104602,109202,109502,110602,111902,112402,112802,113902,114702,114902
,115102,115202,123902,125002,1233703,1233803,1234303,1235003,1235103,1318803,1319103,1319203
,1320303,1320503,1320603,1320903,1321003,1321103,1322703,1322803,1324803,1324903,1325003,1325103
,1325403,1325603,1325903,1326003,1326103,1326503,1327403,1327903,1328403,1328603,1328803,1328903
,1329303,1329503,1330103,1330203,1330903,500,600,700,800,1300,1400,1600,2100,2200,2300,2600,2700
,3700,4900,5100,5200,5500,6000,1299403,1300003,1305803,1306003,1306203,1306903,1307003,1307103
,1307403,1307503,1307803,1308403,1308503,1308603,1309803,1310803,1311003,1312603,1312703,1312803
,1312903,1313003,1313403,1313903,1314103,1314703,1314803,1315003,1315203,1315403,1316003,1316303
,1316403,1316503,1317003,1317203,1317303,1317403,1317903,1321803,1321903,1323803,1324103,200,1200
,1500,1700,2000,2900,3300,3500,3600,4000,4200,4400,4500,4600,4700,5800,5900,6100,6200,6300,6400
,6500,6600,6700,6800,7000,7200,7300,7500,7700,7800,8100,63502,109602,6900,70102,70302,70902,71102
,71502,71802,73002]
selected_patients = []
for pat in number_of_seizures:
    if pat[1] >= 4 and pat[0] in selected_patients_fs:
        selected_patients.append(pat)
        
selected_patients = np.array(selected_patients)



    
