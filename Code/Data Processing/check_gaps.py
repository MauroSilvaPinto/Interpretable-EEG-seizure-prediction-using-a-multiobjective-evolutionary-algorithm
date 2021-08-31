"""
a code to check gaps in the patients eeg recordings (original data)
as this concerns a patient criteria selection: the existence of gaps
over an hour duration mean that the correspondent patient should be discarded

this code can not be executed
as the original data from Epilepsiae can not be available online for public use
due to ethical concers
"""

import numpy as np
#import matplotlib.pyplot as plt
import datetime as dt
import os
from scipy import signal

#%% Path setup and patient selection

path = "D:\\O nosso paper\\Data"
sep = os.path.sep
if path[-1] != sep:
    path+=sep

patient_selection = input('Enter patient ID: ')
patient_IDs = patient_selection.split(sep = ',') # allow user to enter multiple IDs separated by commas
patient_fs = int(input('Enter original sampling frequency (Hz): ')) # used for downsampling (if higher than 256Hz)
    
#%% Hyperparameters (e.g. seizure time, sliding window, filtering, ...)
    
# BUILDING SEIZURE DATA:
h_before_onset = dt.timedelta(hours = 4) # how many hours before onset?
h_between_onsets = dt.timedelta(hours = 4.5) # how many hours between seizures (cluster assumption)?
m_postictal = dt.timedelta(minutes = 30) # how many minutes of post-itcal (avoid influence in inter-ictal)?

# SLIDING WINDOW:
fsampling = 256 # sampling frequency (Hz)
    
#%% List all EVTS and patients

evts_list = sorted(os.listdir(path + 'EVTS' + sep))
evts_list = [s for s in evts_list if 'dataEvts' in s] # only files with "dataEvts"
evts_list = [path + 'EVTS' + sep + s for s in evts_list]

patient_list = sorted(os.listdir(path))
patient_list = [s for s in patient_list if 'pat' in s] # only folders with "pat"
patient_list = [path + s + sep for s in patient_list]

#%% Downsample signals to 256Hz (if necessary), generate time vectors and find gaps

# For each signal/header combo, check if fs = 256Hz and downsample if necessary; next, build a datetime vector;
# afterwards, find gap duration in seconds (compared to the previous time vector)
for ID in patient_IDs:
    for pat in patient_list:
        if "pat_" + ID in pat:
            print(f'Generating time vectors and finding gaps for patient {ID}...')
            
            signal_list = sorted(os.listdir(pat))
            signal_list = [s for s in signal_list if 'signalData' in s] # only files with "signalData"
            signal_list = [pat + s for s in signal_list]
            
            header_list = sorted(os.listdir(pat))
            header_list = [s for s in header_list if 'dataHead' in s] # only files with "dataHead"
            header_list = [pat + s for s in header_list]
            
            if len(signal_list) == len(header_list):
                print(f'0/{len(signal_list)}')
                for i in range (len(signal_list)):
                    header = np.load(header_list[i], allow_pickle = True)
                    sig = np.load(signal_list[i], allow_pickle = True)
                    
                    # DOWNSAMPLING
                    if patient_fs != 256:
                        if patient_fs == 400 or patient_fs == 2500: # non-integer downsampling factor
                            n_samples = int(len(sig)/(patient_fs/256))
                            sig = signal.resample(sig, n_samples, axis = 0).astype("float32") # based on FFT
                        else:
                            factor = int(patient_fs/256) # patient_fs must be a multiple of 256!
                            sig = signal.decimate(sig, factor, axis = 0, zero_phase = True).astype("float32") # equivalent to pop_resample()
                            
                        np.save(signal_list[i], sig) # overwrite file with downsampled version
                        
                    recording_start = header.item(0) # datetime (start)
                    duration = sig.shape[0] / fsampling # duration in seconds
                    signal_time = np.arange(0, duration, 1/fsampling) # time vector (seconds) from 0 until duration
                    
                    time = np.array([recording_start + dt.timedelta(seconds=signal_time[i]) for i in range(len(signal_time))])
                    np.save(header_list[i].split("dataHead.npy")[0] + "timeVector",time)
                    print(f'{i+1}/{len(signal_list)}')
                    
            else:
                print(ID + ' missing header or signal!')
        
            # find gap duration after time vectors have been generated for the recordings
            time_list = sorted(os.listdir(pat))
            time_list = [s for s in time_list if 'timeVector' in s] # only files with "timeVector"
            time_list = [pat + s for s in time_list]
            
            np.save(time_list[0].split("timeVector.npy")[0] + "gapSeconds", 0) # first signal does not have a gap
            
            for i in range (1, len(time_list)):
                time_current = np.load(time_list[i], allow_pickle = True)
                time_previous = np.load(time_list[i-1], allow_pickle = True)
                
                gap_duration = time_current[0] - time_previous[-1] # gap = recording_start - end of previous signal
                gap_duration = gap_duration - dt.timedelta(seconds = 1/fsampling) # compensate for time vector correction (not a real gap)
                gap_duration = gap_duration.total_seconds()
                
                if gap_duration != 0:
                    print(f"Gap of {gap_duration} seconds found in " + time_list[i])
                np.save(time_list[i].split("timeVector.npy")[0] + "gapSeconds", gap_duration)
          
#%% Check, for each seizure, if a gap of over 1 hour is found within the data
                        
# First 3 seizures, for training: 4h before each seizure + ictal period; 
# Remaining seizures, for testing: 30 mins after previous offset until onset + ictal period
# If an overly large gap is found, terminate and display a warning

for ID in patient_IDs:
    for EVTS in evts_list:
        if sep + ID in EVTS:
            print(f'Checking seizure data for patient {ID}...')
            all_onsets = np.load(EVTS, allow_pickle = True)[:,1]
            all_offsets = np.load(EVTS, allow_pickle = True)[:,7]
            exogenous = np.load(EVTS, allow_pickle = True)[:, 11:] # pattern, classification, vigilance, medicament, dosage
            
            # find any onsets / offsets that are invalid (offset before onset, rare...)
            annotation_errors = []
            for i in range(len(all_onsets)):
                if all_onsets[i]>all_offsets[i]:
                    annotation_errors.append(i)
            
            # discard seizures that are too close together  
            clusters = []
            for i in range(1,len(all_onsets)):
                if all_onsets[i] - all_offsets[i-1] < h_between_onsets:
                    clusters.append(i)
                    
            # check if the first seizure has enough data before the onset; otherwise, discard it
            not_enough_data = []
            for pat in patient_list:
                if "pat_" + ID in pat:
                    time_list = sorted(os.listdir(pat))
                    time_list = [s for s in time_list if 'timeVector' in s] # only files with "timeVector"
                    time_list = [pat + s for s in time_list]
                    
                    rec_start = np.load(time_list[0], allow_pickle=True)[0]
                    
                    if (all_onsets[0] - rec_start) < h_before_onset:
                        not_enough_data.append(0)
            
            discard = np.unique(annotation_errors + clusters + not_enough_data)
            print(f'Discarding seizures: {discard}')
            
            if discard.size > 0:
                onsets = np.delete(all_onsets, discard)
                offsets = np.delete(all_offsets, discard)
                exogenous = np.delete(exogenous, discard, 0)
            else:
                onsets = all_onsets
                offsets = all_offsets
                exogenous = exogenous
    
    print(f'Number of remaining seizures: {len(onsets)}') # if less than 4 remain, discard the patient as well...
    
    for pat in patient_list:
        
        found_seizures = 0
        
        if "pat_" + ID in pat:
            time_list = sorted(os.listdir(pat))
            time_list = [s for s in time_list if 'timeVector' in s] # only files with "timeVector"
            time_list = [pat + s for s in time_list]
            
            signal_list = sorted(os.listdir(pat))
            signal_list = [s for s in signal_list if 'signalData' in s] # only files with "signalData"
            signal_list = [pat + s for s in signal_list]
            
            gap_list = sorted(os.listdir(pat))
            gap_list = [s for s in gap_list if 'gapSeconds' in s] # only files with "gapSeconds"
            gap_list = [pat + s for s in gap_list]
            
            # reset these for each recording (optimize search)
            t_start = 0
            t_end = 0
            
            if found_seizures > 0: # avoid looking for seizures already found (optimize search)
                onsets = onsets[found_seizures:]
                offsets = offsets[found_seizures:]
            
            for o in range(len(onsets)):
                print(f"Analyzing data for seizure #{o+1}...")
                gap_too_large = False # !!! boolean used to prevent a MemoryError when gaps > 1h
                
                # find beginning of the signal (different for training and testing seizures, read above)
                if found_seizures < 3:
                    # find first signal that is X hours before the onset
                    searching_start = True
                    while searching_start and t_start <= len(time_list):
                        t_vector = np.load(time_list[t_start], allow_pickle=True)
                        gap = np.load(gap_list[t_start]).item(0) # check in case onset - X is in missing data segment
                        
                        if t_vector[0] - dt.timedelta(seconds = gap) <= onsets[o] - h_before_onset and t_vector[-1] > onsets[o] - h_before_onset:
                            
                            print(f"Found it! t_start = {t_start}")
                            
                            searching_start = False
                            
                            t_end = t_start # start looking for offset after onset (optimize search)
                            
                            # Check if the starting point requires generating >1h of data
                            if gap > 0 and t_vector[0] - (onsets[o] - h_before_onset) > dt.timedelta(seconds = 3600):
                                gap_too_large = True
                                break
                        
                        else:
                            t_start+=1
                        
                else:
                    # find first signal that is 30 mins after the previous offset (including discarded ones)
                    original_idx = np.where(all_onsets == onsets[o])[0][0] 
                    
                    if original_idx - 1 in annotation_errors:
                        after_last_offset = all_onsets[original_idx - 1] + m_postictal # use onset instead (rare, but it happens)
                    else:
                        after_last_offset = all_offsets[original_idx - 1] + m_postictal
                    
                    
                    searching_start = True
                    while searching_start and t_start <= len(time_list):
                        t_vector = np.load(time_list[t_start], allow_pickle=True)
                        gap = np.load(gap_list[t_start]).item(0) # check in case onset - X is in missing data segment
                        
                        if t_vector[0] - dt.timedelta(seconds = gap) <= after_last_offset and t_vector[-1] > after_last_offset:
                            
                            print(f"Found it! t_start = {t_start}")
                            
                            searching_start = False
                            
                            t_end = t_start # start looking for offset after onset (optimize search)
                            
                            # Check if the starting point requires generating >1h of data
                            if gap > 0 and t_vector[0] - after_last_offset > dt.timedelta(seconds = 3600):
                                gap_too_large = True
                                break
                        
                        else:
                            t_start+=1
                    
                # find first signal that contains the offset
                searching_end = True
                
                if t_start == len(time_list):
                    searching_end = False # start searching in a different recording (optimize search)
                    
                while searching_end and t_end <= len(time_list):
                    t_vector = np.load(time_list[t_end], allow_pickle=True)
                    
                    if t_vector[0] <= offsets[o] and t_vector[-1] > offsets[o]:
                        print(f"Found it! t_end = {t_end}")
                        
                        searching_end = False
                    
                    else:
                        t_end+=1
                
                if t_start != len(time_list): # find remaining signals between the previous segments and concatenate all of them; check for gaps!
                    
                    for t in range(t_start+1,t_end+1):
                        print(f"Checking! t = {t}")
                        
                        if t==t_start+1:
                            t_vector = np.load(time_list[t], allow_pickle=True)
                            #signal_array = np.load(signal_list[t], allow_pickle = True)[:,electrodes_rows].astype("float32")
                            gap = np.load(gap_list[t])
                            
                            if gap > 0:
                                if gap > 3600:
                                    gap_too_large = True
                                    break
                        
                        elif t==t_end:
                            gap = np.load(gap_list[t])
                            
                            if gap > 0:
                                if gap > 3600:
                                    gap_too_large = True
                                    break
                                
                        else:
                            t_vector = np.load(time_list[t], allow_pickle=True)
                            #signal_array = np.load(signal_list[t], allow_pickle = True)[:,electrodes_rows].astype("float32")
                            gap = np.load(gap_list[t])
                            
                            if gap > 0:
                                if gap > 3600:
                                    gap_too_large = True
                                    break
                                
                    if gap_too_large == True:
                        print("Signal gap is over 1 hour long! Patient should be discarded...")
                        break
                    else:
                        found_seizures+=1

print("\a") # beep when done :)