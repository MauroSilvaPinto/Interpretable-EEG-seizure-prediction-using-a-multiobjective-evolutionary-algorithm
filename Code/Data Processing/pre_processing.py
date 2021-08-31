"""
a code to pre-process and extract fist-level features from raw data from the selected patients.
the output will be the extracted features, chronologically, in 5 second non-overlapping windows
order by patient and seizure.

format output name:pat[patient_number]_seizure[seizure_number]_featureMatrix.npy
example output name: pat102_seizure_1_featureMatrix.npy
    

this code can not be executed
as the original data from Epilepsiae can not be  available online for public use
due to ethical concers
"""

import numpy as np
#import matplotlib.pyplot as plt
import datetime as dt
import os
from scipy import signal, integrate
import pywt

#%% Path setup and patient selection

#path = "/Users/Tiago/Desktop/Research/Data"
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
window_size = fsampling * 5 # in number of samples
overlap = 0 # in number of samples

# FILTERING:
f_notch = 50 # power-line interference
Q = 30
b_notch, a_notch = signal.iirnotch(f_notch, Q, fsampling) # notch filter

f_HPF = 0.5 # remove DC component and breathing artifacts (slow drifts)
order = 4
b_HPF, a_HPF = signal.butter(order, f_HPF, 'highpass', fs = fsampling)

# FEATURE EXTRACTION:

# Features: statistical moments, spectral band power, SEF, wavelets, hjorth parameters (more?)
feature_labels = np.sort(['mean', 'var', 'skew', 'kurt', 'theta', 'delta', 'beta', 'alpha', 'lowgamma', 'highgamma',
                          'h_act', 'h_com', 'h_mob', 'sef50', 'sef75', 'sef90',
                          'a7', 'd7', 'd6', 'd5', 'd4', 'd3', 'd2', 'd1'])
number_of_features = len(feature_labels) # used later to detect number of seizures for each patient

theta_range = [0, 4]
delta_range = [4, 8]
beta_range = [8, 12]
alpha_range = [13, 30]
gamma_range = [30, 128]
low_gamma_range = [30, 79]
high_gamma_range = [79, 128]

mother_wavelet = pywt.Wavelet('db4')
    
#%% List all EVTS and patients

evts_list = sorted(os.listdir(path + 'EVTS' + sep))
evts_list = [s for s in evts_list if 'dataEvts' in s] # only files with "dataEvts"
evts_list = [path + 'EVTS' + sep + s for s in evts_list]

patient_list = sorted(os.listdir(path))
patient_list = [s for s in patient_list if 'pat' in s] # only folders with "pat"
patient_list = [path + s + sep for s in patient_list]

#%% Retrieve electrode labels / rows from data header

for ID in patient_IDs:
    for pat in patient_list:
        if "pat_" + ID in pat:
            print(f'Gathering time vectors and gaps for patient {ID}...')
            
            signal_list = sorted(os.listdir(pat))
            signal_list = [s for s in signal_list if 'signalData' in s] # only files with "signalData"
            signal_list = [pat + s for s in signal_list]
            
            header_list = sorted(os.listdir(pat))
            header_list = [s for s in header_list if 'dataHead' in s] # only files with "dataHead"
            header_list = [pat + s for s in header_list]         
            
            header = np.load(header_list[0], allow_pickle = True)

# Retrieve electrode labels and find which rows correspond to them
electrodes_label = np.array(['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                             'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'])

# !!! Some patients seem to have the header in a different index
if patient_fs == 400:
    header_label = np.array([x.lower() for x in header.item(6)])
else:
    header_label = np.array([x.lower() for x in header.item(6)]) #!!! mudar para 5 depois
electrodes_rows = []
electrodes_rows.append(np.where(header_label == 'fp1')[0][0])
electrodes_rows.append(np.where(header_label == 'fp2')[0][0])
electrodes_rows.append(np.where(header_label == 'f3')[0][0])
electrodes_rows.append(np.where(header_label == 'f4')[0][0])
electrodes_rows.append(np.where(header_label == 'c3')[0][0])
electrodes_rows.append(np.where(header_label == 'c4')[0][0])
electrodes_rows.append(np.where(header_label == 'p3')[0][0])
electrodes_rows.append(np.where(header_label == 'p4')[0][0])
electrodes_rows.append(np.where(header_label == 'o1')[0][0])
electrodes_rows.append(np.where(header_label == 'o2')[0][0])
electrodes_rows.append(np.where(header_label == 'f7')[0][0])
electrodes_rows.append(np.where(header_label == 'f8')[0][0])

try:
    electrodes_rows.append(np.where(header_label == 't3')[0][0])
except:
    electrodes_rows.append(np.where(header_label == 't7')[0][0])
try:
    electrodes_rows.append(np.where(header_label == 't4')[0][0])
except:
    electrodes_rows.append(np.where(header_label == 't8')[0][0])
try:
    electrodes_rows.append(np.where(header_label == 't5')[0][0])
except:
    electrodes_rows.append(np.where(header_label == 'p7')[0][0])
try:
    electrodes_rows.append(np.where(header_label == 't6')[0][0])
except:
    electrodes_rows.append(np.where(header_label == 'p8')[0][0])

electrodes_rows.append(np.where(header_label == 'fz')[0][0])
electrodes_rows.append(np.where(header_label == 'cz')[0][0])
electrodes_rows.append(np.where(header_label == 'pz')[0][0])
          
#%% Concatenate seizure data (before seizure + ictal)
                        
# First 3 seizures, for training: 4h before each seizure + ictal period; 
# Remaining seizures, for testing: 30 mins after previous offset until onset + ictal period
# Signals, time vectors are concatenated; labels (ictal/non-ictal) added; exogenous variables for each seizure added

for ID in patient_IDs:
    for EVTS in evts_list:
        if sep + ID in EVTS:
            print(f'Building seizure data for patient {ID}...')
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
                print(f"Gathering data for seizure #{o+1}...")
                
                # find beginning of the signal (different for training and testing seizures, read above)
                if found_seizures < 3:
                    # find first signal that is X hours before the onset
                    searching_start = True
                    while searching_start and t_start <= len(time_list):
                        t_vector = np.load(time_list[t_start], allow_pickle=True)
                        gap = np.load(gap_list[t_start]).item(0) # check in case onset - X is in missing data segment
                        
                        if t_vector[0] - dt.timedelta(seconds = gap) <= onsets[o] - h_before_onset and t_vector[-1] > onsets[o] - h_before_onset:
                            
                            if gap > 0 and onsets[o] - h_before_onset < t_vector[0]: 
                                gap_time = np.arange(1/fsampling, gap, 1/fsampling)
                                previous_t_vector = np.load(time_list[t_start-1], allow_pickle=True)
                
                                signal_array = np.load(signal_list[t_start], allow_pickle=True)[:,electrodes_rows].astype("float32")
                                previous_signal = np.load(signal_list[t_start-1], allow_pickle=True)[:,electrodes_rows].astype("float32")
                                signal_gap_input = (previous_signal[-1,:] + signal_array[0,:])/2
                                    
                                generated_time = np.array([previous_t_vector[-1] + dt.timedelta(seconds=gap_time[i]) for i in range(len(gap_time))])
                                generated_signal = np.ones((len(gap_time), 19), dtype="float32") * signal_gap_input
                                
                                
                                new_t_vector = np.concatenate((generated_time, t_vector))
                                new_signal_array = np.vstack((generated_signal, signal_array))
                                
                                signal_start_idx = (np.abs(new_t_vector - (onsets[o] - h_before_onset))).argmin() # closest time sample
                                
                                signal_start = new_signal_array[signal_start_idx:,:].astype("float32")
                            
                                time_start = new_t_vector[signal_start_idx:]
                            else:   
                                signal_start_idx = (np.abs(t_vector - (onsets[o] - h_before_onset))).argmin() # closest time sample
                                
                                signal_array = np.load(signal_list[t_start], allow_pickle=True)[:,electrodes_rows].astype("float32")
                                signal_start = signal_array[signal_start_idx:,:].astype("float32")
                            
                                time_start = t_vector[signal_start_idx:]
                            
                            print(f"Found it! t_start = {t_start}")
                            
                            searching_start = False
                            
                            t_end = t_start # start looking for offset after onset (optimize search)
                        
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
                            
                            if gap > 0 and after_last_offset < t_vector[0]: 
                                gap_time = np.arange(1/fsampling, gap, 1/fsampling) # !!! if a MemoryError occurs later, change this
                                previous_t_vector = np.load(time_list[t_start-1], allow_pickle=True)
                                
                                signal_array = np.load(signal_list[t_start], allow_pickle=True)[:,electrodes_rows].astype("float32")
                                previous_signal = np.load(signal_list[t_start-1], allow_pickle=True)[:,electrodes_rows].astype("float32")
                                signal_gap_input = (previous_signal[-1,:] + signal_array[0,:])/2
                                    
                                generated_time = np.array([previous_t_vector[-1] + dt.timedelta(seconds=gap_time[i]) for i in range(len(gap_time))])
                                generated_signal = np.ones((len(gap_time), 19), dtype="float32") * signal_gap_input
                                
                                
                                new_t_vector = np.concatenate((generated_time, t_vector))
                                new_signal_array = np.vstack((generated_signal, signal_array)).astype("float32")
                                
                                signal_start_idx = (np.abs(new_t_vector - (after_last_offset))).argmin() # closest time sample
                                
                                signal_start = new_signal_array[signal_start_idx:,:]
                            
                                time_start = new_t_vector[signal_start_idx:]
                            else:   
                                signal_start_idx = (np.abs(t_vector - (after_last_offset))).argmin() # closest time sample
                                
                                signal_array = np.load(signal_list[t_start], allow_pickle=True)[:,electrodes_rows].astype("float32")
                                signal_start = signal_array[signal_start_idx:,:].astype("float32")
                            
                                time_start = t_vector[signal_start_idx:]
                            
                            print(f"Found it! t_start = {t_start}")
                            
                            searching_start = False
                            
                            t_end = t_start # start looking for offset after onset (optimize search)
                        
                        else:
                            t_start+=1
                    
                # find first signal that contains the offset
                searching_end = True
                
                if t_start == len(time_list):
                    searching_end = False # start searching in a different recording (optimize search)
                    
                while searching_end and t_end <= len(time_list):
                    t_vector = np.load(time_list[t_end], allow_pickle=True)
                    
                    if t_vector[0] <= offsets[o] and t_vector[-1] > offsets[o]:
                        signal_end_idx = (np.abs(t_vector - offsets[o])).argmin() # closest time sample
                        
                        signal_array = np.load(signal_list[t_end], allow_pickle=True)[:,electrodes_rows].astype("float32")
                        signal_end = signal_array[:signal_end_idx,:].astype("float32")
                        
                        time_end = t_vector[:signal_end_idx]
                        
                        print(f"Found it! t_end = {t_end}")
                        
                        searching_end = False
                    
                    else:
                        t_end+=1
                
                if t_start != len(time_list): # find remaining signals between the previous segments and concatenate all of them; check for gaps!

                    if t_start == t_end: # may happen in large files that span several hours...
                        signal_segment = signal_array[signal_start_idx:signal_end_idx]
                        time_segment = t_vector[signal_start_idx:signal_end_idx]
                        
                    for t in range(t_start+1,t_end+1):
                        print(f"Concatenating! t = {t}")
                        
                        if t==t_start+1:
                            t_vector = np.load(time_list[t], allow_pickle=True)
                            signal_array = np.load(signal_list[t], allow_pickle = True)[:,electrodes_rows].astype("float32")
                            gap = np.load(gap_list[t])
                            
                            if gap > 0:
                                # generate vector with missing samples (time and signal)
                                gap_time = np.arange(1/fsampling, gap, 1/fsampling)
                                previous_t_vector = np.load(time_list[t-1], allow_pickle=True)
                                
                                #previous_signal = np.load(signal_list[t-1], allow_pickle=True)[:,0:19]
                                signal_gap_input = (signal_start[-1,:] + signal_array[0,:])/2
                                
                                generated_time = np.array([previous_t_vector[-1] + dt.timedelta(seconds=gap_time[i]) for i in range(len(gap_time))])
                                generated_signal = np.ones((len(gap_time), 19), dtype="float32") * signal_gap_input
                                
                                time_segment = np.concatenate((time_start, generated_time, t_vector))
                                signal_segment = np.vstack((signal_start, generated_signal, signal_array)).astype("float32")
                                
                            else:
                                time_segment = np.concatenate((time_start, t_vector))
                                signal_segment = np.vstack((signal_start, signal_array)).astype("float32")
                        
                        elif t==t_end:
                            gap = np.load(gap_list[t])
                            
                            if gap > 0:
                                # generate vector with missing samples (time and signal)
                                gap_time = np.arange(1/fsampling, gap, 1/fsampling)
                                previous_t_vector = np.load(time_list[t-1], allow_pickle=True)
                                
                                #previous_signal = np.load(signal_list[t-1], allow_pickle=True)[:,0:19]
                                signal_gap_input = (signal_segment[-1,:] + signal_end[0,:])/2
                                
                                generated_time = np.array([previous_t_vector[-1] + dt.timedelta(seconds=gap_time[i]) for i in range(len(gap_time))])
                                generated_signal = np.ones((len(gap_time), 19), dtype="float32") * signal_gap_input
                                
                                time_segment = np.concatenate((time_segment, generated_time, time_end))
                                signal_segment = np.vstack((signal_segment, generated_signal, signal_end)).astype("float32")
                                
                            else:
                                time_segment = np.concatenate((time_segment, time_end))
                                signal_segment = np.vstack((signal_segment, signal_end))[:,:].astype("float32")
                            
                        else:
                            t_vector = np.load(time_list[t], allow_pickle=True)
                            signal_array = np.load(signal_list[t], allow_pickle = True)[:,electrodes_rows].astype("float32")
                            gap = np.load(gap_list[t])
                            
                            if gap > 0:                               
                                # generate vector with missing samples (time and signal)
                                gap_time = np.arange(1/fsampling, gap, 1/fsampling)
                                previous_t_vector = np.load(time_list[t-1], allow_pickle=True)
                                
                                #previous_signal = np.load(signal_list[t-1], allow_pickle=True)[:,0:19]
                                signal_gap_input = (signal_segment[-1,:] + signal_array[0,:])/2
                                
                                generated_time = np.array([previous_t_vector[-1] + dt.timedelta(seconds=gap_time[i]) for i in range(len(gap_time))])
                                generated_signal = np.ones((len(gap_time), 19), dtype="float32") * signal_gap_input
                                
                                time_segment = np.concatenate((time_segment, generated_time, t_vector))
                                signal_segment = np.vstack((signal_segment, generated_signal, signal_array)).astype("float32")
                                
                            else:
                                time_segment = np.concatenate((time_segment, t_vector))
                                signal_segment = np.vstack((signal_segment, signal_array)).astype("float32")
                        
                        
                    # label time_segment and signal_segment: 0 = non-ictal; 2 = ictal
                    ictal_start_idx = (np.abs(time_segment - onsets[o])).argmin() # closest time sample
                    label_segment = np.zeros(time_segment.shape)
                    label_segment[ictal_start_idx:] = 2
                
                    # save each seizure data in "Seizures" folder as: patX_seizureY_signal, patX_seizureY_time, patX_seizureY_label
                    found_seizures+=1
                    print(f'Saving seizure #{o+1}...')
                    np.save(path + 'Seizures' + sep + 'pat' + ID + '_seizure'+ str(found_seizures) + '_timeVector', time_segment)
                    np.save(path + 'Seizures' + sep + 'pat' + ID + '_seizure' + str(found_seizures) + '_signalData', signal_segment)
                    np.save(path + 'Seizures' + sep + 'pat' + ID + '_seizure' + str(found_seizures) + '_labelVector', label_segment)
                    np.save(path + 'Seizures' + sep + 'pat' + ID + '_seizure' + str(found_seizures) + '_exogenousVariables', exogenous[o])

#%% Window segmentation (5 secs, no overlap)
       
# Segment seizure data in windows, create labels for windowed data and extract linear univariate features
seizure_list = sorted(os.listdir(path + 'Seizures' + sep))
seizure_list = [path + 'Seizures' + sep + s for s in seizure_list]

signal_list = [s for s in seizure_list if 'signalData' in s] # only files with "signalData"
#time_list = [s for s in seizure_list if 'timeVector' in s] # only files with "timeVector"
label_list = [s for s in seizure_list if 'labelVector' in s] # only files with "labelVector"

for ID in patient_IDs:
    print(f'Segmenting data for patient {ID}...')                
    for i in range(len(signal_list)):
        if "pat" + ID in signal_list[i]:
            
            sig = np.load(signal_list[i], allow_pickle = True)
            labels = np.load(label_list[i], allow_pickle = True)
            #times = np.load(time_list[i], allow_pickle = True)
            print(f'Splitting signal {signal_list[i].split("Seizures" + sep)[1]}')
            
            windows = []
            windows_label = []

            idx = 0
            while idx + window_size < len(sig):
                win = sig[idx:idx + window_size,:]
                lab = labels[idx:idx + window_size]
                
                # label window: if any ictal samples are present, classify whole window as ictal
                if np.any(lab == 2) == True:
                    windows_label.append(2)
                else:
                    windows_label.append(0)
                
                # apply filters and save window
                win_notch = signal.lfilter(b_notch, a_notch, win)  
                win_filtered = signal.lfilter(b_HPF, a_HPF, win_notch)
                
                windows.append(np.array(win_filtered, dtype="float32"))

                
                idx += window_size + 1
                
            print('Saving windowed signal and labels...')
            np.save(signal_list[i].split('_signalData.npy')[0].replace('Seizures', 'Seizures_windowed')+ '_windowData', windows)
            np.save(signal_list[i].split('_signalData.npy')[0].replace('Seizures', 'Seizures_windowed')+ '_windowLabel', windows_label)
            #np.save(signal_list[i].split('_signalData.npy')[0].replace('Seizures', 'Seizures_windowed')+ '_windowTime', windows_time)

#%% Feature extraction (linear univariate features)

window_list = sorted(os.listdir(path + 'Seizures_windowed' + sep))
window_list = [path + 'Seizures_windowed' + sep + s for s in window_list]

signal_list = [s for s in window_list if 'windowData' in s] # only files with "windowData"

for ID in patient_IDs:
    print(f'Extracting features for patient {ID}...')
    for i in range(len(signal_list)):
        if "pat" + ID in signal_list[i]:
            sig = np.load(signal_list[i], allow_pickle = True)
            print(f'Computing features from {signal_list[i].split("Seizures_windowed" + sep)[1]}')
            
            feature_mean = []
            feature_variance = []
            feature_skewness = []
            feature_kurtosis = []
            
            feature_thetapower = []
            feature_deltapower = []
            feature_betapower = []
            feature_alphapower = []
            #feature_gammapower = []
            feature_lowgammapower = []
            feature_highgammapower = []
            
            feature_hjorth_act = []
            feature_hjorth_mob = []
            feature_hjorth_com = []
            
            feature_sef50 = []
            feature_sef75 = []
            feature_sef90 = []
            
            feature_wavelet_energy_a7 = []
            feature_wavelet_energy_d7 = []
            feature_wavelet_energy_d6 = []
            feature_wavelet_energy_d5 = []
            feature_wavelet_energy_d4 = []
            feature_wavelet_energy_d3 = []
            feature_wavelet_energy_d2 = []
            feature_wavelet_energy_d1 = []
            
            #feature_circadian_rhythm = []
            
            for j in range(sig.shape[0]):
                window = sig[j, :, :]
                
                
                # MEAN
                mean = np.mean(window, axis = 0)
                feature_mean.append(mean)
                
                # VARIANCE
                variance = np.var(window, axis = 0, ddof = 1)
                feature_variance.append(variance)
                
                # SKEWNESS
                sum = 0
                for x in window:
                    sum += (x - mean)**3
                    
                skewness = ((1 / (len(window) - 1)) * sum) / (np.std(window, axis = 0)**3)
                feature_skewness.append(skewness)
                
                # KURTOSIS
                sum = 0
                for x in window:
                    sum += (x - mean)**4
                    
                kurtosis = (((1 / (len(window) - 1)) * sum) / ((len(window) - 1) * np.std(window, axis = 0)**4)) - 3
                feature_kurtosis.append(kurtosis)
                
                # RELATIVE SPECTRAL POWER
                psd = []
                for channel in range(window.shape[1]):
                    freqs, power = signal.welch(window[:,channel], fsampling)
                    psd.append(power)
                
                thetapower = []
                deltapower = []
                betapower = []
                alphapower = []
                gammapower = []
                lowgammapower = []
                highgammapower = []
                for spectrum in psd:
                    theta = integrate.simps(spectrum[theta_range[0]:theta_range[1]+1]) / integrate.simps(spectrum)
                    thetapower.append(theta)
                    
                    delta = integrate.simps(spectrum[delta_range[0] : delta_range[1]+1]) / integrate.simps(spectrum)
                    deltapower.append(delta)
                    
                    beta = integrate.simps(spectrum[beta_range[0] : beta_range[1]+1]) / integrate.simps(spectrum)
                    betapower.append(beta)
                    
                    alpha = integrate.simps(spectrum[alpha_range[0] : alpha_range[1]+1]) / integrate.simps(spectrum)
                    alphapower.append(alpha)
                    
                    #gamma = integrate.simps(spectrum[gamma_range[0] : gamma_range[1]+1]) / integrate.simps(spectrum)
                    #gammapower.append(gamma)
                    
                    low_gamma = integrate.simps(spectrum[low_gamma_range[0] : low_gamma_range[1]+1]) / integrate.simps(spectrum)
                    lowgammapower.append(low_gamma)
                    
                    high_gamma = integrate.simps(spectrum[high_gamma_range[0] : high_gamma_range[1]+1]) / integrate.simps(spectrum)
                    highgammapower.append(high_gamma)
                
                feature_thetapower.append(np.array(thetapower))
                feature_deltapower.append(np.array(deltapower))
                feature_betapower.append(np.array(betapower))
                feature_alphapower.append(np.array(alphapower))
                #feature_gammapower.append(np.array(gammapower))
                feature_lowgammapower.append(np.array(lowgammapower))
                feature_highgammapower.append(np.array(highgammapower))
                
                # HJORTH PARAMETERS
                deriv1 = np.gradient(window, axis = 0)
                deriv2 = np.gradient(deriv1, axis = 0)
                
                hjorth_act = variance
                hjorth_mob = np.sqrt(np.var(deriv1, axis = 0, ddof = 1)/np.var(window, axis = 0, ddof = 1))
                hjorth_com = np.sqrt((np.var(deriv2, axis = 0, ddof = 1)*np.var(window, axis = 0, ddof = 1))/np.var(deriv1, axis = 0, ddof = 1)**2)
                
                feature_hjorth_act.append(hjorth_act)
                feature_hjorth_mob.append(hjorth_mob)
                feature_hjorth_com.append(hjorth_com)
                
                # SPECTRAL EDGE FREQUENCY (50%, 75%, 90%)
                
                sef50percent = []
                sef75percent = []
                sef90percent = []
                for spectrum in psd:
                    power_cum = integrate.cumtrapz(spectrum)
                    sef50 = (np.abs(power_cum - 0.5*integrate.trapz(spectrum))).argmin() # closest freq holding 50% spectral power
                    sef50percent.append(sef50)
                    sef75 = (np.abs(power_cum - 0.75*integrate.trapz(spectrum))).argmin() # closest freq holding 75% spectral power
                    sef75percent.append(sef75)
                    sef90 = (np.abs(power_cum - 0.9*integrate.trapz(spectrum))).argmin() # closest freq holding 90% spectral power
                    sef90percent.append(sef90)
                    
                feature_sef50.append(np.array(sef50percent))
                feature_sef75.append(np.array(sef75percent)) 
                feature_sef90.append(np.array(sef90percent))
                
                # WAVELET COEFFICIENTS (ENERGY)
                
                a7_energy = []; d7_energy = []; d6_energy = []; d5_energy = []
                d4_energy = []; d3_energy = []; d2_energy = []; d1_energy = []
                for channel in range(window.shape[1]):
                    coeffs = pywt.wavedec(window[:, channel], mother_wavelet, level = 8)
                    # coeffs -> [A7, D7, D6, D5, D4, D3, D2, D1]
                    
                    a7_energy.append(np.sum(np.abs(np.power(coeffs[0], 2))))
                    d7_energy.append(np.sum(np.abs(np.power(coeffs[1], 2))))
                    d6_energy.append(np.sum(np.abs(np.power(coeffs[2], 2))))
                    d5_energy.append(np.sum(np.abs(np.power(coeffs[3], 2))))
                    d4_energy.append(np.sum(np.abs(np.power(coeffs[4], 2))))
                    d3_energy.append(np.sum(np.abs(np.power(coeffs[5], 2))))
                    d2_energy.append(np.sum(np.abs(np.power(coeffs[6], 2))))
                    d1_energy.append(np.sum(np.abs(np.power(coeffs[7], 2))))
                    
                feature_wavelet_energy_a7.append(a7_energy)
                feature_wavelet_energy_d7.append(d7_energy)
                feature_wavelet_energy_d6.append(d6_energy)
                feature_wavelet_energy_d5.append(d5_energy)
                feature_wavelet_energy_d4.append(d4_energy)
                feature_wavelet_energy_d3.append(d3_energy)
                feature_wavelet_energy_d2.append(d2_energy)
                feature_wavelet_energy_d1.append(d1_energy)
        
                # CIRCADIAN RHYTHM (seconds of the day, between 0 and 86400 -> normalize to 0-1)
                #circadian = (window_time[0].hour * 3600 + window_time[0].minute * 60) / (24 * 3600)
                #feature_circadian_rhythm.append(np.ones((19)) * circadian)
                
                
            print('Saving features...')
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_mean', feature_mean)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_var', feature_variance)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_skew', feature_skewness)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_kurt', feature_kurtosis)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_theta', feature_thetapower)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_delta', feature_deltapower)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_beta', feature_betapower)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_alpha', feature_alphapower)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_lowgamma', feature_lowgammapower)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_highgamma', feature_highgammapower)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_h_act', feature_hjorth_act)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_h_mob', feature_hjorth_mob)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_h_com', feature_hjorth_com)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_sef50', feature_sef50)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_sef75', feature_sef75)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_sef90', feature_sef90)
            #np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_circadian', feature_circadian_rhythm)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_a7', feature_wavelet_energy_a7)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_d7', feature_wavelet_energy_d7)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_d6', feature_wavelet_energy_d6)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_d5', feature_wavelet_energy_d5)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_d4', feature_wavelet_energy_d4)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_d3', feature_wavelet_energy_d3)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_d2', feature_wavelet_energy_d2)
            np.save(signal_list[i].split('_windowData.npy')[0].replace('Seizures_windowed', 'Features')+ '_d1', feature_wavelet_energy_d1)
            
            
#%% Organize data for the evolutionary framework

feature_list = sorted(os.listdir(path + 'Features' + sep))
feature_list = [path + 'Features' + sep + s for s in feature_list]
window_labels = [s for s in window_list if 'windowLabel' in s] # only files with "windowLabel"
seizure_exogenous = [s for s in seizure_list if 'exogenousVariables' in s] # only files with "exogenousVariables"

# Build array containing: feature values, labels, missingdata/flat percentage (for each window); columns = time, rows = feature/others
# Build label for the array created above; feature values = electrodeX_featureY
for ID in patient_IDs:
    
    # Get required files for all seizures of each patient: feature values, labels, missing/flat info
    feature_list_patient = []
    label_list_patient = []
    flat_list_patient = []
    saturated_list_patient = []
    missing_list_patient = []
    exogenous_list_patient = []
    
    print(f'Organizing data for patient {ID}...')
    for i in range(len(feature_list)):
        if "pat" + ID in feature_list[i]:
            feature_list_patient.append(feature_list[i])        
    for j in range(len(window_labels)):
        if "pat" + ID in window_labels[j]:
            label_list_patient.append(window_labels[j])
    for j in range(len(seizure_exogenous)):
        if "pat" + ID in seizure_exogenous[j]:
            exogenous_list_patient.append(seizure_exogenous[j])
                    
    # used to detect number of seizures for each patient        
    if len(feature_list_patient) % number_of_features == 0:
        seizures_number = len(feature_list_patient) / number_of_features
        
        # build, for each seizure, matrix containing feature values, classification labels (in this order)
        for j in range(0, len(feature_list_patient), number_of_features):
            seizure_features = feature_list_patient[j:j + number_of_features]
            seizure_ID = seizure_features[0].split(sep="_")[1]
            seizure_no = int(seizure_ID.split("seizure")[1])
            
            feature_matrix = np.load(seizure_features[0], allow_pickle = True).T # transpose so that rows = features, columns = window 
            for k in range(1, len(seizure_features)):
                feature_matrix = np.vstack((feature_matrix, np.load(seizure_features[k], allow_pickle = True).T))
            
            # add classification labels
            feature_matrix = np.vstack((feature_matrix, np.load([x for x in label_list_patient if seizure_ID+"_" in x][0], allow_pickle = True).T))
            
            np.save(path + 'Evol2' + sep + 'pat' + ID + '_seizure'+ str(seizure_no) + '_featureMatrix', feature_matrix)
    else:
        print(f'Could not detect number of seizures for patient {ID}! Please update feature labels...')
    
    # build array with exogenous information for all seizures
    exogenous_matrix = []
    for j in range(len(exogenous_list_patient)):
        exogenous_matrix.append(np.load(exogenous_list_patient[j], allow_pickle = True))
    np.save(path + 'Evol2' + sep + 'pat' + ID + '_seizureInfo', exogenous_matrix)

# build legend (same for all patients)
legend = []
for i in range(len(feature_labels)):
    for j in range(len(electrodes_label)):
        legend.append(electrodes_label[j] + '_' + feature_labels[i])
legend.append('class')
np.save(path + 'Evol2' + sep + 'legend', legend) # !!! mudar de volta para Evol depois

print("\a") # beep when done :)