from helper_functions import train_selfONN, test_selfONN

from pathlib import Path
import pandas as pd 

import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import time
import os


Q = [3]
all_patients = [1,2,3,4,5,6,7,8.9,10]
base_path = '../'
num_epochs = 30

results_all = np.zeros((10*len(Q),6), dtype = np.int32)
perc_all = np.zeros((10*len(Q),3), dtype = np.float32)
results_S = np.zeros((10*len(Q),5), dtype = np.int32)
results_V = np.zeros((10*len(Q),5), dtype = np.int32)

count = 0


for pat_num in [1]:

    for q in Q:
        
        
        # Training
        train_selfONN(pat_num, q, epochs = num_epochs)
        
        # Testing
        stats_R, stats_S, stats_V = test_selfONN(pat_num, q, threshold = 0.1)

        #_______________________________________________
        # Saving stats
        #_______________________________________________

        if not os.path.exists(base_path + 'Results'):
            os.makedirs(base_path + 'Results')

        # Save Results for All beats
        results_all[count][0] = pat_num
        results_all[count][1] = q
        results_all[count][2:] = stats_R[:4]
        perc_all[count] = stats_R[4:7]

        df_all = pd.DataFrame(results_all)
        df_all = pd.concat([df_all, pd.DataFrame(perc_all, dtype = np.float32)], axis=1)
        df_all.columns = ['Patient No', 'Q', 'Total Beats', 'TP', 'FN', 'FP', 'Recall', 'Precision', 'F1']
        f = base_path + 'Results/selfONN_all_3a.csv'
        df_all.to_csv (r'{}'.format(f), index = False, header=True)

        
        # Save Results for S beats
        if stats_S != []:
            results_S[count][0] = pat_num
            results_S[count][1] = q
            results_S[count][2:] = stats_S[:3]

            df_S = pd.DataFrame(results_S)
            df_S.columns = ['Patient No', 'Q','Total Beats', 'Detected', 'Missed']
            f = base_path + 'Results/selfONN_S_3a.csv'
            df_S.to_csv (r'{}'.format(f), index = False, header=True)

        # Save Results for V beats
        if stats_V != []:

            results_V[count][0] = pat_num
            results_V[count][1] = q
            results_V[count][2:] = stats_V[:3]

            df_V = pd.DataFrame(results_V)
            df_V.columns = ['Patient No', 'Q','Total Beats', 'Detected', 'Missed']
            f = base_path + 'Results/selfONN_V_3a.csv'
            df_V.to_csv (r'{}'.format(f), index = False, header=True)

        count += 1 
