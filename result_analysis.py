import os
import linecache
import numpy as np
from scipy.io import savemat

FG_mean = np.zeros((12, 9))
accuracy_all = np.zeros((12, 1))
F_all = np.zeros((12, 1))
G_all = np.zeros((12, 1))
FG_mean_all = np.zeros((12, 1))
for lead in range(1, 13):
    root = 'save'
    folder = 'lead'+str(lead)+'_ResNet8_NoDropout_32_WCE'
    directory = os.path.join(root, folder)
    for file in os.listdir(directory):
        if os.path.splitext(file)[1] == ".txt":
            log_file = os.path.join(directory, file)
            for c in range(9):
                FG_mean_c = float(linecache.getline(log_file, 6+c)[21:26])
                FG_mean[lead-1, c] = FG_mean_c
            accuracy_all_lead = float(linecache.getline(log_file, 1)[14:22])
            F_all_lead = float(linecache.getline(log_file, 3)[19:27])
            G_all_lead = float(linecache.getline(log_file, 4)[19:27])
            FG_mean_all_lead = float(linecache.getline(log_file, 5)[16:24])

            accuracy_all[lead-1, 0] = accuracy_all_lead
            F_all[lead-1, 0] = F_all_lead
            G_all[lead-1, 0] = G_all_lead
            FG_mean_all[lead-1, 0] = FG_mean_all_lead

savemat('result.mat', {'FG_mean': FG_mean,
                       'accuracy_all': accuracy_all,
                       'F_all': F_all,
                       'G_all': G_all,
                       'FG_mean_all': FG_mean_all})
