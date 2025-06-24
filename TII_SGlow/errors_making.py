import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import math



model_disc = ['LSTM', 'GRU', 'Seq2Seq', 'TCNN', 'STGCN','Trans']
model_gen = ['VAE', 'CVAE', 'GAN', 'RealNVP', 'Glow', 'SGlow']
model_alba = ['aff_conv_D','E_aff_D','E_aff_conv', 'E_D', 'aff_conv']
for steps in [16, 32, 48, 64]:
    model = model_disc + model_gen + model_alba
    Er_list = []
    for mo in model:
        Er = np.load(f'./output_f4_{steps}/{mo}/Error.npy')
        Er_df = pd.DataFrame(Er, columns=[f'{mo}_ade', f'{mo}_fde'])
        Er_list.append(Er_df)
    Er_DF = pd.concat(Er_list, axis=1)
    file = open(f'/home/user/Documents/Yangkaisen/GMs/Gms_SGlow/output_f4_{steps}/Error_Table_{steps}.csv', 'w')
    Er_DF.to_csv(file, index=False)
    file.close()
