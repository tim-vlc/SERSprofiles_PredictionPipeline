import pandas as pd
import numpy as np
import torch
import sys

sys.path.insert(1, '../NNs')

from vae2 import VariationalAutoencoder

type_ = 'processed' # 'raw' or 'processed'
ratio = 0.2
latent_dims = 64
train_data = pd.read_csv(f'../../CSVs/{type_}_data/{ratio}complete_train_data.csv')

def vae_augment(train_data, type_, latent_dims, device):
    num_pixels = len(train_data.columns) - 2
    
    # Initialize vae
    vae = VariationalAutoencoder(latent_dims, device)
    vae.load_state_dict(torch.load(f"../saved_models/vae_model_state_dict_CC.pth", map_location=torch.device('cpu')))
    
    vae.eval()

    length = len(train_data)
    spectra_array = np.zeros((length, num_pixels))
    labels = train_data['labels']
    patient = train_data['patient#']

    for i in range(length):
        if i % int(0.1 * length) == 0:
            print("[", i, f"/{length}]")
        
        true_spectrum = list(train_data.iloc[i, :-2].to_numpy())
        fake_spectrum = vae(torch.Tensor(true_spectrum).float().unsqueeze(0))

        spectra_array[i, :] = fake_spectrum.detach()

    df2 = pd.DataFrame(spectra_array, columns=train_data.columns[:-2])
    df2['labels'] = list(labels.to_numpy())
    df2['patient#'] = list(patient.to_numpy())
    train_data = pd.concat([train_data, df2], axis=0).copy()
    return train_data

labels = ['IHG', 'IMG', 'ILG', 'MMG', 'MLG', 'SCA', 'PC']

train_data = vae_augment(train_data, type_, latent_dims, device=torch.device('cpu'))

train_data.to_csv('../../CSVs/augmented_data/vae_train_data.csv', index=False)
print(len(train_data))
