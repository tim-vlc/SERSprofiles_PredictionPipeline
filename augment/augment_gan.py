import pandas as pd
import numpy as np
import torch
import sys

sys.path.insert(1, '../NNs')

from gan import Generator, Discriminator

import ramanspy as rp
from ramanspy import Spectrum

type_ = 'processed' # 'raw' or 'processed'
ratio = 0.5
train_data = pd.read_csv(f'../../CSVs/{type_}_data/{ratio}complete_train_data.csv')

pipe = rp.preprocessing.Pipeline([
    rp.preprocessing.denoise.SavGol(window_length=14, polyorder=3),
])

def gan_augment(label, train_data, num_aug, ratio_dict, pipe, ratio, type_):
    num_pixels = len(train_data.columns) - 1
    grade = 0.1 #ratio_dict[label]
    
    # Initialize generator and discriminator
    generator = Generator(num_pixels)
    generator.load_state_dict(torch.load(f"../saved_models/{type_}_data/{ratio}{label}_gen_model.pth", map_location=torch.device('cpu')))

    discriminator = Discriminator(num_pixels)
    discriminator.load_state_dict(torch.load(f"../saved_models/{type_}_data/{ratio}{label}_disc_model.pth", map_location=torch.device('cpu')))
    
    generator.eval()

    spectra_array = np.zeros((num_aug, num_pixels))

    for i in range(num_aug):
        if i % int(0.1 * num_aug) == 0:
            print("[", i, f"/{num_aug}]")
        fake_pred = 0

        while fake_pred < grade:
            # Generate noise
            noise = torch.randn(1, 1000)

            # Generate fake images
            fake_spectrum = generator(noise)
            fake_pred = discriminator(fake_spectrum)[0][0].detach().numpy()


        fake_spectrum = pipe.apply(Spectrum(list(fake_spectrum[0].squeeze().detach().numpy()), range(1, 852))).spectral_data

        spectra_array[i, :] = fake_spectrum

    df2 = pd.DataFrame(spectra_array, columns=train_data.columns[:-1])
    df2['labels'] = label
    train_data = pd.concat([train_data, df2], axis=0).copy()
    return train_data

labels = ['IHG', 'IMG', 'ILG', 'MMG', 'MLG', 'SCA', 'PC']
length_data = len(train_data)
augment_num = 14800
ratio_dict = {'IHG':0.9, 'ILG':0.6, 'MCN':0.9, 'SCA':0.8, 'PC':0.9}

len_dict = ((train_data['labels'].value_counts() * augment_num) / length_data).to_dict()
print(len_dict)
for label in labels:
    length = len_dict[label]

    train_data = gan_augment(label, train_data, int(length), ratio_dict, pipe, ratio, type_)

train_data.to_csv('../../CSVs/augmented_data/gan_train_data.csv', index=False)
print(len(train_data))
