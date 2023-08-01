import pandas as pd
import numpy as np
import torch
from gan import Generator, Discriminator

import ramanspy as rp
from ramanspy import Spectrum

train_data = pd.read_csv('../CSVs/processed_data/train_data.csv')

pipe = rp.preprocessing.Pipeline([
    rp.preprocessing.denoise.SavGol(window_length=14, polyorder=3),
])

def gan_augment(label, train_data, num_aug, ratio_dict, pipe):
    num_pixels = len(train_data.columns) - 1
    grade = ratio_dict[label]
    
    # Initialize generator and discriminator
    generator = Generator()
    generator.load_state_dict(torch.load(f"../saved_models/{label}_gen_model.pth", map_location=torch.device('cpu')))

    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(f"../saved_models/{label}_disc_model.pth", map_location=torch.device('cpu')))
    
    generator.eval()

    spectra_array = np.zeros((num_aug, num_pixels))

    for i in range(num_aug):
        if i % int(0.1 * num_aug) == 0:
            print("[", i, f"/{num_aug}]")
        fake_pred = 0

        while fake_pred < grade:
            # Generate noise
            noise = torch.randn(1, 100)

            # Generate fake images
            fake_spectrum = generator(noise)
            fake_pred = discriminator(fake_spectrum)[0][0].detach().numpy()


        fake_spectrum = pipe.apply(Spectrum(list(fake_spectrum[0].squeeze().detach().numpy()), range(1, 852))).spectral_data

        spectra_array[i, :] = fake_spectrum

    df2 = pd.DataFrame(spectra_array, columns=train_data.columns[:-1])
    df2['labels'] = label
    train_data = pd.concat([train_data, df2], axis=0).copy()
    return train_data

labels = ['IHG', 'ILG', 'MCN', 'SCA', 'PC']
length_data = len(train_data)
augment_num = 30000
ratio_dict = {'IHG':0.3, 'ILG':0.6, 'MCN':0.9, 'SCA':0.8, 'PC':0.9}

len_dict = ((train_data['labels'].value_counts() * augment_num) / length_data).to_dict()
print(len_dict)
for label in labels:
    length = len_dict[label]

    updated_train_data = gan_augment(label, train_data, int(length), ratio_dict, pipe)

updated_train_data.to_csv('ganaug_train_data.csv')
print(len(updated_train_data))