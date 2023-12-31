import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from gan import Generator, Discriminator, train

ratio = 0.1
type_ = 'processed'
pixel_num = 1650 if type_ == 'raw' else 851
#train_data = pd.read_csv(f'../../CSVs/{type_}_data/{ratio}complete_train_data.csv')
data = pd.read_csv('../../complete_processed_data.csv')
train_data = data.sample(frac=ratio, random_state=42)
test_data = data.drop(train_data.index)
test_data.to_csv(f'../../CSVs/augmented_data/{ratio}gan_test_data.csv', index=False)
train_data.to_csv(f'../../CSVs/augmented_data/{ratio}prev_train_data.csv', index=False)

labels = ['IHG', 'IMG', 'ILG', 'MMG', 'MLG', 'SCA', 'PC']

def train_gan(label, train_data, type_, ratio, pixel_num):
    print(f"Getting the {label} dataframe...")
    df = train_data[train_data['labels']==label].iloc[:, :-1]
    df['labels'] = 0
    print("Done!")

    # Split the DataFrame into input features (spectra) and labels
    spectra = df.iloc[:, :-1].values  # Extract all columns except the last one
    labels = df.iloc[:, -1].values   # Extract the last column

    # Convert the data to torch tensors
    spectra_tensor = torch.from_numpy(spectra).float() #.float()
    labels = np.vstack(labels).astype(float)
    labels_tensor = torch.from_numpy(labels)

    # Create a TensorDataset
    dataset = TensorDataset(spectra_tensor, labels_tensor)

    # Set batch size and number of workers
    batch_size = 25
    num_workers = 0

    if len(df) % batch_size == 1:
        batch_size +=1

    # Create data loader
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    g_loss = []
    d_loss = []

    # Initialize generator and discriminator
    device = torch.device("cuda:0")
    generator = Generator(pixel_num).to(device)
    discriminator = Discriminator(pixel_num).to(device)

    # Loss function
    criterion = nn.BCELoss().to(device)

    # Optimizers
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Train the model
    EPOCHS = 200
    print(f"Training the {label} GAN model...")
    train(generator, discriminator, train_loader, EPOCHS, criterion, discriminator_optimizer, generator_optimizer, g_loss, d_loss, device)
    print("Done!")
    torch.save(generator.state_dict(), f"../saved_models/{type_}_data/{ratio}{label}_gen_model.pth")
    torch.save(discriminator.state_dict(), f"../saved_models/{type_}_data/{ratio}{label}_disc_model.pth")

#label = 'PC'
#train_gan(label, train_data, type_, ratio, pixel_num)

for label in labels:
     train_gan(label, train_data, type_, ratio, pixel_num)
