import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from gan import Generator, Discriminator, train

train_data = pd.read_csv('../../CSVs/processed_data/train_data.csv')

labels = ['IHG', 'ILG', 'MCN', 'SCA', 'PC']

def train_gan(label, train_data):
    print(f"Getting the {label} dataframe...")
    df = train_data[train_data['labels']==label]
    df[df.columns[-1]] = 0
    print("Done!")

    # Split the DataFrame into input features (spectra) and labels
    spectra = df.iloc[:, :-1].values  # Extract all columns except the last one
    labels = df.iloc[:, -1].values   # Extract the last column

    # Convert the data to torch tensors
    spectra_tensor = torch.tensor(spectra, dtype=torch.float32)
    labels = np.vstack(labels).astype(float)
    labels_tensor = torch.from_numpy(labels)

    # Create a TensorDataset
    dataset = TensorDataset(spectra_tensor, labels_tensor)

    # Set batch size and number of workers
    batch_size = 20
    num_workers = 0

    # Create data loader
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    g_loss = []
    d_loss = []

    # Initialize generator and discriminator
    device = torch.device("cuda:0")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

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
    torch.save(generator.state_dict(), f"../saved_models/{label}_gen_model.pth")
    torch.save(discriminator.state_dict(), f"../saved_models/{label}_disc_model.pth")

label = 'SCA'
train_gan(label, train_data)

# for label in labels:
#     train_gan(label, train_data, batch_size)
