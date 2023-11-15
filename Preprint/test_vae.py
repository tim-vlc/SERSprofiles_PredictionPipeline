import pandas as pd
import numpy as np
from nn_main import *
import sys

sys.path.insert(1, 'VAE/')

from vae import *
from plots import *

# Set the random seed for reproducible results
torch.manual_seed(0)

# IMPORT DATA
path_to_file = '../../CSVs/diabetes.csv'

data = pd.read_csv(path_to_file)

train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

device = torch.device("cuda:0")

X_test, y_test = test_data.iloc[:,:-1], test_data['labels']
X_train, y_train = train_data.iloc[:,:-1], train_data['labels']
X_train, X_test = torch.tensor(X_train.values), torch.tensor(X_test.values)

d = 64
lr = 1e-4

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

vae = VariationalAutoencoder(latent_dims=d, device=device, verbose=True)

optim_ = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

vae.to(device)
vae.encoder.to(device)
vae.decoder.to(device)

# Train

num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train_epoch(vae,device,X_train,optim_)
    val_loss = test_epoch(vae,device,X_test)
    torch.cuda.empty_cache()
    if epoch % 1 == 0:
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))

plot_latent_2D(X_train, y_train, vae, d, 'PCA', True)
plot_latent_2D(X_train, y_train, vae, d, 'PCA', False)
