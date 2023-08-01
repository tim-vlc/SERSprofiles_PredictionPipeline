import pandas as pd
import torch

from vae2 import VariationalAutoencoder, train_epoch, test_epoch


train_data = pd.read_csv('../../CSVs/processed_data/train_data.csv')
train_data.dropna(inplace=True)
test_data = pd.read_csv('../../CSVs/processed_data/test_data.csv')
test_data.dropna(inplace=True)

X_test, y_test = test_data.iloc[:,:-1], test_data.iloc[:,-1]
X_train, y_train = train_data.iloc[:,:-1], train_data.iloc[:,-1]

X_train, X_test = torch.tensor(X_train.values), torch.tensor(X_test.values)

# Set the random seed for reproducible results
torch.manual_seed(0)

d = 32

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

vae = VariationalAutoencoder(latent_dims=d, device=device)

lr = 1e-3 

optim_ = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

vae.to(device)
vae.encoder.to(device)
vae.decoder.to(device)

# Train

num_epochs = 20

for epoch in range(num_epochs):
    train_loss = train_epoch(vae,device,X_train,optim_)
    val_loss = test_epoch(vae,device,X_test)
    if epoch % 4 == 0:
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))

# Save the model state dictionary
torch.save(vae.state_dict(), '../saved_models/vae_model_state_dict_CC.pth')