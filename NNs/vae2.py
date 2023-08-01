import torch
import torch.nn as nn
import torch.nn.functional as F

# We define a Variational Autoencoder Class, which combines the Encoder and Decoder classes.
# Composition:
#       - 3 Conv Layers
#       - 2 Fully Connected Layers
# Also added batch layers for better features in the latent space.

# ≠ Autoencoder ==> encoder returns mean and variance matrices, used for sample latent vector.
# VAE ==> we obtain the Kullback-Leibler Term (D_{kl})

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):  
        super(VariationalEncoder, self).__init__()
        
        self.fc1 = nn.Linear(851, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, latent_dims)
        self.fc5 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        # x = x.to(device)
        x = x.unsqueeze(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.fc4(x)
        sigma = torch.exp(self.fc5(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(latent_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 851)
        
    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = torch.sigmoid(self.fc4(z))
        return z
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        # x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
    
### Training function
def train_epoch(vae, device, X_train, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    batch = 16
    train_loss = 0.0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for i in range(0, len(X_train), batch):
        batch_X = torch.tensor(X_train[i:i+batch], dtype=torch.float32).to(device)
        
        x_hat = vae(batch_X)
        # Evaluate loss
        loss = ((batch_X - x_hat)**2).sum() + vae.encoder.kl
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            # Print batch loss
            print('[%i] \t partial train loss (single batch): %f' % (i, loss.item()))
        
        train_loss += loss.item()

    return train_loss / len(X_train)

### Testing function
def test_epoch(vae, device, X_test):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0

    with torch.no_grad(): # No need to track the gradients
        for i in range(len(X_test)):
            # Move tensor to the proper device
            x = torch.tensor(X_test[i], dtype=torch.float32).to(device)
            
            # Decode data
            x_test = x.unsqueeze(0)
            x_hat = vae(x_test)

            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(X_test)