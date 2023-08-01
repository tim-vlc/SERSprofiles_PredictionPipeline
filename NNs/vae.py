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
    def __init__(self, latent_dims, device):  
        super(VariationalEncoder, self).__init__()
        
        # 1st Convolutional Layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=20, stride=1)
        )
        
        # 2nd Convolutional Layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=20, stride=1)
        )
        
        # 3rd Convolutional Layer
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=20, stride=1)
        )
        
        self.fc1 = nn.Linear(64 * 794, 512)
        self.fc2 = nn.Linear(512, latent_dims)
        self.fc3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1).to(device)
        self.kl = 0

    def forward(self, x):
        # x = x.to(device)
        x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers
        x = F.relu(self.fc1(x))
        mu = self.fc2(x)
        sigma = torch.exp(self.fc3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(latent_dims, 512)
        self.fc2 = nn.Linear(512, 794*64)

        # 1st Convolutional Layer
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=20, stride=1)
        )
        
        # 2nd Convolutional Layer
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=20, stride=1)
        )
        
        # 3rd Convolutional Layer
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=20, stride=1)
        )
        
    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = z.view(z.size(0), 64, 794)  # Reshape for convolutional layers
        #z = z.unsqueeze(1)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = torch.sigmoid(self.conv3(z))
        return z
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, device):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, device)
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
        batch_X = X_train[i:i+batch].clone().detach().float().to(device)
        
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
            x = X_test[i].clone().detach().float().to(device)
            
            # Decode data
            x_test = x.unsqueeze(0)
            x_hat = vae(x_test)

            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(X_test)