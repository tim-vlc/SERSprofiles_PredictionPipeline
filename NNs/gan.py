import torch.nn as nn
import torch
import numpy as np

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1000, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(256, 708),
            nn.LeakyReLU(0.2),
            nn.Linear(708, 851),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(851, 1608),
            nn.LeakyReLU(0.2),
            nn.Linear(1608, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
# Training loop
def train(generator, discriminator, train_loader, num_epochs, criterion, discriminator_optimizer, generator_optimizer, g_loss, d_loss, device):
    for epoch in range(num_epochs):
        gen_losses = []
        disc_losses = []

        for spectra, _ in train_loader:
            batch_size = spectra.size(0)

            # Generate noise
            noise = torch.randn(batch_size, 1000)

            # Generate fake images
            fake_spectra = generator(noise.to(device))

            # Train discriminator
            real_logits = discriminator(spectra.to(device))
            fake_logits = discriminator(fake_spectra.detach())

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            disc_loss = criterion(real_logits, real_labels) + criterion(fake_logits, fake_labels)

            discriminator_optimizer.zero_grad()
            disc_loss.backward()
            discriminator_optimizer.step()

            # Train generator
            fake_logits = discriminator(fake_spectra)

            gen_loss = criterion(fake_logits, real_labels)

            generator_optimizer.zero_grad()
            gen_loss.backward()
            generator_optimizer.step()

            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())

        g_loss.append(np.mean(gen_losses))
        d_loss.append(np.mean(disc_losses))

        #if epoch % 10 == 0:
        #    with torch.no_grad():
        #        generate_and_save_images(generator, epoch + 1)

        print(f"Epoch {epoch + 1}, Gen Loss: {g_loss[-1]}, Disc Loss: {d_loss[-1]}")