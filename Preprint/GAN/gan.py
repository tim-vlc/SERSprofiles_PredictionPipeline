from libraries import *

# Generator model
class Generator(nn.Module):
    def __init__(self, pixel_num):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1000, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 708),
            nn.LeakyReLU(0.2),
            nn.Linear(708, pixel_num),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
    
# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, pixel_num):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(pixel_num, 813),
            nn.LeakyReLU(0.2),
            nn.Linear(813, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
# Training loop
def train(generator, discriminator, train_loader, num_epochs, criterion, discriminator_optimizer, generator_optimizer, g_loss, d_loss, device, verbose):
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
        if verbose:
            print(f"Epoch {epoch + 1}, Gen Loss: {g_loss[-1]}, Disc Loss: {d_loss[-1]}")

def train_gan(label, train_data, verbose, split):
    pixel_num = len(train_data.columns) - 1
    print(f"Getting the {label} dataframe...")
    df = train_data[train_data['labels']==label]
    df.loc.__setitem__((slice(None), ('labels')), 0) 
    print("Done!")

    # Split the DataFrame into input features (spectra) and labels
    spectra = df.iloc[:, :-1].values  # Extract all columns except the last one
    labels = df['labels'].values   # Extract the last column

    # Convert the data to torch tensors
    spectra_tensor = torch.from_numpy(spectra).float()
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    generator = Generator(pixel_num).to(device)
    discriminator = Discriminator(pixel_num).to(device)

    # Loss function
    criterion = nn.BCELoss().to(device)

    # Optimizers
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Train the model
    EPOCHS = 60
    print(f"Training the {label} GAN model...")
    train(generator, discriminator, train_loader, EPOCHS, criterion, discriminator_optimizer, generator_optimizer, g_loss, d_loss, device, verbose)
    print("Done!")
    torch.save(generator.state_dict(), f"{split}{label}_gen_model.pth")
    torch.save(discriminator.state_dict(), f"{split}{label}_disc_model.pth")
