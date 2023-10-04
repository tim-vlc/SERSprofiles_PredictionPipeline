from vae import *
from gaussian import *
from plots import *
from gaussian import MultiDimensionalGaussian, get_distribution_labels

def augment_vae(num_augment, df_all, split, num_epochs, verbose):
    # Seperate Training and Test set
    # ----------------------------------------------------------
    X = df_all.iloc[:, :-1]
    y = df_all['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=100)

    train_set = X_train.copy()
    train_set['labels'] = y_train

    test_set = X_test.copy()
    test_set['labels'] = y_test

    X_train, X_test = torch.tensor(X_train.values), torch.tensor(X_test.values)

    # Create VAE and setup hyperparameters
    # ----------------------------------------------------------
    torch.manual_seed(0)

    d = 32

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vae = VariationalAutoencoder(latent_dims=d, device=device, verbose=verbose)

    lr = 1e-3 

    optim_ = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)

    vae.to(device)
    vae.encoder.to(device)
    vae.decoder.to(device)

    # Train
    # ----------------------------------------------------------
    for epoch in range(num_epochs):
        train_loss = train_epoch(vae,device,X_train,optim_)
        val_loss = test_epoch(vae,device,X_test)
        torch.cuda.empty_cache()
        if epoch % 1 == 0 and verbose:
            print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))

    # Get distribution of Gaussian Vector per class
    # ----------------------------------------------------------
    distribution_dict = get_distribution_labels(d, train_set, vae, device)

    # Generate fake spectra
    # ----------------------------------------------------------
    labels = train_set['labels'].unique()
    df_list = [train_set]

    for label in labels:
        class_df = train_set[train_set['labels']==label]
        num_samples = int( (len(class_df)/len(test_set)) * num_augment )
        means, variances = distribution_dict[label]

        # Create an instance of the MultiDimensionalGaussian class
        gaussian_vector = MultiDimensionalGaussian(means, variances)

        fake_latent = torch.tensor(gaussian_vector.sample(num_samples).astype(np.float32))
        fake_spectra = vae.decoder(fake_latent).detach().numpy()
        
        df = pd.DataFrame(fake_spectra, columns=train_set.columns.difference(['labels']))
        df['labels'] = label
        df_list.append(df)
    
    aug_set = pd.concat(df_list, axis=0)

    torch.save(vae.state_dict(), f"{split}_vae.pth")

    return aug_set, train_set, test_set
    
