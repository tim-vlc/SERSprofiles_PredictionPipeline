from libraries import * 

def plot_latent_2D(X_test, y_test, vae, d, dimred, guess_labels):
    vae.eval()
    device = vae.device
    
    latent = np.zeros((len(X_test), d))

    with torch.no_grad(): # No need to track the gradients
        for i in range(len(X_test)):
            # Move tensor to the proper device
            x = X_test[i].clone().detach().float().to(device)
            
            x_test = x.unsqueeze(0)
            x_latent = vae.encoder(x_test)
            latent[i, :] = x_latent.detach().cpu().numpy()[0]

    # Assigning random color to each label
    if guess_labels:
        n_clusters = 2

        # Predict the labels for the encoded samples
        clustering = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(latent)
        cat_labels = clustering.predict(latent)
        # Convert integer labels to string
        pred_labels = [str(label) for label in cat_labels]
        labels = list(set(pred_labels))
        color_dict = {label:list(mcolors.TABLEAU_COLORS.keys())[i] for i, label in enumerate(labels)}
        colors = [color_dict[value] for value in list(pred_labels)]
    else:
        labels = list(y_test.unique())
        color_dict = {label:list(mcolors.TABLEAU_COLORS.keys())[i] for i, label in enumerate(labels)}
        colors = [color_dict[value] for value in list(y_test.values)]
    
    if dimred == 'TSNE':
        # Visualize the encoded samples in 2D space using t-SNE
        tsne = TSNE(n_components=2)
        results = tsne.fit_transform(latent)
    elif dimred == 'PCA':
        pca = PCA(n_components=2, svd_solver='full')
        results = pca.fit_transform(latent)
    else:
        print('Wrong dimensionality reduction method.')
        return

    plt.figure(figsize=(8,3))
    plt.scatter(results[:,0],results[:,1],
                c = colors)
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
    plt.legend(handles=legend_handles)

    filename = 'latent2DGuess' if guess_labels else 'latent2DLabels'

    plt.savefig('images/' + filename + '.png')
    
    return

def plot_latent_3D(X_test, y_test, vae, d):
    vae.eval()
    device = vae.device
    
    latent = np.zeros((len(X_test), d))

    with torch.no_grad(): # No need to track the gradients
        for i in range(len(X_test)):
            # Move tensor to the proper device
            x = X_test[i].clone().detach().float().to(device)
            
            x_test = x.unsqueeze(0)
            x_latent = vae.encoder(x_test)
            latent[i, :] = x_latent.detach().numpy()[0]
            
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Visualize the encoded samples in 3D space using t-SNE
    tsne = TSNE(n_components=3)
    results = tsne.fit_transform(latent)
    
    labels = list(y_test.unique())
    color_dict = {label:list(mcolors.TABLEAU_COLORS.keys())[i] for i, label in enumerate(labels)}
    colors = [color_dict[value] for value in list(y_test.values)]

    # Plot the 3D data
    ax.scatter(results[:,0], results[:,1], results[:,2], c=colors, marker='o')

    # Customize the plot
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('3D Scatter Plot')
    ax.view_init(elev=20, azim=30)

    # Display the plot
    plt.savefig('../images/latent3D.png')
    
    return

def real_fake_spectra(reals, fakes, labels):
    plt.figure(figsize=(10, 6))

    for i, (real_spectra, fake_spectra) in enumerate(zip(reals, fakes)):
        for fake_spectrum, real_spectrum in zip(fake_spectra, real_spectra):
            plt.plot(fake_spectrum, label=f'{labels[i]} fake Spectrum')
            plt.plot(real_spectrum, label=f'{labels[i]} real Spectrum')
        
    # Set labels and title
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Real vs. Fake Spectra Comparison')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig('../images/rf_spectra.png')
