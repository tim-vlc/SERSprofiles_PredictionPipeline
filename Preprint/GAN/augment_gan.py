from gan import *

def augment_gan(num_augment, train_set, test_set, verbose, split):
    num_pixels = len(train_set.columns) - 1
    pipe = rp.preprocessing.Pipeline([
        rp.preprocessing.denoise.SavGol(window_length=30, polyorder=3),
    ])
    df_list = [train_set]

    # Train
    # ----------------------------------------------------------
    for label in train_set.labels.unique():
        train_gan(label, train_set, verbose, split)
        class_df = train_set[train_set['labels']==label]
        num_samples = int( (len(class_df)/len(test_set)) * num_augment )

        # Generate fake spectra
        # ----------------------------------------------------------
        generator = Generator(num_pixels)
        generator.load_state_dict(torch.load(f"{split}{label}_gen_model.pth", map_location=torch.device('cpu')))

        with torch.no_grad():
            generator.eval()

            # Generate noise
            noise = torch.randn(num_samples, 1000)

            # Generate fake images
            fake_spectra = generator(noise).detach().numpy()
            fake_spectra = pipe.apply(Spectrum(fake_spectra, range(num_pixels))).spectral_data

        df = pd.DataFrame(fake_spectra, columns=train_set.columns.difference(['labels']))
        df['labels'] = label
        df_list.append(df)
        del df
        del fake_spectra

    aug_set = pd.concat(df_list, axis=0)
    del df_list

    return aug_set
