import numpy as np
import pandas as pd
import random

num_new = 10000 # Number of linearly generated spectra to add per class
num_per_class = 2074 # Number of og samples to take from each class (balance train_data)

ihg_df = pd.read_csv('../CSVs/raw_ihg_data.csv').drop('labels', axis=1)
ilg_df = pd.read_csv('../CSVs/raw_ilg_data.csv').drop('labels', axis=1)
mcn_df = pd.read_csv('../CSVs/raw_mcn_data.csv').drop('labels', axis=1)
pc_df = pd.read_csv('../CSVs/raw_pc_data.csv').drop('labels', axis=1)
sca_df = pd.read_csv('../CSVs/raw_sca_data.csv').drop('labels', axis=1)

print('Seperating train-test data...')
ihg_train = ihg_df.sample(frac=num_per_class/len(ihg_df), random_state=42)
ihg_test = ihg_df.drop(ihg_train.index)
ilg_train = ilg_df.sample(frac=num_per_class//len(ilg_df), random_state=42)
ilg_test = ilg_df.drop(ilg_train.index)
mcn_train = mcn_df.sample(frac=num_per_class//len(mcn_df), random_state=42)
mcn_test = mcn_df.drop(mcn_train.index)
pc_train = pc_df.sample(frac=num_per_class//len(pc_df), random_state=42)
pc_test = pc_df.drop(pc_train.index)
sca_train = sca_df.sample(frac=num_per_class//len(sca_df), random_state=42)
sca_test = sca_df.drop(sca_train.index)

test_data = pd.concat([ihg_test, ilg_test, mcn_test, sca_test, pc_test])
test_data.to_csv('../CSVs/augmented_data/augpro_test_data.csv')
print('Finished seperating train-test and saving test data.')

list_df = [ihg_train, ilg_train, mcn_train, sca_train, pc_train]

dict_df = {'IHG':list_df[0], 'ILG':list_df[1], 'MCN':list_df[2], 'SCA':list_df[3], 'PC':list_df[4]}

def lin_comb_spectra(df, num_new, label):
    # Choose three, four, and five ≠ spectra to make a linear combination of them
    lin_df = df.copy()
    length_df = len(df)

    for i in range(3, 6):
        already_gen = []
        num_added = num_new//3
        spectra_array = np.zeros((num_added, len(df.iloc[0])))

        for j in range(num_added):

            # Choose i random integers between 0 and the number of original spectra
            rnd_i_num = random.sample(range(0, length_df), i)
            while rnd_i_num in already_gen:
                rnd_i_num = random.sample(range(0, length_df), i)
            already_gen.append(rnd_i_num)

            # Initialize new raman spectra generated
            new_rs = np.zeros_like(lin_df.iloc[0].values)

            # Choose i random floats between 0 and 1 summing to 1
            sum_floats = 0.0  # Initialize the sum
            floats = []
            for j in range(i):
                if j == i-1:
                    value = 1-sum_floats
                else:
                    value = random.uniform(0, 1 - sum_floats)
                floats.append(value)
                sum_floats += value

                # Compute the new raman spectra
                new_rs += value * lin_df.iloc[rnd_i_num[j]].values
                
            spectra_array[j, :] = new_rs
        df2 = pd.DataFrame(spectra_array)
        lin_df = pd.concat([lin_df, df2], ignore_index=True)

    lin_df['labels'] = label
    return lin_df

for i, label in enumerate(dict_df.keys()):
    print(f"Augmenting {label} data...")
    list_df[i] = lin_comb_spectra(dict_df[label], num_new, label).copy()
    print("Done!")

print('Saving the augmented data...')
train_data = pd.concat(list_df)
train_data.to_csv('../CSVs/augmented_data/augpro_train_data.csv', index=False)
print('Augmentation is complete.')