import numpy as np
import pandas as pd
import random

ratio = 0.8 # % of linearly generated spectra to add per class

ihg_df = pd.read_csv('../../CSVs/full_ihg_data.csv')
ilg_df = pd.read_csv('../../CSVs/full_ilg_data.csv')
mcn_df = pd.read_csv('../../CSVs/full_mcn_data.csv')
pc_df = pd.read_csv('../../CSVs/full_pc_data.csv')
sca_df = pd.read_csv('../../CSVs/full_sca_data.csv')

processed_df = pd.concat([ihg_df, ilg_df, mcn_df, pc_df, sca_df])
processed_df.dropna(inplace=True)

# Randomly select 80% of the data
train_data = processed_df.sample(frac=ratio, random_state=42)
test_data = processed_df.drop(train_data.index)

train_data.to_csv('../../CSVs/processed_data/train_data.csv', index=False)
test_data.to_csv('../../CSVs/processed_data/test_data.csv', index=False)
