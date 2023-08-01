import numpy as np
import pandas as pd
import random

ratio = 0.8 # % of linearly generated spectra to add per class
type_ = 'raw' # raw or full

ihg_df = pd.read_csv(f'../../CSVs/{type_}_ihg_data.csv')
ilg_df = pd.read_csv(f'../../CSVs/{type_}_ilg_data.csv')
mcn_df = pd.read_csv(f'../../CSVs/{type_}_mcn_data.csv')
pc_df = pd.read_csv(f'../../CSVs/{type_}_pc_data.csv')
sca_df = pd.read_csv(f'../../CSVs/{type_}_sca_data.csv')

processed_df = pd.concat([ihg_df, ilg_df, mcn_df, pc_df, sca_df])
processed_df.dropna(inplace=True)

# Randomly select 80% of the data
train_data = processed_df.sample(frac=ratio, random_state=42)
test_data = processed_df.drop(train_data.index)

folder = 'processed' if type_ == 'full' else 'raw'
train_data.to_csv(f'../../CSVs/{folder}_data/train_data.csv', index=False)
test_data.to_csv(f'../../CSVs/{folder}_data/test_data.csv', index=False)
