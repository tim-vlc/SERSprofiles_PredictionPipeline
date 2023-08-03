import pandas as pd

type_ = 'processed' # raw or processed
ratio = 0.8

data = pd.read_csv(f'../../CSVs/complete_{type_}_data.csv')
# Randomly select ratio of the data
train_data = data.sample(frac=ratio, random_state=42)
test_data = data.drop(train_data.index)

train_data.to_csv(f'../../CSVs/{type_}_data/complete_train_data.csv', index=False)
test_data.to_csv(f'../../CSVs/{type_}_data/complete_test_data.csv', index=False)