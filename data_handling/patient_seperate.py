import pandas as pd
import random

type_ = 'processed' # raw or processed
ratio = 0.2

data = pd.read_csv(f'../../CSVs/complete_{type_}_data.csv')
# Randomly select ratio of the data
ratio_ = ratio - 0.1
labels = list(data['labels'].unique())
num_col = len(data.columns)
test_data = pd.DataFrame(columns= list(range(851)) + ['labels', 'patient#'])

for label in labels:
    patients = list(data[data['labels']==label]['patient#'].unique())
    length = len(patients)
    elements = int(ratio_ * length) + 1
    test_patients = random.sample(patients, elements)
    
    df2 = data[data['patient#'].isin(test_patients)]
    test_data = pd.concat([test_data, df2], axis=0)
train_data = data.drop(test_data.index)

train_data.to_csv(f'../../CSVs/{type_}_data/{ratio}complete_train_data.csv', index=False)
test_data.to_csv(f'../../CSVs/{type_}_data/{ratio}complete_test_data.csv', index=False)
print(len(train_data), len(test_data))
