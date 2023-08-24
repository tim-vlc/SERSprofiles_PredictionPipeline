import pandas as pd
import random

data = pd.read_csv('../../binary/complete_processed_data.csv')

# Replace values by correspondingly important ones
labels = list(data['labels'].unique())
labels.remove('PC')
print(labels)
replace_ = {label:'Control' for label in labels}
replace_['PC'] = 'I-Cancer'

data['labels'].replace(replace_, inplace=True)

cancer_patients = list(data[data['labels']=='I-Cancer']['patient#'].unique())
control_patients = list(data[data['labels']=='Control']['patient#'].unique())
#cancer_patients.remove(149)
#cancer_patients.remove(854)

canpat = [cancer_patients[i] for i in random.sample(range(0, len(cancer_patients)), 1)]
conpat = [control_patients[i] for i in random.sample(range(0, len(control_patients)), 4)]

patients = canpat + conpat
test_data = data[data['patient#'].isin(patients)]
train_data = data.drop(test_data.index).sample(frac=1.).reset_index(drop=True)

print('cancer patients selected: ', canpat)

test_data.to_csv('../../binary/test_binary.csv', index=False)
train_data.to_csv('../../binary/train_binary.csv', index=False)
print(train_data)
print(test_data)
