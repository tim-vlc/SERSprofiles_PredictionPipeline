import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from cnn import CNN

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

type_ = 'processed'
input_size = 1650 if type_ == 'raw' else 851
ratio = 0.8

# IMPORT DATA
path_to_file = '../../CSVs/diabetes.csv'

data = pd.read_csv(path_to_file)

train_data = data.sample(frac=ratio, random_state=42)
test_data = data.drop(train_data.index)

device = torch.device("cuda:0")

X_test, y_test = test_data.iloc[:,:-1], test_data['labels']
X_train, y_train = train_data.iloc[:,:-1], train_data['labels']

output_size = 2

dropratio = 0.15
alpha = 0.0001 # learning rate
batch = 175
ep = 10 # epoch
    
model = CNN(input_size, output_size)
model.to(device)

criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=alpha)

#OneHot the labels
train_encoder = LabelEncoder()
test_encoder = LabelEncoder()

train_labels = train_encoder.fit_transform(y_train)
test_labels = test_encoder.fit_transform(y_test)

onehot_train = OneHotEncoder(sparse_output=False)
onehot_test = OneHotEncoder(sparse_output=False)

label_train = y_train
label_test = y_test

y_train = onehot_train.fit_transform(train_labels.reshape(-1, 1))
y_test = onehot_test.fit_transform(test_labels.reshape(-1, 1))

# Change dataframes to torch.tensors
X_train, y_train, X_test, y_test = (torch.tensor(X_train.values), torch.tensor(y_train),
                                    torch.tensor(X_test.values), torch.tensor(y_test))

best_loss = float('inf')
best_model_state_dict = None
epochs_without_improvement = 0
patience = 4

# Train the CNN
for epoch in range(ep):
    model.train()
    running_loss = 0.0
    for i in range(0, len(X_train), batch):
        batch_X = X_train[i:i+batch].clone().detach().float().to(device)
        batch_y = y_train[i:i+batch].clone().detach().float().to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        torch.cuda.empty_cache()
    
    # Print the average loss for the epoch
    avg_loss = running_loss / (len(X_train) / batch)
    print(f'Epoch [{epoch+1}/{ep}], Loss: {avg_loss:.4f}')

# Calculate the accuracy
correct = 0
total = 0

full_truth = np.array([])
full_outputs = np.array([])
full_predicted = np.array([])

with torch.no_grad():
    for i in range(0, len(X_test), batch):
        batch_X = X_test[i:i+batch].clone().detach().float().to(device)
        batch_y = y_test[i:i+batch].clone()
        outputs = (model(batch_X)).detach().cpu().numpy()

        if i == 0:
            full_outputs = outputs
        else:
            full_outputs = np.concatenate((full_outputs, outputs))

        torch.cuda.empty_cache()

        predicted = np.argmax(outputs, 1)
        truth = np.argmax(batch_y, 1).detach().numpy()

        full_predicted = np.concatenate((full_predicted, predicted))
        full_truth = np.concatenate((full_truth, truth))

        total += truth.shape[0]
        correct += (predicted == truth).sum().item()

print('Accuracy of the network on the test data: %f %%' % (
    100 * correct / total))

#print('Saving model')
#torch.save(model.state_dict(), f"../saved_models/cnn_model.pth")
#print('Saved.')

raw_outputs = []
prediction_list = []
labels_list=[]

raw_outputs.append(full_outputs)
prediction_list.append(full_predicted)
labels_list.append(full_truth)

# Create a dictionary to map encoded labels to original labels
label_mapping = dict(zip(full_truth, label_test))
labels_sorted = dict(sorted(label_mapping.items())).values()

probabilities = np.concatenate([np.asarray((i)) for i in raw_outputs])
prediction_array = np.concatenate([np.asarray(i) for i in prediction_list])
labels_array = np.concatenate([np.asarray(i) for i in labels_list])

test_truth = np.argmax(y_test, 1)

labels = list(np.unique(test_truth))

confusion_array = confusion_matrix(labels_array, prediction_array, labels=labels, sample_weight=None, normalize=None)

plt.figure(figsize = (10,7))
sn.heatmap(confusion_array, annot=True, fmt='d',cmap="OrRd")
plt.xticks([0.5,1.5],labels=labels_sorted)
plt.yticks([0.45,1.45],labels=labels_sorted)
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.title("Confusion matrix of trained CNN for Diabetes SERS Profiles")

plt.savefig('confmat_CNN_diabetes.png')
