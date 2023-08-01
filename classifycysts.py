import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn

from cnn import CNN


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

train_data = pd.read_csv('augmented_train_data.csv')
test_data = pd.read_csv('augmented_test_data.csv')

X_test, y_test = test_data.iloc[:,:-1], test_data.iloc[:,-1]
X_train, y_train = train_data.iloc[:,:-1], train_data.iloc[:,-1]

input_size = 900
output_size = 5

dropratio = 0.15
alpha = 0.0001 # learning rate
batch = 175
ep = 20 # epoch
    
model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=alpha)

#OneHot the labels
train_encoder = LabelEncoder()
test_encoder = LabelEncoder()

train_labels = train_encoder.fit_transform(y_train)
test_labels = test_encoder.fit_transform(y_test)

onehot_train = OneHotEncoder(sparse=False)
onehot_test = OneHotEncoder(sparse=False)

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
        batch_X = torch.tensor(X_train[i:i+batch], dtype=torch.float32)
        batch_y = torch.tensor(y_train[i:i+batch], dtype=torch.float32)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print the average loss for the epoch
    avg_loss = running_loss / (len(X_train) / batch)
    print(f'Epoch [{epoch+1}/{ep}], Loss: {avg_loss:.4f}')


# Calculate the accuracy
correct = 0
total = 0
raw_outputs = []
prediction_list = []
labels_list=[]

outputs = (model(torch.tensor(X_test, dtype=torch.float32))).detach().numpy()
predicted = np.argmax(outputs, 1)

truth = np.argmax(y_test, 1).detach().numpy()

total += truth.shape[0]
correct += (predicted == truth).sum().item()

raw_outputs.append(outputs)
prediction_list.append(predicted)
labels_list.append(truth)

print('Accuracy of the network on the test data: %f %%' % (
    100 * correct / total))