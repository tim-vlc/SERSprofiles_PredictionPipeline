import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim


from nn import NN


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

train_type = 'processed' # 'processed' or 'augmented'
input_size = 851

train_data = pd.read_csv('../../CSVs/augmented_data/gan_train_data.csv') if train_type == 'augmented' else pd.read_csv('../../CSVs/processed_data/train_data.csv')
test_data = pd.read_csv('../../CSVs/processed_data/test_data.csv')

X_test, y_test = test_data.iloc[:,:-1], test_data.iloc[:,-1]
X_train = train_data.iloc[:,1:-1] if train_type == 'augmented' else train_data.iloc[:,:-1]
y_train = train_data.iloc[:,-1]

device = torch.device("cuda:0")

output_size = 5
dense1_output = 512
dense2_output = 256
dense3_output = 64
dense4_output = 20

dropratio = 0.15
alpha = 0.0001 # learning rate
batch = 10
ep = 150 # epoch
    
model = NN(input_size, output_size, dense1_output, dense2_output, dense3_output, dense4_output, dropratio)
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

best_loss = float('inf')
best_model_state_dict = None
epochs_without_improvement = 0
patience = 20

# Train the NN
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
    
    # Print the average loss for the epoch
    avg_loss = running_loss / (len(X_train) / batch)
    print(f'Epoch [{epoch+1}/{ep}], Loss: {avg_loss:.4f}')


# Calculate the accuracy
correct = 0
total = 0
raw_outputs = []
prediction_list = []
labels_list=[]

outputs = (model(X_test.clone().detach().float().to(device))).detach().cpu().numpy()
predicted = np.argmax(outputs, 1)

truth = np.argmax(y_test, 1).detach().numpy()

total += truth.shape[0]
correct += (predicted == truth).sum().item()

raw_outputs.append(outputs)
prediction_list.append(predicted)
labels_list.append(truth)

print('Accuracy of the network on the test data: %f %%' % (
    100 * correct / total))

print('Saving model')
torch.save(model.state_dict(), f"../saved_models/nn_model.pth")
print('Saved.')
