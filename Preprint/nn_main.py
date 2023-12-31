import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import numpy as np


class NN(nn.Module):
    def __init__(self, input_size, output_size, dense1_output, dense2_output, dense3_output, dense4_output, dropratio):
        super(NN, self).__init__()
        
        self.dense1 = nn.Linear(input_size, dense1_output)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropratio)
        
        self.dense2 = nn.Linear(dense1_output, dense2_output)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropratio)
        
        self.dense3 = nn.Linear(dense2_output, dense3_output)
        self.relu3 = nn.ReLU()
        
        self.dense4 = nn.Linear(dense3_output, dense4_output)
        self.relu4 = nn.ReLU()
        
        self.final = nn.Linear(dense4_output, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.relu3(x)
        x = self.dense4(x)
        x = self.relu4(x)
        x = self.final(x)
        x = self.softmax(x)
        return x
    

# FUNCTION FOR TRAINING AND EVALUATING NN
def NeuralNetwork(train_data, test_data):
    X_test, y_test = test_data.iloc[:,:-1], test_data['labels']
    X_train, y_train = train_data.iloc[:,:-1], train_data['labels']
    
    input_size = len(train_data.columns) - 1
    output_size = len(y_train.unique())
    
    dense1_output = 64
    dense2_output = 32
    dense3_output = 16
    dense4_output = 8

    dropratio = 0.15
    alpha = 0.0001 # learning rate
    batch = 10
    ep = 2 # epoch
    
    # Uses the gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

    y_train = onehot_train.fit_transform(train_labels.reshape(-1, 1))
    y_test = onehot_test.fit_transform(test_labels.reshape(-1, 1))

    # Change dataframes to torch.tensors
    X_train, y_train, X_test, y_test = (torch.tensor(X_train.values), torch.tensor(y_train),
                                        torch.tensor(X_test.values), torch.tensor(y_test))

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

    # Calculate the accuracy of the network
    full_truth = np.array([])
    full_outputs = np.array([])
    full_predicted = np.array([])
    
    correct = 0
    total = 0

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
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test data: %f %%' % (
        accuracy))
    return accuracy