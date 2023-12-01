import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import numpy as np

class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        # 1st Convolutional Layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=20, stride=1),
            nn.BatchNorm1d(32), #881 size
            nn.ReLU()
        )
        
        # 2nd Convolutional Layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=20, stride=1),
            nn.BatchNorm1d(64), #862 size
            nn.ReLU()
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(1612 * 64, 20) if self.input_size == 1650 else nn.Linear(813 * 64, 20) # 813 for input 851 or 1612 for 1650
        self.fc2 = nn.Linear(20, self.output_size)         # 5 output neurons for the 5 possible classes

        # Dropout layer with inactivation probability of 0.1
        self.dropout = nn.Dropout(0.1)

        # Softmax activation for classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutional layers
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activation
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = self.fc2(x)

        # Softmax output
        x = self.softmax(x)
        return x
    
# FUNCTION FOR TRAINING AND EVALUATING NN
def ConvolutionalNeuralNetwork(train_data, test_data):
    X_test, y_test = test_data.iloc[:,:-1], test_data['labels']
    X_train, y_train = train_data.iloc[:,:-1], train_data['labels']
    
    input_size = len(train_data.columns) - 1
    output_size = len(y_train.unique())

    alpha = 0.001 # learning rate
    batch = 175
    ep = 10 # epoch

    # Uses the gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

    y_train = onehot_train.fit_transform(train_labels.reshape(-1, 1))
    y_test = onehot_test.fit_transform(test_labels.reshape(-1, 1))

    # Change dataframes to torch.tensors
    X_train, y_train, X_test, y_test = (torch.tensor(X_train.values), torch.tensor(y_train),
                                        torch.tensor(X_test.values), torch.tensor(y_test))

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
