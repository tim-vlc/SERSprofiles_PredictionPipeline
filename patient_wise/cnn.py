# IMPORT RELEVANT LIBRARIES
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

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
        self.fc1 = nn.Linear(813 * 64, 20)
        self.fc2 = nn.Linear(20, self.output_size)

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
        x = self.dropout(nn.LeakyReLU()(self.fc1(x)))
        x = self.fc2(x)

        # Softmax output
        x = self.softmax(x)
        return x
    
# FUNCTION FOR TRAINING AND EVALUATING CNN
def ConvolutionalNeuralNetwork(train_data, test_data):
    X_test, y_test = test_data.iloc[:,:-1], test_data['labels']
    X_train, y_train = train_data.iloc[:,:-1], train_data['labels']

    input_size = len(train_data.columns) - 1
    output_size = len(y_train.unique())

    alpha = 0.0001 # learning rate
    batch = 175
    ep = 20 # epoch

    # Uses the gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device used: ", device)

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
    num_labels = len(labels)
    tickx = list(np.linspace(0.5, num_labels - 0.5, num_labels))
    ticky = list(np.linspace(0.45, num_labels - 0.55, num_labels))

    conf_mat = confusion_matrix(labels_array, prediction_array, labels=labels, sample_weight=None, normalize=None)

    # Set the font size for the labels, title, and text within the boxes
    sn.set(font_scale=1.5)

    # Set the font style for labels and title
    sn.set(font='sans-serif')

    # Create a heatmap with customizations
    plt.figure(figsize=(12, 7))
    sn.heatmap(conf_mat, annot=True, fmt='d', cmap="Blues", cbar_kws={'label': 'Number of guesses'}, annot_kws={'fontsize':11, 'fontweight':'bold'})
    plt.xticks(tickx, labels=[f'${label}$' for label in labels_sorted])
    plt.yticks(ticky, labels=[f'${label}$' for label in labels_sorted])
    plt.ylabel('True label', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=16, fontweight='bold')

    plt.title("Confusion matrix of trained CNN", fontsize=20, fontweight='bold')
    plt.savefig('Confmat_CNN.png', dpi=200)
    plt.show()

    return model, probabilities