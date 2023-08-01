import torch.nn as nn

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