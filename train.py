import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data

# Niet echt nuttige code, wij doen many-to-many sequence labelling

class ManyToManyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ManyToManyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# Take a matrix of all the labeled input data
def train(X, y):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
    # Define model parameters
    input_size = 7

    hidden_size = 32
    num_layers = 1
    output_size = 1
    model = ManyToManyLSTM(input_size, hidden_size, num_layers, output_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    n_epochs = 2000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            y_pred = torch.squeeze(y_pred, 2)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            y_pred = torch.squeeze(y_pred, 2)
            train_rmse = np.sqrt(criterion(y_pred, y_train))
            y_pred = model(X_test)
            y_pred = torch.squeeze(y_pred, 2)
            test_rmse = np.sqrt(criterion(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))




features = np.load('processed_data/features.npy')
labels = np.load('processed_data/labels.npy')

train(features, labels)


