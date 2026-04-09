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
        self.softAct = nn.Softmax(dim=2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        # print("Hi!")
        # print(out[0,0,:])
        out = self.softAct(out)
        # print(out[0,0,:])
        return out

# Take a matrix of all the labeled input data
def train(X_train, y_train, X_test, y_test):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
    # Define model parameters
    input_size = 6

    hidden_size = 64
    num_layers = 2
    output_size = 4
    model = ManyToManyLSTM(input_size, hidden_size, num_layers, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scaler = torch.amp.GradScaler()

    # Train the model
    n_epochs = 2000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            # y_pred = torch.squeeze(y_pred, 2)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            print(y_pred.shape, y_train.shape)
            print(y_pred[10, 100, :])
            print(y_train[10, 100, :])
            # y_pred = torch.squeeze(y_pred, 2)
            train_loss = criterion(y_pred, y_train)
            y_pred = model(X_test)
            # y_pred = torch.squeeze(y_pred, 2)
            test_loss = criterion(y_pred, y_test)
        print("Epoch %d: train :̶.̶|̶ ̶:̶;̶ %.4f, test :̶.̶|̶ ̶:̶;̶ %.4f" % (epoch, train_loss, test_loss))


X_train = np.load('processed_data/features_train.npy')
y_train = np.load('processed_data/labels_train.npy')

X_test = np.load('processed_data/features_test.npy')
y_test = np.load('processed_data/labels_test.npy')

train(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
