from idlelib.colorizer import prog_group_name_to_tag

import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data


class ManyToManyLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size):
        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers
        super(ManyToManyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softAct = nn.Softmax(dim=1)


    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        out = self.softAct(out[:,-1,:])
        return out

def acc(pred, real):
    _, pred = torch.topk(pred, k=1, dim=1)
    _, real = torch.topk(real, k=1, dim=1)
    is_correct = (pred == real).long()
    return sum(is_correct)/len(pred)


# Take a matrix of all the labeled input data
def train(X_train, y_train, X_test, y_test):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    y_train = y_train[:, -1, :]
    y_test = y_test[:, -1, :]

    _, y_test = torch.topk(y_test, k=1, dim=1)
    y_test = y_test.squeeze(1)

    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
    # Define model parameters
    input_size = 6

    hidden_size = 32
    num_layers = 2
    output_size = 4
    model = ManyToManyLSTM(input_size, hidden_size, num_layers, output_size)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # h0, c0 = None, None
    # Train the model
    n_epochs = 2000
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            _, y_batch = torch.topk(y_batch, k=1, dim=1)
            y_batch = y_batch.squeeze(1)
            # y_pred = torch.squeeze(y_pred, 2)
            # print(y_pred.shape)
            # print(y_batch.shape)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        # h0, c0 = h0.detach(), c0.detach()
        # Validation
        if epoch % 20 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            # print(y_pred.shape, y_train.shape)
            # print(y_pred[10, 100, :])
            # print(y_train[10, 100, :])
            # y_pred = torch.squeeze(y_pred, 2)
            train_loss = criterion(y_pred, y_train)
            y_pred = model(X_test)
            # y_pred = torch.squeeze(y_pred, 2)
            print(acc(y_pred, y_test))
            test_loss = criterion(y_pred, y_test)
            for i in range(y_pred.shape[0]):
                print(y_test[i])
                print(y_pred[i])
        print("Epoch %d: train :̶.̶|̶ ̶:̶;̶ %.4f, test :̶.̶|̶ ̶:̶;̶ %.4f" % (epoch, train_loss, test_loss))




X_train = np.load('processed_data/features_train_60.npy')
y_train = np.load('processed_data/labels_train_60.npy')

X_test = np.load('processed_data/features_test_60.npy')
y_test = np.load('processed_data/labels_test_60.npy')

train(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
