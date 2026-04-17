from idlelib.colorizer import prog_group_name_to_tag

import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from ignite import metrics


class ManyToOneLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, device):
        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers
        super(ManyToOneLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, device=device, dropout=0.1)
        self.lstm2 = nn.LSTM(hidden_dim, int(hidden_dim/2), 1, batch_first=True, device=device, dropout=0.1)
        self.fc = nn.Linear(int(hidden_dim/2), output_size)
        self.softAct = nn.Softmax(dim=1)


    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out = self.fc(out)
        out = self.softAct(out[:,-1,:])
        return out

def acc(pred, real):
    _, pred = torch.topk(pred, k=1, dim=1)
    pred = pred.squeeze(1)
    # _, real = torch.topk(real, k=1, dim=1)
    is_correct = (pred == real).long()
    return sum(is_correct)/len(pred)


# Take a matrix of all the labeled input data
def train(X_train, y_train, X_test, y_test, device, batch_size=8):
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    # y_train = y_train[:, -1, :]
    # y_test = y_test[:, -1, :]

    # _, y_test = torch.topk(y_test, k=1, dim=1)
    # y_test = y_test.squeeze(1)

    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    # Define model parameters
    input_size = 5

    hidden_size = 64
    num_layers = 1
    output_size = 3
    model = ManyToOneLSTM(input_size, hidden_size, num_layers, output_size, device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.90, weight_decay=1e-5)#, nesterov=True)

    # h0, c0 = None, None
    # Train the model
    topAcc = 0
    n_epochs = 200
    accuracies = metrics.Accuracy()
    recall = metrics.Recall()
    confusion_matrix = metrics.ConfusionMatrix(output_size)
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            # _, y_batch = torch.topk(y_batch, k=1, dim=1)
            # y_batch = y_batch.squeeze(1)
            # y_pred = torch.squeeze(y_pred, 2)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        # h0, c0 = h0.detach(), c0.detach()
        # Validation
        if epoch % 10 != 0:
            continue
        model.eval()
        
        confusion_matrix.reset()
        with torch.no_grad():
            accuracies.reset()
            y_pred = model(X_train)
            # y_pred = torch.squeeze(y_pred, 2)
            train_loss = criterion(y_pred, y_train)
            y_pred = model(X_test)
            print(y_pred.shape)
            # y_pred = torch.squeeze(y_pred, 2)
            test_loss = criterion(y_pred, y_test)

            print(y_test[-1])
            print(y_pred[-1])
            accuracies.update((y_pred, y_test))
            current_acc = accuracies.compute() #acc(y_pred, y_test)

            print("Accuracy:", current_acc)
            
            recall.reset()
            recall.update((y_pred, y_test))
            current_recall = recall.compute()
            print("recall:", current_recall)

            confusion_matrix.update((y_pred, y_test))
            


        print("Epoch %d: train :̶.̶|̶ ̶:̶;̶ %.4f, test :̶.̶|̶ ̶:̶;̶ %.4f" % (epoch, train_loss, test_loss))
    print("Accuracy:", current_acc)

    confusion = confusion_matrix.compute()
    print("Confusion matrix: ", confusion)
    topAcc = current_acc
    torch.save(model.state_dict(), "./models/best_model"+str(topAcc)+".pt")

if __name__ == "__main__":
    batch_size = 256

    X_train = np.load('processed_data/features_train-v2_onlyStrest.npy')
    y_train = np.load('processed_data/labels_train-v2_onlyStrest.npy')

    X_test = np.load('processed_data/features_test-v2_onlyStrest.npy')
    y_test = np.load('processed_data/labels_test-v2_onlyStrest.npy')

    print(y_test.shape)
    # print(y_test[0,0,:])

    X_train = X_train[:,:,[0,1,2,3,5]]
    X_test = X_test[:,:,[0,1,2,3,5]]
    device = torch.device("cuda")

    train(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, batch_size=batch_size, device=device)