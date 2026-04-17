from idlelib.colorizer import prog_group_name_to_tag

import torch
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.utils.data as data
from ignite import metrics

from train_alternative import ManyToOneLSTM

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

batch_size = 256

X_val = np.load('processed_data/features_val-v2_onlyStrest.npy')
y_val = np.load('processed_data/labels_val-v2_onlyStrest.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)

X_val = X_val[:,:,[0,1,2,3,5]]
device = torch.device("cuda")

input_size = 5
hidden_size = 64
num_layers = 1
output_size = 3

model = ManyToOneLSTM(input_size, hidden_size, num_layers, output_size, device)
model.load_state_dict(torch.load("testiewestie.pt", weights_only=False))

accuracies = metrics.Accuracy()
recall = metrics.Recall()
confusion_matrix = metrics.ConfusionMatrix(output_size)
precision = metrics.Precision()

X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
y_val = torch.tensor(y_val, dtype=torch.long, device=device)
loader = data.DataLoader(data.TensorDataset(X_val, y_val), shuffle=False, batch_size=batch_size)


model.eval()
model.to(device)
for X_batch, y_batch in loader:
    y_pred = model(X_batch)

    accuracies.update((y_pred, y_batch))
    recall.update((y_pred, y_batch))
    precision.update((y_pred, y_batch))
    confusion_matrix.update((y_pred, y_batch))

current_acc = accuracies.compute()
print("Accuracy:", current_acc)

current_recall = recall.compute()
print("Recall:", current_recall)

current_precision = precision.compute()
print("Precison: ", current_precision)

heatmap = confusion_matrix.compute()
heatmap = heatmap.numpy()

class_names = ['Rest', 'Stress', 'Exercise']
 
# Plot the confusion matrix
plot_confusion_matrix(heatmap, classes=class_names, normalize=True)

#df = pd.DataFrame(heatmap)
#print(df.to_csv("heatmap.csv", index=False))

