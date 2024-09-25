#   //heatmap
import torch
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt
import seaborn as sns


#import the data file as df using pandas library
df = pd.read_csv('datasets\\data.csv')


#   19 cols = 1 index + 16 features + 2 targets!
#   Since we have 16 features and 2 targets, an easy way to visualise

#   This dataset is using a heatmap!
figure, ax = plt.subplots(figsize = (20, 12))

#we remove the columns that are problematic in our heat map
df_new = df.drop(['GT Compressor inlet air temperature (T1) [C]\xa0','GT Compressor inlet air pressure (P1) [bar]\xa0 '],axis=1)
title = 'Naval Vessel, engine life heatmap'
plt.title(title, fontsize = 18)

#   We can see that the table is the same diagonally,
#       so we remove the top part using a mask

trim_mask = np.triu(np.ones_like(df_new.corr()))
sns.heatmap(df_new.corr(),
            cbar = True,
            cmap = 'Reds',
            annot = True,
            linewidth = 1,
            ax = ax,
            mask = trim_mask)
plt.show()

# Data Cleaning: Removing unnecessary columns

# Import the data file as df using pandas lib
df = pd.read_csv('datasets\\data.csv')

#   df is 19 cols 1 index + 16 features + 2 targets!
#   we remove the 2 columns that are the same

df = df.drop(['index', 'GT Compressor inlet air temperature (T1)[C]\xa0','GT Compressor inlet air pressure (P1) [bar]\xa0'],axis=1)

#   Now df is 16 cols = 14 features + 2 targets!
X, y1 = df, df[['GT Compressor decay state coefficient ',
                'GT Turbine decay state coefficient']]

#   //hyper parameters 1
seed = 1234
batch_size = 256
learning_rate = 0.000015
hidden_layers = [32, 64, 128, 256, 128, 64, 32, 16, 8, 4]
epochs = 20
criterion = nn.L1Loss()
load = 1

#   //Split and print
#   We split the data for y1 first
X_train, X_test, y_train, y_test = train_test_split(X,y1,test_size=0.2, random_state=seed)

print(X)
print(X_train)

#   //custom data class

class customdata(Dataset):
    def __init__(self, train, target):
        super(customdata, self).__init__()
        self.train = train
        self.target = target

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        datpoint = self.train.iloc[idx].values
        targ = self.target.iloc[idx].values
        datpoint = torch.from_numpy(datpoint.astype(np.float32)).to(device)
        targ = torch.tensor(targ.astype(np.float32)).to(device)
        return datpoint, targ

#   //defining the 2 objects
cd_train = customdata(X_train, y_train)
cd_test = customdata(X_test, y_test)

#   //Using Dataloader to split data in batch size and shuffle it
train_loader = torch.utils.data.DataLoader(cd_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(cd_test, batch_size=batch_size, shuffle=True)

#   //making the model structure

class MLP(nn.Module):
    def __init__(self, layers:list[int]):
        super(MLP, self).__init__()
        self.linears = []
        for i in range(len(layers)-1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
        self.linears.append(nn.ReLU())
        self.linears.pop()
        self.linears = nn.Sequential( *self.linears)

    def forward(self, x):
        for operation in self.linears:
            x = operation(x)
        return x

#hyper parameters 2
input, output = [a.shape[-1] for a in next(iter(train_loader))]

print(f'input={input} and output={output}')

model = MLP(layers=[input, *hidden_layers, output]).to(device)

#   //defining optimizer and scheduler, the load parameter, and training
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

num_totalsteps = len(train_loader)
if load:
    model = torch.load('second_trained_model_LRMNV1.pth').to(device)
    print('model loaded')

for epoch in range(epochs):
    pbar = tqdm(enumerate(iter(train_loader)))

    for i, (data, target) in pbar:
        y_pred = model(data)
        loss = criterion(y_pred, target)
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f'epoch[{epoch+1}/{epochs}],
                            step[{i+1}/{num_totalsteps}],
                            loss: {loss.item():.4f},
                            learningrate = {learning_rate}')
    validateparam = validate(model)
    scheduler.step(torch.tensor(validateparam).sum())

torch.save(model.cpu(), 'second_trained_model_LRMNV1.pth')
#finished
model.to(device)
print(validate(model))

#   //new optimizer for training
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#   //Loading model and printing different evaluations
model = torch.load('second_trained_model_LRMNV1.pth').to(device)

#print(f'the model is {model}')
#print(f'the model parameters are = {list(model.parameters())}')
#print(f'model.statedict() = {model.state_dict()[[]].items()}')
#print(f'model = {model.eval()}')

summary(model)
print(test_loader)

def validate(model):
    predictions = []
    targets = []
    for data, target in iter(all_loader):
        outputs = model(data)
        predictions.append(outputs)
        targets.append(target)
    predictions = torch.cat(predictions, dim=0).movedim(0,-1)
    targets = torch.cat(targets, dim=0).movedim(0, -1)

    print(f'targets = {targets}')
    print(f'preds = {predictions}')

    eval1 = [nn.functional.mse_loss(pred, targ).item() for pred, targ in zip(predictions, targets)]
    eval2 = [nn.functional.soft_margin_loss(pred, targ).item() for pred, targ in zip(predictions, targets)]
    eval3 = [nn.functional.l1_loss(pred, targ).item() for pred, targ in zip(predictions, targets)]
    #print(f'eval1 = {eval1}, eval2 = {eval2}, eval3 = {eval3}')
    return predictions

predictions = validate(model)

print(f'dataset index 1 = \n{X.iloc[1]}')
print(predictions[[1],[1]].item())
print(f'predictions for target 1 = {predictions[[0], [1]].item()} and target 2 = {predictions[[1],[1]].item()}')