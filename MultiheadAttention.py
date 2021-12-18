import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import math 

from torch.nn import MultiheadAttention

class MLP(nn.Module):
  def __init__(self,input_size, hidden1_size, hidden2_size, num_classes):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden1_size)
    self.sigmoid1 = nn.Sigmoid()
    self.fc2 = nn.Linear(hidden1_size, hidden2_size)
    self.sigmoid2 = nn.Sigmoid()
    self.fc3 = nn.Linear(hidden2_size, num_classes)
    self.sigmoid3 = nn.Sigmoid()
  def forward(self,x):
    out = self.fc1(x)
    out = self.sigmoid1(out)
    out = self.fc2(out)
    out = self.sigmoid2(out)
    out = self.fc3(out)
    out = self.sigmoid3(out)
    out = out.reshape(-1)
    return out
    
    
class Transformer(nn.Module):
  def __init__(self, dim, heads, dropout = 0.1):
    super().__init__()
    self.mhsa = nn.MultiheadAttention(dim, heads)
    self.drop = nn.Dropout(dropout)
    self.norm_1 = nn.LayerNorm(dim)
    self.norm_2 = nn.LayerNorm(dim)
    self.feature = 10
    self.linear = nn.Sequential(
        nn.Linear(dim,self.feature),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(self.feature,dim),
        nn.Dropout(dropout)
    )
  def forward(self, x, mask=None):
    out,_ = self.mhsa(x,x,x)
    out = self.drop(out)
    out = self.norm_1(out)
    out = self.norm_2(self.linear(out))
    return out
    

if __name__ == "__main__":
    df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')


    cat_features = [ 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    cont_features = [ 'Balance','EstimatedSalary']

    x_cat = torch.tensor(df[cat_features].values, dtype=torch.float32)
    x_cont = torch.tensor(df[cont_features].values, dtype=torch.float32)
    y = torch.tensor(df['Exited'].values, dtype=torch.float32)
    input_dim = 4
    num_heads = 1

    x_cat = x_cat.long()
    embedding = nn.Embedding(10000, 4)
    x = embedding(x_cat)

    transformer = Transformer(4,1,0.0)
    out = transformer.forward(x)

    x_categories = torch.zeros(10000,4)
    for n in range(10000):
      for i in range(4):
        for j in range(4):
          x_categories[n][i]+=out[n][i][j]/4

    inputOfMLP = torch.cat((x_categories, x_continues),1)

    model = MLP(6,10,10,1)
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion  = nn.BCELoss()
    x_train, x_test, y_train, y_test = train_test_split(inputOfMLP, y, test_size=0.2, random_state=1)
    epochs = 1
    for epoch in range(epochs):
      optimizer.zero_grad()
      #Forward pass
      y_pred = model(x_train)
      #Compute loss
      # y_pred = y_pred.reshape(-1)
      loss = criterion(y_pred,y_train)
      print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
      #Backward pass
      loss.backward(retain_graph=True)
      optimizer.step()


    model.eval()
    y_pred = model(x_test)

    ls = []
    for i in y_pred:
      if i<0.5:
        ls.append(1)
      else:
        ls.append(0)

    print(accuracy_score(ls,y_test))
