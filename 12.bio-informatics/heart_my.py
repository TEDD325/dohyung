import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

INPUT_DIM = 13
MY_EPOCH = 200

df = pd.read_csv('./heart.csv', header=0)
print('\n === RAW DATA === ')
pd.set_option('display.max_columns', None) # *
print(df.head(5), end="\n\n")
# print(df.describe())

df_input = df.loc[:, ~(df.columns == 'target')] # ?
''' 
loc[행, 열]
~: 제외해.
'''
df_output = df['target']

print("df_input")
print(df_input)
print("df_output")
print(df_output, end="\n\n")

names = df_input.columns.tolist()
print("names")
print(names, end="\n\n")

scaler = StandardScaler()
df_input = scaler.fit_transform(df_input)
print("df_input - NORMALIZATION")
print(df_input, end="\n\n")

df_input = pd.DataFrame(df_input, columns=names)
print("df_input - DataFrame")
print(df_input, end="\n\n")

# sns.boxplot(data=df_input, palette='colorblind')
# plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(df_input, df_output, test_size=0.3, random_state=42)
''' 
train_test_split의 shuffle은 디폴트로 True임
'''
print("X_train")
print(X_train, end="\n\n")
print("X_test")
print(X_test, end="\n\n")
print("Y_train")
print(Y_train, end="\n\n")
print("Y_test")
print(Y_test, end="\n\n")

model = nn.Sequential(
    nn.Linear(INPUT_DIM, 1000),
    nn.Tanh(),
    nn.Linear(1000, 1000),
    nn.Tanh(),
    nn.Linear(1000, 1),
    nn.Sigmoid()
)
print("model")
print(model, end="\n\n")

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss() # ?

print("optimizer")
print(optimizer, end="\n\n")
# print("criterion")
# print(criterion, end="\n\n")

X_train = torch.tensor(X_train.values).float()
Y_train = torch.tensor(Y_train.values).float()
print("X_train")
print(X_train, end="\n\n")
print("Y_train")
print(Y_train, end="\n\n")

began = time()
print("\n === Training Begins ===")
for epoch in range(0, MY_EPOCH):
    output = model(X_train)
    # print("output")
    # print(output, end="\n\n")

    output = torch.squeeze(output) # ?
    # print("output")
    # print(output, end="\n\n")

    loss = criterion(output, Y_train)
    # print("loss")
    # print(loss, end="\n\n")

    print("Epoch: ", epoch, "Loss: ", loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("\n === Training Time (in seconds) = {:.1f}".format(time() - began), end="\n\n")

X_test = torch.Tensor(X_test.values).float()
print("X_test")
print(X_test, end="\n\n")

print("\n === Test Begins ===")
with torch.no_grad(): # ?
    pred = model(X_test)

print("pred before detach()")
print(pred, end="\n\n")
print("pred after detach()")
print(pred.detach(), end="\n\n")

print("type of pred: ", type(pred))
pred = pred.detach().numpy() # detach() ?
print("type of pred: ", type(pred))
print("pred")
print(pred, end="\n\n")

pred = (pred > 0.5)
print("pred")
print(pred, end="\n\n")

print("Accuracy: ", f1_score(Y_test, pred))