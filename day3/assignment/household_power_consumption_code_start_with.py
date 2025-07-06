# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# %%
# load data
df = pd.read_csv('data//household_power_consumption.txt', sep=";")
df.head()

# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis=1, inplace=True)
# handle missing values
df.dropna(inplace=True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']

# %%
# data normalization
# We will normalize the numerical columns except 'datetime'
numerical_columns = [col for col in train.columns if col != 'datetime']
train_mean = train[numerical_columns].mean()
train_std = train[numerical_columns].std()

train[numerical_columns] = (train[numerical_columns] - train_mean) / train_std
test[numerical_columns] = (test[numerical_columns] - train_mean) / train_std

# %%
# split X and y
# Let's assume we want to predict 'Global_active_power'
target_column = 'Global_active_power'
X_train = train.drop([target_column, 'datetime'], axis=1).values
y_train = train[target_column].values
X_test = test.drop([target_column, 'datetime'], axis=1).values
y_test = test[target_column].values

# %%
# create custom dataset class
class PowerConsumptionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# create dataloaders
train_dataset = PowerConsumptionDataset(X_train, y_train)
test_dataset = PowerConsumptionDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# build a LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (hn, cn) = self.lstm(x.unsqueeze(1), (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

input_size = X_train.shape[1]
hidden_size = 64
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# %%
# train the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# %%
# evaluate the model on the test set
model.eval()
test_loss = 0.0
predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        test_loss += loss.item()
        predictions.extend(outputs.squeeze().tolist())

print(f'Test Loss: {test_loss / len(test_loader)}')

# %%
# plotting the predictions against the ground truth
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Ground Truth')
plt.plot(predictions, label='Predictions')
plt.title('Predictions vs Ground Truth')
plt.xlabel('Time')
plt.ylabel('Global Active Power')
plt.legend()
plt.show()