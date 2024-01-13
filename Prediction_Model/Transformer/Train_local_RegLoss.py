import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from joblib import dump, load
from tqdm import tqdm
import time
from torch.nn import DataParallel
import os

# Hyperparameters
batch_size = 8
n_steps = 10
# input_size = X_train.shape[2] # 24
output_size = 2
num_heads = 2 # should be a factor of input_size
num_layers = 8
num_epochs = 100
learning_rate = 0.0001
# weight_decay = 0.001 # for Adam optimizer
weight_decay = 0

predicting_index = 1 # Close price

print("batch_size =", batch_size)
print("n_steps =", n_steps)
print("output_size =", output_size)
print("num_heads =", num_heads)
print("num_layers =", num_layers)
print("num_epochs =", num_epochs)
print("learning_rate =", learning_rate)
# print("weight_decay =", weight_decay)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if device == torch.device("cuda"):
    input_filename = "BTCUSDT4Y1MKline_cleaned_24features_nSteps"+str(n_steps)+".csv"
elif device == torch.device("cpu"):
    os.chdir('/Users/faridsoroush/Documents/GitHub/Trading-Software/Prediction_Model/Transformer/')
    # print("Current working directory: {0}".format(os.getcwd()))
    input_filename = "BTCUSDT4Y1MKline_cleaned_24features_nSteps"+str(n_steps)+".csv"
else:
    print("Device not found")

# Read the data for the last 6 months
# 1 months * 30 days * 24 hours * 60 minutes = 43200
# 15 days * 24 hours * 60 minutes = 21600
# 1.5 days = 2160
length = 2048
df = pd.read_csv(input_filename)
# end=1957687
end=1957687
start=end-length
#  df = df.iloc[start:end]
df = df.iloc[start:end]

print("start",start)
print("end",end)
print("df shape",df.shape)

# df=df[['Open Time', 'Close', 'Volume', 'Moving_Avg_Close_10X_n_steps']]
df=df[['Open Time', 'Close']]
# print(df.iloc[end-5:end])
# print(df.shape)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, y_binary):
        self.X = X
        self.y = y
        self.y_binary = y_binary

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.y_binary[idx]

class Transformer(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_size, input_size)
        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.embedding(x.float())
        x = x.permute(1, 0, 2)
        tgt = torch.zeros_like(x).to(device)
        out = self.transformer(x, tgt=tgt)
        out = out.permute(1, 0, 2)
        out = self.fc(out[:, -1, :])
        return out[:, 0], torch.sigmoid(out[:, 1])

# Prepare the data for Transformer
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df.values)
# save the scaler
dump(scaler, 'scaler.joblib') 

X = []
y = []
y_binary = []

for i in range(n_steps, len(scaled_data)-1):
    X.append(scaled_data[i - n_steps:i])
    y.append(scaled_data[i, predicting_index])
X = np.array(X)
X = torch.from_numpy(X).float().to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

for i in range(n_steps, len(scaled_data) - 1):
    y_binary.append(int(scaled_data[i + 1, predicting_index] > scaled_data[i, predicting_index]))
y_binary = torch.tensor(y_binary, dtype=torch.float32).to(device)


train_size = int(len(X) * 0.9)
val_size = int(len(X) * 0.09) # additional
# adjusted indices to create validation set
X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]
y_train_binary, y_val_binary, y_test_binary = y_binary[:train_size], y_binary[train_size:train_size + val_size], y_binary[train_size + val_size:]

# Convert data to PyTorch tensors and create data loaders
train_dataset = TimeSeriesDataset(X_train, y_train, y_train_binary)
val_dataset = TimeSeriesDataset(X_val, y_val, y_val_binary)
test_dataset = TimeSeriesDataset(X_test, y_test, y_test_binary)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = X_train.shape[2]
model = Transformer(input_size, output_size, num_heads, num_layers)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model)
model.to(device)

criterion_regression = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

model.train()
best_val_loss = float('inf')
best_val_accuracy=1

for epoch in range(num_epochs):
    train_accuracy = 0
    total_loss = 0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for i, (batch_X, batch_y, batch_y_binary) in loop:
        optimizer.zero_grad()
        batch_y = batch_y.float().to(device)
        batch_X = batch_X.to(device)
        outputs, outputs_binary = model(batch_X.float())
        loss_regression = criterion_regression(outputs, batch_y)
        loss = loss_regression
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # update progress bar with current loss
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss = loss.item())

    # Print average loss after every epoch
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}')

    # Validate every X epochs and save the best model
    if (epoch + 1) % 5 == 0:
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_true_positives = 0
        val_predicted_positives = 0
        val_loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=True)
        with torch.no_grad():
            for i, (batch_X, batch_y_continuous, batch_y_binary) in val_loop:
                batch_y_continuous = batch_y_continuous.float().to(device)
                batch_y_binary = batch_y_binary.to(device)
                batch_X = batch_X.to(device)
                outputs, outputs_binary = model(batch_X.float())
                loss_regression = criterion_regression(outputs, batch_y_continuous)
                loss = loss_regression
                val_loss += loss.item()

                val_true_positives += ((outputs_binary > 0.6) & (batch_y_binary == 1)).float().sum()
                val_predicted_positives += (outputs_binary > 0.6).float().sum()

                val_loop.set_description(f"Validation Epoch [{epoch+1}/{num_epochs}]")
                val_loop.set_postfix(val_loss = loss.item())

        val_loss /= len(val_loader)

        validation_precision = val_true_positives / (val_predicted_positives + 1e-10) # added a small value to avoid division by zero
        # print(f'Validation accuracy: {val_accuracy}, Validation precision: {validation_precision}')

        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy

            print(f'Saving the new best model at epoch {epoch+1} with validation loss {best_val_loss}')
            output_filename = f'model_parameters_{n_steps}_{num_layers}_{num_heads}_{num_epochs}_{batch_size}_{learning_rate}.pth'
            torch.save(model.state_dict(), output_filename)
        else:
            print(f'Validation loss did not improve from {best_val_loss}')
            print(f'Current validation accuracy: {val_accuracy}')

print("Training finished. Loading best model for evaluation...")

# Load the best model
model_path = output_filename
model = Transformer(input_size, output_size, num_heads, num_layers).to(device)
model.load_state_dict(torch.load(model_path))

# Testing
model.eval()
test_preds = []
test_actuals = []
with torch.no_grad():
    test_accuracy = 0
    for i, (batch_X, batch_y_continuous, batch_y_binary) in enumerate(test_loader):
        batch_y_continuous = batch_y_continuous.float().to(device)
        batch_y_binary = batch_y_binary.to(device)
        batch_X = batch_X.to(device)
        outputs, outputs_binary = model(batch_X.float())
        print(f'Test accuracy: {test_accuracy}')
        test_preds.append(outputs.detach().cpu().numpy())
        test_actuals.append(batch_y_continuous.detach().cpu().numpy())

test_preds = np.concatenate(test_preds)
test_actuals = np.concatenate(test_actuals)

mse_test = mean_squared_error(test_actuals, test_preds)
print(f"Mean Squared Error on Test Data: {mse_test}")
print(f"Average Test Accuracy: {ave_test_accuracy}")