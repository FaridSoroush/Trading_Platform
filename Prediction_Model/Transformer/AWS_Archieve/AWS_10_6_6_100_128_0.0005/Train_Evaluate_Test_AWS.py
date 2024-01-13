import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from joblib import dump, load

# Hyperparameters
batch_size = 128
n_steps = 10
# input_size = X_train.shape[2] # 12
output_size = 1
num_heads = 6
num_layers = 6
num_epochs = 100
learning_rate = 0.001
predicting_index = 4

#########################################################################################
# Train a Simple Transformer based on BTC/USDT for the past 6 years, every 1 min, Kline #
#########################################################################################

#Sample Data Input:
# Open Time,Open,High,Low,Close,Volume,Close Time,Quote Asset Volume,Number of Trades,Taker Buy Base Asset Volume,Taker Buy Quote Asset Volume,Ignore
# 2022-06-22 18:52:00,20225.73000000,20225.73000000,20225.73000000,20225.73000000,0.03700000,1655923979999,748.35201000,2,0.03700000,748.35201000,0
# 2022-06-22 18:53:00,20230.57000000,20230.57000000,20204.21000000,20209.86000000,1.22668200,1655924039999,24797.38425961,32,0.17690200,3575.93964732,0
# 2022-06-22 18:54:00,20206.52000000,20220.85000000,20199.92000000,20209.78000000,0.44545500,1655924099999,9002.12230745,22,0.23799000,4810.13309021,0

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
        return out

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load, convert and sort the data
input_filename = "BTCUSDT6Y1MKline.csv"
df = pd.read_csv(input_filename)
df['Open Time'] = pd.to_datetime(df['Open Time'])
df['Open Time'] = df['Open Time'].values.astype(np.int64) // 10 ** 9
df = df.sort_values('Open Time')
print("Importing and sorting: Done")

# Prepare the data for Transformer
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)
# save the scaler
dump(scaler, 'scaler.joblib') 

X = []
y = []
for i in range(n_steps, len(scaled_data)):
    X.append(scaled_data[i - n_steps:i])
    y.append(scaled_data[i, predicting_index])
X = np.array(X)
X = torch.from_numpy(X).float().to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1) # additional
# adjusted indices to create validation set
X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Convert data to PyTorch tensors and create data loaders
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val) # additional
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # additional
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = X_train.shape[2]
model = Transformer(input_size, output_size, num_heads, num_layers).to(device)
print("Instantiating the Transformer Model: Done")

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training Transformer started:")

# Train the Transformer model
model.train()

best_val_loss = float('inf')

for epoch in range(num_epochs):
    total_loss = 0

    for i, (batch_X, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_y = batch_y.float().to(device)  # Convert batch_y to float type and move to device
        batch_X = batch_X.to(device)  # Move batch_X to device
        outputs = model(batch_X.float()).squeeze(1)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Print progress every 1000 batches
        # if (i + 1) % 1000 == 0:  
        #     print(f'Epoch: {epoch + 1}/{num_epochs}, Batch: {i + 1}/{len(train_loader)}, Loss: {loss.item()}')

    # Print average loss after every epoch
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}')

    # Evaluate every 5 epochs and save the best model
    if (epoch + 1) % 5 == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_y = batch_y.float().to(device)
                batch_X = batch_X.to(device)
                outputs = model(batch_X.float()).squeeze(1)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        model.train()  # switch back to training mode
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Saving the new best model at epoch {epoch+1} with loss {best_val_loss}')
            output_filename = f'model_parameters_{n_steps}_{num_layers}_{num_heads}_{num_epochs}_{batch_size}_{learning_rate}.pth'
            torch.save(model.state_dict(), output_filename)

print("Training finished. Loading best model for evaluation...")

# Load the best model
model_path = output_filename
model = Transformer(input_size, output_size, num_heads, num_layers).to(device)
model.load_state_dict(torch.load(model_path))

# Evaluation
model.eval()
test_preds = []
test_actuals = []
with torch.no_grad():
    for i, (batch_X, batch_y) in enumerate(test_loader):
        batch_y = batch_y.float().to(device)
        batch_X = batch_X.to(device)
        outputs = model(batch_X.float()).squeeze(1)
        test_preds.append(outputs.detach().cpu().numpy())
        test_actuals.append(batch_y.detach().cpu().numpy())

test_preds = np.concatenate(test_preds)
test_actuals = np.concatenate(test_actuals)

mse_test = mean_squared_error(test_actuals, test_preds)
print(f"Mean Squared Error on Test Data: {mse_test}")
