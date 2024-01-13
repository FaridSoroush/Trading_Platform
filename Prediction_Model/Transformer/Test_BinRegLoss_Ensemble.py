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
# input_size = X_train.shape[2] # 24
output_size = 2
num_heads = 2 # should be a factor of input_size
num_layers = 2
num_epochs = 500
learning_rate = 0.001
# weight_decay = 0.001 # for Adam optimizer
weight_decay = 0.001

predicting_index = 1 # Close price

weight_price_A = 0
weight_price_B = 0
weight_price_C = 1 - weight_price_A - weight_price_B
weight_binary_A = 0.4
weight_binary_B =0.3
weight_binary_C = 1 - weight_binary_A - weight_binary_B

class Ensemble(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        
    def forward(self, x):

        xA = x[:, :, :2]  
        xC = x 
        
        outA, logitsA = self.modelA(xA)
        outB, logitsB = self.modelB(xA)
        outC, logitsC = self.modelC(xC)

        out = weight_price_A * outA + weight_price_B * outB + weight_price_C * outC
        logits = weight_binary_A * logitsA + weight_binary_B * logitsB + weight_binary_C * logitsC
        out_binary = torch.sigmoid(logits)

        print("out_binary: ", out_binary)

        return out, out_binary


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

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, y_binary):
        self.X = X
        self.y = y
        self.y_binary = y_binary

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.y_binary[idx]


device = "cpu"

# input_filename = "BTCUSDT4Y1MKline.csv"
input_filename = "BTCUSDT4Y1MKline_cleaned_24features_nSteps10.csv"
df = pd.read_csv(input_filename)
df['Open Time'] = pd.to_datetime(df['Open Time'])
df['Open Time'] = df['Open Time'].values.astype(np.int64) // 10 ** 9
df = df.sort_values('Open Time')

df=df[['Open Time', 'Close', 'Volume','Moving_Avg_Close_10X_n_steps']]
# Use the loaded scaler to transform the new data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df.values)

print("Importing, sorting, scaling: Done")

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

modelC = Transformer(4, output_size, num_heads, num_layers)
modelB = Transformer(2, output_size, num_heads, num_layers)
modelA = Transformer(2, output_size, num_heads, num_layers)
modelC.load_state_dict(torch.load('Exp10_model_parameters_10_2_2_500_128_0.001_4inputs.pth', map_location=torch.device('cpu')))
modelB.load_state_dict(torch.load('Exp9_model_parameters_10_2_2_500_128_0.001.pth', map_location=torch.device('cpu')))
modelA.load_state_dict(torch.load('Exp8_model_parameters_10_2_2_100_256_0.001.pth', map_location=torch.device('cpu')))

model = Ensemble(modelA, modelB, modelC)
model.to(device)

# switch the model to evaluation mode
model.eval()

# initialize arrays to store predictions and actual values
test_preds = []
test_actuals = []
accuracy=0
# perform the testing
with torch.no_grad():
    for i, (batch_X, batch_y_continuous, batch_y_binary) in enumerate(test_loader):
        batch_y_continuous = batch_y_continuous.float().to(device)
        batch_y_binary = batch_y_binary.to(device)
        batch_X = batch_X.to(device)
        outputs, outputs_binary = model(batch_X.float())
        accuracy += (outputs_binary.round() == batch_y_binary).float().mean()
        test_preds.append(outputs.detach().cpu().numpy())
        test_actuals.append(batch_y_continuous.detach().cpu().numpy())

# compute the accuracy
accuracy = accuracy / len(test_loader)
print(f"Accuracy on Test Data: {accuracy}")

# convert lists to numpy arrays
test_preds = np.concatenate(test_preds)
test_actuals = np.concatenate(test_actuals)

# compute mean squared error on test data
mse_test = mean_squared_error(test_actuals, test_preds)
# print(f"Mean Squared Error on Test Data: {mse_test}")

# Create dummy arrays
n_features = df.shape[1] # assuming df is your original DataFrame before scaling
dummy_preds = np.zeros((len(test_preds), n_features))
dummy_actuals = np.zeros((len(test_actuals), n_features))

# Replace the 'Close' price column with your predictions/actuals
dummy_preds[:, predicting_index] = test_preds
dummy_actuals[:, predicting_index] = test_actuals

# Now you can inverse transform
test_preds_inv = scaler.inverse_transform(dummy_preds)[:, predicting_index]
test_actuals_inv = scaler.inverse_transform(dummy_actuals)[:, predicting_index]

# # Compute mean squared error on inverse transformed (original scale) data
# mse_test_inv = mean_squared_error(test_actuals_inv, test_preds_inv)
# print(f"Mean Squared Error on Test Data (original scale): {mse_test_inv}")

# Compute errors on inverse transformed (original scale) data
errors_inv = test_actuals_inv - test_preds_inv
# Convert errors to percentage terms
percentage_errors = np.abs(errors_inv / test_actuals_inv) * 100
# Compute average of the percentage errors
average_percentage_error = np.mean(percentage_errors)
print(f"Average Error on Test Data (percentage terms): {average_percentage_error}%")
