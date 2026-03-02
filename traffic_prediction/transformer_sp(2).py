import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import csv
import time
start_time = time.time()
# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Usage:")
    print(f"Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB")
    print(f"Cached:    {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB")

# Load data
npz_data = np.load(r'C:\Users\Kelsey\PycharmProjects\computer_vision\traffic_prediction\pems08.npz')  # Update the path as necessary
data = npz_data['data']  # Shape: (17856, 170, 3)
distance_df = pd.read_csv('distance.csv')  # Update the path as necessary

# Data shape
num_time_steps, num_sensors, num_features = data.shape

# Build adjacency matrix
G = nx.Graph()
for _, row in distance_df.iterrows():
    G.add_edge(int(row['from']), int(row['to']), weight=row['cost'])
adj_matrix = nx.adjacency_matrix(G, nodelist=range(num_sensors)).todense()


# Normalize adjacency matrix
def normalize_adj(adj):
    adj = np.array(adj)
    D = np.diag(np.sum(adj, axis=1))
    D_inv = np.linalg.inv(D)
    adj_norm = D_inv.dot(adj)
    return adj_norm


adj_norm = torch.tensor(normalize_adj(adj_matrix), dtype=torch.float32).to(device)

# Normalize data
mean = data.mean(axis=(0, 1), keepdims=True)
std = data.std(axis=(0, 1), keepdims=True)
data = (data - mean) / std
data = torch.tensor(data, dtype=torch.float32).to(device)


# Define PyTorch Dataset
class TrafficDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.shape[0] - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y


dataset = TrafficDataset(data, 12)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define model components
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, adj):
        out = torch.matmul(adj, X)
        out = self.linear(out)
        return out


class SpatialExtractor(nn.Module):
    def __init__(self, num_sensors, input_dim, hidden_dim):
        super(SpatialExtractor, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

    def forward(self, X, adj):
        out = self.gcn1(X, adj)
        out = torch.relu(out)
        out = self.gcn2(out, adj)
        return out


class TemporalExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(TemporalExtractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, X):
        out, _ = self.lstm(X)
        return out


class ConvTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim, seq_len):
        super(ConvTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Embedding(seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        src = self.input_linear(src) * np.sqrt(src.size(-1))
        position = torch.arange(src.size(1), device=src.device).unsqueeze(0).repeat(src.size(0), 1)
        src = src + self.pos_encoder(position)

        # Transformer part
        transformer_out = self.transformer_encoder(src.permute(1, 0, 2))
        transformer_out = transformer_out.permute(1, 2, 0)

        # Convolution part
        conv_out = self.conv1d(src.permute(0, 2, 1))

        # Gated fusion
        combined = torch.cat([transformer_out, conv_out], dim=1)
        gate_weights = torch.sigmoid(self.gate(combined.permute(0, 2, 1)))
        fused_out = gate_weights * transformer_out.permute(0, 2, 1) + (1 - gate_weights) * conv_out.permute(0, 2, 1)

        # Final processing
        output = self.fc(fused_out[:, -1, :])  # Take the last timestep
        return output


class SpatioTemporalModel(nn.Module):
    def __init__(self, num_sensors, input_dim, hidden_dim, num_heads, num_layers, output_dim, seq_len):
        super(SpatioTemporalModel, self).__init__()
        self.spatial_extractor = SpatialExtractor(num_sensors, input_dim, hidden_dim)
        self.temporal_extractor = TemporalExtractor(hidden_dim * num_sensors, hidden_dim, num_layers)
        self.transformer = ConvTransformer(hidden_dim, hidden_dim, num_heads, num_layers, output_dim * num_sensors,
                                           seq_len)
        self.num_sensors = num_sensors
        self.output_dim = output_dim

    def forward(self, X, adj):
        batch_size, seq_len, num_sensors, num_features = X.size()
        spatial_out = []
        for t in range(seq_len):
            spatial_out.append(self.spatial_extractor(X[:, t, :, :], adj))
        spatial_out = torch.stack(spatial_out, dim=1)
        spatial_out = spatial_out.view(batch_size, seq_len, -1)
        temporal_out = self.temporal_extractor(spatial_out)
        output = self.transformer(temporal_out)
        return output.view(batch_size, self.num_sensors, self.output_dim)


# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# Initialize the model, optimizer, and loss function
model = SpatioTemporalModel(num_sensors, num_features, 64, 8, 2, num_features, 12).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train the model
num_epochs = 100
train_losses = []
test_losses = []
early_stopping = EarlyStopping(patience=10, verbose=True)

# Create CSV file to log losses
with open('training_log.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Test Loss"])

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x, adj_norm)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate the model
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                output = model(x, adj_norm)
                loss = criterion(output, y)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

        # Log to CSV
        writer.writerow([epoch + 1, avg_train_loss, avg_test_loss])

        # Early Stopping
        early_stopping(avg_test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

# Plot the losses
if len(train_losses) > 0 and len(test_losses) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the plot
    plt.savefig('loss_convergence.png')
    print("Loss convergence plot saved as 'loss_convergence.png'")
else:
    print("No losses collected. Check the training loop.")

# Print final test loss
if len(test_losses) > 0:
    print(f'Final Test Loss: {test_losses[-1]:.4f}')
else:
    print("No test losses collected. Check the evaluation in the training loop.")

# Load the best model
model.load_state_dict(torch.load('checkpoint.pt'))
print("Loaded the best model")

# Final evaluation on the test set
model.eval()
final_test_loss = 0
with torch.no_grad():
    for x, y in test_loader:
        output = model(x, adj_norm)
        loss = criterion(output, y)
        final_test_loss += loss.item()

final_avg_test_loss = final_test_loss / len(test_loader)
print(f'Final Test Loss (Best Model): {final_avg_test_loss:.4f}')

# Print final GPU memory usage
if device.type == 'cuda':
    print(f"Final GPU Memory Usage:")
    print(f"Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB")
    print(f"Cached:    {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB")

#record the training time
import time
end_time = time.time()
print("Time taken to train the model: ", end_time-start_time, "seconds")

