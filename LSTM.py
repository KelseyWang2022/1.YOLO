import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
start_time= time.time()
# Load data
data = np.load(r'C:\Users\Kelsey\PycharmProjects\computer_vision\staeformer\data\PEMS08\data.npz')['data']
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, data.shape[2])).reshape(data.shape)

# Function to create sequences
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in)].reshape(n_steps_in, -1))  # Reshape to merge site and features
        y.append(data[(i + n_steps_in):(i + n_steps_in + n_steps_out)].reshape(n_steps_out, -1))
    return np.array(X), np.array(y)

n_steps_in, n_steps_out = 12, 12
X, y = create_sequences(data_scaled, n_steps_in, n_steps_out)

# Split and convert to tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

# Model
class BiLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)  # Apply the linear layer to each time step
        return out

model = BiLSTM(X_train.shape[2], 256, 4, y_train.shape[2]).cuda()  # Adjust for your device

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU if using one
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Train the model

train_model(model, train_loader, criterion, optimizer, 10)
end_time = time.time()
print('Time:',end_time-start_time)
# Save the model
torch.save(model.state_dict(), 'model.pth')

# Predictions
model.eval()
with torch.no_grad():
    predictions = model(X_test_t.cuda())
    print(predictions)
