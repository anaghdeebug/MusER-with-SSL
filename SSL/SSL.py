import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

from x_transformers.x_transformers import (
    TokenEmbedding,
)

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(AbsolutePositionalEmbedding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        max_len = x.size(1)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(np.log(10000.0) / self.d_model))
        sin_part = torch.sin(position * div_term)
        cos_part = torch.cos(position * div_term)
        pos_encoding = torch.zeros(1, max_len, self.d_model).to(x.device)
        pos_encoding[:, :, 0::2] = sin_part
        pos_encoding[:, :, 1::2] = cos_part
        return pos_encoding

class SectionPredictionModel(nn.Module):
    def __init__(self, emb_dim=256):
        super(SectionPredictionModel, self).__init__()

        self.pos_emb = AbsolutePositionalEmbedding(emb_dim)
        self.n_tokens = [3, 2048, 13, 129, 128, 33, 2]
        self.token_emb = nn.ModuleList([TokenEmbedding(emb_dim, n) for n in self.n_tokens])
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=8, num_encoder_layers=8)
        self.fc = nn.Linear(emb_dim, 3)

    def forward(self, seq):
        pos_emb = self.pos_emb(seq)
        x = sum(emb(seq[..., i]) for i, emb in enumerate(self.token_emb)) + pos_emb
        x = self.transformer(x, x)  
        x = x.mean(dim=1)  
        output = self.fc(x)
        output = torch.sigmoid(output)
        return output

class MusicDataset(Dataset):
    def __init__(self, data_folder, labels_file):
        self.data_folder = data_folder
        with open(labels_file, 'r') as file:
            self.labels = [float(line.strip()[:-3]) for line in file.readlines()]

        self.file_list = os.listdir(data_folder)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_folder, file_name)
        data = np.load(file_path)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return data, label

data_folder = "D:\preprocess-2.0/train_dataset_7"
labels_file = "D:\preprocess-2.0\SSL/train_list.txt"

dataset = MusicDataset(data_folder, labels_file)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# target_labels = torch.tensor([0.3, 0.6, 0.9], dtype=torch.float32).to(device)
random_labels = torch.rand((len(dataloader.dataset), 3)).to(device) * 0.1 + torch.tensor([0.3, 0.6, 0.9]).to(device)

model = SectionPredictionModel().to(device) 

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3  
for epoch in range(num_epochs):
    model.train()
    for i, (batch_data, _) in enumerate(dataloader):

        batch_data = batch_data.to(device)

        optimizer.zero_grad()

        batch_random_labels = random_labels[i * batch_data.size(0): (i + 1) * batch_data.size(0)]

        predictions = model(batch_data)

        loss = criterion(predictions.squeeze(), batch_random_labels)

        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Batch Loss: {loss.item()}')

torch.save(model.state_dict(), "D:\preprocess-2.0/trained_model.pth")