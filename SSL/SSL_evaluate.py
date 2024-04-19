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

model = SectionPredictionModel()
model.load_state_dict(torch.load("D:\preprocess-2.0/trained_model.pth"))
model.eval()

test_data_folder = 'D:\preprocess-2.0\muspy_numpy_dir'
labels_file = 'D:\preprocess-2.0\SSL/test_list.txt'
test_file_list = os.listdir(test_data_folder)

test_dataset = MusicDataset(test_data_folder, labels_file)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for i, (test_data, _) in enumerate(test_dataloader):
    with torch.no_grad():

        predictions = model(test_data)

    s_1, s_2, s_3 = predictions.squeeze().tolist()
    s_1, s_2, s_3 = sorted([s_1, s_2, s_3])

    total_length = test_data.size(1)
    s_1_position = int(s_1 * total_length)
    s_2_position = int(s_2 * total_length)
    s_3_position = int(s_3 * total_length)

    print(f"Example {i+1}:")
    print("Total Length of the Piece:", total_length)
    print("Exposition ends at beat:", s_1_position)
    print("Development ends at beat:", s_2_position)
    print("Recapitulation ends at beat:", s_3_position)
    print('-' * 30)
