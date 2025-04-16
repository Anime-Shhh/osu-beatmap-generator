import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Initialize model, dataset, and dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerDecoder().to(device)
dataset = OsuDataset('data/spectrograms', 'data/parsed_beatmaps')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for specs, tokens in dataloader:
        specs, tokens = specs.to(device), tokens.to(device)
        
        optimizer.zero_grad()
        time_logits, type_logits, grid_x_logits, grid_y_logits = model(specs, tokens)
        
        # Compute loss
        loss = (
            criterion(time_logits.view(-1, model.max_tokens), tokens[:, 1:, 0].view(-1)) +
            criterion(type_logits.view(-1, 3), tokens[:, 1:, 1].view(-1)) +
            criterion(grid_x_logits.view(-1, 16), tokens[:, 1:, 2].view(-1)) +
            criterion(grid_y_logits.view(-1, 12), tokens[:, 1:, 3].view(-1))
        ) / 4
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')