import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import OsuDataset
from model import TransformerDecoder
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize model, dataset, and dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerDecoder().to(device)
dataset = OsuDataset('data/spectrograms', 'data/parsed_beatmaps')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: [item for item in x if item is not None])

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    for specs, tokens in dataloader:
        if specs is None or tokens is None:
            logger.warning("Skipping invalid batch")
            continue
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
        batch_count += 1
    
    if batch_count > 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss / batch_count}')
    else:
        print(f'Epoch {epoch+1}, No valid batches processed')

# Save the model
torch.save(model.state_dict(), 'output/model.pth')