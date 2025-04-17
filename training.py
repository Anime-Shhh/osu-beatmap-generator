import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import OsuDataset
from model import TransformerDecoder
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collate_fn(batch):
    valid_items = [item for item in batch if item is not None]
    if not valid_items:
        return None
    specs = torch.stack([item[0] for item in valid_items])
    tokens = torch.stack([item[1] for item in valid_items])
    return specs, tokens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerDecoder(max_tokens=1000).to(device)
dataset = OsuDataset('data/spectrograms', 'data/parsed_beatmaps', max_tokens=1000)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    for batch in dataloader:
        if batch is None:
            logger.warning("Skipping invalid batch")
            continue
        specs, tokens = batch
        specs, tokens = specs.to(device), tokens.to(device)
        
        # Clamp tokens to prevent invalid indices
        tokens[:, :, 0] = torch.clamp(tokens[:, :, 0], 0, model.max_tokens - 1)  # time_bin
        tokens[:, :, 1] = torch.clamp(tokens[:, :, 1], 0, 2)                      # type
        tokens[:, :, 2] = torch.clamp(tokens[:, :, 2], 0, 15)                     # grid_x
        tokens[:, :, 3] = torch.clamp(tokens[:, :, 3], 0, 11)                     # grid_y
        
        logger.debug(f"Batch shapes: specs {specs.shape}, tokens {tokens.shape}")
        logger.debug(f"Batch token ranges: time_bin [{tokens[:, :, 0].min().item()}, {tokens[:, :, 0].max().item()}], "
                     f"type [{tokens[:, :, 1].min().item()}, {tokens[:, :, 1].max().item()}], "
                     f"grid_x [{tokens[:, :, 2].min().item()}, {tokens[:, :, 2].max().item()}], "
                     f"grid_y [{tokens[:, :, 3].min().item()}, {tokens[:, :, 3].max().item()}]")
        
        optimizer.zero_grad()
        time_logits, type_logits, grid_x_logits, grid_y_logits = model(specs, tokens)
        
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

torch.save(model.state_dict(), 'output/model.pth')