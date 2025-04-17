import torch
import torch.nn as nn
import math
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim=128, d_model=128, nhead=4, num_layers=1, max_tokens=1000):
        super(TransformerDecoder, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.max_tokens = max_tokens
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        
        self.out_time = nn.Linear(d_model, max_tokens)
        self.out_type = nn.Linear(d_model, 3)
        self.out_grid_x = nn.Linear(d_model, 16)
        self.out_grid_y = nn.Linear(d_model, 12)

        self.time_emb = nn.Parameter(torch.randn(max_tokens, d_model // 4))
        self.type_emb = nn.Parameter(torch.randn(3, d_model // 4))
        self.grid_x_emb = nn.Parameter(torch.randn(16, d_model // 4))
        self.grid_y_emb = nn.Parameter(torch.randn(12, d_model // 4))

    def forward(self, src, tgt):
        src = src.permute(0, 2, 1)
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        
        tgt_emb = self.embed_target(tgt[:, :-1, :])
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1) - 1).to(tgt.device)
        output = self.decoder(tgt_emb, src, tgt_mask)
        
        time_logits = self.out_time(output)
        type_logits = self.out_type(output)
        grid_x_logits = self.out_grid_x(output)
        grid_y_logits = self.out_grid_y(output)
        
        return time_logits, type_logits, grid_x_logits, grid_y_logits

    def embed_target(self, tgt):
        logger.debug(f"tgt shape: {tgt.shape}")
        time_indices = torch.clamp(tgt[:, :, 0], 0, self.max_tokens - 1)
        type_indices = torch.clamp(tgt[:, :, 1], 0, 2)
        grid_x_indices = torch.clamp(tgt[:, :, 2], 0, 15)
        grid_y_indices = torch.clamp(tgt[:, :, 3], 0, 11)
        logger.debug(f"Indices ranges: time [{time_indices.min().item()}, {time_indices.max().item()}], "
                     f"type [{type_indices.min().item()}, {type_indices.max().item()}], "
                     f"x [{grid_x_indices.min().item()}, {grid_x_indices.max().item()}], "
                     f"y [{grid_y_indices.min().item()}, {grid_y_indices.max().item()}]")
        
        emb = torch.cat([
            self.time_emb[time_indices],
            self.type_emb[type_indices],
            self.grid_x_emb[grid_x_indices],
            self.grid_y_emb[grid_y_indices]
        ], dim=-1)
        return self.pos_encoder(emb)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerDecoder().to(device)