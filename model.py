import torch
import torch.nn as nn
import math

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim=128, d_model=128, nhead=4, num_layers=1, max_tokens=300):
        super(TransformerDecoder, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_tokens = max_tokens
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        
        # Output layers
        self.out_time = nn.Linear(d_model, max_tokens)  # Time bin
        self.out_type = nn.Linear(d_model, 3)  # Circle, slider, spinner
        self.out_grid_x = nn.Linear(d_model, 16)  # Grid x
        self.out_grid_y = nn.Linear(d_model, 12)  # Grid y

    def forward(self, src, tgt):
        # src: spectrogram (batch, mel_bins, frames)
        # tgt: target tokens (batch, max_tokens, 4)
        
        # Project spectrogram to d_model
        src = src.permute(0, 2, 1)  # (batch, frames, mel_bins)
        src = self.input_proj(src)  # (batch, frames, d_model)
        src = self.pos_encoder(src)
        
        # Prepare target
        tgt_emb = self.embed_target(tgt[:, :-1, :])  # Shift right
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1) - 1)
        
        # Decode
        output = self.decoder(tgt_emb, src, tgt_mask)
        
        # Predict
        time_logits = self.out_time(output)
        type_logits = self.out_type(output)
        grid_x_logits = self.out_grid_x(output)
        grid_y_logits = self.out_grid_y(output)
        
        return time_logits, type_logits, grid_x_logits, grid_y_logits

    def embed_target(self, tgt):
        # Simple embedding for tokens (time_bin, type, grid_x, grid_y)
        time_emb = nn.Parameter(torch.randn(self.max_tokens, self.d_model // 4))
        type_emb = nn.Parameter(torch.randn(3, self.d_model // 4))
        grid_x_emb = nn.Parameter(torch.randn(16, self.d_model // 4))
        grid_y_emb = nn.Parameter(torch.randn(12, self.d_model // 4))
        
        emb = torch.cat([
            time_emb[tgt[:, :, 0]],
            type_emb[tgt[:, :, 1]],
            grid_x_emb[tgt[:, :, 2]],
            grid_y_emb[tgt[:, :, 3]]
        ], dim=-1)
        return self.pos_encoder(emb)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask.to(device)

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

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerDecoder().to(device)