"""
OsuMapper: Audio Spectrogram Transformer encoder + autoregressive
TransformerDecoder for beatmap token generation with hybrid
discrete + residual output heads.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTModel, ASTConfig

from .tokenizer import TOTAL_VOCAB, PAD, BOS, NUM_DIFF_BINS
from .dataset import N_MELS, N_FEATURES


# ---------------------------------------------------------------------------
# Temporal upsampling for encoder outputs
# ---------------------------------------------------------------------------
class TemporalUpsampleHead(nn.Module):
    """1D conv stack to increase temporal resolution of AST embeddings."""

    def __init__(self, d_model: int = 768, upsample_factor: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode="linear", align_corners=False)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> transpose to [B, D, T]
        x = x.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.upsample(x)
        x = self.act(self.conv2(x))
        x = x.transpose(1, 2)  # [B, T', D]
        return self.norm(x)


# ---------------------------------------------------------------------------
# Audio Encoder
# ---------------------------------------------------------------------------
class AudioEncoder(nn.Module):
    """
    Pretrained AST-based encoder with temporal upsampling.
    Freezes most of AST, trains last few layers + upsample head.
    """

    def __init__(
        self,
        pretrained: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        d_model: int = 768,
        unfreeze_last_n: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        try:
            self.ast = ASTModel.from_pretrained(pretrained)
        except Exception:
            config = ASTConfig(
                hidden_size=d_model,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.ast = ASTModel(config)

        for param in self.ast.parameters():
            param.requires_grad = False
        for layer in self.ast.encoder.layer[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.upsample = TemporalUpsampleHead(d_model)
        self.mel_proj = nn.Linear(N_FEATURES, d_model)
        self.mel_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self._fallback_warned = False
        self._ast_pos_len = getattr(self.ast.config, "max_length", None)

    def _resize_ast_position_embeddings(self, mel_len: int) -> None:
        """Resize AST position embeddings to match the current mel frame length."""
        config = self.ast.config
        patch = config.patch_size
        time_stride = config.time_stride
        freq_stride = config.frequency_stride

        freq_out = (config.num_mel_bins - patch) // freq_stride + 1
        old_time_out = (config.max_length - patch) // time_stride + 1
        new_time_out = (mel_len - patch) // time_stride + 1
        if new_time_out <= 0:
            raise ValueError(f"Invalid mel length for AST patches: {mel_len}")

        pos = self.ast.embeddings.position_embeddings  # [1, old_patches+2, hidden]
        if pos.shape[1] == (freq_out * new_time_out + 2):
            return

        cls_dist = pos[:, :2, :]
        pos_no_cls = pos[:, 2:, :].reshape(1, freq_out, old_time_out, -1).permute(0, 3, 1, 2)
        pos_no_cls = F.interpolate(
            pos_no_cls,
            size=(freq_out, new_time_out),
            mode="bilinear",
            align_corners=False,
        )
        pos_no_cls = pos_no_cls.permute(0, 2, 3, 1).reshape(1, freq_out * new_time_out, -1)
        new_pos = torch.cat([cls_dist, pos_no_cls], dim=1)

        self.ast.embeddings.position_embeddings = nn.Parameter(new_pos)
        self.ast.config.max_length = mel_len

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 1, N_FEATURES, T] mel spectrogram + onset strength

        Returns:
            encoder_out: [B, T', D] temporal embeddings
        """
        B = mel.shape[0]

        if mel.dim() == 4:
            mel = mel.squeeze(1)  # [B, N_FEATURES, T]
        mel = mel.transpose(1, 2)  # [B, T, N_FEATURES]

        mel_len = mel.shape[1]
        if self._ast_pos_len is not None and mel_len != self._ast_pos_len:
            with torch.no_grad():
                self._resize_ast_position_embeddings(mel_len)
            print(
                f"AST position embeddings resized for mel_len={mel_len} (was {self._ast_pos_len})",
                flush=True,
            )
            self._ast_pos_len = mel_len

        projected = self.mel_proj(mel)  # [B, T, D] -- includes onset channel

        mel_for_ast = mel[:, :, :N_MELS]  # strip onset for AST (expects 128 bins)

        try:
            ast_out = self.ast(input_values=mel_for_ast).last_hidden_state
            encoder_out = self.upsample(ast_out)
        except Exception as exc:
            encoder_out = self.upsample(projected)
            if not self._fallback_warned:
                print(
                    "WARNING: AST forward failed; using projected mel fallback path. "
                    f"Error: {exc}"
                )
                self._fallback_warned = True

        proj_aligned = F.interpolate(
            projected.transpose(1, 2),
            size=encoder_out.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        gate = self.mel_gate(proj_aligned)
        encoder_out = encoder_out + gate * proj_aligned

        return encoder_out


# ---------------------------------------------------------------------------
# Autoregressive Decoder
# ---------------------------------------------------------------------------
class BeatmapDecoder(nn.Module):
    """
    Transformer decoder with cross-attention to audio encoder output.
    Outputs discrete token logits + continuous residual predictions.
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embed = nn.Embedding(TOTAL_VOCAB, d_model, padding_idx=PAD)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.discrete_head = nn.Linear(d_model, TOTAL_VOCAB)
        self.time_res_head = nn.Linear(d_model, 1)
        self.x_res_head = nn.Linear(d_model, 1)
        self.y_res_head = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

    def forward(
        self,
        encoder_out: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_out: [B, S, D] from audio encoder
            tgt_tokens: [B, T] target token ids (shifted right)
            tgt_padding_mask: [B, T] True where padded

        Returns:
            logits: [B, T, TOTAL_VOCAB]
            time_res: [B, T, 1]
            x_res: [B, T, 1]
            y_res: [B, T, 1]
        """
        B, T = tgt_tokens.shape
        device = tgt_tokens.device

        tok_emb = self.token_embed(tgt_tokens)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embed(positions)
        tgt = tok_emb + pos_emb

        causal_mask = self._causal_mask(T, device)

        out = self.transformer(
            tgt=tgt,
            memory=encoder_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        out = self.norm(out)

        logits = self.discrete_head(out)
        time_res = self.time_res_head(out)
        x_res = self.x_res_head(out)
        y_res = self.y_res_head(out)

        return logits, time_res, x_res, y_res


# ---------------------------------------------------------------------------
# Conditioning Encoder  (difficulty + BPM -> single conditioning token)
# ---------------------------------------------------------------------------
class ConditioningEncoder(nn.Module):
    """Embeds difficulty bin and log-normalized BPM into a single [B,1,D] token."""

    def __init__(self, d_model: int = 768, num_difficulties: int = NUM_DIFF_BINS):
        super().__init__()
        self.diff_embed = nn.Embedding(num_difficulties, d_model)
        self.bpm_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, difficulty_id: torch.Tensor, bpm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            difficulty_id: [B] int tensor  (0..4)
            bpm:           [B, 1] float tensor (raw BPM)
        Returns:
            cond: [B, 1, D]
        """
        bpm_log = torch.log(bpm.clamp(min=1.0)) / math.log(300.0)
        cond = self.diff_embed(difficulty_id) + 0.5 * self.bpm_proj(bpm_log)
        return self.norm(cond).unsqueeze(1)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class OsuMapper(nn.Module):
    """
    End-to-end model: mel + difficulty + BPM -> beatmap tokens + residuals.
    """

    def __init__(
        self,
        d_model: int = 768,
        encoder_pretrained: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        encoder_unfreeze_last_n: int = 2,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.encoder = AudioEncoder(
            pretrained=encoder_pretrained,
            d_model=d_model,
            unfreeze_last_n=encoder_unfreeze_last_n,
        )
        self.conditioning = ConditioningEncoder(d_model)
        self.decoder = BeatmapDecoder(
            d_model=d_model,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            dim_feedforward=decoder_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

    def forward(
        self,
        mel: torch.Tensor,
        tgt_tokens: torch.Tensor,
        difficulty_id: torch.Tensor,
        bpm: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        encoder_out = self.encoder(mel)
        cond = self.conditioning(difficulty_id, bpm)  # [B, 1, D]
        memory = torch.cat([cond, encoder_out], dim=1)  # [B, 1+T, D]
        logits, time_res, x_res, y_res = self.decoder(
            memory, tgt_tokens, tgt_padding_mask
        )
        return {
            "logits": logits,
            "time_residuals": time_res,
            "x_residuals": x_res,
            "y_residuals": y_res,
        }

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        difficulty_id: torch.Tensor,
        bpm: torch.Tensor,
        max_len: int = 512,
        temperature: float = 1.0,
        beam_size: int = 1,
    ) -> tuple[list[int], list[tuple[float, float, float]]]:
        """
        Greedy / temperature-sampled autoregressive generation.

        Args:
            mel:           [B, 1, N_MELS, T] or [1, N_MELS, T]
            difficulty_id: [B] int tensor
            bpm:           [B, 1] float tensor
        Returns:
            tokens: list of generated token ids
            residuals: list of (time_res, x_res, y_res) tuples
        """
        self.eval()
        device = mel.device

        if mel.dim() == 3:
            mel = mel.unsqueeze(0)

        encoder_out = self.encoder(mel)
        cond = self.conditioning(difficulty_id, bpm)
        memory = torch.cat([cond, encoder_out], dim=1)

        tokens = [BOS]
        residuals_out = [(0.0, 0.0, 0.0)]
        from .tokenizer import EOS

        for _ in range(max_len):
            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, t_res, x_res, y_res = self.decoder(memory, tgt)

            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)

            if beam_size <= 1:
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(probs).item()

            tokens.append(next_token)
            residuals_out.append((
                t_res[0, -1, 0].item(),
                x_res[0, -1, 0].item(),
                y_res[0, -1, 0].item(),
            ))

            if next_token == EOS:
                break

        return tokens, residuals_out


# ---------------------------------------------------------------------------
# GPU-adaptive batch size
# ---------------------------------------------------------------------------
def get_adaptive_batch_size() -> int:
    """Auto-detect GPU type and return appropriate batch size."""
    if not torch.cuda.is_available():
        return 4

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    name = torch.cuda.get_device_name(0).lower()

    if vram_gb >= 70 or "a100" in name:
        return 32
    elif vram_gb >= 40 or "a6000" in name:
        return 24
    elif vram_gb >= 20 or "a5000" in name:
        return 16
    elif vram_gb >= 14 or "a4000" in name or "4500" in name:
        return 16
    else:
        return 4
