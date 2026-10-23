"""Auxiliary head-to-head distance supervision for the original token model."""
import torch
import torch.nn.functional as F
import hashlib
import json
from pathlib import Path
from .tokenizer import PAD, POS_OFFSET, TIME_OFFSET, TOTAL_VOCAB, PLAYFIELD_DIAGONAL

DISTANCE_SCHEMA = "legacy-head-distance/1"


def corpus_fingerprint(split_shards):
    """Hash immutable shard bytes once per run, independently of absolute paths."""
    rows = []
    for split, paths in sorted(split_shards.items()):
        for path in sorted(paths):
            with open(path, 'rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            rows.append((split, Path(path).name, digest))
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def distance_config(weight):
    if not 0 <= weight < float('inf'):
        raise ValueError('distance_loss_weight must be finite and nonnegative')
    return dict(schema=DISTANCE_SCHEMA, loss_weight=float(weight), head_hidden_dim=64,
                normalization_px=PLAYFIELD_DIAGONAL, target='head-to-head',
                activation='softplus', loss='smooth_l1', beta=1.0)


def distance_loss(prediction, targets, mask):
    """Inputs are already shifted to the next-token prediction axis.

    Boolean selection keeps masked values (including NaNs) out of the objective.
    An empty selection gives a differentiable zero with zero parameter gradients.
    """
    prediction = prediction.squeeze(-1).float()
    if prediction.shape != targets.shape or targets.shape != mask.shape:
        raise ValueError('distance prediction, target and mask must align')
    if mask.dtype != torch.bool:
        raise ValueError('distance mask must be boolean')
    pred, target = prediction[mask], targets.float()[mask]
    count = mask.sum()
    loss = F.smooth_l1_loss(pred, target, reduction='sum') / count.clamp_min(1)
    mae_px = (pred - target).abs().sum() * PLAYFIELD_DIAGONAL / count.clamp_min(1)
    return loss, mae_px, count


def shifted_distance_loss(outputs, batch):
    prediction = outputs['distance_predictions']
    return distance_loss(prediction, batch['distance_targets'][:, 1:].to(prediction.device),
                         batch['distance_mask'][:, 1:].to(prediction.device))


class TeacherForcedMetrics:
    """Accumulate by eligible tokens, not unweighted averages of padded batches."""
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def add(self, name, total, count):
        self.sums[name] = self.sums.get(name, 0.) + float(total)
        self.counts[name] = self.counts.get(name, 0) + int(count)

    @torch.no_grad()
    def update(self, outputs, batch):
        device = outputs['logits'].device
        truth = batch['tokens'][:, 1:].to(device)
        valid = truth != PAD
        logits = outputs['logits'].float()
        self.add('token_ce', F.cross_entropy(logits[valid], truth[valid], reduction='sum'), valid.sum())
        self.add('token_accuracy', ((logits.argmax(-1) == truth) & valid).sum(), valid.sum())
        masks = {'time': (truth >= TIME_OFFSET) & (truth < TOTAL_VOCAB),
                 'x': (truth >= POS_OFFSET) & (truth < TIME_OFFSET),
                 'y': (truth >= POS_OFFSET) & (truth < TIME_OFFSET)}
        for key, mask in masks.items():
            error = outputs[key+'_residuals'].squeeze(-1).float() - batch[key+'_residuals'][:, 1:].to(device)
            self.add(key+'_residual_mae', error[mask].abs().sum(), mask.sum())
        loss, mae, count = shifted_distance_loss(outputs, batch)
        self.add('distance_loss', loss*count, count)
        self.add('distance_mae_px', mae*count, count)

    def result(self):
        result = {k: self.sums[k]/self.counts[k] if self.counts[k] else None for k in self.sums}
        result['distance_target_count'] = self.counts.get('distance_loss', 0)
        return result
