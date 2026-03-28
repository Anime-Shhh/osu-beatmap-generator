"""Quick diagnostic: load checkpoint, run 5 val batches, dump evidence to debug log."""
import json
import os
import sys
import time
import glob
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import OsuMapper
from src.dataset import OsuCachedDataset, collate_fn, N_MELS
from src.tokenizer import (
    PAD, TOTAL_VOCAB, TIME_OFFSET, NUM_BEAT_BINS,
    POS_OFFSET, NUM_POS_BINS, _NUM_SPECIAL,
    detokenize_to_hitobjects,
)
from src.metrics import compute_all_metrics

LOG = "/common/home/asj102/personalCS/osu_beatmap_generator/.cursor/debug-559ead.log"
CKPT = "/common/users/asj102/osu_project/models/best.pt"
SHARD_DIR = "/common/users/asj102/osu_project/data/shards"

def cat(t):
    if TIME_OFFSET <= t < TIME_OFFSET + NUM_BEAT_BINS: return "time"
    if POS_OFFSET <= t < POS_OFFSET + NUM_POS_BINS: return "pos"
    if 4 <= t <= 13: return "type"
    if t < _NUM_SPECIAL: return "special"
    return "unknown"

def log(entry):
    with open(LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model = OsuMapper()
    skip = {k for k in cleaned if "position_embeddings" in k and cleaned[k].shape != dict(model.named_parameters()).get(k, cleaned[k]).shape}
    filtered = {k: v for k, v in cleaned.items() if k not in skip}
    model.load_state_dict(filtered, strict=False)
    for k in skip:
        print(f"Skipped loading {k} (shape mismatch: ckpt={cleaned[k].shape})")
    model.to(device).eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}")

    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "shard-*.tar")))
    shards = [s for s in shards if os.path.getsize(s) > 0]
    val_shards = shards[-max(1, len(shards)//10):]
    print(f"Using {len(val_shards)} val shards out of {len(shards)} total")

    ds = OsuCachedDataset(shard_pattern="", shards=val_shards, shuffle=False)
    dl = DataLoader(ds, batch_size=16, collate_fn=collate_fn, num_workers=2)

    ce = nn.CrossEntropyLoss(ignore_index=PAD)

    for step, batch in enumerate(dl):
        if step >= 5:
            break

        mel = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        difficulty_id = batch["difficulty_id"].to(device)
        bpm = batch["bpm"].to(device)

        tgt_input = tokens[:, :-1]
        tgt_output = tokens[:, 1:]
        padding_mask = tgt_input == PAD

        with torch.no_grad():
            outputs = model(mel, tgt_input, difficulty_id, bpm, padding_mask)
            logits = outputs["logits"]
            loss = ce(logits.reshape(-1, TOTAL_VOCAB), tgt_output.reshape(-1))

            if step == 0:
                mel_no_onset = mel.clone()
                mel_no_onset[:, :, -1, :] = 0.0
                outputs_no = model(mel_no_onset, tgt_input, difficulty_id, bpm, padding_mask)
                diff = (outputs["logits"].float() - outputs_no["logits"].float()).abs().mean()
                print(f"[onset zero-out test] logit diff = {diff:.6f}")
                log({"sessionId": "559ead", "runId": "onset_test", "hypothesisId": "onset_path",
                     "message": "onset_zero_out_diff", "data": {"diff": round(diff.item(), 6)},
                     "timestamp": int(time.time()*1000)})

        pred_tokens_batch = logits.argmax(dim=-1)

        for b in range(min(2, tokens.shape[0])):
            pred_t = pred_tokens_batch[b].cpu().tolist()
            true_t = tgt_output[b].cpu().tolist()
            true_t = [t for t in true_t if t != PAD]
            pred_t = pred_t[:len(true_t)]

            sample_bpm = bpm[b].item()
            sample_ms_per_beat = 60000.0 / sample_bpm

            pred_time = [t for t in pred_t if TIME_OFFSET <= t < TIME_OFFSET + NUM_BEAT_BINS]
            true_time = [t for t in true_t if TIME_OFFSET <= t < TIME_OFFSET + NUM_BEAT_BINS]
            pred_pos = [t for t in pred_t if POS_OFFSET <= t < POS_OFFSET + NUM_POS_BINS]
            true_pos = [t for t in true_t if POS_OFFSET <= t < POS_OFFSET + NUM_POS_BINS]
            pred_type = [t for t in pred_t if 4 <= t <= 13]
            true_type = [t for t in true_t if 4 <= t <= 13]

            pred_objs = detokenize_to_hitobjects(pred_t, ms_per_beat=sample_ms_per_beat)
            true_objs = detokenize_to_hitobjects(true_t, ms_per_beat=sample_ms_per_beat)

            metrics = compute_all_metrics(pred_t, true_t, ms_per_beat=sample_ms_per_beat)
            mae = metrics["timing_mae_ms"]

            pred_bins = [t - TIME_OFFSET for t in pred_time]
            true_bins = [t - TIME_OFFSET for t in true_time]
            time_tok_acc = sum(1 for p, t in zip(pred_t, true_t) if p == t and TIME_OFFSET <= t < TIME_OFFSET + NUM_BEAT_BINS) / max(len(true_time), 1)
            all_acc = sum(1 for p, t in zip(pred_t, true_t) if p == t) / max(len(true_t), 1)

            cat_match = {"time_at_time": 0, "pos_at_time": 0, "type_at_time": 0, "other_at_time": 0,
                         "time_at_pos": 0, "pos_at_pos": 0, "type_at_pos": 0, "other_at_pos": 0,
                         "time_at_type": 0, "pos_at_type": 0, "type_at_type": 0, "other_at_type": 0}
            for p, t in zip(pred_t, true_t):
                pc, tc = cat(p), cat(t)
                key = f"{pc}_at_{tc}"
                if key in cat_match:
                    cat_match[key] += 1

            bin_err = [abs(p - t) for p, t in zip(pred_bins, true_bins)] if pred_bins and true_bins else []
            mean_bin_err = sum(bin_err) / len(bin_err) if bin_err else -1

            pred_times_ms = sorted([o["time"] for o in pred_objs])
            true_times_ms = sorted([o["time"] for o in true_objs])

            run_id = f"step{step}_b{b}"
            log({"sessionId": "559ead", "runId": run_id, "hypothesisId": "H1", "message": "token_breakdown",
                 "data": {"seq_len": len(true_t), "pred_time_count": len(pred_time), "true_time_count": len(true_time),
                          "pred_pos_count": len(pred_pos), "true_pos_count": len(true_pos),
                          "pred_type_count": len(pred_type), "true_type_count": len(true_type),
                          "pred_obj_count": len(pred_objs), "true_obj_count": len(true_objs)},
                 "timestamp": int(time.time()*1000)})
            log({"sessionId": "559ead", "runId": run_id, "hypothesisId": "H2", "message": "time_bin_dist",
                 "data": {"pred_bins": pred_bins[:30], "true_bins": true_bins[:30],
                          "pred_bin_set": sorted(set(pred_bins)), "true_bin_set": sorted(set(true_bins)),
                          "mean_bin_error": round(mean_bin_err, 2)},
                 "timestamp": int(time.time()*1000)})
            log({"sessionId": "559ead", "runId": run_id, "hypothesisId": "H3", "message": "bpm_and_timing",
                 "data": {"bpm": round(sample_bpm, 2), "ms_per_beat": round(sample_ms_per_beat, 2),
                          "pred_times_ms": [round(t, 1) for t in pred_times_ms[:15]],
                          "true_times_ms": [round(t, 1) for t in true_times_ms[:15]],
                          "timing_mae": round(mae, 2) if mae != float("inf") else -1,
                          "val_loss": round(loss.item(), 4)},
                 "timestamp": int(time.time()*1000)})
            log({"sessionId": "559ead", "runId": run_id, "hypothesisId": "H4", "message": "accuracy",
                 "data": {"time_tok_accuracy": round(time_tok_acc, 4), "overall_tok_accuracy": round(all_acc, 4),
                          "hit_f1": round(metrics["hit_f1"], 4), "edit_distance": round(metrics["edit_distance"], 4)},
                 "timestamp": int(time.time()*1000)})
            log({"sessionId": "559ead", "runId": run_id, "hypothesisId": "H5", "message": "first_30_tokens",
                 "data": {"pred_first30": pred_t[:30], "true_first30": true_t[:30]},
                 "timestamp": int(time.time()*1000)})
            log({"sessionId": "559ead", "runId": run_id, "hypothesisId": "H1H2", "message": "category_confusion",
                 "data": cat_match,
                 "timestamp": int(time.time()*1000)})
            log({"sessionId": "559ead", "runId": run_id, "hypothesisId": "H2", "message": "bin_level_errors",
                 "data": {"per_pos_bin_err": bin_err[:20], "mean_bin_error": round(mean_bin_err, 2)},
                 "timestamp": int(time.time()*1000)})

            print(f"[step={step} b={b}] loss={loss.item():.4f} mae={mae:.1f}ms f1={metrics['hit_f1']:.4f} "
                  f"acc={all_acc:.3f} time_acc={time_tok_acc:.3f} pred_objs={len(pred_objs)} true_objs={len(true_objs)}")

    print(f"\nDiagnostics written to {LOG}")

if __name__ == "__main__":
    main()
