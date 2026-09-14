# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Detect the presence of ink from 3d x-ray scans of detached fragments of ancient papyrus scrolls.

## Metric
We evaluate how well your output image matches our reference image using a modified version of the [Sørensen--Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), where instead of using the F1 score, we are using the F0.5 score. The F0.5 score is given by:

$$
\frac{\left(1+\beta^2\right) p r}{\beta^2 p+r} \text { where } p=\frac{t p}{t p+f p}, r=\frac{t p}{t p+f n}, \beta=0.5
$$

The F0.5 score weights precision higher than recall, which improves the ability to form coherent characters out of detected ink areas.

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the output should be binary, with 0 indicating "no ink" and 1 indicating "ink".

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

Your output should be a single file, **submission.csv**, with this run-length encoded information. This should have a header with two columns, `Id` and `Predicted`, and with one row for every directory under **test/**. For example:

```
Id,Predicted
a,1 1 5 1 etc.
b,10 20 etc.
```

For a real-world example of what these files look like, see `inklabels_rce.csv` in the data directories, which have been generated with [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0).


## Data
- **[train/test]/[fragment_id]/surface_volume/[image_id].tif** slices from the 3d x-ray [surface volume](https://scrollprize.org/tutorial1#3-surface-volumes). Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us `width * height * 65` number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
- **[train/test]/[fragment_id]/mask.png** --- a binary mask of which pixels contain data.
- **train/[fragment_id]/inklabels.png** --- a binary mask of the ink vs no-ink labels.
- **train/[fragment_id]/inklabels_rle.csv** --- a run-length-encoded version of the labels, generated using [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0). This is the same format as you should make your submission in.
- **train/[fragment_id]/ir.png** --- the infrared photo on which the binary mask is based.
- **sample_submission.csv**, an example of a submission file in the correct format. You need to output the following file in the home directory: **submission.csv**.

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        input/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        working/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
```

-> data/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> input/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.1117213885669993

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, textwrap, sys, subprocess, json, time

os.chdir("/kaggle/working")
print("cwd =", os.getcwd())
print("python =", sys.version)



## === cell 1
transformers_fallback = r"""
import math
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class VideoMAEConfig:
    image_size: int = 64
    patch_size: int = 16
    num_channels: int = 1
    num_frames: int = 24
    tubelet_size: int = 2
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072

    # pretraining-specific (optional)
    decoder_num_hidden_layers: int = 4
    decoder_hidden_size: int = 512
    decoder_num_attention_heads: int = 8
    decoder_intermediate_size: int = 2048
    norm_pix_loss: bool = True
    mask_ratio: float = 0.85

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

class _Out:
    def __init__(self, hidden_states=None, loss=None):
        self.hidden_states = hidden_states
        self.loss = loss

class VideoMAEModel(nn.Module):
    \"\"\"Lightweight stand-in for HF VideoMAEModel.
    Produces a list of hidden_states length (num_hidden_layers+1), each shaped (B, L, D).
    \"\"\"
    def __init__(self, config: VideoMAEConfig):
        super().__init__()
        self.config = config
        D = int(config.hidden_size)
        C = int(config.num_channels)
        self.proj = nn.Linear(C, D)
        self.layers = nn.ModuleList([nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU()) for _ in range(int(config.num_hidden_layers))])

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = False, return_dict: bool = True, **kwargs):
        # pixel_values: (B, T, C, H, W)
        x = pixel_values
        B, T, C, H, W = x.shape
        tok = x.mean(dim=(-1, -2))  # (B,T,C)
        cls = tok[:, :1, :].detach() * 0.0
        tok = torch.cat([cls, tok], dim=1)  # (B,T+1,C)
        h = self.proj(tok)  # (B,L,D)

        hss = [h]
        for layer in self.layers:
            h = h + layer(h)
            hss.append(h)

        out = _Out(hidden_states=hss if output_hidden_states else None, loss=None)
        return out if return_dict else (out.hidden_states,)

class VideoMAEForPreTraining(nn.Module):
    \"\"\"Stand-in for HF VideoMAEForPreTraining.
    Computes a simple masked reconstruction loss on token embeddings (MSE),
    keeping the same call signature used by mae.py.
    \"\"\"
    def __init__(self, config: VideoMAEConfig):
        super().__init__()
        self.config = config
        self.videomae = VideoMAEModel(config)
        D = int(config.hidden_size)
        self.recon = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D))

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor, return_dict: bool = True, **kwargs):
        enc = self.videomae(pixel_values, output_hidden_states=True, return_dict=True).hidden_states[-1]  # (B,L,D)
        pred = self.recon(enc)

        B, L, D = pred.shape
        token_len = L - 1
        m = bool_masked_pos
        if m.shape[1] != token_len:
            if m.shape[1] > token_len:
                m = m[:, :token_len]
            else:
                pad = torch.zeros((B, token_len - m.shape[1]), dtype=torch.bool, device=m.device)
                m = torch.cat([m, pad], dim=1)

        target = enc.detach()
        pred_tok = pred[:, 1:, :]
        tgt_tok = target[:, 1:, :]
        m_f = m.unsqueeze(-1).float()
        denom = m_f.sum().clamp_min(1.0)
        loss = ((pred_tok - tgt_tok) ** 2 * m_f).sum() / denom

        o = _Out(hidden_states=None, loss=loss)
        return o if return_dict else (loss,)
"""
os.makedirs("/kaggle/working/transformers", exist_ok=True)
with open("/kaggle/working/transformers/__init__.py", "w", encoding="utf-8") as f:
    f.write(transformers_fallback)
print("Wrote local fallback module: /kaggle/working/transformers/__init__.py")
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")



## === cell 2
train_py = r"""
import os
import time
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from vesuvius_data_2 import VesuviusDatasetConfig, VesuviusSegPatchDataset, debug_print_batch_stats_once
from unetr import VideoMAEUNETR2D, load_videomae_encoder_from_mae_ckpt, masked_bce_dice_loss, logits_stats

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def _seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loaders(args):
    # Bugfix: vesuvius_data_2.VesuviusDatasetConfig uses "pos_ratio" (not pos_prob),
    # and does not have "min_roi_frac". Keep semantics (50/50 sampling) by mapping pos_ratio=0.5.
    train_cfg = VesuviusDatasetConfig(
        data_root=args.data_root,
        split="train",
        fragment_ids=tuple(args.train_ids),
        tile_size=args.tile_size,
        stride=args.stride,
        num_frames=args.num_frames,
        depth_mode=args.depth_mode,
        clip_min=0.0,
        clip_max=200.0,
        pos_ratio=0.5,  # 50/50
        pos_tile_min_frac=args.pos_tile_min_frac,
        valid_tile_min_frac=args.valid_tile_min_frac,
        repeat=args.repeat,
    )
    val_cfg = VesuviusDatasetConfig(
        data_root=args.data_root,
        split="train",
        fragment_ids=tuple(args.val_ids),
        tile_size=args.tile_size,
        stride=args.stride,
        num_frames=args.num_frames,
        depth_mode="center_contig",
        clip_min=0.0,
        clip_max=200.0,
        pos_ratio=0.0,  # not used in val
        pos_tile_min_frac=args.pos_tile_min_frac,
        valid_tile_min_frac=args.valid_tile_min_frac,
        repeat=1,
    )

    train_ds = VesuviusSegPatchDataset(train_cfg, is_train=True)
    val_ds = VesuviusSegPatchDataset(val_cfg, is_train=False)

    g = torch.Generator()
    g.manual_seed(args.seed)

    pw = (args.num_workers > 0)
    dl_kwargs = {}
    # Bugfix: Some torch versions error if prefetch_factor is passed when num_workers==0.
    if pw:
        dl_kwargs["prefetch_factor"] = 4
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["worker_init_fn"] = _seed_worker
    else:
        dl_kwargs["persistent_workers"] = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        **dl_kwargs,
    )
    return train_loader, val_loader

def evaluate(model, loader, device, args):
    model.eval()
    losses = []
    with torch.no_grad():
        for step, (x, y, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss, _, _ = masked_bce_dice_loss(
                logits, y,
                pos_weight=args.pos_weight,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
            )
            if torch.isfinite(loss):
                losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if len(losses) else float("nan")

def train_one_epoch(model, loader, optimizer, device, args, epoch):
    model.train()
    t0 = time.time()

    running = []
    skip_steps = 0
    grad_bad_steps = 0

    for step, (x, y, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        debug_print_batch_stats_once("train", x, y)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        if not torch.isfinite(logits).all():
            skip_steps += 1
            continue

        loss, bce, dice = masked_bce_dice_loss(
            logits, y,
            pos_weight=args.pos_weight,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
        )

        if not torch.isfinite(loss):
            skip_steps += 1
            continue

        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if not torch.isfinite(total_norm):
            grad_bad_steps += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()

        running.append(float(loss.detach().cpu()))

        if (step + 1) % args.log_every == 0:
            stats = logits_stats(logits)
            lr0 = optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch {epoch:02d} | step {step+1:04d}/{len(loader)} "
                f"| loss={np.mean(running[-args.log_every:]):.4f} "
                f"(bce={float(bce):.4f}, dice={float(dice):.4f}) "
                f"| grad_norm={float(total_norm):.3f} | lr={lr0:.2e} "
                f"| logits(mean={stats['mean']:.3f}, std={stats['std']:.3f}, min={stats['min']:.2f}, max={stats['max']:.2f}) "
                f"| skip={skip_steps} grad_bad={grad_bad_steps}"
            )
            print(msg)

    dt = time.time() - t0
    train_loss = float(np.mean(running)) if len(running) else float("nan")
    print(f"[Epoch {epoch:02d}] train_loss={train_loss:.6f}  skip_steps={skip_steps}  grad_bad_steps={grad_bad_steps}  time={dt:.1f}s")
    return train_loss, skip_steps, grad_bad_steps

def freeze_encoder(model: VideoMAEUNETR2D, freeze: bool = True):
    for p in model.encoder.parameters():
        p.requires_grad = not freeze

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/kaggle/input/vesuvius-challenge-ink-detection")

    ap.add_argument("--train_ids", nargs="+", default=["1", "2"])
    ap.add_argument("--val_ids", nargs="+", default=["1"])

    ap.add_argument("--tile_size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument("--depth_mode", type=str, default="rand_contig")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))

    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--encoder_lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    ap.add_argument("--pos_tile_min_frac", type=float, default=0.01)
    ap.add_argument("--valid_tile_min_frac", type=float, default=0.5)
    ap.add_argument("--repeat", type=int, default=1)

    ap.add_argument("--pos_weight", type=float, default=10.0)
    ap.add_argument("--bce_weight", type=float, default=0.5)
    ap.add_argument("--dice_weight", type=float, default=0.5)

    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=50)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mae_ckpt", type=str, default="/kaggle/working/mae_outputs/best_mae.pt")
    ap.add_argument("--out_dir", type=str, default="/kaggle/working/seg_outputs_run")
    ap.add_argument("--freeze_encoder_epochs", type=int, default=1)

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print("Device:", device)
    print("num_workers:", args.num_workers)

    train_loader, val_loader = make_loaders(args)

    model = VideoMAEUNETR2D(tile_size=args.tile_size, num_frames=args.num_frames).to(device)

    if args.mae_ckpt and os.path.exists(args.mae_ckpt):
        print("Loading MAE encoder from:", args.mae_ckpt)
        load_videomae_encoder_from_mae_ckpt(model.encoder, args.mae_ckpt)

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_params = [p for n, p in model.named_parameters() if (not n.startswith("encoder.")) and p.requires_grad]

    if args.freeze_encoder_epochs > 0:
        freeze_encoder(model, True)
        enc_params = []
        print(f"[Warmup] Freeze encoder for {args.freeze_encoder_epochs} epoch(s)")

    optimizer = torch.optim.AdamW(
        [
            {"params": dec_params, "lr": args.lr},
            {"params": enc_params, "lr": args.encoder_lr},
        ],
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")

    for xb, yb, _ in val_loader:
        debug_print_batch_stats_once("val", xb, yb)
        break

    for epoch in range(1, args.epochs + 1):
        if (epoch == args.freeze_encoder_epochs + 1) and (args.freeze_encoder_epochs > 0):
            freeze_encoder(model, False)
            enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
            dec_params = [p for n, p in model.named_parameters() if (not n.startswith("encoder.")) and p.requires_grad]
            optimizer = torch.optim.AdamW(
                [
                    {"params": dec_params, "lr": args.lr},
                    {"params": enc_params, "lr": args.encoder_lr},
                ],
                weight_decay=args.weight_decay,
            )
            print("[Warmup] Encoder unfrozen. Optimizer rebuilt.")

        train_loss, skip_steps, grad_bad_steps = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        val_loss = evaluate(model, val_loader, device, args)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | skip={skip_steps} | grad_bad={grad_bad_steps}")

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.out_dir, "best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_path)
            print("  [Saved] best ->", ckpt_path)

if __name__ == "__main__":
    main()
"""
with open("train.py", "w", encoding="utf-8") as f:
    f.write(train_py)
print("Wrote train.py")



## === cell 3
ves1 = r"""
import os
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import tifffile
except Exception:
    tifffile = None

IGNORE_INDEX = 127  # keep consistent with your losses

_MEMMAP_CACHE: Dict[str, np.ndarray] = {}

def _read_slice_memmap(path: str) -> np.ndarray:
    # Bugfix: tifffile.memmap can fail depending on file structure / runtime;
    # fall back to imread to prevent MAE crashing.
    arr = _MEMMAP_CACHE.get(path, None)
    if arr is not None:
        return arr
    if tifffile is not None:
        try:
            arr = tifffile.memmap(path, mode="r")
        except Exception:
            arr = tifffile.imread(path)
    else:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(path)
    _MEMMAP_CACHE[path] = arr
    return arr

def _load_png_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def _normalize_volume(x: np.ndarray, clip_min=0.0, clip_max=200.0) -> np.ndarray:
    # Bugfix: keep original dtype for correct uint16 handling (previous code converted to float
    # before checking dtype, so the uint16 scaling branch never executed).
    x = np.asarray(x)
    orig_dtype = x.dtype
    x = np.clip(x, clip_min, clip_max).astype(np.float32)
    if orig_dtype == np.uint16:
        denom = float(np.iinfo(np.uint16).max)
    else:
        denom = 255.0
    x = x / denom
    x = (x - 0.5) / 0.5
    return x.astype(np.float32)

def sample_depth_indices(total_slices: int, num_frames: int, mode: str) -> List[int]:
    if num_frames > total_slices:
        raise ValueError(f"num_frames={num_frames} > total_slices={total_slices}")

    if mode == "rand_contig":
        s = random.randint(0, total_slices - num_frames)
        return list(range(s, s + num_frames))

    if mode == "center_contig":
        s = (total_slices - num_frames) // 2
        return list(range(s, s + num_frames))

    if mode == "odd_subsample":
        odds = list(range(1, total_slices, 2))
        if len(odds) >= num_frames:
            idx = np.linspace(0, len(odds) - 1, num_frames).round().astype(int)
            return [odds[i] for i in idx]
        return sample_depth_indices(total_slices, num_frames, "rand_contig")

    if mode == "rand_stride2":
        stride = 2
        needed = 1 + (num_frames - 1) * stride
        if needed <= total_slices:
            s = random.randint(0, total_slices - needed)
            return [s + i * stride for i in range(num_frames)]
        return sample_depth_indices(total_slices, num_frames, "rand_contig")

    raise ValueError(f"Unknown depth mode: {mode}")

def build_tile_coords(roi_mask: np.ndarray, tile_size: int, stride: int, min_roi_frac: float = 0.05) -> List[Tuple[int, int]]:
    h, w = roi_mask.shape
    roi = (roi_mask > 0).astype(np.uint8)

    coords = []
    tile_area = tile_size * tile_size
    thr = int(tile_area * min_roi_frac)

    integ = cv2.integral(roi)

    def rect_sum(x1, y1, x2, y2):
        return integ[y2, x2] - integ[y1, x2] - integ[y2, x1] + integ[y1, x1]

    for y in range(0, h - tile_size + 1, stride):
        y2 = y + tile_size
        for x in range(0, w - tile_size + 1, stride):
            x2 = x + tile_size
            s = rect_sum(x, y, x2, y2)
            if s >= thr:
                coords.append((x, y))
    return coords

@dataclass
class VesuviusDatasetConfig:
    data_root: str = "/kaggle/input/vesuvius-challenge-ink-detection"
    split: str = "train"
    # Bugfix: this dataset only contains train fragments 1 and 2 in this environment.
    fragment_ids: Tuple[str, ...] = ("1", "2")

    tile_size: int = 64
    stride: int = 64

    num_frames: int = 24
    depth_mode: str = "rand_contig"
    total_slices: int = 65

    clip_min: float = 0.0
    clip_max: float = 200.0

    repeat: int = 1

    # NOTE: kept for backward compatibility with earlier notebooks.
    pos_prob: float = 0.5
    min_roi_frac: float = 0.05

    # Bugfix: mae.py passes mask_ratio into VesuviusDatasetConfig; include it to avoid
    # "unexpected keyword argument 'mask_ratio'" while preserving MAE logic.
    mask_ratio: float = 0.85

class VesuviusMAEPatchDataset(Dataset):
    def __init__(self, cfg: VesuviusDatasetConfig, is_train: bool = True):
        self.cfg = cfg
        self.is_train = is_train

        self.slice_paths: Dict[str, List[str]] = {}
        self.roi_masks: Dict[str, np.ndarray] = {}
        self.coords_all: List[Tuple[str, int, int]] = []

        for fid in cfg.fragment_ids:
            fid = str(fid).replace("Frag", "")
            base = os.path.join(cfg.data_root, cfg.split, fid)
            vol_dir = os.path.join(base, "surface_volume")
            mask_path = os.path.join(base, "mask.png")

            paths = sorted(glob.glob(os.path.join(vol_dir, "*.tif")))
            if len(paths) == 0:
                raise FileNotFoundError(vol_dir)
            self.slice_paths[fid] = paths

            roi = _load_png_gray(mask_path)
            self.roi_masks[fid] = roi

            coords = build_tile_coords(roi, cfg.tile_size, cfg.stride, cfg.min_roi_frac)
            for (x, y) in coords:
                self.coords_all.append((fid, x, y))

        if len(self.coords_all) == 0:
            raise RuntimeError("No tiles found. Check mask.png / tile_size / stride.")

        self._len = len(self.coords_all) * max(1, cfg.repeat)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        fid, x, y = self.coords_all[idx % len(self.coords_all)]
        paths = self.slice_paths[fid]
        total = len(paths)

        depth_mode = self.cfg.depth_mode if self.is_train else "center_contig"
        z_idx = sample_depth_indices(total, self.cfg.num_frames, depth_mode)

        tile = np.empty((self.cfg.tile_size, self.cfg.tile_size, len(z_idx)), dtype=np.float32)
        for i, z in enumerate(z_idx):
            arr = _read_slice_memmap(paths[z])
            patch = np.asarray(arr[y:y+self.cfg.tile_size, x:x+self.cfg.tile_size], dtype=np.float32)
            tile[..., i] = patch

        tile = _normalize_volume(tile, self.cfg.clip_min, self.cfg.clip_max)
        video = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(1)
        return video

# Note: VesuviusSegPatchDataset is intentionally not included in this file;
# segmentation training uses vesuvius_data_2.py.
"""
with open("vesuvius_data_1.py", "w", encoding="utf-8") as f:
    f.write(ves1)
print("Wrote vesuvius_data_1.py (robust tif reading + corrected uint16 normalization)")



## === cell 4
ves2 = r"""
import os
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import tifffile
import cv2

IGNORE_INDEX = 127

_DEBUG_PRINTED = set()

def debug_print_batch_stats_once(prefix: str, x: torch.Tensor, y: torch.Tensor, ignore_index: int = IGNORE_INDEX):
    key = str(prefix)
    if key in _DEBUG_PRINTED:
        return
    _DEBUG_PRINTED.add(key)

    with torch.no_grad():
        valid = (y != float(ignore_index)).float()
        valid_frac = float(valid.mean().cpu().item())

        pos = (y > 0.5).float()
        pos_frac_all = float(pos.mean().cpu().item())
        pos_frac_valid = float((pos * valid).sum().cpu().item() / (valid.sum().cpu().item() + 1e-6))

        uniq = torch.unique(y).detach().cpu().tolist()

    print(f"[DebugBatch:{prefix}] x={tuple(x.shape)} y={tuple(y.shape)} uniq={uniq}")
    print(f"[DebugBatch:{prefix}] valid_frac={valid_frac:.6f}  pos_frac_all={pos_frac_all:.6f}  pos_frac_valid={pos_frac_valid:.6f}")

class _FragVolumeCache:
    def __init__(self, max_frags: int = 2):
        self.max_frags = int(max_frags)
        self.cache: Dict[str, List[np.ndarray]] = {}
        self.order: List[str] = []

    def get(self, frag_key: str):
        v = self.cache.get(frag_key)
        if v is None:
            return None
        try:
            self.order.remove(frag_key)
        except ValueError:
            pass
        self.order.append(frag_key)
        return v

    def put(self, frag_key: str, vol: List[np.ndarray]):
        if frag_key in self.cache:
            self.cache[frag_key] = vol
            try:
                self.order.remove(frag_key)
            except ValueError:
                pass
            self.order.append(frag_key)
            return
        self.cache[frag_key] = vol
        self.order.append(frag_key)
        if len(self.order) > self.max_frags:
            old = self.order.pop(0)
            self.cache.pop(old, None)

_FRAG_VOL_CACHE = _FragVolumeCache(max_frags=2)

def get_fragment_volume_memmaps(slice_paths: List[str]) -> List[np.ndarray]:
    frag_key = f"{slice_paths[0]}|{slice_paths[-1]}|{len(slice_paths)}"
    vol = _FRAG_VOL_CACHE.get(frag_key)
    if vol is not None:
        return vol
    vol = [tifffile.memmap(p, mode="r") for p in slice_paths]
    _FRAG_VOL_CACHE.put(frag_key, vol)
    return vol

def load_png_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def normalize_tile(tile: np.ndarray, clip_min=0.0, clip_max=200.0) -> np.ndarray:
    tile = np.clip(tile, clip_min, clip_max)
    tile = tile / 255.0
    tile = (tile - 0.5) / 0.5
    return tile.astype(np.float32)

def sample_depth_indices(total_slices: int, num_frames: int, mode: str) -> List[int]:
    if num_frames > total_slices:
        raise ValueError(f"num_frames={num_frames} > total_slices={total_slices}")

    if mode == "rand_contig":
        s = random.randint(0, total_slices - num_frames)
        return list(range(s, s + num_frames))

    if mode == "center_contig":
        s = (total_slices - num_frames) // 2
        return list(range(s, s + num_frames))

    if mode == "odd_subsample":
        odds = list(range(1, total_slices, 2))
        if len(odds) >= num_frames:
            idx = np.linspace(0, len(odds) - 1, num_frames).round().astype(int)
            return [odds[i] for i in idx]
        return sample_depth_indices(total_slices, num_frames, "rand_contig")

    if mode == "rand_stride2":
        stride = 2
        needed = 1 + (num_frames - 1) * stride
        if needed <= total_slices:
            s = random.randint(0, total_slices - needed)
            return [s + i * stride for i in range(num_frames)]
        return sample_depth_indices(total_slices, num_frames, "rand_contig")

    raise ValueError(f"Unknown depth mode: {mode}")

def integral_image_uint8(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0).astype(np.int64)
    ii = np.pad(m, ((1, 0), (1, 0)), constant_values=0)
    ii = ii.cumsum(0).cumsum(1)
    return ii

def rect_sum(ii: np.ndarray, x: int, y: int, sz: int) -> int:
    x2 = x + sz
    y2 = y + sz
    return int(ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x])

@dataclass
class VesuviusDatasetConfig:
    data_root: str = "/kaggle/input/vesuvius-challenge-ink-detection"
    split: str = "train"
    fragment_ids: Tuple[str, ...] = ("1", "2", "3")

    tile_size: int = 64
    stride: int = 64

    num_frames: int = 24
    depth_mode: str = "rand_contig"
    total_slices: int = 65

    clip_min: float = 0.0
    clip_max: float = 200.0

    pos_ratio: float = 0.5
    pos_tile_min_frac: float = 0.01
    valid_tile_min_frac: float = 0.5
    repeat: int = 1

class VesuviusSegPatchDataset(Dataset):
    def __init__(self, cfg: VesuviusDatasetConfig, is_train: bool = True):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train

        self.slice_paths: Dict[str, List[str]] = {}
        self.ink_labels: Dict[str, np.ndarray] = {}
        self.roi_masks: Dict[str, np.ndarray] = {}

        self.coords_all: List[Tuple[str, int, int]] = []
        self.coords_pos: List[Tuple[str, int, int]] = []
        self.coords_neg: List[Tuple[str, int, int]] = []

        ts = cfg.tile_size
        tile_area = ts * ts
        pos_thr = int(tile_area * cfg.pos_tile_min_frac)
        valid_thr = int(tile_area * cfg.valid_tile_min_frac)

        for raw_fid in cfg.fragment_ids:
            fid = str(raw_fid).replace("Frag", "")
            base = os.path.join(cfg.data_root, cfg.split, fid)
            vol_dir = os.path.join(base, "surface_volume")
            mask_path = os.path.join(base, "mask.png")
            ink_path = os.path.join(base, "inklabels.png")

            paths = sorted(glob.glob(os.path.join(vol_dir, "*.tif")))
            if len(paths) == 0:
                raise FileNotFoundError(vol_dir)

            roi = load_png_gray(mask_path)
            ink = load_png_gray(ink_path)

            self.slice_paths[fid] = paths
            self.roi_masks[fid] = roi
            self.ink_labels[fid] = ink

            H, W = roi.shape
            gx = (W - ts) // cfg.stride + 1
            gy = (H - ts) // cfg.stride + 1
            total_grid = gx * gy

            print(f"[Index] frag={fid} scanning tiles... grid={gx}x{gy} (~{total_grid}) pos_thr={pos_thr}/{tile_area} valid_thr={valid_thr}/{tile_area}")

            roi01 = (roi > 0).astype(np.uint8)
            ink01 = ((ink > 0) & (roi > 0)).astype(np.uint8)

            ii_roi = integral_image_uint8(roi01)
            ii_ink = integral_image_uint8(ink01)

            for y in range(0, H - ts + 1, cfg.stride):
                for x in range(0, W - ts + 1, cfg.stride):
                    vcnt = rect_sum(ii_roi, x, y, ts)
                    if vcnt < valid_thr:
                        continue

                    pcnt = rect_sum(ii_ink, x, y, ts)
                    self.coords_all.append((fid, x, y))
                    if pcnt >= pos_thr:
                        self.coords_pos.append((fid, x, y))
                    else:
                        self.coords_neg.append((fid, x, y))

            print(f"[Index] frag={fid} done. tiles_all={len(self.coords_all)} tiles_pos={len(self.coords_pos)} (pos_tile_min_frac={cfg.pos_tile_min_frac}, pos_ratio={cfg.pos_ratio})")

        if len(self.coords_all) == 0:
            raise RuntimeError("No tiles found. Check mask.png / tile_size / stride / valid_tile_min_frac.")
        if self.is_train and (len(self.coords_pos) == 0 or len(self.coords_neg) == 0):
            print("[WARN] pos or neg tiles empty. You may need to adjust pos_tile_min_frac / valid_tile_min_frac.")

        self._len = len(self.coords_all) * max(1, int(cfg.repeat))

    def __len__(self):
        return self._len

    def pick_coord(self, idx: int) -> Tuple[str, int, int]:
        if self.is_train and len(self.coords_pos) > 0 and len(self.coords_neg) > 0:
            if random.random() < float(self.cfg.pos_ratio):
                return random.choice(self.coords_pos)
            return random.choice(self.coords_neg)
        return self.coords_all[idx % len(self.coords_all)]

    def __getitem__(self, idx: int):
        fid, x, y = self.pick_coord(idx)
        paths = self.slice_paths[fid]
        total = len(paths)

        depth_mode = self.cfg.depth_mode if self.is_train else "center_contig"
        z_idx = sample_depth_indices(total, self.cfg.num_frames, depth_mode)

        vol = get_fragment_volume_memmaps(paths)

        tile = np.empty((self.cfg.tile_size, self.cfg.tile_size, len(z_idx)), dtype=np.float32)
        for i, z in enumerate(z_idx):
            arr = vol[z]
            patch = np.asarray(arr[y:y+self.cfg.tile_size, x:x+self.cfg.tile_size], dtype=np.float32)
            tile[..., i] = patch

        tile = normalize_tile(tile, self.cfg.clip_min, self.cfg.clip_max)
        video = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(1)

        ink = self.ink_labels[fid][y:y+self.cfg.tile_size, x:x+self.cfg.tile_size]
        roi = self.roi_masks[fid][y:y+self.cfg.tile_size, x:x+self.cfg.tile_size]
        mask = (ink > 0).astype(np.uint8)
        mask[roi == 0] = IGNORE_INDEX
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        return video, mask_t, (fid, x, y)
"""
with open("vesuvius_data_2.py", "w", encoding="utf-8") as f:
    f.write(ves2)
print("Wrote vesuvius_data_2.py")



## === cell 5
unetr_py = r"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEConfig, VideoMAEModel

IGNORE_INDEX = 127

def _strip_prefix(k: str) -> str:
    for p in ["model.", "net.", "encoder.", "videomae.", "module.", "model_state."]:
        if k.startswith(p):
            return k[len(p):]
    return k

def _pick_state_dict(ckpt: dict):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            return ckpt["model_state"]
    return ckpt

@torch.no_grad()
def load_videomae_encoder_from_mae_ckpt(encoder: VideoMAEModel, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _pick_state_dict(ckpt)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {type(state)}")

    new_state = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        kk = _strip_prefix(k)

        if any(s in kk for s in ["decoder", "mask_token", "decoder_pos_embed", "encoder_to_decoder"]):
            continue

        for p in ["videomae.videomae.", "model.videomae.", "videomae."]:
            if kk.startswith(p):
                kk = kk[len(p):]
                break

        new_state[kk] = v

    missing, unexpected = encoder.load_state_dict(new_state, strict=False)
    print(f"[load_videomae_encoder_from_mae_ckpt] loaded from {ckpt_path}")
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if len(unexpected) > 0:
        print("  unexpected (first 5):", unexpected[:5])
    return missing, unexpected

def _gn_groups(ch: int, max_groups: int = 8) -> int:
    g = min(max_groups, ch)
    while g > 1:
        if ch % g == 0:
            return g
        g -= 1
    return 1

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, max_groups=8):
        super().__init__()
        g = _gn_groups(out_ch, max_groups)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, max_groups=8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvGNAct(out_ch + skip_ch, out_ch, max_groups=max_groups)
        self.conv2 = ConvGNAct(out_ch, out_ch, max_groups=max_groups)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv2(self.conv1(x))
        return x

class VideoMAEUNETR2D(nn.Module):
    def __init__(self, tile_size=64, num_frames=24, intermediate_size=3072, use_layers=(3, 6, 9, 12), ch=256, max_gn_groups=8):
        super().__init__()
        tubelet_size = 2 if (num_frames % 2 == 0) else 1

        self.vcfg = VideoMAEConfig(
            image_size=tile_size,
            patch_size=16,
            num_channels=1,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=intermediate_size,
        )
        self.encoder = VideoMAEModel(self.vcfg)
        self.use_layers = tuple(use_layers)

        self.patch = self.vcfg.patch_size
        self.Hp = tile_size // self.patch
        self.Wp = tile_size // self.patch
        self.Tp = num_frames // tubelet_size
        self.D = self.vcfg.hidden_size

        self.proj = nn.ModuleDict({str(l): nn.Conv2d(self.D, ch, kernel_size=1) for l in self.use_layers})

        self.up1 = UpBlock(ch, ch, ch, max_groups=max_gn_groups)
        self.up2 = UpBlock(ch, ch, ch, max_groups=max_gn_groups)
        self.up3 = UpBlock(ch, ch, ch, max_groups=max_gn_groups)
        self.up4 = UpBlock(ch, 0,  ch, max_groups=max_gn_groups)
        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    def tokens_to_2d(self, hs: torch.Tensor) -> torch.Tensor:
        B, L, D = hs.shape
        n_hw = self.Hp * self.Wp
        expected = self.Tp * n_hw

        if L == expected + 1:
            x = hs[:, 1:, :]
            Tp = self.Tp
        elif L == expected:
            x = hs
            Tp = self.Tp
        else:
            if (L - 1) % n_hw == 0:
                x = hs[:, 1:, :]
                Tp = (L - 1) // n_hw
            elif L % n_hw == 0:
                x = hs
                Tp = L // n_hw
            else:
                raise RuntimeError(f"Cannot reshape tokens: hs={hs.shape}, Hp={self.Hp},Wp={self.Wp},Tp={self.Tp}")

        x = x.reshape(B, Tp, self.Hp, self.Wp, D)
        x = x.mean(dim=1)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            out = self.encoder(video.float(), output_hidden_states=True, return_dict=True)
            hss = out.hidden_states

            feats = {}
            for l in self.use_layers:
                f2d = self.tokens_to_2d(hss[l])
                feats[l] = self.proj[str(l)](f2d)

        x = feats[self.use_layers[-1]]
        x = self.up1(x, feats[self.use_layers[-2]])
        x = self.up2(x, feats[self.use_layers[-3]])
        x = self.up3(x, feats[self.use_layers[-4]])
        x = self.up4(x, None)
        return self.head(x)

def masked_bce_dice_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    pos_weight: float = 10.0,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    eps: float = 1e-6,
):
    logits = logits.float()
    y = y.float()

    valid = (y != float(ignore_index)).float()
    y_bin = (y > 0.5).float() * valid

    pw = torch.as_tensor([pos_weight], device=logits.device, dtype=torch.float32)

    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_bin, reduction="none", pos_weight=pw)
    bce = (bce * valid).sum() / (valid.sum() + eps)

    p = torch.sigmoid(logits) * valid
    inter = (p * y_bin).sum()
    den = p.sum() + y_bin.sum()
    dice = (2.0 * inter + eps) / (den + eps)
    dice_loss = 1.0 - dice

    loss = bce_weight * bce + dice_weight * dice_loss
    return loss, bce.detach(), dice_loss.detach()

@torch.no_grad()
def logits_stats(logits: torch.Tensor) -> dict:
    t = logits.float()
    return {
        "mean": float(t.mean().cpu()),
        "std": float(t.std().cpu()),
        "min": float(t.min().cpu()),
        "max": float(t.max().cpu()),
    }
"""
with open("unetr.py", "w", encoding="utf-8") as f:
    f.write(unetr_py)
print("Wrote unetr.py")



## === cell 6
mae_py = r"""
import os
import csv
import math
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

from transformers import VideoMAEConfig, VideoMAEForPreTraining
from vesuvius_data_1 import VesuviusDatasetConfig, VesuviusMAEPatchDataset

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def _seed_worker(worker_id: int):
    import random as _random
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)

def save_history_csv(path: str, rows):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "lr"])
        for r in rows:
            w.writerow(r)

def build_lr_scheduler(optimizer, total_steps, warmup_steps, min_lr=1e-6):
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def lr_at(step):
        if step < warmup_steps:
            scale = (step + 1) / max(1, warmup_steps)
        else:
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * t))
        return scale

    def step_fn(step):
        scale = lr_at(step)
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = min_lr + (base_lrs[i] - min_lr) * scale

    return step_fn

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default="/kaggle/input/vesuvius-challenge-ink-detection")
    ap.add_argument("--train_ids", nargs="+", default=["1", "2"])
    ap.add_argument("--valid_ids", nargs="+", default=["1"])

    ap.add_argument("--tile_size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)

    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument(
        "--depth_mode", type=str, default="rand_contig",
        choices=["rand_contig", "center_contig", "odd_subsample", "rand_stride2"]
    )

    ap.add_argument("--mask_ratio", type=float, default=0.85)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    ap.add_argument("--repeat", type=int, default=8)

    ap.add_argument("--num_workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))

    ap.add_argument("--out_dir", type=str, default="/kaggle/working/mae_outputs")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--fp16", dest="fp16", action="store_true")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.set_defaults(fp16=True)

    return ap.parse_args()

@torch.no_grad()
def make_bool_masked_pos(batch_size: int, num_patches: int, mask_ratio: float, device: torch.device):
    num_mask = int(mask_ratio * num_patches)
    num_mask = max(1, min(num_mask, num_patches - 1))

    scores = torch.rand((batch_size, num_patches), device=device)
    idx = scores.topk(k=num_mask, dim=1, largest=True, sorted=False).indices
    bool_masked_pos = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
    bool_masked_pos.scatter_(1, idx, True)
    return bool_masked_pos

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    print("num_workers =", args.num_workers)

    train_cfg = VesuviusDatasetConfig(
        data_root=args.data_root,
        split="train",
        fragment_ids=tuple(args.train_ids),
        tile_size=args.tile_size,
        stride=args.stride,
        num_frames=args.num_frames,
        depth_mode=args.depth_mode,
        repeat=args.repeat,
        mask_ratio=args.mask_ratio,
    )
    valid_cfg = VesuviusDatasetConfig(
        data_root=args.data_root,
        split="train",
        fragment_ids=tuple(args.valid_ids),
        tile_size=args.tile_size,
        stride=args.stride,
        num_frames=args.num_frames,
        depth_mode="center_contig",
        repeat=1,
        mask_ratio=args.mask_ratio,
    )

    train_ds = VesuviusMAEPatchDataset(train_cfg, is_train=True)
    val_ds = VesuviusMAEPatchDataset(valid_cfg, is_train=False)

    g = torch.Generator()
    g.manual_seed(args.seed)

    pw = (args.num_workers > 0)
    dl_kwargs = {}
    # Bugfix: only pass prefetch_factor/persistent_workers when using workers.
    if pw:
        dl_kwargs["prefetch_factor"] = 4
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["worker_init_fn"] = _seed_worker

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        generator=g,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        **dl_kwargs,
    )

    tubelet_size = 2 if (args.num_frames % 2 == 0) else 1

    vcfg = VideoMAEConfig(
        image_size=args.tile_size,
        patch_size=16,
        num_channels=1,
        num_frames=args.num_frames,
        tubelet_size=tubelet_size,

        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,

        decoder_num_hidden_layers=4,
        decoder_hidden_size=512,
        decoder_num_attention_heads=8,
        decoder_intermediate_size=2048,

        norm_pix_loss=True,
        mask_ratio=args.mask_ratio,
    )

    model = VideoMAEForPreTraining(vcfg).to(device)
    model.train()

    patch = vcfg.patch_size
    tube = vcfg.tubelet_size
    Hp = args.tile_size // patch
    Wp = args.tile_size // patch
    Tp = args.num_frames // tube
    num_patches = Hp * Wp * Tp
    print(f"[info] patch_size={patch} tubelet_size={tube} Hp={Hp} Wp={Wp} Tp={Tp} num_patches={num_patches}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    lr_step = build_lr_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr=args.min_lr
    )

    use_amp = bool(args.fp16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_val = float("inf")
    global_step = 0

    history_path = os.path.join(args.out_dir, f"training_history_videomae_{args.tile_size}_{args.num_frames}.csv")
    best_ckpt_path = os.path.join(args.out_dir, "best_mae.pt")
    last_ckpt_path = os.path.join(args.out_dir, "last_mae.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"[Train] epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = batch.to(device, non_blocking=True)

            lr_step(global_step)
            optimizer.zero_grad(set_to_none=True)

            B = batch.size(0)
            bool_masked_pos = make_bool_masked_pos(B, num_patches, args.mask_ratio, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(pixel_values=batch, bool_masked_pos=bool_masked_pos)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            global_step += 1

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=float(np.mean(train_losses)), lr=optimizer.param_groups[0]["lr"])

        train_loss = float(np.mean(train_losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"[Val] epoch {epoch}/{args.epochs}", leave=False)
            for batch in pbar:
                batch = batch.to(device, non_blocking=True)
                B = batch.size(0)
                bool_masked_pos = make_bool_masked_pos(B, num_patches, args.mask_ratio, device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(pixel_values=batch, bool_masked_pos=bool_masked_pos)
                    loss = out.loss

                val_losses.append(loss.item())
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(val_loss=float(np.mean(val_losses)))

        val_loss = float(np.mean(val_losses))
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr_now:.2e}")

        history.append([epoch, train_loss, val_loss, lr_now])
        save_history_csv(history_path, history)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": vcfg.to_dict(),
                "args": vars(args),
            },
            last_ckpt_path
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": vcfg.to_dict(),
                    "args": vars(args),
                },
                best_ckpt_path
            )
            print("  -> saved BEST to", best_ckpt_path)

    print("Done.")
    print("Best ckpt:", best_ckpt_path)
    print("History csv:", history_path)

if __name__ == "__main__":
    main()
"""
with open("mae.py", "w", encoding="utf-8") as f:
    f.write(mae_py)
print("Wrote mae.py")



## === cell 7
infer_py = r"""
import os
import glob
import argparse
from typing import List, Tuple, Optional, Any
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import tifffile
import csv
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from unetr import VideoMAEUNETR2D, load_videomae_encoder_from_mae_ckpt

def rle_encode(mask01: np.ndarray) -> str:
    m = mask01.astype(np.uint8)
    pixels = m.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = changes.copy()
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def _list_tif_slices(frag_root: str) -> List[str]:
    vol_dir = os.path.join(frag_root, "surface_volume")
    if not os.path.isdir(vol_dir):
        alt = os.path.join(frag_root, "surface_volumn")
        if os.path.isdir(alt):
            vol_dir = alt
    paths = sorted(glob.glob(os.path.join(vol_dir, "*.tif")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No tif slices found in: {vol_dir}")
    return paths

def write_submission_csv(path: str, ids: List[str], rles: List[str]) -> None:
    clean = []
    for s in rles:
        s = "" if s is None else str(s)
        s = re.sub(r"\s+", " ", s).strip()
        clean.append(s)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(["Id", "Predicted"])
        for fid, rle in zip(ids, clean):
            w.writerow([str(fid), rle])

def _choose_z_indices(total_slices: int, num_frames: int, mode: str, start_z: Optional[int]) -> List[int]:
    if num_frames > total_slices:
        raise ValueError(f"num_frames={num_frames} > total_slices={total_slices}")
    if start_z is not None:
        s = max(0, min(int(start_z), total_slices - num_frames))
        return list(range(s, s + num_frames))
    if mode == "front_contig":
        return list(range(0, num_frames))
    s = (total_slices - num_frames) // 2
    return list(range(s, s + num_frames))

def load_mask_png(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 0).astype(np.uint8)

def build_coords(mask01: np.ndarray, tile_size: int, stride: int) -> List[Tuple[int, int]]:
    H, W = mask01.shape
    ys = list(range(0, max(H - tile_size + 1, 1), stride))
    xs = list(range(0, max(W - tile_size + 1, 1), stride))
    if len(ys) == 0: ys = [0]
    if len(xs) == 0: xs = [0]
    y_last = max(H - tile_size, 0)
    x_last = max(W - tile_size, 0)
    if ys[-1] != y_last: ys.append(y_last)
    if xs[-1] != x_last: xs.append(x_last)

    coords: List[Tuple[int, int]] = []
    for y in ys:
        for x in xs:
            if mask01[y:y+tile_size, x:x+tile_size].sum() > 0:
                coords.append((x, y))
    return coords

def normalize_patch_u16_to_model(patch_u16: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    x = patch_u16.astype(np.float32)
    x = np.clip(x, clip_min, clip_max)
    x = x / 255.0
    x = (x - 0.5) / 0.5
    return x.astype(np.float16)

class MemmapVolume:
    def __init__(self, slice_paths: List[str], z_idx: List[int]):
        self.paths = slice_paths
        self.z_idx = list(z_idx)
        self.vol = [tifffile.memmap(self.paths[z], mode="r") for z in self.z_idx]

    def get_patch_u16(self, x0: int, y0: int, ts: int) -> np.ndarray:
        out = np.empty((len(self.z_idx), ts, ts), dtype=np.uint16)
        for i, sl in enumerate(self.vol):
            out[i] = np.asarray(sl[y0:y0+ts, x0:x0+ts], dtype=np.uint16)
        return out

class TestPatchDataset(Dataset):
    def __init__(self, vol: MemmapVolume, coords: List[Tuple[int, int]], tile_size: int, clip_min: float, clip_max: float):
        self.vol = vol
        self.coords = coords
        self.tile_size = int(tile_size)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx: int):
        x0, y0 = self.coords[idx]
        ts = self.tile_size
        patch_u16 = self.vol.get_patch_u16(x0, y0, ts)
        patch = normalize_patch_u16_to_model(patch_u16, self.clip_min, self.clip_max)
        x = torch.from_numpy(patch).unsqueeze(1)
        coord = torch.tensor([x0, y0], dtype=torch.int32)
        return x, coord

def _strip_outer_prefix(k: str) -> str:
    for p in ("model.", "net.", "module.", "pl_module.", "lit_model.", "vmae."):
        if k.startswith(p):
            return k[len(p):]
    return k

def find_ckpt_path(ckpt_dir_or_file: str) -> str:
    p = ckpt_dir_or_file.rstrip("/")
    if os.path.isfile(p) and p.lower().endswith((".pt", ".pth", ".ckpt")):
        return p
    if not os.path.isdir(p):
        raise FileNotFoundError(f"ckpt_dir not found: {p}")

    exts = ("*.pt", "*.pth", "*.ckpt")
    cands = []
    for ext in exts:
        cands.extend(glob.glob(os.path.join(p, "**", ext), recursive=True))
        cands.extend(glob.glob(os.path.join(p, ext)))

    cands = sorted(list(set(cands)))
    if len(cands) == 0:
        raise FileNotFoundError(f"No .pt/.pth/.ckpt found under: {p}")

    def score(path: str):
        name = os.path.basename(path).lower()
        s = 0
        if "best" in name: s += 30
        if "last" in name: s += 20
        if "final" in name: s += 10
        mtime = int(os.path.getmtime(path))
        return (s, mtime)

    cands = sorted(cands, key=score, reverse=True)
    return cands[0]

def _extract_state_and_hp(ckpt_obj: Any):
    hp = {}
    if isinstance(ckpt_obj, dict):
        if "hyper_parameters" in ckpt_obj and isinstance(ckpt_obj["hyper_parameters"], dict):
            hp = ckpt_obj["hyper_parameters"]
        for key in ("state_dict", "model_state", "model", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key], hp
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj, hp
    raise TypeError(f"Unrecognized checkpoint format: {type(ckpt_obj)}")

def load_seg_model(ckpt_path: str, tile_size: int, num_frames: int, mae_ckpt: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state, hp = _extract_state_and_hp(ckpt)

    tile_size = int(hp.get("tile_size", tile_size))
    num_frames = int(hp.get("num_frames", num_frames))

    net = VideoMAEUNETR2D(tile_size=tile_size, num_frames=num_frames)

    if mae_ckpt and os.path.exists(mae_ckpt):
        try:
            load_videomae_encoder_from_mae_ckpt(net.encoder, mae_ckpt)
        except Exception as e:
            print(f"[WARN] failed to load mae_ckpt='{mae_ckpt}': {e}")

    new_state = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            new_state[_strip_outer_prefix(k)] = v

    missing, unexpected = net.load_state_dict(new_state, strict=False)
    print(f"[CKPT] {ckpt_path}")
    print(f"  loaded with missing={len(missing)} unexpected={len(unexpected)}")
    return net, tile_size, num_frames

@torch.no_grad()
def predict_fragment(net, vol, mask01, tile_size, stride, clip_min, clip_max, batch_size, num_workers, use_amp):
    device = next(net.parameters()).device
    coords = build_coords(mask01, tile_size=tile_size, stride=stride)
    print(f"[Predict] HxW={mask01.shape} tiles={len(coords)} tile={tile_size} stride={stride}")

    ds = TestPatchDataset(vol=vol, coords=coords, tile_size=tile_size, clip_min=clip_min, clip_max=clip_max)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    H, W = mask01.shape
    pred = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    ts = int(tile_size)

    for xb, cb in tqdm(dl, desc="tiles", leave=False):
        xb = xb.to(device, non_blocking=True)
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = net(xb)
        else:
            logits = net(xb)

        probs = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy().astype(np.float32)
        cb = cb.cpu().numpy().astype(np.int32)
        for i in range(probs.shape[0]):
            x0, y0 = int(cb[i, 0]), int(cb[i, 1])
            pred[y0:y0+ts, x0:x0+ts] += probs[i]
            cnt[y0:y0+ts, x0:x0+ts] += 1.0

    pred = pred / np.maximum(cnt, 1.0)
    pred = pred * mask01.astype(np.float32)
    return pred

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/kaggle/input/vesuvius-challenge-ink-detection")
    ap.add_argument("--ckpt_dir", type=str, default="/kaggle/working/seg_outputs_run3")
    ap.add_argument("--mae_ckpt", type=str, default="/kaggle/working/mae_outputs/best_mae.pt")
    ap.add_argument("--out_csv", type=str, default="submission.csv")
    ap.add_argument("--tile_size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--num_frames", type=int, default=24)
    ap.add_argument("--depth_mode", type=str, default="center_contig", choices=["center_contig", "front_contig"])
    ap.add_argument("--start_z", type=int, default=-1)
    ap.add_argument("--clip_min", type=float, default=0.0)
    ap.add_argument("--clip_max", type=float, default=200.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--no_amp", action="store_true")
    if argv is None:
        import sys
        argv = sys.argv[1:]
        if ("ipykernel" in sys.argv[0]) or ("colab_kernel_launcher" in sys.argv[0]):
            argv = []
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    ckpt_path = find_ckpt_path(args.ckpt_dir)
    start_z = None if args.start_z < 0 else int(args.start_z)

    net, tile_size, num_frames = load_seg_model(ckpt_path, args.tile_size, args.num_frames, args.mae_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()

    sub = pd.read_csv(os.path.join(args.data_root, "sample_submission.csv"))
    frag_ids = sub["Id"].astype(str).tolist()

    test_root = os.path.join(args.data_root, "test")
    out_rle = []

    for fid in frag_ids:
        frag_root = os.path.join(test_root, str(fid))
        mask01 = load_mask_png(os.path.join(frag_root, "mask.png"))
        slice_paths = _list_tif_slices(frag_root)
        z_idx = _choose_z_indices(len(slice_paths), num_frames, args.depth_mode, start_z)
        print(f"\n=== Fragment {fid} | slices={len(slice_paths)} use={z_idx[0]}..{z_idx[-1]} (T={num_frames}) ===")
        vol = MemmapVolume(slice_paths=slice_paths, z_idx=z_idx)

        prob = predict_fragment(
            net=net, vol=vol, mask01=mask01,
            tile_size=int(tile_size), stride=int(args.stride),
            clip_min=float(args.clip_min), clip_max=float(args.clip_max),
            batch_size=int(args.batch_size), num_workers=int(args.num_workers),
            use_amp=(not args.no_amp),
        )
        pred_bin = (prob > float(args.threshold)).astype(np.uint8)
        out_rle.append(rle_encode(pred_bin))

    write_submission_csv(args.out_csv, frag_ids, out_rle)

    print("\n[CHECK] first 2 lines:")
    with open(args.out_csv, "r", encoding="utf-8") as f:
        for _ in range(2):
            print(f.readline().rstrip("\n"))
    print("[CHECK] line count =", sum(1 for _ in open(args.out_csv, "r", encoding="utf-8")))
    print(f"\nsaved: {args.out_csv}")

if __name__ == "__main__":
    main()
"""
with open("infer_submit.py", "w", encoding="utf-8") as f:
    f.write(infer_py)
print("Wrote infer_submit.py")




## === cell 8
def run_cmd(cmd):
    print("Running:", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print("\n[ERROR] Command failed. Output:\n")
        print(p.stdout)
        raise subprocess.CalledProcessError(p.returncode, cmd)
    return p.stdout


mae_ckpt = "/kaggle/working/mae_outputs/best_mae.pt"
if not os.path.exists(mae_ckpt):
    cmd = [
        sys.executable,
        "mae.py",
        "--data_root",
        "/kaggle/input/vesuvius-challenge-ink-detection",
        "--train_ids",
        "2",
        "--valid_ids",
        "1",
        "--tile_size",
        "64",
        "--stride",
        "96",
        "--num_frames",
        "24",
        "--depth_mode",
        "rand_contig",
        "--mask_ratio",
        "0.85",
        "--batch_size",
        "32",
        "--epochs",
        "13",
        "--lr",
        "1e-4",
        "--weight_decay",
        "0.05",
        "--warmup_epochs",
        "2",
        "--repeat",
        "1",
        "--out_dir",
        "/kaggle/working/mae_outputs",
        "--num_workers",
        "4",
    ]
    run_cmd(cmd)
else:
    print("[Skip] Found existing MAE checkpoint:", mae_ckpt)

cmd = [sys.executable, "train.py", "--num_workers", "4"]
run_cmd(cmd)

cmd = [
    sys.executable,
    "infer_submit.py",
    "--ckpt_dir",
    "/kaggle/working/seg_outputs_run",
    "--mae_ckpt",
    "/kaggle/working/mae_outputs/best_mae.pt",
    "--tile_size",
    "64",
    "--stride",
    "64",
    "--num_frames",
    "24",
    "--depth_mode",
    "center_contig",
    "--batch_size",
    "8",
    "--num_workers",
    "0",
    "--out_csv",
    "submission.csv",
]
run_cmd(cmd)

assert os.path.exists("submission.csv"), "submission.csv was not created"
print("submission.csv size:", os.path.getsize("submission.csv"), "bytes")
print("Done.")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_55/2484765191.py in <cell line: 0>()
     48         "4",
     49     ]
---> 50     run_cmd(cmd)
     51 else:
     52     print("[Skip] Found existing MAE checkpoint:", mae_ckpt)

/tmp/ipykernel_55/2484765191.py in run_cmd(cmd)
      6         print("\n[ERROR] Command failed. Output:\n")
      7         print(p.stdout)
----> 8         raise subprocess.CalledProcessError(p.returncode, cmd)
      9     return p.stdout
     10 

CalledProcessError: Command '['/usr/bin/python3', 'mae.py', '--data_root', '/kaggle/input/vesuvius-challenge-ink-detection', '--train_ids', '2', '--valid_ids', '1', '--tile_size', '64', '--stride', '96', '--num_frames', '24', '--depth_mode', 'rand_contig', '--mask_ratio', '0.85', '--batch_size', '32', '--epochs', '13', '--lr', '1e-4', '--weight_decay', '0.05', '--warmup_epochs', '2', '--repeat', '1', '--out_dir', '/kaggle/working/mae_outputs', '--num_workers', '4']' returned non-zero exit status 1.
