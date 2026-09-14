# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8960411000302206

# 6. Current score

0.17451

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it always produces a valid `submission.csv` by (1) making the model checkpoint loading robust (so `vit_model` is always defined, even if the external dataset path is missing) and (2) ensuring prediction rows align 1:1 with `sample_submission.csv` (correct length, unique `image_id`s, and correct ordering). I also remove the data loader shuffle to keep deterministic ordering and avoid accidental duplication/misalignment when using TTA. The core inference logic (ViT forward + softmax + argmax, optional TTA averaging) is preserved; changes are limited to stability, alignment, and preventing runtime errors.'
- What this solution (achieved 0.17451) has done: 'I fix the runtime error by ensuring the ViT model and the test-time transforms use the same expected input size: torchvision’s `vit_b_16` defaults to `image_size=224`, but your pipeline resizes to 384. The minimal, score-improving correction is to set `img_size=224` and simplify the crop so every test image deterministically becomes 224×224, preventing the assertion failure and avoiding unintended distribution shift. I also make TTA deterministic-safe by using only shape-preserving, non-random flips/rotations (still TTA averaging, same core logic) so predictions are stable and aligned. Finally, I keep the submission alignment logic intact so the produced `submission.csv` matches `sample_submission.csv` exactly.'
- What this solution (achieved 0.17451) has done: 'Your current score suggests the model weights being used at inference are likely not the trained Cassava checkpoint (often falling back to ImageNet-initialized ViT with a random 5-class head), which yields near-random accuracy. I make the checkpoint loading deterministic and stricter: first search for likely Cassava-trained ViT checkpoints, then load them into a ViT-B/16 with a 5-class head using `strict=True` where possible (and only fall back to `strict=False` if keys are slightly mismatched). I also fix a small but important inference detail: average TTA probabilities per sample correctly by reshaping to `[B, n_ttas, C]` instead of splitting, which is safer and avoids edge-case mis-aggregation. These are minimal changes that preserve your core logic (ViT + softmax + TTA mean + argmax) but should move accuracy substantially toward the target by actually using the intended trained weights.'
- What this solution (achieved 0.17451) has done: 'Your score is far below the target, so the most likely issue is that inference is still using the wrong weights (ImageNet backbone + random 5-class head), which produces near-random predictions. I keep your ViT+softmax+TTA+argmax logic intact, but make checkpoint loading more “Cassava-aware” by (1) prioritizing checkpoints that look like fine-tuned cassava classifiers and (2) automatically adapting common head key names (e.g., `head.weight` vs `heads.head.weight`) while requiring the final head shape to match 5 classes. This increases the chance we actually load a real cassava-trained ViT state_dict (instead of a partial/non-matching one) without changing model architecture or inference semantics. I also ensure `torch.inference_mode()` is used for safe speed/determinism (no numerical approximation changes intended) and keep submission alignment unchanged.'
- What this solution (achieved 0.17451) has done: 'Your current gap to the target is large (0.1745 vs 0.8960), and the most likely reason is still that inference is not using a real Cassava-finetuned checkpoint, so predictions remain near-random. I keep your exact ViT+softmax+TTA-mean+argmax inference logic, but make checkpoint loading “shape-aware” for torchvision ViT by (1) recognizing common checkpoint formats (including Lightning) and (2) remapping not only head keys but also the MLP head key path used by torchvision (`heads.head.*`) vs other implementations (`head.*`, `classifier.*`, `fc.*`) while ensuring the final head has 5 outputs. I also switch TTA transforms from `Random*` to deterministic equivalents (functional flips/rotate) to avoid any hidden nondeterminism while keeping the same set of TTAs. These minimal changes should substantially increase the chance you actually load the intended trained weights and therefore move accuracy toward the target.'
- What this solution (achieved 0.17451) has done: 'Your current score (0.17451) is far below the target (0.89604), so we should improve accuracy, not reduce it. The most likely cause is still that you’re *not actually loading a Cassava-finetuned checkpoint* (your loader usually falls back to ImageNet ViT with a random 5-class head), which score near-random. I keep your exact inference core (ViT -> softmax -> TTA mean -> argmax) and only make checkpoint loading more robust by (1) allowing common key-path differences for torchvision ViT (encoder vs vit.* vs transformer.*) and (2) accepting checkpoints that don’t include the head by loading the backbone strictly and the head non-strictly, instead of rejecting them outright. I also ensure the DataLoader uses `persistent_workers` only when valid (avoids worker issues) but keep ordering/semantics identical.'
- What this solution (achieved 0.19395) has done: 'We need to move accuracy up substantially toward the target, and your current 0.174 suggests the Cassava finetuned weights still aren’t being loaded, so we make checkpoint selection stricter and prefer only checkpoints that actually contain a 5-class head (or at least a clearly ViT-shaped backbone) instead of accepting weak “backbone-only” loads that effectively behave like an untrained head. To keep core logic identical (ViT forward → softmax → TTA mean → argmax), we won’t change the model, transforms, TTA set, or inference flow—only the checkpoint discovery/loading criteria and key remapping robustness so the intended finetuned weights are actually used. We also add a lightweight sanity print of the loaded head weight shape to confirm we’re not silently running an ImageNet/random head. These minimal changes are directly aimed at improving score by ensuring inference uses meaningful Cassava-trained parameters.'
- What this solution (achieved 0.17451) has done: 'Your score is far below the target, so we should increase accuracy with the smallest changes that make the biggest difference without changing the core ViT+softmax+TTA-mean+argmax logic. The main likely issue is still that you’re not actually loading the intended Cassava-finetuned weights; the current loader scans all of `/kaggle/input/**` (slow/noisy) and often falls back to ImageNet ViT with a randomly initialized 5-class head, which performs near-random. I restrict checkpoint search to the competition dataset directory and `/kaggle/input/*` top-level datasets, and I only accept checkpoints that contain a valid 5-class head (reject backbone-only loads) to avoid submitting random-head predictions. I also add a lightweight sanity check that prints the head weight std after loading (to detect accidental random head) while keeping transforms, TTA set, and inference semantics unchanged.'
- What this solution (achieved 0.17451) has done: 'We need to move accuracy up a lot toward the 0.896 target, and the current ~0.17 strongly suggests the model is still effectively “untrained for Cassava” at inference (fallback ImageNet backbone + randomly initialized 5-class head, or a wrong/partial checkpoint). I keep your exact core inference (ViT-B/16 → softmax → TTA mean → argmax) and only make checkpoint loading actually succeed by (1) allowing backbone-only checkpoints but then *building* the 5-class head from the checkpoint when it matches, and (2) adding robust key remapping for torchvision ViT’s `heads.head.*` and common Lightning `state_dict` formats. I also ensure we never silently accept a random head by verifying the loaded head parameters changed from initialization (simple checksum via std/mean) and prefer checkpoints that contain a 5-class head but still fall back to “best available” backbone+head when needed. No training is introduced; transforms/TTA/data order/submission alignment remain the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.17451) has done: 'We need to move accuracy up a lot toward the 0.896 target, so we should fix the most likely cause of near-random predictions: the inference is still effectively using an untrained/random 5-class head because no real Cassava-finetuned checkpoint is being found/loaded. I keep your exact core inference (torchvision ViT-B/16 → softmax → TTA mean → argmax) and only make checkpoint discovery/loading actually find a plausible cassava checkpoint by (1) preferring checkpoints inside `/kaggle/input/**` and `/kaggle/working/**`, (2) accepting common “best.pth”/“model.pth” style checkpoints even when they don’t contain the head under the exact torchvision key, and (3) allowing safe head-key remapping plus partial loading, but only when the head is verifiably 5-class (otherwise reject to avoid random-head submissions). I also add a tiny runtime sanity check: if no suitable 5-class head checkpoint is found, we still write a valid submission but clearly print that we’re in fallback mode (so you can confirm why the score would stay low).'
- What this solution (achieved 0.17451) has done: 'Your gap to the target is very large (0.1745 vs 0.8960), and this kind of score usually happens when inference is effectively using an ImageNet ViT backbone with a random/incorrect 5-class head because the finetuned Cassava checkpoint is not being found/loaded. I keep your exact inference core (torchvision ViT-B/16 → softmax → optional deterministic TTA mean → argmax) and focus only on making checkpoint discovery/loading actually succeed by (1) restricting and prioritizing likely locations/files, (2) supporting common serialization formats (Lightning, timm-style), and (3) more complete key remapping into torchvision ViT’s parameter names (including patch embedding/positional/class token differences that currently prevent most real ViT checkpoints from loading). I also ensure the loader prefers checkpoints that contain a valid 5-class head, but if only a backbone is found, it still load it (better than pure ImageNet) while clearly reporting what happened. Submission writing/alignment remains unchanged and the script still produces `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True


def _load_vit_model(device, num_classes: int):
    """
    Change (score improvement toward target):
    - Your score (~0.17) strongly suggests the finetuned Cassava checkpoint still isn't being
      loaded (model behaves close to random / wrong head).
    - Keep the exact ViT-B/16 inference (softmax -> (TTA mean) -> argmax), but make checkpoint
      loading much more likely to succeed by:
        (1) prioritizing likely roots/files (avoid noisy recursive scan of all inputs first)
        (2) extracting state_dict from common checkpoint formats
        (3) remapping common key names (timm/Lightning/custom) into torchvision ViT keys,
            including patch embedding + cls/pos embedding + encoder naming.
      This does NOT change model architecture or inference semantics; it only helps load
      the intended weights, which should move accuracy toward the target.
    """
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    def build_base():
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
        return model

    search_roots_primary = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/working",
    ]
    search_roots_secondary = [
        "/kaggle/input",
    ]

    exts = ("*.pt", "*.pth", "*.bin", "*.ckpt")

    def find_ckpts(roots, recursive: bool):
        ckpts = []
        for root in roots:
            if recursive:
                for ext in exts:
                    ckpts.extend(
                        glob.glob(os.path.join(root, "**", ext), recursive=True)
                    )
            else:
                for ext in exts:
                    ckpts.extend(glob.glob(os.path.join(root, ext), recursive=False))
                    ckpts.extend(
                        glob.glob(os.path.join(root, "*", ext), recursive=False)
                    )
                    ckpts.extend(
                        glob.glob(os.path.join(root, "*", "*", ext), recursive=False)
                    )
        return ckpts

    all_ckpts = []
    all_ckpts.extend(find_ckpts(search_roots_primary, recursive=False))
    all_ckpts.extend(find_ckpts(search_roots_primary, recursive=True))
    all_ckpts.extend(find_ckpts(search_roots_secondary, recursive=False))
    all_ckpts.extend(find_ckpts(search_roots_secondary, recursive=True))

    def ckpt_priority(p: str) -> int:
        lname = os.path.basename(p).lower()
        parent = os.path.dirname(p).lower()
        s = 10_000
        for kw, bonus in [
            ("cassava", 5000),
            ("leaf", 1800),
            ("disease", 1800),
            ("cls", 300),
            ("class", 300),
            ("vit", 1200),
            ("deit", 700),
            ("finetune", 900),
            ("finetuned", 900),
            ("best", 700),
            ("final", 500),
            ("fold", 200),
        ]:
            if kw in lname or kw in parent:
                s -= bonus
        if "imagenet" in lname:
            s += 2500
        if "optimizer" in lname or "sched" in lname or "scheduler" in lname:
            s += 1500
        if "ema" in lname and "model" not in lname:
            s += 400
        s += min(600, p.count(os.sep) * 8)
        return s

    candidates = sorted(list(set(all_ckpts)), key=ckpt_priority)

    def _extract_state_dict(obj):
        if isinstance(obj, dict):
            for k in (
                "state_dict",
                "model_state_dict",
                "model",
                "net",
                "weights",
                "params",
            ):
                if k in obj and isinstance(obj[k], dict):
                    return obj[k]
            return obj
        return None

    def _strip_prefixes(k: str) -> str:
        prefixes = (
            "module.",
            "model.",
            "net.",
            "backbone.",
            "student.",
            "teacher.",
            "ema.",
            "model_ema.",
        )
        nk = k
        changed = True
        while changed:
            changed = False
            for pfx in prefixes:
                if nk.startswith(pfx):
                    nk = nk[len(pfx) :]
                    changed = True
        return nk

    def _rename_common_vit_namespaces(cleaned: dict) -> dict:
        """
        Change: make key mapping cover the most common ViT checkpoint namespaces so we can
        actually load finetuned weights instead of falling back.
        """
        keys = list(cleaned.keys())

        if any(k.startswith("encoder.") for k in keys) or any(
            k.startswith("conv_proj.") for k in keys
        ):
            return cleaned

        out = dict(cleaned)

        if any(k.startswith("vit.") for k in keys):
            out2 = {}
            for k, v in out.items():
                out2[k[len("vit.") :]] = v if k.startswith("vit.") else v
            out = out2

        if any(k.startswith("transformer.") for k in out.keys()):
            out2 = {}
            for k, v in out.items():
                if k.startswith("transformer."):
                    out2[k[len("transformer.") :]] = v
                else:
                    out2[k] = v
            out = out2

        out2 = {}
        for k, v in out.items():
            nk = k
            if nk.startswith("patch_embed.proj."):
                nk = "conv_proj." + nk[len("patch_embed.proj.") :]
            elif nk.startswith("patch_embed."):
                nk = nk
            elif nk.startswith("blocks."):
                nk = "encoder.layers." + nk[len("blocks.") :]
            elif nk.startswith("norm."):
                nk = "encoder.ln." + nk[len("norm.") :]
            elif nk == "pos_embed":
                nk = "encoder.pos_embedding"
            elif nk == "cls_token":
                nk = "class_token"
            out2[nk] = v
        out = out2

        return out

    def _clean_and_adapt_keys(state):
        cleaned = {}
        for k, v in state.items():
            cleaned[_strip_prefixes(k)] = v

        cleaned = _rename_common_vit_namespaces(cleaned)

        remap = {
            "head.weight": "heads.head.weight",
            "head.bias": "heads.head.bias",
            "classifier.weight": "heads.head.weight",
            "classifier.bias": "heads.head.bias",
            "fc.weight": "heads.head.weight",
            "fc.bias": "heads.head.bias",
            "mlp_head.weight": "heads.head.weight",
            "mlp_head.bias": "heads.head.bias",
            "head.fc.weight": "heads.head.weight",
            "head.fc.bias": "heads.head.bias",
            "head.linear.weight": "heads.head.weight",
            "head.linear.bias": "heads.head.bias",
            "cls_head.weight": "heads.head.weight",
            "cls_head.bias": "heads.head.bias",
        }
        for src, dst in remap.items():
            if src in cleaned and dst not in cleaned:
                cleaned[dst] = cleaned[src]
        return cleaned

    def _get_head_shapes(cleaned):
        w = cleaned.get("heads.head.weight", None)
        b = cleaned.get("heads.head.bias", None)
        ws = tuple(w.shape) if hasattr(w, "shape") else None
        bs = tuple(b.shape) if hasattr(b, "shape") else None
        return ws, bs

    def _head_present_and_correct(cleaned, num_classes: int) -> bool:
        ws, bs = _get_head_shapes(cleaned)
        if ws is None or bs is None:
            return False
        return (len(ws) == 2) and (ws[0] == num_classes) and (bs[0] == num_classes)

    best_backbone_candidate = None
    best_backbone_score = 10_000

    last_exception = None
    for ckpt_path in candidates:
        if not os.path.exists(ckpt_path):
            continue
        try:
            obj = torch.load(ckpt_path, map_location="cpu")

            if isinstance(obj, torch.nn.Module):
                m = obj
                hw = getattr(getattr(m, "heads", None), "head", None)
                if hw is None or not hasattr(hw, "weight"):
                    continue
                if tuple(hw.weight.shape)[0] != num_classes:
                    continue
                m = m.to(device)
                m.eval()
                print(f"Loaded full torch.nn.Module (5-class head) from: {ckpt_path}")
                print(
                    f"Using head weight shape: {tuple(m.heads.head.weight.shape)}; std={float(m.heads.head.weight.std().cpu()):.6f}"
                )
                return m

            state = _extract_state_dict(obj)
            if state is None or not isinstance(state, dict):
                continue

            cleaned = _clean_and_adapt_keys(state)

            model = build_base()

            if _head_present_and_correct(cleaned, num_classes):
                try:
                    model.load_state_dict(cleaned, strict=True)
                except Exception:
                    model_keys = set(model.state_dict().keys())
                    filtered = {k: v for k, v in cleaned.items() if k in model_keys}
                    if not _head_present_and_correct(filtered, num_classes):
                        raise RuntimeError(
                            "Filtered state_dict lost valid 5-class head."
                        )
                    model.load_state_dict(filtered, strict=False)

                model.to(device).eval()
                head_w = model.heads.head.weight.detach().float().cpu()
                print(
                    f"Loaded Cassava-like ViT weights (with 5-class head) from: {ckpt_path}"
                )
                print(
                    f"Using head weight shape: {tuple(model.heads.head.weight.shape)}; mean={float(head_w.mean()):.6f}; std={float(head_w.std()):.6f}"
                )
                return model

            backbone_keys = (
                "encoder.layers.0.",
                "conv_proj.",
                "class_token",
                "encoder.pos_embedding",
            )
            if any(
                any(k.startswith(bk) for k in cleaned.keys()) for bk in backbone_keys
            ):
                score = ckpt_priority(ckpt_path)
                if score < best_backbone_score:
                    best_backbone_score = score
                    best_backbone_candidate = (ckpt_path, cleaned)

        except Exception as e:
            last_exception = e
            continue

    if best_backbone_candidate is not None:
        ckpt_path, cleaned = best_backbone_candidate
        model = build_base()
        model_keys = set(model.state_dict().keys())
        filtered = {
            k: v
            for k, v in cleaned.items()
            if k in model_keys and k not in ("heads.head.weight", "heads.head.bias")
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        model.to(device).eval()
        print(
            f"Loaded backbone-only ViT weights (no 5-class head in ckpt) from: {ckpt_path}"
        )
        print(
            f"Missing keys count={len(missing)}, unexpected keys count={len(unexpected)}"
        )
        print(
            f"Head remains initialized for 5 classes: shape={tuple(model.heads.head.weight.shape)}; std={float(model.heads.head.weight.std().cpu()):.6f}"
        )
        return model

    if last_exception is not None:
        print(
            f"No suitable checkpoint loaded; falling back to ImageNet ViT + random 5-class head. Last error: {last_exception}"
        )
    else:
        print(
            "No suitable checkpoint found; falling back to ImageNet ViT + random 5-class head."
        )
    model = build_base().to(device).eval()
    print(
        f"Fallback model head weight shape: {tuple(model.heads.head.weight.shape)}; std={float(model.heads.head.weight.std().cpu()):.6f}"
    )
    return model


vit_model = _load_vit_model(device, num_classes)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test images.

    Args:
        data_dir: base directory to the images.
        transform: transforms applied to image.
        ttas: list of transforms (each applied then followed by `transform`) for TTA.
    """

    def __init__(self, data_dir, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        self.ttas = ttas

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:

    def _hflip(x):
        return v2.functional.horizontal_flip(x)

    def _vflip(x):
        return v2.functional.vertical_flip(x)

    def _rot180(x):
        return v2.functional.rotate(
            x, angle=180, interpolation=InterpolationMode.BILINEAR
        )

    ttas = [
        (lambda x: x),
        _hflip,
        _vflip,
        _rot180,
    ]
else:
    ttas = None

test_dataset = CassavaDataset(test_dir, transform=test_transforms, ttas=ttas)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

vit_model.eval()

with torch.inference_mode():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        filenames = list(filenames)

        if tta:
            n_ttas = len(ttas)
            bsz = len(filenames)

            inputs = torch.cat(inputs, dim=0).to(device, non_blocking=True)
            probs = normalizer(vit_model(inputs))

            probs = probs.view(n_ttas, bsz, -1).permute(1, 0, 2).contiguous()
            mean_probs = probs.mean(dim=1)
            pred_labels = torch.argmax(mean_probs, dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            probs = normalizer(vit_model(inputs))
            pred_labels = torch.argmax(probs, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 4
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)

pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})
pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first")

merged = sample[["image_id"]].merge(pred_df, on="image_id", how="left")

if merged["label"].isna().any():
    if pred_df["label"].notna().any():
        fill_label = int(pred_df["label"].mode().iloc[0])
    else:
        fill_label = 0
    merged["label"] = merged["label"].fillna(fill_label)

merged["label"] = merged["label"].astype(int)

submission_path = "submission.csv"
merged.to_csv(submission_path, index=False)

print(f"Wrote {submission_path} with shape={merged.shape} (should be {sample.shape}).")
merged.head()



## === cell 5
merged
