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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.7805983680870353

# 6. Current score

0.59716

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.1491) has done: 'I fix the immediate runtime/import failure by removing the TensorFlow/Keras dependency that triggers the `MessageFactory.GetPrototype` error, and I also remove the unused TFRecord parsing code. Next, I make the solution robust to missing external model weight files by falling back to a torchvision ViT with ImageNet weights (same inference loop, no training), so the notebook always runs end-to-end. Finally, I generate the submission using `sample_submission.csv` as the authoritative test image order/length (and only predict those ids), which fixes the “Invalid submission length” issue and guarantees correct formatting with a `.csv` suffix.'
- What this solution (achieved 0.179) has done: 'I fix the ViT loading crash by keeping the ImageNet-weighted fallback at its required `image_size=224` (torchvision enforces this), while still allowing custom cassava checkpoints to use `vit_image_size=512` if compatible. Then I make inference robust by ensuring `vit_model` is always defined (so cell 5 can run), and I guarantee the submission lengths match by deriving `test_image_ids` from `sample_submission.csv` and hard-aligning `predictions` to that length right before writing. These changes are execution-unblocking and should also increase score versus random/empty output by producing real model predictions. The core approach (inference-only ViT, argmax class) is preserved.'
- What this solution (achieved 0.19544) has done: 'Your current score is far below the target, and the biggest issue is that the ImageNet ViT fallback is being used with a randomly initialized classification head (5-way), which makes predictions near-random. To move accuracy toward the target with minimal core-logic change (still inference-only ViT + argmax), we build the ViT so it can load ImageNet weights while cleanly replacing the head, use the weights’ own preprocessing (which matches training normalization), and force `use_224_preprocess` whenever we’re not actually using a compatible custom 512px cassava checkpoint. These changes keep the same model family and inference semantics, but should substantially improve accuracy versus the current random-head behavior. The submission writing and `sample_submission.csv` alignment are preserved.'
- What this solution (achieved 0.19507) has done: 'Your current score is far below the target because the fallback ViT uses ImageNet features but a randomly initialized 5-class head, making predictions near-random. To move accuracy toward the target while preserving the same “ViT inference-only + argmax” core logic, I keep the torchvision ViT backbone pretrained and replace the head with a classifier initialized from the backbone’s own class-token features (a nearest-centroid/prototype head computed once from the provided labeled training set). This is still the same model family and inference semantics (single forward pass + argmax), but it makes the head non-random and aligned to cassava classes, which should substantially increase accuracy. I also add safe, minimal test-time augmentation (horizontal flip averaging) to stabilize predictions without changing training loops. Submission alignment via `sample_submission.csv` remains unchanged.'
- What this solution (achieved 0.64126) has done: 'I fix the crash in prototype feature extraction by using the correct final normalization layer name for torchvision’s ViT (`encoder.ln` instead of the non-existent `ln`). I also make the feature extractor robust across torchvision variants by falling back to `ln`/`norm` if present, without changing the overall “pretrained ViT backbone + prototype head + argmax” approach. This unblocks end-to-end execution so the prototype head is actually installed (instead of failing mid-way), which should substantially improve accuracy versus the current near-random head. Submission generation remain aligned to `sample_submission.csv` and still write `submission.csv`.'
- What this solution (achieved 0.59529) has done: 'Your current score (0.64126) is below the target (0.78060), so we should improve accuracy with minimal, low-risk changes that keep the same core approach (pretrained ViT backbone + prototype/centroid head + argmax). The biggest easy gain is making the prototype head consistent with its intended “nearest centroid” semantics by L2-normalizing both the extracted features and the class centroids, and then using cosine similarity (implemented by normalized dot product) via the existing linear head. Additionally, we install the prototype head with a small scaling factor (temperature) to sharpen logits without changing the prediction rule (still argmax). These changes keep the model, inference loop, and loss/training-free approach intact, but typically improve centroid-classifier accuracy noticeably on Cassava.'
- What this solution (achieved 0.59529) has done: 'To move your score upward toward 0.7806 with minimal logic change, I keep the same “pretrained ViT backbone + prototype/centroid head + argmax” approach but make the prototype estimates less noisy and more class-balanced. Concretely: (1) compute centroids using per-class caps (so majority classes don’t dominate and minority classes get enough representation), and (2) build centroids from a small, deterministic multi-crop average per image (original + horizontal flip) to better match your test-time augmentation and reduce feature variance. These changes preserve the model, no-training inference pipeline, and submission semantics, but typically give a noticeable accuracy lift versus single-view, imbalanced prototype building. Submission formatting and `sample_submission.csv` alignment remain unchanged.'
- What this solution (achieved 0.5938) has done: 'We keep your same core approach (pretrained torchvision ViT backbone → extract class-token features → build per-class cosine centroids from train images → install as linear head with logit scaling → test-time 2-view average → argmax). To move accuracy closer to the 0.7806 target, the minimal high-impact fix is to make the feature space used for prototypes match the feature space used by the classifier head: we apply the model’s `heads.pre_logits` to the extracted class-token features (and use that same representation for centroids and inference), because torchvision ViT’s classification head expects pre-logits features. We also ensure the forward pass used for inference uses the same normalized pre-logits features by bypassing the random head and directly computing cosine similarity to centroids (still argmax; same semantics), which avoids any mismatch from unused/extra layers. These are small, deterministic changes and should improve score without changing the overall pipeline or adding training.'
- What this solution (achieved 0.59716) has done: 'We keep your current “pretrained torchvision ViT backbone + pre-logits feature extraction + cosine prototypes + 2-view flip average + argmax” core logic intact, but fix two high-impact issues that can depress accuracy: (1) prototypes are currently computed from a per-class *prefix* after sorting by `image_id`, which can bias centroids; we instead do a deterministic per-class shuffle before capping to make prototypes more representative. (2) Your test loop runs image-by-image (no batching), which can subtly change normalization/statistics handling and is slower; we batch inference deterministically while preserving the exact same preprocessing, 2-view averaging, and cosine scoring, which typically improves stability and helps score move upward toward the target. All I/O paths and the submission format/order (from `sample_submission.csv`) remain unchanged.'
- What this solution (achieved 0.59716) has done: 'We keep your same core “pretrained torchvision ViT backbone → extract pre-logits features → build cosine centroids from train → 2-view flip average at test → argmax” pipeline, but fix two small mismatches that commonly suppress accuracy. First, we make the feature extractor apply the same post-encoder layernorm as torchvision’s own `forward_features` (`encoder.ln` on the full token sequence, then take class token), which better matches the representation used by the ViT head. Second, we compute the prototype feature dimensionality using the actual preprocess resolution in use (224 vs 512), preventing silent shape/representation inconsistencies when a custom checkpoint is used. These are minimal, deterministic changes aimed to move your score upward toward the 0.7806 target without changing the overall approach.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models

torch.manual_seed(0)
np.random.seed(0)



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_data_directory = f"{DATA_DIR}/test_images"
train_data_directory = f"{DATA_DIR}/train_images"
train_csv_path = f"{DATA_DIR}/train.csv"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

vit_model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"
resnet_model_path = "/kaggle/input/resnet_cassava/keras/default/1/resnet_cassava.keras"

vit_image_size = 512



## === cell 2
vit_preprocess = transforms.Compose(
    [
        transforms.Resize((vit_image_size, vit_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)




## === cell 3
def _strip_prefix_if_present(
    state_dict, prefixes=("module.", "model.", "net.", "encoder.")
):
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        new_sd[nk] = v
    return new_sd


def _load_vit_model(device, num_classes, vit_image_size, vit_model_path):
    """
    Keep core logic (ViT inference-only, argmax), but ensure we always have a strong pretrained backbone.
    Returns:
      vit_model, use_224_preprocess (bool), weights_source (str)
    """
    use_224_preprocess = True
    weights_source = None

    if os.path.exists(vit_model_path):
        vit_model = models.vit_l_16(weights=None, image_size=vit_image_size)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )

        state = torch.load(vit_model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _strip_prefix_if_present(state)

        model_sd = vit_model.state_dict()
        filtered = {
            k: v
            for k, v in state.items()
            if (k in model_sd and model_sd[k].shape == v.shape)
        }

        vit_model.load_state_dict(filtered, strict=False)

        if len(filtered) > 50:
            use_224_preprocess = False
            weights_source = f"custom cassava weights @ {vit_image_size}px (loaded {len(filtered)}/{len(state)} tensors)"
        else:
            weights = models.ViT_L_16_Weights.DEFAULT
            vit_model = models.vit_l_16(weights=weights)  # 224px
            vit_model.heads.head = torch.nn.Linear(
                vit_model.heads.head.in_features, num_classes
            )
            use_224_preprocess = True
            weights_source = "torchvision ImageNet weights @ 224px (fallback; checkpoint missing/incompatible)"
    else:
        weights = models.ViT_L_16_Weights.DEFAULT
        vit_model = models.vit_l_16(weights=weights)  # 224px
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        use_224_preprocess = True
        weights_source = (
            "torchvision ImageNet weights @ 224px (fallback; checkpoint missing)"
        )

    vit_model.to(device)
    vit_model.eval()
    print(f"ViT loaded using: {weights_source}")
    return vit_model, use_224_preprocess, weights_source


vit_model, use_224_preprocess, weights_source = _load_vit_model(
    device, num_classes, vit_image_size, vit_model_path
)



## === cell 4
use_resnet = False
resnet_model = None

if os.path.exists(resnet_model_path):
    use_resnet = False

print(
    f"ResNet enabled: {use_resnet} (model file exists: {os.path.exists(resnet_model_path)})"
)



## === cell 5
vit_preprocess_224 = models.ViT_L_16_Weights.DEFAULT.transforms()


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _get_vit_prelogits_features(model, x):
    """
    Change (accuracy-relevant, minimal): apply encoder.ln the same way torchvision does
    (layernorm over the full token sequence), then take CLS token and apply heads.pre_logits.
    This better matches the representation expected by the head and stabilizes prototypes.
    """
    with torch.no_grad():
        y = model._process_input(x)  # (B, N, hidden_dim)
        n = y.shape[0]
        batch_class_token = model.class_token.expand(n, -1, -1)
        y = torch.cat([batch_class_token, y], dim=1)  # (B, 1+N, hidden_dim)

        y = model.encoder(
            y
        )  # for torchvision ViT, encoder includes the transformer blocks

        if hasattr(model, "encoder") and hasattr(model.encoder, "ln"):
            y = model.encoder.ln(y)

        feats = y[:, 0]  # CLS token after post-encoder LN

        if not (hasattr(model, "encoder") and hasattr(model.encoder, "ln")):
            if hasattr(model, "ln"):
                feats = model.ln(feats)
            elif hasattr(model, "norm"):
                feats = model.norm(feats)

        if hasattr(model, "heads") and hasattr(model.heads, "pre_logits"):
            feats = model.heads.pre_logits(feats)

    return feats


proto_use_224 = use_224_preprocess

train_df = pd.read_csv(train_csv_path)

per_class_cap = 3000  # keep unchanged (prior behavior)

train_df = train_df.reset_index(drop=True)
train_df_balanced = (
    train_df.groupby("label", group_keys=False)
    .apply(lambda g: g.sample(frac=1.0, random_state=0).head(per_class_cap))
    .reset_index(drop=True)
)

_probe_size = 224 if proto_use_224 else vit_image_size
with torch.no_grad():
    _tmp = torch.zeros((1, 3, _probe_size, _probe_size), device=device)
    _d = int(_get_vit_prelogits_features(vit_model, _tmp).shape[-1])

hidden_dim = _d
class_sums = torch.zeros((num_classes, hidden_dim), dtype=torch.float32, device=device)
class_counts = torch.zeros((num_classes,), dtype=torch.long, device=device)

missing_train = 0
batch_imgs = []
batch_labels = []
batch_size = 32 if torch.cuda.is_available() else 16

for row in tqdm(
    train_df_balanced.itertuples(index=False),
    total=len(train_df_balanced),
    desc="Build prototypes (balanced + 2-view)",
):
    img_id = str(row.image_id)
    y = int(row.label)
    img_path = os.path.join(train_data_directory, img_id)
    if not os.path.exists(img_path):
        missing_train += 1
        continue

    img = Image.open(img_path).convert("RGB")

    if proto_use_224:
        t1 = vit_preprocess_224(img)
        t2 = vit_preprocess_224(img.transpose(Image.FLIP_LEFT_RIGHT))
    else:
        t1 = vit_preprocess(img)
        t2 = vit_preprocess(img.transpose(Image.FLIP_LEFT_RIGHT))

    batch_imgs.append(torch.stack([t1, t2], dim=0))  # (2,C,H,W)
    batch_labels.append(y)

    if len(batch_imgs) >= batch_size:
        x = torch.cat(batch_imgs, dim=0).to(device, non_blocking=True)  # (2B,C,H,W)
        feats = _get_vit_prelogits_features(vit_model, x).float()
        feats = _l2_normalize(feats)
        feats = feats.view(-1, 2, feats.shape[-1]).mean(dim=1)  # (B, D)

        labels_t = torch.tensor(batch_labels, device=device, dtype=torch.long)
        for c in range(num_classes):
            m = labels_t == c
            if m.any():
                class_sums[c] += feats[m].sum(dim=0)
                class_counts[c] += int(m.sum().item())
        batch_imgs, batch_labels = [], []

if batch_imgs:
    x = torch.cat(batch_imgs, dim=0).to(device, non_blocking=True)  # (2B,C,H,W)
    feats = _get_vit_prelogits_features(vit_model, x).float()
    feats = _l2_normalize(feats)
    feats = feats.view(-1, 2, feats.shape[-1]).mean(dim=1)  # (B, D)

    labels_t = torch.tensor(batch_labels, device=device, dtype=torch.long)
    for c in range(num_classes):
        m = labels_t == c
        if m.any():
            class_sums[c] += feats[m].sum(dim=0)
            class_counts[c] += int(m.sum().item())
    batch_imgs, batch_labels = [], []

if missing_train:
    print(
        f"Warning: {missing_train} train images were missing while building prototypes."
    )

centroids = torch.zeros_like(class_sums)
for c in range(num_classes):
    if class_counts[c] > 0:
        centroids[c] = class_sums[c] / class_counts[c].float()

centroids = _l2_normalize(centroids)
logit_scale = 20.0  # keep unchanged to preserve prior behavior

with torch.no_grad():
    vit_model.heads.head = torch.nn.Linear(hidden_dim, num_classes, bias=True).to(
        device
    )
    vit_model.heads.head.weight.copy_(centroids * logit_scale)
    vit_model.heads.head.bias.zero_()

print(
    "Prototype head installed from train set features (per-class shuffled cap + 2-view cosine prototypes in pre-logits space + logit scaling)."
)
print("Prototype class counts:", class_counts.detach().cpu().tolist())



## === cell 6
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].astype(str).tolist()

predictions = [0] * len(test_image_ids)
missing_images = 0

centroids_t = centroids  # (C,D), already L2-normalized
scale_t = float(logit_scale)

test_batch = 64 if torch.cuda.is_available() else 16
i = 0
pbar = tqdm(total=len(test_image_ids), desc="Predict (batched)")
while i < len(test_image_ids):
    batch_ids = test_image_ids[i : i + test_batch]

    xs = []
    valid_positions = []
    for j, image_name in enumerate(batch_ids):
        image_path = os.path.join(test_data_directory, image_name)
        if not os.path.exists(image_path):
            missing_images += 1
            continue

        image = Image.open(image_path).convert("RGB")
        if use_224_preprocess:
            x1 = vit_preprocess_224(image)
            x2 = vit_preprocess_224(image.transpose(Image.FLIP_LEFT_RIGHT))
        else:
            x1 = vit_preprocess(image)
            x2 = vit_preprocess(image.transpose(Image.FLIP_LEFT_RIGHT))

        xs.append(torch.stack([x1, x2], dim=0))  # (2,C,H,W)
        valid_positions.append(j)

    if xs:
        x = torch.cat(xs, dim=0).to(device, non_blocking=True)  # (2B,C,H,W)
        with torch.no_grad():
            feats = _get_vit_prelogits_features(vit_model, x).float()  # (2B,D)
            feats = _l2_normalize(feats)
            feats = feats.view(-1, 2, feats.shape[-1]).mean(dim=1)  # (B,D)
            logits = (feats @ centroids_t.T) * scale_t  # (B,C)
            preds = (
                torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
            )

        for k, j in enumerate(valid_positions):
            predictions[i + j] = int(preds[k])

    i += test_batch
    pbar.update(len(batch_ids))

pbar.close()

if missing_images:
    print(
        f"Warning: {missing_images} test images were missing; filled with label=0 to keep valid length."
    )

n_expected = len(test_image_ids)
if len(predictions) != n_expected:
    print(
        f"Warning: predictions length {len(predictions)} != expected {n_expected}. Fixing by trim/pad."
    )
    if len(predictions) > n_expected:
        predictions = predictions[:n_expected]
    else:
        predictions = predictions + [0] * (n_expected - len(predictions))



## === cell 7
submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print(f"Rows: {len(submission_df)} (expected {len(sample_sub)})")
