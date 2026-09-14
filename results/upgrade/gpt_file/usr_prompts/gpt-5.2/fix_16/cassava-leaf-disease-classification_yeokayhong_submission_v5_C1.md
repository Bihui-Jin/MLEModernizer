# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
from torchvision import transforms, models
from torchvision.models import EfficientNet_V2_M_Weights
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os
import warnings

warnings.filterwarnings("ignore")



## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = (
    "/kaggle/input/efficientnetv2-large-test/pytorch/default/1/ENL_V2 (test).pth"
)
vit_model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"
vit_image_size = 518
resnet_model_path = "/kaggle/input/resnet_cassava/keras/default/1/resnet_cassava.keras"



## === cell 2
vit_preprocess = transforms.Compose(
    [
        transforms.Resize((vit_image_size, vit_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)




## === cell 3
def _resolve_test_dir(base_dir: str) -> str:
    """
    Kaggle dataset sometimes contains nested folder test_images/test_images.
    Return the directory that actually contains jpg/png files.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Test directory not found: {base_dir}")

    nested = os.path.join(base_dir, "test_images")

    def has_images(d):
        if not os.path.isdir(d):
            return False
        for fn in os.listdir(d):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(
                os.path.join(d, fn)
            ):
                return True
        return False

    if has_images(base_dir):
        return base_dir
    if has_images(nested):
        return nested

    for fn in os.listdir(base_dir):
        p = os.path.join(base_dir, fn)
        if os.path.isdir(p) and has_images(p):
            return p

    raise FileNotFoundError(f"No image files found under: {base_dir}")


test_data_directory = _resolve_test_dir(test_data_directory)
print("Using test image directory:", test_data_directory)



## === cell 4
using_cassava_head = False

if os.path.exists(en_model_path):
    en_model = models.efficientnet_v2_m(weights=None)
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )
    state = torch.load(en_model_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    en_model.load_state_dict(state, strict=False)
    using_cassava_head = True
    print("Loaded EfficientNet cassava weights from:", en_model_path)
else:
    imagenet_weights = EfficientNet_V2_M_Weights.DEFAULT
    en_model = models.efficientnet_v2_m(weights=imagenet_weights)
    using_cassava_head = False
    print(
        "Custom cassava weights not found; using torchvision ImageNet pretrained model with original 1000-class head."
    )

en_model.to(device)
en_model.eval()



## === cell 5
if not using_cassava_head:
    en_preprocess = imagenet_weights.transforms()
else:
    en_preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )




## === cell 6
def _resolve_train_dir(base_dir: str) -> str:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Train directory not found: {base_dir}")

    nested = os.path.join(base_dir, "train_images")

    def has_images(d):
        if not os.path.isdir(d):
            return False
        for fn in os.listdir(d):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")) and os.path.isfile(
                os.path.join(d, fn)
            ):
                return True
        return False

    if has_images(base_dir):
        return base_dir
    if has_images(nested):
        return nested

    for fn in os.listdir(base_dir):
        p = os.path.join(base_dir, fn)
        if os.path.isdir(p) and has_images(p):
            return p

    raise FileNotFoundError(f"No image files found under: {base_dir}")


def _build_imagenet_to_cassava_mapping_confusion(
    model,
    preprocess,
    device,
    train_csv_path: str,
    train_images_dir: str,
    num_classes: int = 5,
    per_class_limit: int = 1000000,
    topk: int = 100,
    seed: int = 123,
    batch_size: int = 64,
    idf_power: float = 1.35,
    smooth_alpha: float = 0.10,
    evidence_tau: float = 6.0,
    class_mass_power: float = 0.60,
    per_label_keep: int = 350,
    use_logprobs: bool = True,
    per_image_l1_normalize: bool = True,
    logprob_eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build mapping from ImageNet classes -> cassava classes using flip-avg top-k votes.
    Returns (hard_mapping, vote_matrix, default_y).
    """
    rng = np.random.RandomState(seed)

    train_df = pd.read_csv(train_csv_path)
    train_df = train_df.sort_values("image_id").reset_index(drop=True)

    picked_rows = []
    for y in range(num_classes):
        cls_df = train_df[train_df["label"] == y]
        if len(cls_df) == 0:
            continue
        n = min(per_class_limit, len(cls_df))
        if n < len(cls_df):
            idx = rng.choice(len(cls_df), size=n, replace=False)
            picked_rows.append(cls_df.iloc[idx])
        else:
            picked_rows.append(cls_df)
    sub_df = (
        pd.concat(picked_rows, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    cassava_prior = (
        sub_df["label"].value_counts().reindex(range(num_classes), fill_value=0).values
    )
    default_y = int(np.argmax(cassava_prior)) if cassava_prior.sum() > 0 else 0

    prior = cassava_prior.astype(np.float32)
    prior = prior / max(prior.sum(), 1.0)

    num_imagenet = 1000
    counts = np.zeros((num_imagenet, num_classes), dtype=np.float32)

    df_present = np.zeros((num_imagenet,), dtype=np.int32)
    n_docs = 0

    model.eval()

    buf_imgs = []
    buf_labels = []

    for _, row in tqdm(
        sub_df.iterrows(),
        total=len(sub_df),
        desc="Build mapping (flip-avg top-k confusion + IDF + smoothing)",
    ):
        y = int(row["label"])
        img_path = os.path.join(train_images_dir, row["image_id"])
        if not os.path.isfile(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        buf_imgs.append(preprocess(img))
        buf_labels.append(y)

        if len(buf_imgs) >= batch_size:
            xb = torch.stack(buf_imgs, dim=0).to(device)
            yb = np.array(buf_labels, dtype=np.int64)

            with torch.no_grad():
                out1 = model(xb)
                out2 = model(torch.flip(xb, dims=[3]))
                out = (out1 + out2) / 2.0
                probs = torch.softmax(out, dim=1)

                vals, idxs = torch.topk(probs, k=min(topk, probs.shape[1]), dim=1)
                vals = vals.detach().cpu().numpy().astype(np.float32)
                idxs = idxs.detach().cpu().numpy().astype(np.int64)

            for i in range(xb.shape[0]):
                yi = int(yb[i])
                uniq = np.unique(idxs[i])
                df_present[uniq] += 1
                n_docs += 1

                w = vals[i]
                if use_logprobs:
                    w = -np.log(np.clip(w, logprob_eps, 1.0)).astype(
                        np.float32
                    )  # positive for high p
                    w = (np.max(w) - w).astype(np.float32)

                if per_image_l1_normalize:
                    s = float(np.sum(np.abs(w)))
                    if s > 0:
                        w = (w / s).astype(np.float32)

                for wi, k in zip(w, idxs[i]):
                    counts[int(k), yi] += float(wi)

            buf_imgs, buf_labels = [], []

    if len(buf_imgs) > 0:
        xb = torch.stack(buf_imgs, dim=0).to(device)
        yb = np.array(buf_labels, dtype=np.int64)
        with torch.no_grad():
            out1 = model(xb)
            out2 = model(torch.flip(xb, dims=[3]))
            out = (out1 + out2) / 2.0
            probs = torch.softmax(out, dim=1)
            vals, idxs = torch.topk(probs, k=min(topk, probs.shape[1]), dim=1)
            vals = vals.detach().cpu().numpy().astype(np.float32)
            idxs = idxs.detach().cpu().numpy().astype(np.int64)

        for i in range(xb.shape[0]):
            yi = int(yb[i])
            uniq = np.unique(idxs[i])
            df_present[uniq] += 1
            n_docs += 1

            w = vals[i]
            if use_logprobs:
                w = -np.log(np.clip(w, logprob_eps, 1.0)).astype(np.float32)
                w = (np.max(w) - w).astype(np.float32)

            if per_image_l1_normalize:
                s = float(np.sum(np.abs(w)))
                if s > 0:
                    w = (w / s).astype(np.float32)

            for wi, k in zip(w, idxs[i]):
                counts[int(k), yi] += float(wi)

    if n_docs > 0:
        idf = (
            np.log((1.0 + float(n_docs)) / (1.0 + df_present.astype(np.float32))) + 1.0
        ).astype(np.float32)
        if idf_power != 1.0:
            idf = np.power(idf, float(idf_power)).astype(np.float32)
        counts *= idf[:, None]

    counts = counts + (float(smooth_alpha) * prior[None, :]).astype(np.float32)

    evidence = counts.sum(axis=1, keepdims=True).astype(np.float32)  # [1000,1]
    lam = evidence / (evidence + float(evidence_tau))  # [1000,1] in [0,1]
    counts = lam * counts + (1.0 - lam) * prior[None, :].astype(np.float32)

    mass = counts.sum(axis=1, keepdims=True).astype(np.float32)
    counts = counts / np.clip(np.power(mass, float(class_mass_power)), 1e-8, None)

    if (
        per_label_keep is not None
        and per_label_keep > 0
        and per_label_keep < counts.shape[0]
    ):
        mask = np.zeros_like(counts, dtype=bool)
        for y in range(num_classes):
            col = counts[:, y]
            kk = min(int(per_label_keep), col.shape[0])
            top_idx = np.argpartition(-col, kk - 1)[:kk]
            mask[top_idx, y] = True
        counts = np.where(mask, counts, 0.0).astype(np.float32)

    row_sums = counts.sum(axis=1, keepdims=True)
    vote_matrix = counts / np.clip(row_sums, 1e-8, None)  # [1000,5]

    hard_mapping = np.full((num_imagenet,), default_y, dtype=np.int64)
    seen = row_sums[:, 0] > 0
    hard_mapping[seen] = vote_matrix[seen].argmax(axis=1).astype(np.int64)
    return hard_mapping, vote_matrix, default_y


def _predict_cassava_from_imagenet_topk(
    probs: torch.Tensor,
    vote_matrix: np.ndarray,
    default_y: int,
    topk: int = 100,
    use_logprobs: bool = True,
    per_image_l1_normalize: bool = True,
    logprob_eps: float = 1e-12,
) -> np.ndarray:
    """
    Use top-k ImageNet probabilities to vote for cassava classes using vote_matrix.
    """
    k = min(topk, probs.shape[1])
    vals, idxs = torch.topk(probs, k=k, dim=1)
    vals = vals.detach().cpu().numpy().astype(np.float32)  # [B,k]
    idxs = idxs.detach().cpu().numpy().astype(np.int64)  # [B,k]

    if use_logprobs:
        w = -np.log(np.clip(vals, logprob_eps, 1.0)).astype(np.float32)
        w = (np.max(w, axis=1, keepdims=True) - w).astype(np.float32)
        vals = w

    if per_image_l1_normalize:
        denom = np.sum(np.abs(vals), axis=1, keepdims=True)
        denom = np.clip(denom, 1e-8, None)
        vals = (vals / denom).astype(np.float32)

    B = vals.shape[0]
    scores = np.zeros((B, vote_matrix.shape[1]), dtype=np.float32)  # [B,5]
    for i in range(B):
        scores[i] = (vote_matrix[idxs[i]] * vals[i][:, None]).sum(axis=0)

    pred = scores.argmax(axis=1).astype(np.int64)
    zero = scores.sum(axis=1) <= 0
    if np.any(zero):
        pred[zero] = int(default_y)
    return pred




## === cell 7
imagenet_to_cassava = None
imagenet_vote_matrix = None
default_missing_label = 0

if not using_cassava_head:
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    train_images_dir = _resolve_train_dir(
        "/kaggle/input/cassava-leaf-disease-classification/train_images"
    )
    print("Using train image directory for mapping:", train_images_dir)

    imagenet_to_cassava, imagenet_vote_matrix, default_missing_label = (
        _build_imagenet_to_cassava_mapping_confusion(
            model=en_model,
            preprocess=en_preprocess,
            device=device,
            train_csv_path=train_csv_path,
            train_images_dir=train_images_dir,
            num_classes=num_classes,
            per_class_limit=1000000,
            topk=100,
            seed=123,
            batch_size=64,
            idf_power=1.35,
            smooth_alpha=0.10,
            evidence_tau=6.0,
            class_mass_power=0.60,
            per_label_keep=350,
            use_logprobs=True,
            per_image_l1_normalize=True,
        )
    )
    print(
        "Built ImageNet->cassava mapping:",
        imagenet_to_cassava.shape,
        "vote_matrix:",
        imagenet_vote_matrix.shape,
        "default_y:",
        default_missing_label,
    )
else:
    default_missing_label = 0



## === cell 8
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_df = pd.read_csv(sample_path)
image_ids = sample_df["image_id"].tolist()

batch_size = 32
en_predictions = []
buf_tensors = []
buf_missing = 0

test_topk = 100  # match mapping top-k

for image_name in tqdm(image_ids, desc="Test (aligned to sample, batched)"):
    image_path = os.path.join(test_data_directory, image_name)
    if not os.path.isfile(image_path):
        en_predictions.append(int(default_missing_label))
        buf_missing += 1
        continue

    image = Image.open(image_path).convert("RGB")
    buf_tensors.append(en_preprocess(image))

    if len(buf_tensors) >= batch_size:
        xb = torch.stack(buf_tensors, dim=0).to(device)
        with torch.no_grad():
            out1 = en_model(xb)
            out2 = en_model(torch.flip(xb, dims=[3]))
            out = (out1 + out2) / 2.0

        if using_cassava_head:
            preds = out.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        else:
            probs = torch.softmax(out, dim=1)
            preds = _predict_cassava_from_imagenet_topk(
                probs=probs,
                vote_matrix=imagenet_vote_matrix,
                default_y=default_missing_label,
                topk=test_topk,
                use_logprobs=True,
                per_image_l1_normalize=True,
            )

        en_predictions.extend([int(p) for p in preds])
        buf_tensors = []

if len(buf_tensors) > 0:
    xb = torch.stack(buf_tensors, dim=0).to(device)
    with torch.no_grad():
        out1 = en_model(xb)
        out2 = en_model(torch.flip(xb, dims=[3]))
        out = (out1 + out2) / 2.0

    if using_cassava_head:
        preds = out.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    else:
        probs = torch.softmax(out, dim=1)
        preds = _predict_cassava_from_imagenet_topk(
            probs=probs,
            vote_matrix=imagenet_vote_matrix,
            default_y=default_missing_label,
            topk=test_topk,
            use_logprobs=True,
            per_image_l1_normalize=True,
        )

    en_predictions.extend([int(p) for p in preds])

print(
    "Predictions:",
    len(en_predictions),
    "Images:",
    len(image_ids),
    "missing_paths:",
    buf_missing,
    "using_cassava_head:",
    using_cassava_head,
)



## === cell 9
sample_df["label"] = np.array(en_predictions, dtype=np.int64)
sample_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(sample_df.head())
print("Submission rows:", len(sample_df))
