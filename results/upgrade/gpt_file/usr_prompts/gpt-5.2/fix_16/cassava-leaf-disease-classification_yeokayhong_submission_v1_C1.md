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

geopandas==0.14.4
numpy==1.26.4
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
tqdm==4.67.1

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

0.8845572680568148

# 6. Current score

0.12182

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12182) has done: 'The failure is because the notebook references a private `/kaggle/input/vit_l_cassava/...` checkpoint that is not present in your environment, and the fallback search finds no `.pth` anywhere under `/kaggle/input`, so inference cannot start. To make the pipeline run end-to-end and still keep the core inference logic (ViT + argmax) intact, the minimal fix is to stop hard-failing when no checkpoint exists and instead load an official torchvision ViT-H pretrained weight (ImageNet) as a safe fallback. Since the torchvision ViT-H pretrained head is 1000-way, we keep your 5-class head and simply leave it randomly initialized when no cassava checkpoint is available; this yield a valid submission (score likely low, but at least it runs). The code also adds deterministic settings and ensures the submission matches `sample_submission.csv` ordering exactly.'
- What this solution (achieved 0.12182) has done: 'Your current score is far below the target because the model usually runs with a randomly initialized 5-class head (no cassava checkpoint found), which makes predictions close to random. The smallest change that legitimately improves accuracy (without changing the model architecture or inference semantics) is to add a lightweight “zero-training” head initialization using the provided `train.csv`: compute per-class mean features from the frozen ViT backbone and classify by nearest class mean (a simple prototypical classifier). This keeps the same ViT model and argmax decision rule, but replaces random head weights with data-driven weights derived from the competition’s training images. The rest of the pipeline (paths, transforms, CSV ordering/format) stays the same, and it still writes `submission.csv`.'
- What this solution (achieved 0.58259) has done: 'Main bottlenecks are (1) the recursive `/kaggle/input` checkpoint search via `os.walk` and (2) per-image Python overhead during inference (PIL open/transform/CPU→GPU copy one-by-one). I remove the expensive recursive search in favor of a small set of equivalent deterministic fallback paths, and I switch test-time inference to a `Dataset+DataLoader` with pinned memory and multiple workers so decoding/transforms overlap GPU compute while preserving identical transforms/model forward/argmax semantics. I also avoid pre-scanning every test file with `os.path.isfile` (another full Python loop) and instead let the dataset raise immediately if something is missing. All changes keep the same model, transforms, and prediction logic (head argmax or prototype cosine similarity argmax), just executed in batches and with faster I/O.'
- What this solution (achieved 0.59641) has done: 'Main bottlenecks are (1) using the extremely large `vit_h_14` backbone and (2) slow per-sample PIL decoding in Python; together they can exceed 600s even for just inference. To preserve identical model logic/semantics while speeding up, the script below (a) uses `torch.inference_mode()` and enables TF32 on CUDA for faster matmuls with negligible float differences, (b) compiles the model on PyTorch 2.x CUDA when available (same computation graph, faster execution), (c) vectorizes prototype accumulation to remove the per-class Python loop, and (d) accelerates dataloading by using `torchvision.io.read_image` + `decode_jpeg` (fast C++ backend) and keeps workers persistent. All paths, transforms, architecture, and prediction logic (either head argmax or nearest-prototype argmax) remain unchanged.'
- What this solution (achieved 0.5852) has done: 'To move accuracy up toward your 0.8846 target without changing the model family or inference semantics, the main issue is that your image preprocessing doesn’t match the pretrained ViT-H weights you fall back to (wrong resize/crop policy and wrong normalization), which suppresses feature quality and makes the prototype classifier weak. I switch transforms to the official `ViT_H_14_Weights` preprocessing when using torchvision pretrained weights, while keeping your exact ViT-H backbone + (prototype argmax OR head argmax) logic intact. I also compute prototypes from the backbone *before* compiling the model (compile can change feature extraction accessibility), then compile only for fast test inference; this is a stability fix that should improve prototype quality and avoid edge failures. Everything still runs end-to-end and writes `submission.csv` with the required columns and ordering from `sample_submission.csv`.'
- What this solution (achieved 0.18348) has done: 'Your score gap is large (0.5852 vs target 0.8846), and the most likely cause is still that you’re not actually using a cassava-trained checkpoint (so features don’t align well) plus the prototype classifier is relatively weak without any domain adaptation. To move accuracy up without changing the core model family or decision rule, I (1) add a fast, minimal fine-tuning of only the 5-class head (backbone frozen) on `train.csv` using the same preprocessing you already use, then (2) use the trained head for test-time argmax (same evaluation semantics), keeping the prototype fallback only if training is impossible. This is a small, legitimate change that usually boosts accuracy a lot on Cassava while keeping architecture/loss/prediction logic intact, and it stays within the 600s budget by training only the head for a single epoch. I also ensure we use the official ViT-H weights preprocessing consistently and keep submission ordering identical to `sample_submission.csv`.'
- What this solution (achieved 0.12182) has done: 'The main timeout driver is unnecessary heavy work before inference: (1) `torch.compile` introduces large one-time compile overhead that can exceed the whole 10-minute budget for ViT-H, and (2) the torchvision-pretrained fallback path does a full 1-epoch head fine-tune over ~18k images plus prototype computation, which is far too slow. The optimized script keeps identical inference semantics when the provided cassava checkpoint exists (the intended path), but makes the fallback path “safe-fast” by skipping training/prototypes and just running the pretrained model head (only used when the checkpoint is missing). Additionally, it removes per-image PIL conversion when using `torchvision.io` (apply transforms directly to tensors), and tunes DataLoader settings to reduce CPU overhead without changing outputs.'
- What this solution (achieved 0.12182) has done: 'I remove the only expensive “fallback training” path (1-epoch head fine-tune) and make checkpoint loading strict and fast so the run is always pure inference, which is what timed out. I also speed up input by always using the torchvision tensor decode+resize+normalize path (avoiding PIL) and by increasing DataLoader throughput (more workers, larger batch, pinned memory, persistent workers). These changes preserve the exact model architecture, loss, and inference semantics; they only eliminate unnecessary training and reduce overhead in image decoding/transforms. Determinism is kept via fixed seeds and deterministic settings.'
- What this solution (achieved 0.12182) has done: 'Main bottlenecks are (1) accidentally doing a full 18.7k-image training epoch when the cassava checkpoint is missing (torchvision fallback), and (2) slow PIL-based decoding/resize/normalize in the DataLoader. To fit 600s without changing core logic, I (a) stop the expensive fallback head-finetune and instead run pure inference (same inference semantics as usual; just avoids an unintended extra training pass), and (b) force the fast torchvision tensor decode+resize+normalize path for both modes by using an equivalent tensor transform pipeline (so preprocessing stays the same math). I also tune DataLoader settings (workers/prefetch/persistent/pin) and enable channels_last on GPU to reduce overhead while preserving outputs up to negligible FP differences. No model architecture, loss, or training loop mechanics are altered beyond removing the fallback training that causes the timeout.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_data_directory = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"

image_size = 518
num_classes = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_grad_enabled(False)
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

val_transforms = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)




## === cell 1
def _find_checkpoint_fallback(preferred_path: str) -> str | None:
    """
    Return None if no checkpoint is available under /kaggle/input.
    """
    if os.path.isfile(preferred_path):
        return preferred_path

    base_inputs = [
        "/kaggle/input/vit_l_cassava/pytorch/default/1",
        "/kaggle/input/vit_l_cassava",
        "/kaggle/input",
    ]
    candidate_files = [
        "model_weights_3.pth",
        "model_weights.pth",
        "model.pth",
        "weights.pth",
        "best.pth",
        "checkpoint.pth",
    ]

    candidates = []
    for b in base_inputs:
        for fn in candidate_files:
            fp = os.path.join(b, fn)
            if os.path.isfile(fp):
                candidates.append(fp)

    if candidates:
        chosen = candidates[0]
        print(
            f"[WARN] Checkpoint not found at '{preferred_path}'. Using fallback checkpoint: {chosen}"
        )
        return chosen

    print(
        f"[WARN] Checkpoint not found at '{preferred_path}', and no fallback .pth files were found in expected locations. "
        "Falling back to torchvision pretrained ViT weights."
    )
    return None


def _load_state_dict_robust(
    model: torch.nn.Module, ckpt_path: str, device: torch.device
) -> None:
    """
    Handle checkpoints saved as raw state_dict or wrapped dict (e.g., {'state_dict': ...}),
    stripping common prefixes like 'module.' and 'model.'.
    """
    try:
        obj = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            sd = obj
    else:
        raise RuntimeError(
            f"Unsupported checkpoint object type: {type(obj)} at {ckpt_path}"
        )

    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(
            f"[WARN] Missing keys when loading checkpoint (showing up to 20): {missing[:20]}"
        )
    if unexpected:
        print(
            f"[WARN] Unexpected keys when loading checkpoint (showing up to 20): {unexpected[:20]}"
        )


resolved_model_path = _find_checkpoint_fallback(model_path)

using_torchvision_pretrained = resolved_model_path is None
vit_weights = None

if using_torchvision_pretrained:
    try:
        vit_weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    except Exception:
        vit_weights = models.ViT_H_14_Weights.DEFAULT

    model = models.vit_h_14(weights=vit_weights, image_size=image_size)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    print("[INFO] Using torchvision pretrained backbone (no cassava checkpoint found).")

else:
    model = models.vit_h_14(weights=None, image_size=image_size)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    _load_state_dict_robust(model, resolved_model_path, device)
    print(f"[INFO] Loaded cassava checkpoint from: {resolved_model_path}")

model.to(device)
model.eval()

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)



## === cell 2
from torch.utils.data import Dataset, DataLoader


def _seed_worker(worker_id: int):
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class CassavaTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str, tfm):
        self.image_ids = df["image_id"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.root_dir = root_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        y = self.labels[idx]
        image_path = os.path.join(self.root_dir, image_id)
        img = Image.open(image_path).convert("RGB")
        x = self.tfm(img)
        return x, y


if using_torchvision_pretrained:
    print(
        "[INFO] Checkpoint missing: skipping fallback head-only fine-tuning to meet 600s timeout; running inference only."
    )



## === cell 3
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].astype(str).tolist()

try:
    from torchvision.io import read_file, decode_jpeg

    _TV_IO_OK = True
except Exception:
    _TV_IO_OK = False

_HAS_TENSOR_TRANSFORMS = True
try:
    from torchvision.transforms import functional as F
    from torchvision.transforms.functional import InterpolationMode
except Exception:
    _HAS_TENSOR_TRANSFORMS = False


def _apply_val_transforms_fast(img_chw_uint8: torch.Tensor) -> torch.Tensor:
    x = img_chw_uint8  # uint8 [C,H,W], RGB
    x = F.resize(
        x,
        (image_size, image_size),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    x = x.to(dtype=torch.float32).div_(255.0)
    x = F.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return x


class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, root_dir, tfm):
        self.image_ids = image_ids
        self.root_dir = root_dir
        self.tfm = tfm

        self._use_fast_tensor_path = _TV_IO_OK and _HAS_TENSOR_TRANSFORMS

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.root_dir, image_id)

        if self._use_fast_tensor_path:
            data = read_file(image_path)
            img = decode_jpeg(data, mode="RGB")  # uint8, [C,H,W]
            x = _apply_val_transforms_fast(img)
            return image_id, x

        img = Image.open(image_path).convert("RGB")
        x = self.tfm(img)
        return image_id, x


test_ds = CassavaTestDataset(test_image_ids, test_data_directory, val_transforms)

cpu_cnt = os.cpu_count() or 0
if device.type == "cuda":
    num_workers = min(8, cpu_cnt)
    batch_size = 32
else:
    num_workers = min(4, cpu_cnt)
    batch_size = 2

pin_memory = device.type == "cuda"

g = torch.Generator()
g.manual_seed(seed)

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
)

test_predictions = []
test_image_ids_out = []

model.eval()
with torch.inference_mode():
    for batch_ids, batch_x in tqdm(test_loader, desc="Test", total=len(test_loader)):
        if device.type == "cuda":
            batch_x = batch_x.contiguous(memory_format=torch.channels_last)
        batch_x = batch_x.to(device, non_blocking=(device.type == "cuda"))

        output = model(batch_x)  # [B, C]
        pred = output.argmax(dim=1).to("cpu").numpy().astype(int)

        test_predictions.extend(pred.tolist())
        test_image_ids_out.extend(list(batch_ids))

submission_df = pd.DataFrame(
    {"image_id": test_image_ids_out, "label": test_predictions}
)
submission_df["image_id"] = submission_df["image_id"].astype(str)
submission_df["label"] = submission_df["label"].astype(int)

submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(0).astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Submission shape:", submission_df.shape)
