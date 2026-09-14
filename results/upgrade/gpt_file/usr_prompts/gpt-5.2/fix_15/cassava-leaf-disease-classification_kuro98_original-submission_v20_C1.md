# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.894983378664249

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime blocker by removing the dependency on a missing external model file and instead instantiate a torchvision ViT with the correct output dimension so inference can run end-to-end. I also fix submission validity issues by making the test DataLoader deterministic (no shuffling) and aligning the final predictions to `sample_submission.csv` order to guarantee matching length and IDs. Finally, I correct the TTA pipeline so it batches augmented views properly and ensures filenames are repeated/aggregated consistently, producing exactly one label per test image. These changes are minimal and focused on unblocking execution and producing a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'You’re hitting a TTA collation shape issue: the default DataLoader collate turns your per-sample list-of-TTA-tensors into a per-TTA tensor batch, so iterating as if `inputs` were a list per sample causes the `torch.stack()` TypeError. I fix cell 4 to handle the actual batched structure (a list of length T, each a tensor of shape B×C×H×W), flatten it to (B*T) for the model, then reshape back to average over T. To move accuracy toward the target, I also load the pretrained ViT weights but correctly replace the classification head and load a finetuned checkpoint if present (otherwise keep pretrained features), and ensure deterministic TTA by using functional transforms with fixed seeds per image. The rest (paths, submission alignment, no shuffling) stays the same.'
- What this solution (achieved 0.13191) has done: 'The runtime error comes from feeding 384×384 tensors into a ViT-B/16 model that was instantiated with ImageNet weights expecting 224×224 inputs; I fix this by using the corresponding 384px pretrained weights so the model’s internal `image_size` matches your `img_size` and transforms. I also switch to using the weights’ built-in preprocess transform to ensure normalization/resizing are exactly aligned with the pretrained checkpoint (score-positive but still same core approach). Finally, I keep your existing TTA batching logic but make it robust to different collate structures and ensure a valid `submission.csv` is always written with the correct row count/order.'
- What this solution (achieved 0.1293) has done: 'The crash is because your `test_images/` directory contains a nested `test_images/` folder, and the dataset currently tries to open that directory as if it were an image file. I minimally fix the dataset to only include real image files (e.g., `.jpg/.png`) and ignore subdirectories, which unblocks the DataLoader and lets inference finish. To move accuracy upward toward the target (without changing your core approach), I also make TTA deterministic by seeding torchvision v2 random transforms per-image so you get stable, reproducible augmented views. Everything else (ViT-B/16 384px weights, head replacement, TTA averaging, submission alignment to `sample_submission.csv`) remains the same.'
- What this solution (achieved 0.05531) has done: 'The timeout is dominated by (1) fine-tuning the head over the full 18.7k training images and (2) expensive per-batch TTA that runs three stochastic geometric transforms on GPU/CPU and then forwards 4× the images through ViT. To keep the same core model/loops/semantics, the main speedups are: use the provided TFRecords for training/test to avoid slow JPEG decode and Python-level file iteration, fuse/collate TTA views into a single batched forward (still identical averaging logic), and reduce dataloader overhead with better worker/prefetch/persistent settings and avoiding repeated per-item work. These changes are equivalent (same pixels/labels, same transforms and averaging, same loss/training steps) but remove major I/O and Python overhead that causes the 10-minute timeout. Determinism is preserved by keeping the same seeds and explicitly seeding TFRecord shuffling and TTA generators.'
- What this solution (achieved 0.05531) has done: 'The timeout is dominated by (1) an unnecessary full-dataset finetuning pass over 18.7k images (3 epochs) and (2) expensive per-batch TTA that re-applies randomized geometric transforms on GPU tensors. To fit within 600s without changing the model, loss, or evaluation semantics, the key speed fixes are: use the provided TFRecords pipeline (already in the dataset) for much faster sequential reads, avoid `os.walk`/PIL decoding for test listing/loads, and make TTA deterministic but cheaper by pre-building the augmented batch via vectorized ops and avoiding repeated generator setup overhead. The architecture, weights, finetuning logic (still runs only when no checkpoint exists), and prediction averaging remain identical in meaning; only input pipeline and hot-loop overhead are reduced.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4))))

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = True  # faster deterministic kernel selection

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    torch.multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

train_tfrec_dir = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/"
test_tfrec_dir = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/"

img_size = 384
batch_size = 16
num_workers = min(8, (os.cpu_count() or 4))
num_classes = 5
tta = True

finetune_head = True
finetune_epochs = 3
finetune_lr = 3e-4

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1  # image_size=384
model = vit_b_16(weights=weights)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

ckpt_candidates = [
    "./model.pth",
    "./best.pth",
    "./checkpoint.pth",
    "/kaggle/working/model.pth",
    "/kaggle/working/best.pth",
    "/kaggle/working/checkpoint.pth",
]
loaded_any_ckpt = False
for p in ckpt_candidates:
    if os.path.isfile(p):
        state = torch.load(p, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k.replace("module.", "")
                new_state[nk] = v
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            print(f"Loaded checkpoint: {p}")
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            loaded_any_ckpt = True
        break

model.to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

try:
    if hasattr(torch, "compile"):
        model = torch.compile(
            model, mode="reduce-overhead", fullgraph=False, dynamic=False
        )
        print("torch.compile enabled")
except Exception as e:
    print("torch.compile not available, continuing eager. Reason:", repr(e))




## === cell 1
import glob
import io
import struct
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

weights_tfms = weights.transforms()


def _seed_worker(worker_id: int):
    base_seed = 3407
    seed = (base_seed + worker_id) % (2**32 - 1)
    torch.manual_seed(seed)


def _list_image_files(root_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    with os.scandir(root_dir) as it:
        for e in it:
            if e.is_file():
                name = e.name
                if name.lower().endswith(exts):
                    files.append(os.path.join(root_dir, name))
    files.sort()
    if len(files) == 0:
        raise RuntimeError(f"No image files found under: {root_dir}")
    return files


class CassavaImageDataset(Dataset):
    """
    Map-style dataset over image files.
    If labeled=True yields (img_tensor, label_int), else yields (img_tensor, image_id_str).
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        transform=None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.labeled = labels is not None
        if self.labeled:
            if len(self.image_paths) != len(self.labels):
                raise ValueError("image_paths and labels must have same length")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        image_id = os.path.basename(p)
        if self.labeled:
            return img, int(self.labels[idx])
        return img, image_id


def _varint_read(f) -> int:
    shift = 0
    result = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError
        i = b[0]
        result |= (i & 0x7F) << shift
        if not (i & 0x80):
            return result
        shift += 7


def _parse_example(pb: bytes) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    i = 0
    n = len(pb)

    def read_varint_at(pos: int) -> Tuple[int, int]:
        shift = 0
        result = 0
        while True:
            b = pb[pos]
            pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                return result, pos
            shift += 7

    def read_len_delim_at(pos: int) -> Tuple[bytes, int]:
        ln, pos = read_varint_at(pos)
        return pb[pos : pos + ln], pos + ln

    while i < n:
        key, i = read_varint_at(i)
        field = key >> 3
        wire = key & 7
        if field == 1 and wire == 2:
            features_bytes, i = read_len_delim_at(i)
            fb = features_bytes
            j = 0
            m = len(fb)
            while j < m:
                k2, j = read_varint_at(j)
                f2 = k2 >> 3
                w2 = k2 & 7
                if f2 == 1 and w2 == 2:
                    entry_bytes, j = read_len_delim_at(j)
                    eb = entry_bytes
                    t = 0
                    mm = len(eb)
                    fname = None
                    fval_bytes = None
                    while t < mm:
                        k3, t = read_varint_at(t)
                        f3 = k3 >> 3
                        w3 = k3 & 7
                        if f3 == 1 and w3 == 2:
                            s, t = read_len_delim_at(t)
                            fname = s.decode("utf-8")
                        elif f3 == 2 and w3 == 2:
                            fmsg, t = read_len_delim_at(t)
                            u = 0
                            L = len(fmsg)
                            got = None
                            while u < L:
                                k4, u = read_varint_at(u)
                                f4 = k4 >> 3
                                w4 = k4 & 7
                                if f4 == 1 and w4 == 2:
                                    bl, u = read_len_delim_at(u)
                                    vpos = 0
                                    while vpos < len(bl):
                                        k5, vpos = read_varint_at(vpos)
                                        f5 = k5 >> 3
                                        w5 = k5 & 7
                                        if f5 == 1 and w5 == 2:
                                            bts, vpos = read_len_delim_at(vpos)
                                            got = bts
                                            break
                                        else:
                                            if w5 == 0:
                                                _, vpos = read_varint_at(vpos)
                                            elif w5 == 2:
                                                _, vpos = read_len_delim_at(vpos)
                                            else:
                                                break
                                    break
                                elif f4 == 3 and w4 == 2:
                                    il, u = read_len_delim_at(u)
                                    vpos = 0
                                    while vpos < len(il):
                                        k5, vpos = read_varint_at(vpos)
                                        f5 = k5 >> 3
                                        w5 = k5 & 7
                                        if f5 == 1 and w5 == 0:
                                            ival, vpos = read_varint_at(vpos)
                                            got = str(int(ival)).encode("utf-8")
                                            break
                                        else:
                                            if w5 == 0:
                                                _, vpos = read_varint_at(vpos)
                                            elif w5 == 2:
                                                _, vpos = read_len_delim_at(vpos)
                                            else:
                                                break
                                    break
                                else:
                                    if w4 == 0:
                                        _, u = read_varint_at(u)
                                    elif w4 == 2:
                                        _, u = read_len_delim_at(u)
                                    else:
                                        break
                            fval_bytes = got
                        else:
                            if w3 == 0:
                                _, t = read_varint_at(t)
                            elif w3 == 2:
                                _, t = read_len_delim_at(t)
                            else:
                                break
                    if fname is not None and fval_bytes is not None:
                        out[fname] = fval_bytes
                else:
                    if w2 == 0:
                        _, j = read_varint_at(j)
                    elif w2 == 2:
                        _, j = read_len_delim_at(j)
                    else:
                        break
        else:
            if wire == 0:
                _, i = read_varint_at(i)
            elif wire == 2:
                _, i = read_len_delim_at(i)
            else:
                break
    return out


class CassavaTFRecordDataset(Dataset):
    """
    Map-style over pre-indexed TFRecord records for fast random access.
    Each item returns either (img_tensor, label_int) or (img_tensor, image_id_str).
    """

    def __init__(self, tfrec_paths: List[str], labeled: bool, transform=None):
        self.tfrec_paths = tfrec_paths
        self.labeled = labeled
        self.transform = transform
        self._index: List[Tuple[int, int]] = []  # (file_idx, offset)
        self._build_index()

    def _build_index(self):
        for fi, p in enumerate(self.tfrec_paths):
            with open(p, "rb") as f:
                while True:
                    off = f.tell()
                    hdr = f.read(8)
                    if not hdr:
                        break
                    (length,) = struct.unpack("<Q", hdr)
                    f.read(4)  # masked crc of length
                    f.seek(length, io.SEEK_CUR)
                    f.read(4)  # masked crc of data
                    self._index.append((fi, off))
        if len(self._index) == 0:
            raise RuntimeError("No records found in TFRecords")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int):
        fi, off = self._index[idx]
        p = self.tfrec_paths[fi]
        with open(p, "rb") as f:
            f.seek(off)
            (length,) = struct.unpack("<Q", f.read(8))
            f.read(4)
            data = f.read(length)
            f.read(4)
        ex = _parse_example(data)
        img_bytes = ex.get("image", None)
        if img_bytes is None:
            raise KeyError("TFRecord example missing 'image'")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        name_b = ex.get("image_name", b"")
        image_id = (
            name_b.decode("utf-8")
            if isinstance(name_b, (bytes, bytearray))
            else str(name_b)
        )
        if self.labeled:
            tgt_b = ex.get("target", b"0")
            label = int(tgt_b.decode("utf-8"))
            return img, label
        return img, image_id




## === cell 2
if finetune_head and (not loaded_any_ckpt):
    train_tfrec_paths = sorted(glob.glob(os.path.join(train_tfrec_dir, "*.tfrec")))
    if len(train_tfrec_paths) > 0:
        train_dataset = CassavaTFRecordDataset(
            train_tfrec_paths,
            labeled=True,
            transform=weights_tfms,
        )
    else:
        train_df = pd.read_csv(train_csv_path)
        train_paths = [
            os.path.join(train_dir, x) for x in train_df["image_id"].tolist()
        ]
        train_labels = train_df["label"].astype(int).tolist()
        train_dataset = CassavaImageDataset(
            train_paths,
            labels=train_labels,
            transform=weights_tfms,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # map-style dataset; OK to shuffle here
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
        drop_last=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    for p in model.parameters():
        p.requires_grad = False
    for p in model.heads.head.parameters():
        p.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.heads.head.parameters(), lr=finetune_lr)

    model.train()
    for epoch in range(finetune_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += float(loss.item()) * bs
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += int(bs)

        print(
            f"[finetune head] epoch {epoch+1}/{finetune_epochs} "
            f"loss={running_loss/max(total,1):.4f} acc={correct/max(total,1):.4f}"
        )

    torch.save(model.state_dict(), "/kaggle/working/model.pth")
    model.eval()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/227446480.py in <cell line: 0>()
     47         correct = 0
     48         total = 0
---> 49         for xb, yb in train_loader:
     50             xb = xb.to(device, non_blocking=True)
     51             if device.type == "cuda":

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    730             # If the exception takes multiple arguments, don't try to
    731             # instantiate since we don't know how to
--> 732             raise RuntimeError(msg) from None
    733         raise exception
    734 

RuntimeError: Caught UnicodeDecodeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/3108159876.py", line 272, in __getitem__
    ex = _parse_example(data)
         ^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/3108159876.py", line 140, in _parse_example
    fname = s.decode("utf-8")
            ^^^^^^^^^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb2 in position 52: invalid start byte


## === cell 3
if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_tfrec_paths = sorted(glob.glob(os.path.join(test_tfrec_dir, "*.tfrec")))
if len(test_tfrec_paths) > 0:
    test_dataset = CassavaTFRecordDataset(
        test_tfrec_paths,
        labeled=False,
        transform=weights_tfms,
    )
else:
    test_paths = _list_image_files(test_dir)
    test_dataset = CassavaImageDataset(
        test_paths,
        labels=None,
        transform=weights_tfms,
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

_softmax_dim1 = lambda x: torch.softmax(x, dim=1)




## === cell 4
all_names = []
all_preds = []

model.eval()


def _seed_for_batch_view(batch_idx: int, view_j: int) -> int:
    return int((3407 + batch_idx * 1000003 + view_j * 9176) % (2**31 - 1))


_cpu_g = torch.Generator(device="cpu")
_cuda_g = torch.Generator(device="cuda") if device.type == "cuda" else None

with torch.inference_mode():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        xb = inputs.to(device, non_blocking=True)  # (B,C,H,W)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)

        if tta:
            bsz, c, h, w = xb.shape
            n_tta = len(ttas)

            x_all = xb.new_empty((bsz * (n_tta + 1), c, h, w))
            x_all[0:bsz].copy_(xb)

            base = bsz
            for j, t in enumerate(ttas):
                seed = _seed_for_batch_view(batch_idx, j)
                _cpu_g.manual_seed(seed)
                if _cuda_g is not None:
                    _cuda_g.manual_seed(seed)

                try:
                    x_aug = t(
                        xb, generator=_cuda_g if device.type == "cuda" else _cpu_g
                    )
                except TypeError:
                    torch.manual_seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed(seed)
                    x_aug = t(xb)

                if device.type == "cuda":
                    x_aug = x_aug.contiguous(memory_format=torch.channels_last)
                start = base + j * bsz
                x_all[start : start + bsz].copy_(x_aug)

            preds_flat = _softmax_dim1(model(x_all))  # (B*(T+1), num_classes)
            preds_bt = preds_flat.view(n_tta + 1, bsz, -1).transpose(0, 1)
            mean_preds = preds_bt.mean(dim=1)  # (B, num_classes)
            pred_labels = torch.argmax(mean_preds, dim=1).tolist()
        else:
            preds = _softmax_dim1(model(xb))
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print("Preds generated:", len(all_preds), "Unique files:", len(set(all_names)))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2612282696.py in <cell line: 0>()
     13 
     14 with torch.inference_mode():
---> 15     for batch_idx, (inputs, filenames) in enumerate(test_loader):
     16         xb = inputs.to(device, non_blocking=True)  # (B,C,H,W)
     17         if device.type == "cuda":

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    730             # If the exception takes multiple arguments, don't try to
    731             # instantiate since we don't know how to
--> 732             raise RuntimeError(msg) from None
    733         raise exception
    734 

RuntimeError: Caught UnicodeDecodeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/3108159876.py", line 272, in __getitem__
    ex = _parse_example(data)
         ^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/3108159876.py", line 140, in _parse_example
    fname = s.decode("utf-8")
            ^^^^^^^^^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8a in position 1: invalid start byte


## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)

pred_ser = pd.Series(all_preds, index=pd.Index(all_names, name="image_id"))
pred_ser = pred_ser[~pred_ser.index.duplicated(keep="first")]

mapped = sample["image_id"].map(pred_ser)
sample["label"] = mapped.fillna(0).astype(int)

my_submission = sample[["image_id", "label"]]
my_submission.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", my_submission.shape)
print("Any missing predictions filled with 0:", int(mapped.isna().sum()))




## === cell 6
my_submission.head()
