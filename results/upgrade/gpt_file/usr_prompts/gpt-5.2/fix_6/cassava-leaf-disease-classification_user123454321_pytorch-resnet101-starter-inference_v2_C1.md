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

3.9

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8732245391356905

# 6. Current score

0.2571

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12444) has done: 'I fix the two runtime blockers: the missing external weight file (so inference never runs) and the hard-coded `.cuda()` call that crashes on CPU-only environments. To move accuracy toward the target (current is far below target), I keep your ResNet101-based model but load ImageNet pretrained weights when the custom checkpoint isn’t available, which is a minimal change that should greatly improve predictions versus random initialization. I also ensure inference uses `device` consistently and runs under `torch.inference_mode()` to prevent empty predictions and speed up safely. Finally, I write a properly aligned `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.72422) has done: 'Your current score (0.12444) is far below the target (0.8732), and the main remaining issue is that you are effectively submitting an ImageNet model with a randomly initialized 5-class head, which yields near-random predictions. To move accuracy sharply toward the target while keeping your core ResNet101 logic, the smallest legitimate improvement is to train only the final linear layer on the provided `train.csv` images (freezing the backbone) for a short, fixed number of epochs. This preserves the architecture and loss semantics while producing a head aligned to cassava labels, which should dramatically reduce the gap without introducing early stopping or approximations. I also switch to the official ResNet101 weight transforms (mean/std + crop/resize) to better match pretrained expectations, and keep the submission writing/ordering identical.'
- What this solution (achieved 0.72982) has done: 'Your score gap to the target is large (0.72422 vs 0.87322), so we should improve generalization with minimal, metric-aligned changes while keeping your ResNet101 + linear head training logic intact. The smallest reliable boost here is to add a proper train/validation split and train the linear head for a fixed number of epochs while selecting the best epoch by validation accuracy (still a fixed training procedure; no early stopping), then use that best head for test inference. I also make the dataloading deterministic and add standard image-reading robustness (`ImageFile.LOAD_TRUNCATED_IMAGES`) to avoid rare PIL read issues that can silently harm training. All I/O paths and the model architecture/loss stay the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.2571) has done: 'I focus on eliminating the two biggest sources of wasted time: (1) unnecessary training when the checkpoint is present, and (2) slow image I/O + dataloading overhead that bottlenecks the GPU/CPU pipeline. The core model, transforms, loss, optimizer, and training loops remain identical; changes are limited to faster DataLoader settings, deterministic seeding, and enabling safe PyTorch compilation/cudnn TF32 where available (negligible float diffs). I also reduce per-sample pandas overhead in `__getitem__` by pre-extracting arrays of file names/labels (same semantics). Finally, inference is sped up by larger batch size (same predictions) and persistent workers/prefetching to keep the device fed.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import torchvision.models as models
import torch.utils.data as data
from PIL import Image, ImageFile
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True



## === cell 1
input_path = "/kaggle/input/cassava-leaf-disease-classification/"



## === cell 2
df = pd.read_csv(os.path.join(input_path, "train.csv"))



## === cell 3
num_classes = len(df.label.unique())
num_classes




## === cell 4
class Model(nn.Module):
    def __init__(self, use_imagenet_pretrained=True):
        super(Model, self).__init__()
        if use_imagenet_pretrained:
            weights = models.ResNet101_Weights.IMAGENET1K_V2
        else:
            weights = None
        backbone = models.resnet101(weights=weights)
        self.base = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.dense(x)




## === cell 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 6
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass



## === cell 7
model = Model(use_imagenet_pretrained=True).to(device)



## === cell 8
ckpt_path = "/kaggle/input/resnet-cassava-model/model_101_20.pth"
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")
else:
    print(
        f"Checkpoint not found: {ckpt_path}. Will train a linear head on top of the ImageNet-pretrained backbone."
    )



## === cell 9
submission_df = pd.read_csv(os.path.join(input_path, "sample_submission.csv"))



## === cell 10
submission_df.head()




## === cell 11
class ImageDataset(data.Dataset):
    def __init__(self, df, image_dir, transforms, return_label=True):
        super().__init__()
        df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.return_label = return_label
        self.image_ids = df["image_id"].to_numpy()
        self.labels = df["label"].to_numpy(dtype=np.int64) if return_label else None

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_ids[index])
        img = Image.open(img_path).convert("RGB")
        x = self.transforms(img)
        if self.return_label:
            y = int(self.labels[index])
            return x, y
        return x

    def __len__(self):
        return len(self.image_ids)




## === cell 12
imagenet_weights = models.ResNet101_Weights.IMAGENET1K_V2

norm_mean = list(imagenet_weights.transforms().mean)
norm_std = list(imagenet_weights.transforms().std)

train_transforms = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ]
)
test_transforms = imagenet_weights.transforms()



## === cell 13
if not os.path.exists(ckpt_path):
    for p in model.base.parameters():
        p.requires_grad = False

    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    ((train_idx, val_idx),) = splitter.split(df["image_id"], df["label"])
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = ImageDataset(
        df=train_df,
        image_dir=os.path.join(input_path, "train_images"),
        transforms=train_transforms,
        return_label=True,
    )
    val_dataset = ImageDataset(
        df=val_df,
        image_dir=os.path.join(input_path, "train_images"),
        transforms=test_transforms,
        return_label=True,
    )

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    label_counts = train_df["label"].value_counts().sort_index()
    class_weights = 1.0 / (label_counts.values.astype(np.float64) + 1e-12)
    sample_weights = class_weights[train_df["label"].values]
    sampler = data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=g,
    )

    nw = min(8, (os.cpu_count() or 2))
    common_loader_kwargs = dict(
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )
    if common_loader_kwargs["prefetch_factor"] is None:
        common_loader_kwargs.pop("prefetch_factor")

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,  # sampler controls sampling
        sampler=sampler,
        **common_loader_kwargs,
    )
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        **common_loader_kwargs,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.dense.parameters(), lr=3e-3, weight_decay=1e-2)

    epochs = 8  # fixed count (no early stopping)
    best_val_acc = -1.0
    best_state = None

    _compiled = False
    if torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
            _compiled = True
        except Exception:
            _compiled = False

    for ep in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for xb, yb in train_dataloader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_acc = correct / max(total, 1)

        model.eval()
        v_correct = 0
        v_total = 0
        with torch.inference_mode():
            for xb, yb in val_dataloader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                preds = logits.argmax(1)
                v_correct += (preds == yb).sum().item()
                v_total += xb.size(0)
        val_acc = v_correct / max(v_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"epoch {ep+1}/{epochs} | loss {running_loss/max(total,1):.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f} | best_val_acc {best_val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        print(f"Loaded best epoch weights by val_acc={best_val_acc:.4f}.")

    full_dataset = ImageDataset(
        df=df.reset_index(drop=True),
        image_dir=os.path.join(input_path, "train_images"),
        transforms=train_transforms,
        return_label=True,
    )

    full_counts = df["label"].value_counts().sort_index()
    full_class_weights = 1.0 / (full_counts.values.astype(np.float64) + 1e-12)
    full_sample_weights = full_class_weights[df["label"].values]
    full_sampler = data.WeightedRandomSampler(
        weights=torch.as_tensor(full_sample_weights, dtype=torch.double),
        num_samples=len(full_sample_weights),
        replacement=True,
        generator=g,
    )

    full_loader = data.DataLoader(
        full_dataset,
        batch_size=32,
        shuffle=False,
        sampler=full_sampler,
        **common_loader_kwargs,
    )

    optimizer = torch.optim.AdamW(model.dense.parameters(), lr=3e-3, weight_decay=1e-2)
    model.train()
    for ep in range(epochs):
        running_loss = 0.0
        total = 0
        for xb, yb in full_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)

        print(f"refit epoch {ep+1}/{epochs} | loss {running_loss/max(total,1):.4f}")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
/tmp/ipykernel_55/889666561.py in <cell line: 0>()
     96 
     97             optimizer.zero_grad(set_to_none=True)
---> 98             logits = model(xb)
     99             loss = criterion(logits, yb)
    100             loss.backward()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in RETURN_VALUE(self, inst)
   3046 
   3047     def RETURN_VALUE(self, inst):
-> 3048         self._return(inst)
   3049 
   3050     def RETURN_CONST(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _return(self, inst)
   3031         )
   3032         log.debug("%s triggered compile", inst.opname)
-> 3033         self.output.compile_subgraph(
   3034             self,
   3035             reason=GraphCompileReason(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_subgraph(self, tx, partial_convert, reason)
   1099             # optimization to generate better code in a common case
   1100             self.add_output_instructions(
-> 1101                 self.compile_and_call_fx_graph(
   1102                     tx, list(reversed(stack_values)), root, output_replacements
   1103                 )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_and_call_fx_graph(self, tx, rv, root, replaced_outputs)
   1380 
   1381             with self.restore_global_state():
-> 1382                 compiled_fn = self.call_user_compiler(gm)
   1383 
   1384             from torch.fx._lazy_graph_module import _LazyGraphModule

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in call_user_compiler(self, gm)
   1430             dynamo_compile_column_us="aot_autograd_cumulative_compile_time_us",
   1431         ):
-> 1432             return self._call_user_compiler(gm)
   1433 
   1434     def _call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1481             raise e
   1482         except Exception as e:
-> 1483             raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
   1484                 e.__traceback__
   1485             ) from None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1460             if config.verify_correctness:
   1461                 compiler_fn = WrapperBackend(compiler_fn)
-> 1462             compiled_fn = compiler_fn(gm, self.example_inputs())
   1463             _step_logger()(logging.INFO, f"done compiler function {name}")
   1464             assert callable(compiled_fn), "compiler_fn did not return callable"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_dynamo.py in __call__(self, gm, example_inputs, **kwargs)
    128                     raise
    129         else:
--> 130             compiled_gm = compiler_fn(gm, example_inputs)
    131 
    132         return compiled_gm

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in __call__(self, model_, inputs_)
   2338         from torch._inductor.compile_fx import compile_fx
   2339 
-> 2340         return compile_fx(model_, inputs_, config_patches=self.config)
   2341 
   2342     def get_compiler_config(self):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1550     if config_patches:
   1551         with config.patch(config_patches):
-> 1552             return compile_fx(
   1553                 model_,
   1554                 example_inputs_,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1861             unlift_effect_tokens=True
   1862         ):
-> 1863             return aot_autograd(
   1864                 fw_compiler=fw_compiler,
   1865                 bw_compiler=bw_compiler,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/backends/common.py in __call__(self, gm, example_inputs, **kwargs)
     81             # NB: NOT cloned!
     82             with enable_aot_logging(), patch_config:
---> 83                 cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
     84                 counters["aot_autograd"]["ok"] += 1
     85                 return disable(cg)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations, inference_compiler, cudagraphs)
   1153         )
   1154     else:
-> 1155         compiled_fn = dispatch_and_compile()
   1156 
   1157     if isinstance(mod, torch._dynamo.utils.GmWrapper):

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in dispatch_and_compile()
   1129         functional_call = create_functional_call(mod, params_spec, params_len)
   1130         with compiled_autograd._disable():
-> 1131             compiled_fn, _ = create_aot_dispatcher_function(
   1132                 functional_call,
   1133                 fake_flat_args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    578 ) -> Tuple[Callable, ViewAndMutationMeta]:
    579     with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
--> 580         return _create_aot_dispatcher_function(
    581             flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
    582         )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in _create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    828         compiler_fn = choose_dispatcher(needs_autograd, aot_config)
    829 
--> 830         compiled_fn, fw_metadata = compiler_fn(
    831             flat_fn,
    832             _dup_fake_script_obj(fake_flat_args),

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py in aot_dispatch_autograd(flat_fn, flat_args, aot_config, fw_metadata)
    447             if fake_mode is not None and fake_mode.shape_env is not None:
    448                 tensorify_python_scalars(fx_g, fake_mode.shape_env, fake_mode)
--> 449             fw_module, bw_module = aot_config.partition_fn(
    450                 fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs
    451             )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in partition_fn(gm, joint_inputs, **kwargs)
   1777             cuda_context = get_cuda_device_context(gm)
   1778             with cuda_context:
-> 1779                 _recursive_joint_graph_passes(gm)
   1780             return min_cut_rematerialization_partition(
   1781                 gm, joint_inputs, **kwargs, compiler="inductor"

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in _recursive_joint_graph_passes(gm)
    320             subgraph = getattr(gm, subgraph_name)
    321             _recursive_joint_graph_passes(subgraph)
--> 322         joint_graph_passes(gm)
    323 
    324 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/joint_graph.py in joint_graph_passes(graph)
    466             maybe_count = GraphTransformObserver(
    467                 graph, f"pass_pattern_{i}"
--> 468             ).apply_graph_pass(patterns.apply)
    469             count += maybe_count if maybe_count is not None else 0
    470 

/usr/local/lib/python3.11/dist-packages/torch/fx/passes/graph_transform_observer.py in apply_graph_pass(self, pass_fn)
     68         with self:
     69             if not self._check_disable_pass():
---> 70                 return pass_fn(self.gm.graph)
     71 
     72         return None

/usr/local/lib/python3.11/dist-packages/torch/_inductor/pattern_matcher.py in apply(self, gm)
   1771                     if os.environ.get("TORCHINDUCTOR_PATTERN_MATCH_DEBUG") == node.name:
   1772                         log.warning("%s%s %s %s", node, node.args, m, entry.pattern)
-> 1773                     if is_match(m) and entry.extra_check(m):
   1774                         count += 1
   1775                         entry.apply(m, graph, node)  # type: ignore[arg-type]

/usr/local/lib/python3.11/dist-packages/torch/_inductor/pattern_matcher.py in check_fn(match)
   1350             specific_pattern_match = specific_pattern.match(node)
   1351 
-> 1352             if is_match(specific_pattern_match) and extra_check(specific_pattern_match):
   1353                 # trace the pattern using the shapes from the user program
   1354                 match.replacement_graph = trace_fn(replace_fn, args)

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in should_pad_mm(match)
    706 def should_pad_mm(match: Match) -> bool:
    707     mat1, mat2 = fetch_fake_tensors(match, ("mat1", "mat2"))
--> 708     return should_pad_common(mat1, mat2) and should_pad_bench(
    709         match, mat1, mat2, torch.ops.aten.mm
    710     )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in should_pad_bench(*args, **kwargs)
    384 def should_pad_bench(*args, **kwargs):
    385     with dynamo_timed("pad_mm_benchmark"):
--> 386         return _should_pad_bench(*args, **kwargs)
    387 
    388 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in _should_pad_bench(match, mat1, mat2, op, input)
    592 
    593         if ori_time is None:
--> 594             ori_time = do_bench(orig_bench_fn)
    595             set_cached_base_mm_benchmark_time(ori_time_key, ori_time)
    596 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/benchmarking.py in wrapper(self, *args, **kwargs)
     64             "benchmarking." + self.__class__.__name__ + "." + fn.__name__
     65         ] += 1
---> 66         return fn(self, *args, **kwargs)
     67 
     68     return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/benchmarking.py in benchmark_gpu(self, _callable, **kwargs)
    200         elif "return_mode" in kwargs:
    201             return self.triton_do_bench(_callable, **kwargs)
--> 202         return self.triton_do_bench(_callable, **kwargs, return_mode="median")
    203 
    204 

/usr/local/lib/python3.11/dist-packages/triton/testing.py in do_bench(fn, warmup, rep, grad_to_none, quantiles, return_mode)
    115     di = runtime.driver.active.get_device_interface()
    116 
--> 117     fn()
    118     di.synchronize()
    119 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/fx_passes/pad_mm.py in orig_bench_fn()
    561         def orig_bench_fn():
    562             if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
--> 563                 op(mat1, mat2)
    564             else:
    565                 op(input, mat1, mat2)

/usr/local/lib/python3.11/dist-packages/torch/_ops.py in __call__(self, *args, **kwargs)
   1121         if self._has_torchbind_op_overload and _must_dispatch_in_python(args, kwargs):
   1122             return _call_overload_packet_from_python(self, args, kwargs)
-> 1123         return self._op(*args, **(kwargs or {}))
   1124 
   1125     # TODO: use this to make a __dir__

BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


## === cell 14
test_dataset = ImageDataset(
    df=submission_df,
    image_dir=os.path.join(input_path, "test_images"),
    transforms=test_transforms,
    return_label=False,
)

nw = min(8, (os.cpu_count() or 2))
test_loader_kwargs = dict(
    shuffle=False,
    batch_size=128 if torch.cuda.is_available() else 64,
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(nw > 0),
    prefetch_factor=4 if nw > 0 else None,
)
if test_loader_kwargs["prefetch_factor"] is None:
    test_loader_kwargs.pop("prefetch_factor")

test_dataloader = data.DataLoader(
    dataset=test_dataset,
    **test_loader_kwargs,
)



## === cell 15
model.eval()
test_preds = []

with torch.inference_mode():
    for x in test_dataloader:
        x = x.to(device, non_blocking=True)
        y_pred = model(x)
        test_preds.extend(y_pred.argmax(1).cpu().numpy().tolist())

test_preds = np.asarray(test_preds, dtype=np.int64)
len(test_preds), submission_df.shape



## === cell 16
if len(test_preds) != len(submission_df):
    raise RuntimeError(
        f"Prediction length {len(test_preds)} != submission length {len(submission_df)}"
    )

submission_df["label"] = test_preds



## === cell 17
submission_df[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df[["image_id", "label"]].shape)
