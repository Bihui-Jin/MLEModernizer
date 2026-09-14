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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7485687903970452

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, copy, time, sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn import preprocessing
from sklearn.metrics import f1_score

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True  # enable TF‑32 for faster matmul on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True  # same for CUDA matmul
torch.set_float32_matmul_precision("high")
torch.manual_seed(42)  # deterministic seed for reproducibility




## === cell 1
BATCH = 512
EPOCHS = 5
WEIGHT_DECAY = 0.0
LR = 1e-5
IM_SIZE = 224
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"




## === cell 2
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
try:
    labeled_train_df = pd.read_csv("../input/train-labeled/train.csv")
except Exception:
    labeled_train_df = None




## === cell 3
le = preprocessing.LabelEncoder()
le.fit(train_df["labels"])
train_df["label_id"] = le.transform(train_df["labels"])

if labeled_train_df is not None:
    my_dict = {}
    NUM_CL = len(train_df["labels"].value_counts())
    for i in range(NUM_CL):
        eq = (
            labeled_train_df.loc[labeled_train_df["label_id"] == i].values[0][1]
            != train_df.loc[train_df["label_id"] == i].values[0][1]
        )
        if eq:
            for l in range(NUM_CL):
                eq2 = (
                    labeled_train_df.loc[labeled_train_df["label_id"] == i].values[0][1]
                    == train_df.loc[train_df["label_id"] == l].values[0][1]
                )
                if eq2:
                    my_dict[l] = i
    train_df = train_df.replace({"label_id": my_dict})

class_map = {row[0]: row[1] for row in train_df[["label_id", "labels"]].values.tolist()}
NUM_CL = len(class_map)




## === cell 4
train_transform = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
val_transform = train_transform  # same steps for validation & test




## === cell 5
class GetData(Dataset):
    def __init__(self, root_dir, filenames, labels, transform):
        self.root = root_dir
        self.fnames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if self.labels is None:
            return img, self.fnames[idx]  # test mode
        else:
            return img, self.labels[idx]  # train/val mode




## === cell 6
full_dataset = GetData(
    TRAIN_DIR, train_df["image"].values, train_df["label_id"].values, train_transform
)

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

num_workers = min(8, os.cpu_count() or 1)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=8,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=8,
)




## === cell 7
model = torchvision.models.resnext101_32x8d()
model.fc = nn.Linear(model.fc.in_features, NUM_CL, bias=True)

pretrained_path = os.path.join("../input/resnet-model/ResNext16.pth")
if os.path.isfile(pretrained_path):
    try:
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    except Exception:
        pass  # fall back to random initialization

model = model.to(DEVICE)

if DEVICE.type == "cuda":
    model = torch.compile(model, mode="max-autotune")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scaler = torch.cuda.amp.GradScaler()




## === cell 8
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(DEVICE, non_blocking=True), targets.to(
            DEVICE, non_blocking=True
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{EPOCHS} – Train loss: {epoch_loss:.4f}")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(targets.numpy())

val_f1 = f1_score(all_labels, all_preds, average="macro")
print(f"Final validation F1: {val_f1:.4f}")




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/446395477.py in <cell line: 0>()
      8         optimizer.zero_grad()
      9         with torch.cuda.amp.autocast():
---> 10             logits = model(imgs)
     11             loss = criterion(logits, targets)
     12         scaler.scale(loss).backward()

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

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in forward(self, x)
    282         return x
    283 
--> 284     def forward(self, x: Tensor) -> Tensor:
    285         return self._forward_impl(x)
    286 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    743             )
    744             try:
--> 745                 return fn(*args, **kwargs)
    746             finally:
    747                 _maybe_set_eval_frame(prior)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in forward(*runtime_args)
   1182         full_args.extend(params_flat)
   1183         full_args.extend(runtime_args)
-> 1184         return compiled_fn(full_args)
   1185 
   1186     # Just for convenience

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in runtime_wrapper(args)
    308                 True
    309             ), torch.enable_grad():
--> 310                 all_outs = call_func_at_runtime_with_args(
    311                     compiled_fn, args_, disable_amp=disable_amp, steal_args=True
    312                 )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    124     with context():
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:
    128             # TODO: Please remove soon

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in g(args)
     98 def make_boxed_func(f):
     99     def g(args):
--> 100         return f(*args)
    101 
    102     g._boxed_call = True  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/torch/autograd/function.py in apply(cls, *args, **kwargs)
    573             # See NOTE: [functorch vjp and autograd interaction]
    574             args = _functorch.utils.unwrap_dead_wrappers(args)
--> 575             return super().apply(*args, **kwargs)  # type: ignore[misc]
    576 
    577         if not is_setup_ctx_defined:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in forward(ctx, *deduped_flat_tensor_args)
   1583                 # - Note that donated buffer logic requires (*saved_tensors, *saved_symints) showing up last
   1584                 #   in the fw output order.
-> 1585                 fw_outs = call_func_at_runtime_with_args(
   1586                     CompiledFunction.compiled_fw,
   1587                     args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    124     with context():
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:
    128             # TODO: Please remove soon

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in wrapper(runtime_args)
    488                 )
    489                 return out
--> 490             return compiled_fn(runtime_args)
    491 
    492         return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in inner_fn(args)
    670                 old_args.clear()
    671 
--> 672             outs = compiled_fn(args)
    673 
    674             # Inductor cache DummyModule can return None

/usr/local/lib/python3.11/dist-packages/torch/_inductor/output_code.py in __call__(self, inputs)
    464         assert self.current_callable is not None
    465         try:
--> 466             return self.current_callable(inputs)
    467         finally:
    468             AutotuneCacheBundler.end_compile()

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in run(new_inputs)
   1206             ), dynamo_utils.preserve_rng_state():
   1207                 compiled_fn = cudagraphify_fn(model, new_inputs, static_input_idxs)
-> 1208         return compiled_fn(new_inputs)
   1209 
   1210     return run

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in deferred_cudagraphify(inputs)
    396         copy_misaligned_inputs(inputs, check_input_idxs)
    397 
--> 398         fn, out = cudagraphify(model, inputs, new_static_input_idxs, *args, **kwargs)
    399         fn = align_inputs_from_check_idxs(fn, inputs_to_check=check_input_idxs)
    400         fn_cache[int_key] = fn

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in cudagraphify(model, inputs, static_input_idxs, device_index, is_backward, is_inference, stack_traces, constants, placeholders, mutated_input_idxs)
    426     )
    427 
--> 428     return manager.add_function(
    429         model,
    430         inputs,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in add_function(self, model, inputs, static_input_idxs, stack_traces, mode, constants, placeholders, mutated_input_idxs)
   2251         # container needs to set clean up when fn dies
   2252         get_container(self.device_index).add_strong_reference(fn)
-> 2253         return fn, fn(inputs)
   2254 
   2255     @property

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in run(self, new_inputs, function_id)
   1945         assert self.graph is not None, "Running CUDAGraph after shutdown"
   1946         self.mode = self.id_to_mode[function_id]
-> 1947         out = self._run(new_inputs, function_id)
   1948 
   1949         # The forwards are only pending following invocation, not before

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in _run(self, new_inputs, function_id)
   2053                 log_pt2_compile_event=True,
   2054             ):
-> 2055                 out = self.run_eager(new_inputs, function_id)
   2056 
   2057             return out

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in run_eager(self, new_inputs, function_id)
   2217         self.path_state = ExecutionState.WARMUP
   2218         self.update_generation()
-> 2219         return node.run(new_inputs)
   2220 
   2221     def new_graph_id(self) -> GraphID:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in run(self, new_inputs)
    641             self.device_index, self.cuda_graphs_pool, self.stream
    642         ), get_history_recording():
--> 643             out = self.wrapped_function.model(new_inputs)
    644 
    645         # We need to know which outputs are allocated within the cudagraph pool

/tmp/torchinductor_root/t6/ct6wohj7fkhv2fd6wo4e2jniqwppr53gtxoec5lf5lgwnck2mmli.py in call(args)
   5361         del primals_406
   5362         del primals_407
-> 5363         buf481 = empty_strided_cuda((512, 1024, 14, 14), (200704, 196, 14, 1), torch.float16)
   5364         # Topologically Sorted Source Nodes: [out_211, out_212], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
   5365         stream0 = get_raw_stream(0)

OutOfMemoryError: CUDA out of memory. Tried to allocate 196.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 88.88 MiB is free. Process 2341196 has 47.41 GiB memory in use. Of the allocated memory 47.05 GiB is allocated by PyTorch, with 390.33 MiB allocated in private pools (e.g., CUDA Graphs), and 19.80 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 9
test_fnames = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
test_dataset = GetData(TEST_DIR, test_fnames, None, val_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=8,
)

model.eval()
pred_records = []
with torch.no_grad():
    for imgs, fnames in test_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        pred_ids = torch.argmax(probs, dim=1).cpu().numpy()
        for fname, pid in zip(fnames, pred_ids):
            pred_records.append([fname, pid])

pred_df = pd.DataFrame(pred_records, columns=["image", "label_id"])
pred_df["labels"] = pred_df["label_id"].map(class_map)

submission = pred_df[["image", "labels"]]
submission.to_csv("submission.csv", index=False)
print('Submission file "submission.csv" written with', len(submission), "rows.")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/1029865013.py in <cell line: 0>()
     15 with torch.no_grad():
     16     for imgs, fnames in test_loader:
---> 17         imgs = imgs.to(DEVICE, non_blocking=True)
     18         with torch.cuda.amp.autocast():
     19             logits = model(imgs)

OutOfMemoryError: CUDA out of memory. Tried to allocate 294.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 84.88 MiB is free. Process 2341196 has 47.41 GiB memory in use. Of the allocated memory 47.05 GiB is allocated by PyTorch, with 390.33 MiB allocated in private pools (e.g., CUDA Graphs), and 19.80 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
