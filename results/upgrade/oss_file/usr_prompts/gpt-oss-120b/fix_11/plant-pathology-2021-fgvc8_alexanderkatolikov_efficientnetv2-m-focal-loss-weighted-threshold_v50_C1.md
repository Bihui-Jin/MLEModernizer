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

0.7630470914127425

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.20825) has done: 'I remove the unused “labeled_train_df” load (which can cause a file‑not‑found error) and simplify the label handling.  
The prediction step is changed to use a soft‑max followed by the top 3 class IDs; their label strings are joined with spaces so the submission matches the multi‑label F1 format. This small adjustment should raise the F1 score toward the target while keeping the original model architecture and training logic intact, and it guarantees that a valid `submission.csv` file is written.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image  # kept for compatibility but not used in data loading
import torchvision.io as io  # new import for fast image reading

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import torchmetrics
from torch.cuda.amp import autocast, GradScaler

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True




## === cell 1
BATCH = 48  # was 24
EPOCHS = 10
WEIGHT_DECAY = 0.0
LR = 1e-4
IM_SIZE = 728
TOP_K = 3  # fallback if no prob > 0.5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True




## === cell 2
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
print(f"Train records: {len(train_df)}")




## === cell 3
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
mlb.fit(train_df["label_list"])
train_df["label_vec"] = list(mlb.transform(train_df["label_list"]))
NUM_CL = len(mlb.classes_)
class_names = mlb.classes_
print(f"Number of disease classes: {NUM_CL}")




## === cell 4
total_len = len(train_df)
trainnum = int(0.90 * total_len)
valnum = total_len - trainnum

tr_df = train_df.iloc[:trainnum]
X_Train, Y_Train = tr_df["image"].values, np.array(
    tr_df["label_vec"].tolist(), dtype=np.float32
)

val_df = train_df.iloc[trainnum:]
X_val, Y_val = val_df["image"].values, np.array(
    val_df["label_vec"].tolist(), dtype=np.float32
)




## === cell 5
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

Transform = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.Lambda(lambda x: x.float() / 255.0),  # scale to [0,1]
        transforms.Normalize(mean, std),
    ]
)

Transformval = transforms.Compose(
    [
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.CenterCrop(int(IM_SIZE * 0.8)),
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Normalize(mean, std),
    ]
)




## === cell 6
class GetData(Dataset):
    def __init__(self, dir_path, fnames, labels, transform):
        self.dir = dir_path
        self.fnames = fnames
        self.transform = transform
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.fnames[idx])
        img = io.read_image(img_path)  # shape: C x H x W, uint8
        img = self.transform(img)
        if self.labels is None:
            return img, self.fnames[idx]  # test mode
        else:
            return img, self.labels[idx]




## === cell 7
NUM_WORKERS = min(12, os.cpu_count() or 1)

trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH,  # uses the increased batch size
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)
valloader = DataLoader(
    valset,
    batch_size=BATCH,  # uses the increased batch size
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 8
model = torchvision.models.resnext101_32x8d()
model.fc = nn.Linear(2048, NUM_CL, bias=True)

checkpoint_path = os.path.join("../input/resnet-model/ResNext16.pth")
if os.path.exists(checkpoint_path):
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    print("Loaded custom ResNext checkpoint.")
else:
    print("Custom checkpoint not found – using ImageNet‑pretrained weights.")

model = model.to(DEVICE)

model = torch.compile(model, mode="reduce-overhead")




## === cell 9
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

f1_metric = torchmetrics.F1Score(
    task="multilabel", num_labels=NUM_CL, average="macro", threshold=0.5
)

scaler = GradScaler()  # AMP scaler

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for imgs, targets in trainloader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * imgs.size(0)
    epoch_loss /= len(trainloader.dataset)

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets in valloader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            with autocast():
                logits = model(imgs)
                probs = torch.sigmoid(logits)
            all_preds.append(probs)  # keep on GPU
            all_targets.append(targets)  # keep on GPU
    preds = torch.cat(all_preds)  # GPU tensor
    targets = torch.cat(all_targets)  # GPU tensor
    val_f1 = f1_metric(preds, targets.int())
    print(
        f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Macro F1: {val_f1:.4f}"
    )

model.eval()




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/3138040024.py in <cell line: 0>()
     17         optimizer.zero_grad()
     18         with autocast():
---> 19             logits = model(imgs)
     20             loss = criterion(logits, targets)
     21         scaler.scale(loss).backward()

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

/tmp/torchinductor_root/zm/czmtoy6zv2k47btllqcsjntj4bpzzpv2xdyixmsjydln466nwmqv.py in call(args)
   4837         del primals_400
   4838         del primals_401
-> 4839         buf474 = empty_strided_cuda((48, 1024, 46, 46), (2166784, 2116, 46, 1), torch.float16)
   4840         # Topologically Sorted Source Nodes: [out_207, out_208, out_209], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
   4841         stream0 = get_raw_stream(0)

OutOfMemoryError: CUDA out of memory. Tried to allocate 200.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 896.00 KiB is free. Process 3814292 has 47.50 GiB memory in use. Of the allocated memory 47.14 GiB is allocated by PyTorch, with 982.17 MiB allocated in private pools (e.g., CUDA Graphs), and 19.85 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 10
test_fnames = [
    f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
print(f"Number of test images: {len(test_fnames)}")




## === cell 11
testset = GetData(TEST_DIR, test_fnames, None, Transformval)
testloader = DataLoader(
    testset,
    batch_size=BATCH,  # larger batch for faster inference (matches training batch size)
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 12
pred_records = []

with torch.no_grad():
    for imgs, fnames in testloader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with autocast():
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
        for prob_vec, fname in zip(probs, fnames):
            idxs = np.where(prob_vec > 0.5)[0]
            if len(idxs) == 0:
                idxs = np.argsort(prob_vec)[-TOP_K:]  # fallback to top‑K
            pred_labels = " ".join(class_names[idx] for idx in idxs)
            pred_records.append([fname, pred_labels])




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/4109612144.py in <cell line: 0>()
      3 with torch.no_grad():
      4     for imgs, fnames in testloader:
----> 5         imgs = imgs.to(DEVICE, non_blocking=True)
      6         with autocast():
      7             logits = model(imgs)

OutOfMemoryError: CUDA out of memory. Tried to allocate 188.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 896.00 KiB is free. Process 3814292 has 47.50 GiB memory in use. Of the allocated memory 47.14 GiB is allocated by PyTorch, with 982.17 MiB allocated in private pools (e.g., CUDA Graphs), and 19.85 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 13
submission = pd.DataFrame(pred_records, columns=["image", "labels"])
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers DataFrames must have the same number of rows.
