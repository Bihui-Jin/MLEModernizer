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

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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

0.8884859474161378

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob, json, random, copy, pathlib
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet152, ResNet152_Weights

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")



## === cell 1
train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(BASE_PATH, "train_images", x)
)

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=42,
)




## === cell 2
class CassavaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["image_path"].values
        self.has_labels = "label" in df.columns
        if self.has_labels:
            self.labels = df["label"].values
        else:
            self.labels = None
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx]) if self.has_labels else -1
        return img, label




## === cell 3
IMG_SIZE = 512
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_transform = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(degrees=90)], p=0.5),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=mean, std=std),
    ]
)

val_transform = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=mean, std=std),
    ]
)

test_transform = val_transform  # same as validation



## === cell 4
BATCH_SIZE = 96
NUM_WORKERS = min(8, os.cpu_count() or 8)

train_dataset = CassavaDataset(train_df, transform=train_transform)
val_dataset = CassavaDataset(val_df, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)



## === cell 5
weights = ResNet152_Weights.IMAGENET1K_V1
model = resnet152(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes
model = model.to(DEVICE)

model = torch.compile(model, mode="max-autotune")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scaler = torch.cuda.amp.GradScaler()



## === cell 6
EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for imgs, lbls in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        lbls = lbls.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * imgs.size(0)

    epoch_loss /= len(train_loader.dataset)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Acc: {acc:.4f}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/2300101422.py in <cell line: 0>()
     13 
     14         scaler.scale(loss).backward()
---> 15         scaler.step(optimizer)
     16         scaler.update()
     17 

/usr/local/lib/python3.11/dist-packages/torch/amp/grad_scaler.py in step(self, optimizer, *args, **kwargs)
    455         ), "No inf checks were recorded for this optimizer."
    456 
--> 457         retval = self._maybe_opt_step(optimizer, optimizer_state, *args, **kwargs)
    458 
    459         optimizer_state["stage"] = OptState.STEPPED

/usr/local/lib/python3.11/dist-packages/torch/amp/grad_scaler.py in _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs)
    350         retval: Optional[float] = None
    351         if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
--> 352             retval = optimizer.step(*args, **kwargs)
    353         return retval
    354 

/usr/local/lib/python3.11/dist-packages/torch/optim/optimizer.py in wrapper(*args, **kwargs)
    491                             )
    492 
--> 493                 out = func(*args, **kwargs)
    494                 self._optimizer_step_code()
    495 

/usr/local/lib/python3.11/dist-packages/torch/optim/optimizer.py in _use_grad(self, *args, **kwargs)
     89             torch.set_grad_enabled(self.defaults["differentiable"])
     90             torch._dynamo.graph_break()
---> 91             ret = func(self, *args, **kwargs)
     92         finally:
     93             torch._dynamo.graph_break()

/usr/local/lib/python3.11/dist-packages/torch/optim/adam.py in step(self, closure)
    232             beta1, beta2 = group["betas"]
    233 
--> 234             has_complex = self._init_group(
    235                 group,
    236                 params_with_grad,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    743             )
    744             try:
--> 745                 return fn(*args, **kwargs)
    746             finally:
    747                 _maybe_set_eval_frame(prior)

/usr/local/lib/python3.11/dist-packages/torch/optim/adam.py in _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps)
    172                     )
    173                     # Exponential moving average of gradient values
--> 174                     state["exp_avg"] = torch.zeros_like(
    175                         p, memory_format=torch.preserve_format
    176                     )

OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 896.00 KiB is free. Process 1686436 has 47.51 GiB memory in use. Of the allocated memory 47.00 GiB is allocated by PyTorch, with 46.22 GiB allocated in private pools (e.g., CUDA Graphs), and 149.10 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 7
test_paths = glob.glob(os.path.join(BASE_PATH, "test_images", "*.jpg"))
test_df = pd.DataFrame({"image_path": test_paths})
test_dataset = CassavaDataset(test_df, transform=test_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)



## === cell 8
model.eval()
all_preds = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())

submission = pd.DataFrame(
    {"image_id": [os.path.basename(p) for p in test_paths], "label": all_preds}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
/tmp/ipykernel_55/2658646857.py in <cell line: 0>()
      5         imgs = imgs.to(DEVICE, non_blocking=True)
      6         with torch.cuda.amp.autocast():
----> 7             outputs = model(imgs)
      8         preds = torch.argmax(outputs, dim=1).cpu().numpy()
      9         all_preds.extend(preds.tolist())

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

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py in aot_dispatch_base(flat_fn, flat_args, aot_config, fw_metadata)
    201                 assert isinstance(fw_module, GraphModule)
    202                 tensorify_python_scalars(fw_module, fake_mode.shape_env, fake_mode)
--> 203             compiled_fw = compiler(fw_module, updated_flat_args)
    204 
    205         if fakified_out_wrapper.needs_post_compile:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in __call__(self, gm, example_inputs)
    487         example_inputs: Sequence[InputType],
    488     ) -> OutputCode:
--> 489         return self.compiler_fn(gm, example_inputs)
    490 
    491 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in fw_compiler_base(gm, example_inputs, is_inference)
   1739                     model_outputs_node.meta["user_visible_output_idxs"] = []
   1740 
-> 1741                 return inner_compile(
   1742                     gm,
   1743                     example_inputs,

/usr/lib/python3.11/contextlib.py in inner(*args, **kwds)
     79         def inner(*args, **kwds):
     80             with self._recreate_cm():
---> 81                 return func(*args, **kwds)
     82         return inner
     83 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx_inner(gm, example_inputs, **kwargs)
    567         )
    568 
--> 569         return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
    570             gm,
    571             example_inputs,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_aot.py in debug_wrapper(gm, example_inputs, **kwargs)
    100             # Call the compiler_fn - which is either aot_autograd or inductor
    101             # with fake inputs
--> 102             inner_compiled_fn = compiler_fn(gm, example_inputs)
    103         except Exception as e:
    104             # TODO: Failures here are troublesome because no real inputs,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in _compile_fx_inner(gm, example_inputs, **graph_kwargs)
    683             TritonBundler.begin_compile()
    684             try:
--> 685                 mb_compiled_graph = fx_codegen_and_compile(
    686                     gm, example_inputs, inputs_to_check, **graph_kwargs
    687                 )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in fx_codegen_and_compile(gm, example_inputs, inputs_to_check, **graph_kwargs)
   1127     scheme: FxCompile = _InProcessFxCompile()
   1128 
-> 1129     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
   1130 
   1131 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in codegen_and_compile(self, gm, example_inputs, inputs_to_check, graph_kwargs)
   1042                                 )
   1043                         else:
-> 1044                             compiled_fn = graph.compile_to_module().call
   1045 
   1046                     num_bytes, nodes_num_elem, node_runtimes = graph.count_bytes()

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in compile_to_module(self)
   2025             dynamo_compile_column_us="inductor_code_gen_cumulative_compile_time_us",
   2026         ):
-> 2027             return self._compile_to_module()
   2028 
   2029     def _compile_to_module(self) -> ModuleType:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in _compile_to_module(self)
   2031 
   2032         code, linemap = (
-> 2033             self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
   2034         )
   2035         if config.triton.autotune_at_compile_time:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in codegen(self)
   1962             self.init_wrapper_code()
   1963 
-> 1964             self.scheduler = Scheduler(self.operations)
   1965             V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)
   1966 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in __init__(self, nodes)
   1796     def __init__(self, nodes: List[ir.Operation]) -> None:
   1797         with dynamo_timed("Scheduler.__init__"):
-> 1798             self._init(nodes)
   1799 
   1800     def _init(self, nodes: List[ir.Operation]) -> None:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in _init(self, nodes)
   1868         if config._pre_fusion_custom_pass is not None:
   1869             self.nodes = config._pre_fusion_custom_pass(self.nodes)
-> 1870         self.nodes = self.fuse_nodes(self.nodes)
   1871         if config.reorder_for_peak_memory:
   1872             from .memory import reorder_for_peak_memory

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in fuse_nodes(self, nodes)
   2375                     old_len,
   2376                 )
-> 2377                 nodes = self.fuse_nodes_once(nodes)
   2378                 new_len = len(nodes)
   2379                 fusion_log.debug(

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in fuse_nodes_once(self, nodes)
   2672                 node1, node2
   2673             ):
-> 2674                 if not self.speedup_by_fusion(node1, node2):
   2675                     continue
   2676                 fusion_log.debug(

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in speedup_by_fusion(self, node1, node2)
   2572 
   2573             _, ms1 = multi_node.get_min_choice()
-> 2574             ms2, path2 = self.benchmark_fused_nodes(node_list_2)
   2575 
   2576             min_ms_fused = float("inf")

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in benchmark_fused_nodes(self, nodes)
   2413         backend = self.get_backend(device)
   2414         with dynamo_timed("benchmark_fused_nodes"):
-> 2415             return backend.benchmark_fused_nodes(nodes)
   2416 
   2417     def finalize_multi_template_buffers(self) -> None:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/codegen/cuda_combined_scheduling.py in benchmark_fused_nodes(self, nodes)
     90 
     91     def benchmark_fused_nodes(self, nodes):
---> 92         return self._triton_scheduling.benchmark_fused_nodes(nodes)
     93 
     94     def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/codegen/triton.py in benchmark_fused_nodes(self, nodes)
   3656                 return ms, mod.__file__
   3657 
-> 3658             args = mod.get_args()
   3659             call = mod.call
   3660             wrapped_jit_function = mod.triton_

/tmp/torchinductor_root/gg/cggeijpbkhdnk2a4njrcuy2c2lfoitrumdyb5k2ltax5c64snhkl.py in get_args()
     70 
     71 def get_args():
---> 72     arg_0 = rand_strided((96, 256, 128, 128), (4194304, 1, 32768, 256), device='cuda:0', dtype=torch.float16)
     73     arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
     74     arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/testing.py in rand_strided(size, stride, dtype, device, extra_size)
    390             )
    391         else:
--> 392             buffer = torch.randn(needed_size, dtype=dtype, device=device)
    393     else:
    394         buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)

BackendCompilerFailed: backend='inductor' raised:
OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 896.00 KiB is free. Process 1686436 has 47.51 GiB memory in use. Of the allocated memory 47.00 GiB is allocated by PyTorch, with 46.22 GiB allocated in private pools (e.g., CUDA Graphs), and 149.10 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
