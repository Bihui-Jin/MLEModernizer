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

0.8986098519190088

# 6. Current score

0.10837

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10463) has done: 'Implemented a robust dataset loader that skips directory entries, preventing PIL from trying to open a folder as an image. This resolves the `IsADirectoryError` and ensures the number of predictions matches the expected test set size, allowing a valid `submission.csv` to be generated.'
- What this solution (achieved 0.76794) has done: 'I replace the missing pretrained models with a simple ImageNet‑pretrained EfficientNet backbone, freeze its weights, and train only a small linear head on the provided training set for one epoch. This keeps the original architecture (two feature extractors whose outputs are concatenated) while adding a lightweight fine‑tuning step that should raise accuracy well above the random‑guess baseline and move the score toward the target.'
- What this solution (achieved 0.10837) has done: 'I compile the models with torch.compile to reduce the per‑step compute cost, enable TF‑32 and high‑precision matrix math, and increase the data‑loader workers to better utilize CPU cores. These changes keep the exact architecture, loss, optimizer and training schedule, so model behavior and accuracy remain unchanged while runtime is shortened.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
import torchvision.models as models
from torch.cuda import amp  # mixed precision

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

eff_img_size = 528
vit_img_size = 384
batch_size = 16
num_workers = min(8, os.cpu_count() or 1)
num_classes = 5
tta = False  # disabled to simplify inference


def get_backbone():
    model = models.efficientnet_b0(weights="DEFAULT")
    model.classifier = nn.Identity()
    return model.to(device)


try:
    vit_model = torch.load(
        "/kaggle/input/vit-v1-update/vit_v1_1.pt", map_location=device
    ).to(device)
except FileNotFoundError:
    print("vit model not found, using EfficientNet backbone.")
    vit_model = get_backbone()

try:
    eff_model = torch.load(
        "/kaggle/input/efficient-net/vit_cont_3.pt", map_location=device
    ).to(device)
except FileNotFoundError:
    print("efficient model not found, using EfficientNet backbone.")
    eff_model = get_backbone()

linear_head = nn.Linear(2 * 1280, num_classes).to(device)

if hasattr(torch, "compile"):
    vit_model = torch.compile(vit_model)
    eff_model = torch.compile(eff_model)
    linear_head = torch.compile(linear_head)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data (test mode)."""

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        transform=None,
        ttas=None,
        img_size=384,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.images = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, f)) and not f.startswith(".")
            ]
        )
        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.ttas is not None and self.transform is not None:
            vit_img = [self.transform(t(vit_img)) for t in self.ttas]
            eff_img = [self.transform(t(eff_img)) for t in self.ttas]
        elif self.transform:
            vit_img = self.transform(vit_img)
            eff_img = self.transform(eff_img)

        return vit_img, eff_img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
class CassavaTrainDataset(VisionDataset):
    """Dataset for training – returns images and integer labels."""

    def __init__(
        self,
        csv_path,
        img_dir,
        vit_size,
        efficient_size,
        transform=None,
        img_size=384,
    ):
        super().__init__(root=img_dir)
        self.df = pd.read_csv(csv_path)
        self.image_ids = self.df["image_id"].tolist()
        self.labels = self.df["label"].tolist()
        self.transform = transform
        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        label = self.labels[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.transform:
            vit_img = self.transform(vit_img)
            eff_img = self.transform(eff_img)

        return vit_img, eff_img, label

    def __len__(self):
        return len(self.image_ids)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=30),  # corrected: removed unsupported 'p' argument
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir, vit_img_size, eff_img_size, transform=test_transforms, ttas=ttas
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size * 2,  # increased inference batch size for speed
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

train_csv = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"

train_dataset = CassavaTrainDataset(
    train_csv,
    train_dir,
    vit_img_size,
    eff_img_size,
    transform=train_transforms,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 4
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(linear_head.parameters())
    + list(vit_model.parameters())
    + list(eff_model.parameters()),
    lr=1e-3,
)

scaler = amp.GradScaler()  # mixed‑precision scaler

vit_model.train()
eff_model.train()
linear_head.train()

epochs = 5
for epoch in range(epochs):
    epoch_loss = 0.0
    for vit_inputs, eff_inputs, labels in train_loader:
        vit_inputs = vit_inputs.to(device, non_blocking=True)
        eff_inputs = eff_inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with amp.autocast():
            vit_feats = vit_model(vit_inputs)  # [B, 1280]
            eff_feats = eff_model(eff_inputs)  # [B, 1280]
            logits = linear_head(torch.cat([vit_feats, eff_feats], dim=1))  # [B, 5]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, loss: {epoch_loss/len(train_loader):.4f}")

vit_model.eval()
eff_model.eval()
linear_head.eval()
torch.cuda.empty_cache()  # free unused memory




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
/tmp/ipykernel_55/318531153.py in <cell line: 0>()
     24         with amp.autocast():
     25             vit_feats = vit_model(vit_inputs)  # [B, 1280]
---> 26             eff_feats = eff_model(eff_inputs)  # [B, 1280]
     27             logits = linear_head(torch.cat([vit_feats, eff_feats], dim=1))  # [B, 5]
     28             loss = criterion(logits, labels)

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
    676 
    677             with TracingContext.report_output_strides() as fwd_output_strides:
--> 678                 compiled_fw_func = aot_config.fw_compiler(fw_module, adjusted_flat_args)
    679 
    680             if not hasattr(compiled_fw_func, "_boxed_call"):

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
   2066             )
   2067         with dynamo_timed("PyCodeCache.load_by_key_path", log_pt2_compile_event=True):
-> 2068             mod = PyCodeCache.load_by_key_path(
   2069                 key,
   2070                 path,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/codecache.py in load_by_key_path(cls, key, path, linemap, attrs)
   2757             linemap = []
   2758 
-> 2759         mod = _reload_python_module(key, path)
   2760 
   2761         # unzip into separate lines/nodes lists

/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/compile_tasks.py in _reload_python_module(key, path)
     43         mod.__file__ = path
     44         mod.key = key  # type: ignore[attr-defined]
---> 45         exec(code, mod.__dict__, mod.__dict__)
     46         sys.modules[mod.__name__] = mod
     47         return mod

/tmp/torchinductor_root/gk/cgkvf4vvrblt7g6th6fnxipg337r4pjcdvqpqqsp4qtqv3yx3qrb.py in <module>
   7770 
   7771 
-> 7772 async_compile.wait(globals())
   7773 del async_compile
   7774 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/async_compile.py in wait(self, scope)
    303                     if isinstance(result, (Future, CodeCacheFuture)):
    304                         try:
--> 305                             scope[key] = result.result()
    306                         except BrokenProcessPool as e:
    307                             raise RuntimeError(

/usr/local/lib/python3.11/dist-packages/torch/_inductor/codecache.py in result(self)
   3239         if self.future is not None:
   3240             # If the worker failed this will throw an exception.
-> 3241             result = self.future.result()
   3242             assert result is None
   3243             self.future = None

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

BackendCompilerFailed: backend='inductor' raised:
SubprocException: An exception occurred in a subprocess:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_worker/subproc_pool.py", line 279, in do_job
    result = job()
             ^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/compile_tasks.py", line 68, in _worker_compile_triton
    load_kernel().precompile(warm_cache_only=True)
  File "/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/triton_heuristics.py", line 293, in precompile
    compiled_binary, launcher = self._precompile_config(
                                ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/triton_heuristics.py", line 493, in _precompile_config
    binary = triton.compile(*compile_args, **compile_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/triton/compiler/compiler.py", line 273, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/triton/compiler/compiler.py", line 100, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
triton.compiler.errors.CompilationError: at 17:45:
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 22)
    x1 = xindex // 22
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + 16*((((r2 + (8/11)*x0 + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2))) // (1 + (triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + (triton_helpers.div_floor_integer((-1) + ks0,  2)) + (triton_helpers.div_floor_integer((-1) + ks1,  2)))) % 16)) + x1*(triton_helpers.div_floor_integer((-1) + ks0,  2)) + x1*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + (triton_helpers.div_floor_integer((-1) + ks1,  2))*((((r2 + (8/11)*x0 + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2))) // (1 + (triton_helpers.div_floor_integer((-1) + ks1,  2)))) % (1 + (triton_helpers.div_floor_integer((-1) + ks0,  2))))) + 16*(triton_helpers.div_floor_integer((-1) + ks0,  2))*((((r2 + (8/11)*x0 + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2))) // (1 + (triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + (triton_helpers.div_floor_integer((-1) + ks0,  2)) + (triton_helpers.div_floor_integer((-1) + ks1,  2)))) % 16)) + 16*(triton_helpers.div_floor_integer((-1) + ks1,  2))*((((r2 + (8/11)*x0 + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2))) // (1 + (triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + (triton_helpers.div_floor_integer((-1) + ks0,  2)) + (triton_helpers.div_floor_integer((-1) + ks1,  2)))) % 16)) + x1*(triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + 16*(triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2))*((((r2 + (8/11)*x0 + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2))) // (1 + (triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + (triton_helpers.div_floor_integer((-1) + ks0,  2)) + (triton_helpers.div_floor_integer((-1) + ks1,  2)))) % 16)) + ((r2 % (1 + (triton_helpers.div_floor_integer((-1) + ks1,  2))))) + ((((r2 + (8/11)*x0 + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks1,  2)) + (8/11)*x0*(triton_helpers.div_floor_integer((-1) + ks0,  2))*(triton_helpers.div_floor_integer((-1) + ks1,  2))) // (1 + (triton_helpers.div_floor_integer((-1) + ks1,  2)))) % (1 + (triton_helpers.div_floor_integer((-1) + ks0,  2)))))), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
                                             ^
TypeError('unexpected type fp32')


Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


## === cell 5
all_names = []
all_preds = []

with torch.no_grad():
    for batch_idx, (vit_inputs, eff_inputs, filenames) in enumerate(test_loader):
        vit_inputs = vit_inputs.to(device, non_blocking=True)
        eff_inputs = eff_inputs.to(device, non_blocking=True)

        with amp.autocast():
            vit_feats = vit_model(vit_inputs)
            eff_feats = eff_model(eff_inputs)
            logits = linear_head(torch.cat([vit_feats, eff_feats], dim=1))
            probs = normalizer(logits)

        pred_labels = torch.argmax(probs, dim=1).cpu().tolist()
        all_names.extend(filenames)
        all_preds.extend(pred_labels)




## === cell 6
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
my_submission.to_csv("submission.csv", index=False)
my_submission
