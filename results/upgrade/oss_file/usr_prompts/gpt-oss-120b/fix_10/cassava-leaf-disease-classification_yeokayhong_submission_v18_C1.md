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

0.8732245391356905

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from torchvision import models, transforms, io
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
import os

from PIL import Image

torch.set_num_threads(os.cpu_count() or 1)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")




## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # speeds up convolutional kernels on GPU

num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/4/efficientnet_v2_l_480_8591_ISP_CBP.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/5/vit_h_14_518_8718_CBP.pth"
)
vit_image_size = 518

model_select = "vit"  # choose either "vit" or "en"

if model_select == "vit":
    model_image_size = vit_image_size
elif model_select == "en":
    model_image_size = en_image_size
else:
    raise ValueError("model_select must be 'vit' or 'en'")




## === cell 2
def invert_square_pad(img):
    width, height = img.size

    center_width, center_height = width // 2, height // 2
    top_left = img.crop((0, 0, center_width, center_height))
    top_right = img.crop((center_width, 0, width, center_height))
    bottom_left = img.crop((0, center_height, center_width, height))
    bottom_right = img.crop((center_width, center_height, width, height))

    top_combined = Image.new("RGB", (width, center_height))
    top_combined.paste(bottom_right, (0, 0))
    top_combined.paste(bottom_left, (center_width, 0))

    bottom_combined = Image.new("RGB", (width, center_height))
    bottom_combined.paste(top_right, (0, 0))
    bottom_combined.paste(top_left, (center_width, 0))

    flipped_img = Image.new("RGB", (width, height))
    flipped_img.paste(top_combined, (0, 0))
    flipped_img.paste(bottom_combined, (0, center_height))

    img = flipped_img.copy()
    del top_combined, bottom_combined, flipped_img

    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,  # left
        (max_side - height) // 2,  # top
        (max_side - width) - (max_side - width) // 2,  # right
        (max_side - height) - (max_side - height) // 2,  # bottom
    )

    padded_img = TF.pad(img, padding, padding_mode="reflect")
    return padded_img




## === cell 3
val_transforms = transforms.Compose(
    [
        transforms.Resize((model_image_size, model_image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)




## === cell 4
if model_select == "vit":
    try:
        vit_model = models.vit_h_14(weights=None, image_size=model_image_size)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
        vit_state = torch.load(vit_model_path, map_location=device, weights_only=True)
        vit_model.load_state_dict(vit_state)
    except Exception as e:
        print(f"Custom ViT weights not loaded ({e}); using random‑init ViT.")
        vit_model = models.vit_h_14(weights=None, image_size=model_image_size)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
    if hasattr(torch, "compile"):
        vit_model = torch.compile(vit_model)
    vit_model.to(device)
    vit_model.eval()

if model_select == "en":
    try:
        en_model = models.efficientnet_v2_l(weights=None)
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )
        en_state = torch.load(en_model_path, map_location=device, weights_only=True)
        en_model.load_state_dict(en_state)
    except Exception as e:
        print(
            f"Custom EfficientNetV2 weights not loaded ({e}); using random‑init EfficientNetV2."
        )
        en_model = models.efficientnet_v2_l(weights=None)
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )
    if hasattr(torch, "compile"):
        en_model = torch.compile(en_model)
    en_model.to(device)
    en_model.eval()




## === cell 5
sample_sub = pd.read_csv(sample_submission_path)
image_ids = sample_sub["image_id"].tolist()
total_len = len(image_ids)

existing_indices = []
existing_ids = []
for idx, img_name in enumerate(image_ids):
    if os.path.exists(os.path.join(test_data_directory, img_name)):
        existing_indices.append(idx)
        existing_ids.append(img_name)

name_to_orig_idx = {
    img_name: idx for img_name, idx in zip(existing_ids, existing_indices)
}


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, ids, base_dir, img_size):
        self.ids = ids
        self.base_dir = base_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.ids)

    def _load_tensor(self, path):
        img = io.read_image(path)  # (C, H, W), uint8
        img = TF.convert_image_dtype(img, dtype=torch.float)  # to [0,1]
        img = TF.resize(img, [self.img_size, self.img_size])
        img = TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return img

    def __getitem__(self, i):
        img_name = self.ids[i]
        img_path = os.path.join(self.base_dir, img_name)
        img_tensor = self._load_tensor(img_path)
        return img_tensor, img_name


batch_size = 256  # larger batch reduces forward‑pass overhead
num_workers = min(8, os.cpu_count() or 1)  # better overlap of I/O and compute

test_dataset = TestDataset(existing_ids, test_data_directory, model_image_size)
loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,  # enables faster transfer to GPU when available
)

predictions = [0] * total_len

model = vit_model if model_select == "vit" else en_model

with torch.inference_mode():
    for batch_imgs, batch_names in tqdm(loader, desc="Predicting"):
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        outputs = model(batch_imgs)
        batch_preds = torch.argmax(outputs, dim=1).cpu().tolist()
        for img_name, pred in zip(batch_names, batch_preds):
            orig_idx = name_to_orig_idx[img_name]  # O(1) lookup
            predictions[orig_idx] = int(pred)

submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/3922398536.py in <cell line: 0>()
     57     for batch_imgs, batch_names in tqdm(loader, desc="Predicting"):
     58         batch_imgs = batch_imgs.to(device, non_blocking=True)
---> 59         outputs = model(batch_imgs)
     60         batch_preds = torch.argmax(outputs, dim=1).cpu().tolist()
     61         for img_name, pred in zip(batch_names, batch_preds):

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

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in forward(self, x)
    287         return x
    288 
--> 289     def forward(self, x: torch.Tensor):
    290         # Reshape and permute the input tensor
    291         x = self._process_input(x)

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
    321                 if grad_enabled:
    322                     torch._C._set_grad_enabled(False)
--> 323                 all_outs = call_func_at_runtime_with_args(
    324                     compiled_fn, args, disable_amp=disable_amp, steal_args=True
    325                 )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    124     with context():
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:
    128             # TODO: Please remove soon

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in inner_fn(args)
    670                 old_args.clear()
    671 
--> 672             outs = compiled_fn(args)
    673 
    674             # Inductor cache DummyModule can return None

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in wrapper(runtime_args)
    488                 )
    489                 return out
--> 490             return compiled_fn(runtime_args)
    491 
    492         return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_inductor/output_code.py in __call__(self, inputs)
    464         assert self.current_callable is not None
    465         try:
--> 466             return self.current_callable(inputs)
    467         finally:
    468             AutotuneCacheBundler.end_compile()

/usr/local/lib/python3.11/dist-packages/torch/_inductor/utils.py in run(new_inputs)
   2126     def run(new_inputs: List[InputType]):
   2127         copy_misaligned_inputs(new_inputs, inputs_to_check)
-> 2128         return model(new_inputs)
   2129 
   2130     return run

/tmp/torchinductor_root/kq/ckqn3awetsjxfzh7ij363ngyfigls4abqdiqsydyemnkre2esddt.py in call(args)
   1448         del arg18_1
   1449         # Topologically Sorted Source Nodes: [input_7, x_8, _native_multi_head_attention_1], Original ATen: [aten.add, aten.native_layer_norm, aten._native_multi_head_attention]
-> 1450         buf24 = torch.ops.aten._native_multi_head_attention.default(buf21, buf22, buf23, 1280, 16, arg20_1, arg19_1, arg21_1, arg22_1, None, False)
   1451         del arg19_1
   1452         del arg20_1

/usr/local/lib/python3.11/dist-packages/torch/_ops.py in __call__(self, *args, **kwargs)
    721     # that are named "self". This way, all the aten ops can be called by kwargs.
    722     def __call__(self, /, *args, **kwargs):
--> 723         return self._op(*args, **kwargs)
    724 
    725     # Use positional-only argument to avoid naming collision with aten ops arguments

OutOfMemoryError: CUDA out of memory. Tried to allocate 28.64 GiB. GPU 0 has a total capacity of 47.53 GiB of which 5.24 GiB is free. Process 594906 has 42.27 GiB memory in use. Of the allocated memory 23.26 GiB is allocated by PyTorch, and 18.69 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
