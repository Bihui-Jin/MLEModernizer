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

albumentations==2.0.8
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

0.8856

# 6. Current score

0.09865

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.26233) has done: 'I removed the unsupported `A.Cutout` transform, which caused the pipeline definition to fail and consequently prevented `sub_aug` from being created. With the augmentation pipeline now valid, the inference loop runs correctly and produces a proper `submission.csv` file. No other logic was altered, preserving the original model architecture and inference strategy.'
- What this solution (achieved 0.10949) has done: 'The plan is to make the model actually use the fine‑tuned weights (the original script fell back to ImageNet‑only weights because the checkpoint path was wrong) and to give the model inputs that match its expected size.  
1. Try to load the checkpoint from the original path; if it does not exist, recursively search the `../input/` folder for the first `.pth` file and load it.  
2. Add an explicit resize to `224×224` after the random crop so the network receives the size it was trained on.  
These two tiny, targeted changes keep the core architecture and inference logic intact while should lift the accuracy well toward the target score.'
- What this solution (achieved 0.13602) has done: 'I adjust the checkpoint loader to handle typical checkpoint dict structures, and simplify the inference‑time augmentation (remove the random crop and disable dropout) so the model receives deterministic inputs. These minimal changes keep the architecture and overall pipeline intact while ensuring the fine‑tuned weights are actually used, which should raise the accuracy toward the target score.'
- What this solution (achieved 0.31241) has done: 'Implemented missing imports, corrected module references, and ensured paths and data handling work correctly. Added comprehensive imports at the start, kept the original model loading and checkpoint logic unchanged, and retained the inference pipeline to generate a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.17377) has done: 'Implemented a quick fine‑tuning stage to replace the fallback‑only ImageNet weights with a model trained on the provided cassava data.  
The script now:  
1. Loads any available fine‑tuned checkpoint (searches the entire `/kaggle/input` tree).  
2. If no checkpoint is found, it fine‑tunes the ResNeXt‑50 backbone on the training set for a few epochs (freezing then unfreezing layers) using lightweight Albumentations augmentations.  
3. Saves the best model and uses it for test‑time augmentation inference, finally writing a proper `submission.csv`.'
- What this solution (achieved 0.12369) has done: 'Implemented a fix for the Albumentations `RandomResizedCrop` initialization which caused a validation error under the current library version. The transform now uses named arguments (`height` and `width`) to satisfy the expected tuple‑based schema. No other logic was altered, preserving the original model architecture, training loop, and inference pipeline, ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.16143) has done: 'Implemented a fix for the Albumentations `RandomResizedCrop` initialization which caused a validation error under the current Albumentations 2.x version. The transform now uses the correct `size` argument, allowing the training pipeline to run without errors and produce a proper model fine‑tuning, which should improve validation accuracy and move the Kaggle score toward the target.'
- What this solution (achieved 0.43199) has done: 'Implemented two key fixes to get the pipeline running and improve model quality:  
1. Corrected the `RandomResizedCrop` usage to the new Albumentations signature (`height` / `width`).  
2. Made checkpoint handling robust – the script now tracks whether a fine‑tuned checkpoint was successfully loaded and skips the short training phase if it is, avoiding unnecessary re‑training. This keeps the core architecture unchanged while ensuring the best available weights are used, which should raise validation accuracy toward the target.'
- What this solution (achieved 0.09865) has done: 'Implemented two key fixes and modest enhancements:  
1. Corrected the Albumentations `RandomResizedCrop` call to the new signature (`size` argument) so the training pipeline runs without validation errors.  
2. Strengthened the fine‑tuning by extending training to 8 epochs and added a lightweight horizontal‑flip augmentation to the test‑time augmentation pipeline, giving the model a modest accuracy boost while preserving the original architecture and overall logic.'

# 9. Code solution

## === cell 0
import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader




## === cell 1
model_path = "../input/rn-tta-calr-ft-ofasf/model(11).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 5)
model = model.to(device)
model.eval()


def load_checkpoint(path):
    """Search for a .pth file under /kaggle/input and load it into `model`."""

    def attempt(p):
        try:
            ckpt = torch.load(p, map_location=device)
            if isinstance(ckpt, dict):
                if "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                elif "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)
            print(f"✅ Loaded checkpoint from {p}")
            return True
        except Exception as e:
            print(f"⚠️ Failed loading {p}: {e}")
            return False

    if os.path.isfile(path) and attempt(path):
        return True
    for p in glob.glob("/kaggle/input/**/*.pth", recursive=True):
        if attempt(p):
            return True
    return False


checkpoint_loaded = load_checkpoint(model_path)
if not checkpoint_loaded:
    print("🔎 No checkpoint found – will fine‑tune from ImageNet weights.")

if not checkpoint_loaded:

    class CassavaDataset(Dataset):
        def __init__(self, df, img_dir, transforms):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.transforms = transforms

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["image_id"])
            image = np.array(Image.open(img_path).convert("RGB"))
            aug = self.transforms(image=image)
            img = aug["image"]  # already normalized, shape HWC
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            label = int(row["label"])
            return img, label

    df = pd.read_csv(train_csv_path)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_tf = A.Compose(
        [
            A.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
        ]
    )

    val_tf = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
        ]
    )

    train_dataset = CassavaDataset(train_df, train_images_path, train_tf)
    val_dataset = CassavaDataset(val_df, train_images_path, val_tf)

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(8):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1} – Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"🏆 Loaded best fine‑tuned weights (val acc={best_val_acc:.4f})")
    else:
        print("⚠️ No improvement during fine‑tuning.")

model.eval()  # ensure eval mode before inference




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
size
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1169185555.py in <cell line: 0>()
     77     train_tf = A.Compose(
     78         [
---> 79             A.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
     80             A.HorizontalFlip(p=0.5),
     81             A.RandomBrightnessContrast(p=0.3),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
size
  Input should be a valid tuple [type=tuple_type, input_value=224, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 2
sub_aug = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
    ],
    p=1.0,
)

sample_sub = pd.read_csv(sample_sub_path)

predictions = []
softmax = torch.nn.functional.softmax

for _, row in sample_sub.iterrows():
    prob_sum = torch.zeros(5, device=device)
    img_path = os.path.join(test_images_path, row.image_id)

    pil_img = Image.open(img_path).convert("RGB")
    img_np = np.array(pil_img)

    for _ in range(5):
        aug_out = sub_aug(image=img_np)
        aug_img = aug_out["image"]
        tensor_img = torch.from_numpy(aug_img).permute(2, 0, 1).float().to(device)

        with torch.no_grad():
            logits = model(tensor_img.unsqueeze(0)).squeeze(0)
            probs = softmax(logits, dim=0)
            prob_sum += probs

    avg_probs = prob_sum / 5.0
    pred_label = avg_probs.argmax().item()
    predictions.append([row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print("✅ Submission file created – preview:")
print(sub_df.head())
