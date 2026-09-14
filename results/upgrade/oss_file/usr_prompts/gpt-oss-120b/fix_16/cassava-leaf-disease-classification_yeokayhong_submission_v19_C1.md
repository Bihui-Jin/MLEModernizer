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

0.8878815352070112

# 6. Current score

0.24626

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10912) has done: 'I filter the test directory so only actual image files are loaded, avoiding the nested “test_images” folder that caused an IsADirectoryError. The dataset initialization now walks the directory tree and keeps only files with common image extensions, guaranteeing that the number of predictions matches the number of test images and the submission CSV has the correct length.'
- What this solution (achieved 0.24738) has done: 'I fixed the model‑weight loading to avoid the missing `ViT_H_14_Weights.IMAGENET1K_V1` enum by falling back to a no‑weight instantiation, and I simplified the test‑image discovery to list only files directly under the test folder (preventing nested folder duplication). These changes eliminate the import error, define `model` correctly, and ensure the number of predictions matches the expected submission length, producing a valid `submission.csv`.'
- What this solution (achieved 0.24701) has done: 'I simplify the test‑time preprocessing to match the model’s expected input better. The original pipeline resized the image to a large side, padded it, then resized again, which can distort features and hurt accuracy. By directly resizing to the model’s input size, converting to a float tensor, and normalising with ImageNet statistics, we keep the visual information intact while preserving the core model logic. This modest change should raise the validation accuracy toward the target without altering the architecture or training procedure.'
- What this solution (achieved 0.24626) has done: 'I revert to a more careful preprocessing that preserves aspect ratio: resize the longer side to the model size, pad to a square, then resize to the exact input dimensions before normalising. This matches the model’s expected input better than a direct stretch resize and should raise the validation‑style accuracy toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.24701) has done: 'I simplify the test‑time preprocessing so the image is resized directly to the model’s expected input size, removing the extra max‑side resize and padding steps that can distort features. This change keeps the core model and inference unchanged while aligning the input more closely with how the pretrained weights were trained, which should raise the validation‑style accuracy and move the Kaggle score toward the target.'
- What this solution (achieved 0.24626) has done: 'I (1) ensure the vision‑transformer is instantiated with ImageNet pretrained weights when the custom checkpoint is missing (instead of starting from random weights), and (2) improve test‑time preprocessing by preserving aspect ratio: resize the longer side to the model size, pad to a square, then resize to the exact input dimensions. These small, targeted changes keep the core model unchanged while aligning the inputs with how the pretrained model was trained, which should raise the validation‑style accuracy and move the Kaggle score nearer the target.'

# 9. Code solution

## === cell 0
import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader  # added for efficient batching

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(42)

torch.set_num_threads(max(1, os.cpu_count() // 2))
torch.set_float32_matmul_precision("high")




## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/4/efficientnet_v2_l_480_8591_ISP_CBP.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/6/vit_h_14_518_8860_base.pth"
)
vit_image_size = 518

model_select = "vit"

if model_select == "vit":
    model_image_size = vit_image_size
elif model_select == "en":
    model_image_size = en_image_size
else:
    raise ValueError("model_select must be 'vit' or 'en'")




## === cell 2
def resize_max_side(img, size):
    width, height = img.size
    if max(width, height) <= size:
        return img
    if width > height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)
    return TF.resize(img, (new_height, new_width))


def pad_to_square(img):
    width, height = img.size
    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,
        (max_side - height) // 2,
        (max_side - width) - (max_side - width) // 2,
        (max_side - height) - (max_side - height) // 2,
    )
    return TF.pad(img, padding, padding_mode="reflect")




## === cell 3
val_transforms = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 4
if model_select == "vit":
    try:
        vit_model = models.vit_h_14(
            weights=models.ViT_H_14_Weights.IMAGENET1K_V1, image_size=model_image_size
        )
    except AttributeError:
        vit_model = models.vit_h_14(weights=None, image_size=model_image_size)

    if os.path.exists(vit_model_path):
        vit_model.load_state_dict(
            torch.load(vit_model_path, map_location=device, weights_only=True)
        )
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )
    vit_model.to(device)
    vit_model.eval()
    model = vit_model
elif model_select == "en":
    if os.path.exists(en_model_path):
        en_model = models.efficientnet_v2_l(weights=None)
        en_model.load_state_dict(
            torch.load(en_model_path, map_location=device, weights_only=True)
        )
    else:
        en_model = models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
        )
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )
    en_model.to(device)
    en_model.eval()
    model = en_model

if hasattr(torch, "compile"):
    model = torch.compile(model, mode="reduce-overhead")
model.eval()

if device.type == "cpu":
    example_input = torch.randn(1, 3, model_image_size, model_image_size, device=device)
    model = torch.jit.trace(model, example_input)




## === cell 5
def get_image_filenames(root_dir):
    """Return a sorted list of image file names that are directly inside root_dir."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    files = [
        f
        for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f))
        and os.path.splitext(f)[1].lower() in valid_exts
    ]
    return sorted(files)


class TestDataset(Dataset):
    def __init__(self, file_names, root_dir, transform, target_size):
        self.file_names = file_names
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        path = os.path.join(self.root_dir, name)
        img = Image.open(path).convert("RGB")
        img = resize_max_side(img, self.target_size)
        img = pad_to_square(img)
        img = TF.resize(img, (self.target_size, self.target_size))
        tensor = self.transform(img)
        return tensor, name


sorted_filenames = get_image_filenames(test_data_directory)
test_dataset = TestDataset(
    sorted_filenames, test_data_directory, val_transforms, model_image_size
)

batch_size = 64 if device.type == "cuda" else 32

if device.type == "cuda":
    max_workers = min(16, max(1, os.cpu_count() // 2))
    pin_mem = True
else:
    max_workers = 0
    pin_mem = False

loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=max_workers,
    pin_memory=pin_mem,
    persistent_workers=max_workers > 0,
    prefetch_factor=2,
)

predictions = []
image_ids = []

with torch.inference_mode():
    for batch_imgs, batch_names in tqdm(loader, desc="Test"):
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        flipped_imgs = torch.flip(batch_imgs, dims=[-1])
        combined = torch.cat([batch_imgs, flipped_imgs], dim=0)

        logits_combined = model(combined)

        half = batch_imgs.size(0)
        logits_orig, logits_flip = torch.split(logits_combined, half, dim=0)

        logits = (logits_orig + logits_flip) / 2

        preds = torch.argmax(logits, dim=1).cpu().tolist()
        predictions.extend(preds)
        image_ids.extend(batch_names)




## === cell 6
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
