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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
timm==1.0.19
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7316677251078982

# 6. Current score

0.08987

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18592) has done: 'I make the script robust by handling missing pretrained model files: if a model checkpoint cannot be found it fall back to a standard ImageNet‑pretrained timm model, fine‑tune it quickly on the training set, and use the resulting model(s) for inference. This fixes the file‑not‑found errors, ensures the list of tensors is never empty, and produces a valid `submission.csv`. The core architecture (ResNet‑18) and overall pipeline remain unchanged, while the minimal fine‑tuning should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.08987) has done: 'The fix adds the required `task='multiclass'` argument when creating the `CohenKappa` metric, which removes the TypeError and enables proper validation scoring. The training epochs are increased slightly (to 5) to give the model a bit more learning without altering the core architecture, helping lift the quadratic weighted kappa toward the target score while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import timm
from torchmetrics import CohenKappa




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 3
train_csv = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root = "/kaggle/input/aptos2019-blindness-detection/train_images"
test_csv = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root = "/kaggle/input/aptos2019-blindness-detection/test_images"




## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b5": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/efficientnet_b5.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/inception_v4.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v1/4/seresnext101_32x4d.pth",
}
model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext101_32x4d": "seresnext101_32x4d",
}
models_list = []
loaded_keys = []

for key, path in model_paths.items():
    name = model_names[key]
    try:
        model = timm.create_model(name, pretrained=False, num_classes=5)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded checkpoint for {key}")
    except Exception as e:
        print(f"Could not load {key} from {path}: {e}")
        print(f"Falling back to ImageNet pretrained {name}")
        model = timm.create_model(name, pretrained=True, num_classes=5)
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_keys.append(key)




## === cell 5
train_dataset = BlindnessDataset(
    train_csv, train_root, transform=train_transform, test=False
)
val_len = int(0.1 * len(train_dataset))
train_len = len(train_dataset) - val_len
train_set, val_set = random_split(
    train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)

fine_tune_model = models_list[0]
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fine_tune_model.parameters(), lr=1e-4)

kappa_metric = CohenKappa(num_classes=5, task="multiclass", weights="quadratic").to(
    device
)

epochs = 5
fine_tune_model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = fine_tune_model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / train_len
    fine_tune_model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            probs = torch.softmax(fine_tune_model(imgs), dim=1)
            preds = probs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_true.append(lbls.cpu())
    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)
    qwk = kappa_metric(all_preds, all_true).item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val QWK: {qwk:.4f}")
    fine_tune_model.train()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in wrapped_func(*args, **kwargs)
    548                 try:
--> 549                     update(*args, **kwargs)
    550                 except RuntimeError as err:

/usr/local/lib/python3.11/dist-packages/torchmetrics/classification/confusion_matrix.py in update(self, preds, target)
    285         confmat = _multiclass_confusion_matrix_update(preds, target, self.num_classes)
--> 286         self.confmat += confmat
    287 

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2157659435.py in <cell line: 0>()
     44     all_preds = torch.cat(all_preds)
     45     all_true = torch.cat(all_true)
---> 46     qwk = kappa_metric(all_preds, all_true).item()
     47     print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val QWK: {qwk:.4f}")
     48     fine_tune_model.train()

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

/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in forward(self, *args, **kwargs)
    313             self._forward_cache = self._forward_full_state_update(*args, **kwargs)
    314         else:
--> 315             self._forward_cache = self._forward_reduce_state_update(*args, **kwargs)
    316 
    317         return self._forward_cache

/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in _forward_reduce_state_update(self, *args, **kwargs)
    382 
    383         # calculate batch state and compute batch value
--> 384         self.update(*args, **kwargs)
    385         batch_val = self.compute()
    386 

/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in wrapped_func(*args, **kwargs)
    550                 except RuntimeError as err:
    551                     if "Expected all tensors to be on" in str(err):
--> 552                         raise RuntimeError(
    553                             "Encountered different devices in metric calculation (see stacktrace for details)."
    554                             " This could be due to the metric class not being on the same device as input."

RuntimeError: Encountered different devices in metric calculation (see stacktrace for details). This could be due to the metric class not being on the same device as input. Instead of `metric=MulticlassCohenKappa(...)` try to do `metric=MulticlassCohenKappa(...).to(device)` where device corresponds to the device of the input.

## === cell 6
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b5": 0.952,
    "inception_resnet_v2": 0.880,
    "inception_v4": 0.902,
    "seresnext101_32x4d": 0.9697,
}
filtered_scores = {k: v for k, v in validation_scores.items() if k in loaded_keys}
total_score = sum(filtered_scores.values())
weights = {k: v / total_score for k, v in filtered_scores.items()}




## === cell 7
test_dataset = BlindnessDataset(
    test_csv, test_root, transform=test_transform, test=True
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

all_outputs = []
with torch.no_grad():
    for batch_imgs in tqdm(test_loader, desc="Inference"):
        batch_imgs = batch_imgs.to(device)
        batch_preds = []
        for key, model in zip(loaded_keys, models_list):
            w = weights[key]
            logits = model(batch_imgs)
            probs = torch.softmax(logits, dim=1) * w
            batch_preds.append(probs.unsqueeze(0))
        stacked = torch.cat(batch_preds)  # shape (n_models, batch, 5)
        weighted_sum = torch.sum(stacked, dim=0)  # shape (batch, 5)
        all_outputs.append(weighted_sum.cpu().numpy())
all_outputs = np.concatenate(all_outputs, axis=0)
final_predictions = np.argmax(all_outputs, axis=1)




## === cell 8
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv)["id_code"], "diagnosis": final_predictions}
)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
