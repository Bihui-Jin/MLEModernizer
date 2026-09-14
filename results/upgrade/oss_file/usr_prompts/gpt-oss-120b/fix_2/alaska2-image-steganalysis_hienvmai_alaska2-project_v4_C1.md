# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.13

# 3. Installed packages

albumentations==2.0.8
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Code solution

## === cell 0
I will remove the problematic Albumentations import and replace it with a lightweight custom transform that resizes, normalizes, and converts images to tensors. I’ll also ensure the list of image paths is a plain Python list before creating the DataFrame to avoid the pandas TypeError, and I’ll apply a sigmoid to the model outputs when creating the submission so the scores are in probability space. These fixes will let the script run end‑to‑end and produce a valid `submission.csv` while keeping the original model and training logic intact.

```


## === cell 1
!pip install --upgrade pip
!pip install -qU timm



## === cell 2
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from tqdm.notebook import tqdm
import timm
import warnings
warnings.filterwarnings('ignore')



## === cell 3
class CFG:
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_path = '/kaggle/input/alaska2-image-steganalysis/'
    subset_size = 75000
    image_size = 512
    
    model_name = 'tf_efficientnet_b2_ns' 
    num_classes = 1
    
    n_folds = 5
    fold_to_train = 0
    epochs = 1
    train_batch_size = 16
    valid_batch_size = 32
    weight_decay = 1e-6
    
    lr = 1e-4
    T_0 = 5
    eta_min = 1e-6
    
    checkpoint_save_path = '/kaggle/working/latest_checkpoint.pth'
    best_model_save_path = f'/kaggle/working/best_model_fold_{fold_to_train}.pth'
    
    checkpoint_load_path = None

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(CFG.seed)



## === cell 4
def alaska_weighted_auc(y_true, y_pred):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    if len(fpr) < 2:
        return 0.5
    areas = np.array([0.0] * len(weights))
    for i, lower in enumerate(tpr_thresholds[:-1]):
        upper = tpr_thresholds[i+1]
        mask = (tpr >= lower) & (tpr < upper)
        if np.any(mask):
            idx = np.where(mask)[0]
            start_idx, end_idx = idx[0], idx[-1]
            tpr_slice = np.concatenate([[lower], tpr[start_idx:end_idx+1], [upper]])
            fpr_slice = np.concatenate([[np.interp(lower, tpr, fpr)], fpr[start_idx:end_idx+1], [np.interp(upper, tpr, fpr)]])
            tpr_slice, uniq = np.unique(tpr_slice, return_index=True)
            fpr_slice = fpr_slice[uniq]
            areas[i] = np.trapz(fpr_slice, tpr_slice)
    return np.sum(areas * weights) / np.sum(weights)



## === cell 5
image_folders = ['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']
subset_files = []

print(f"Creating a subset of {CFG.subset_size} images from each folder...")
for folder in image_folders:
    folder_path = os.path.join(CFG.data_path, folder)
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    random.shuffle(all_files)
    subset_files.extend(all_files[:CFG.subset_size])
    print(f"  - Took {len(all_files[:CFG.subset_size])} images from {folder}")

subset_files = list(subset_files)   # ensure plain Python list
df = pd.DataFrame({'image_path': subset_files})
df['label'] = df['image_path'].apply(lambda x: 0 if 'Cover' in x else 1)
df['image_id'] = df['image_path'].apply(os.path.basename)

skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
    df.loc[val_idx, 'fold'] = fold

print("\nSubset dataset distribution:")
print(df['label'].value_counts())
print("\nFold distribution:")
print(df.groupby('fold')['label'].value_counts())



## === cell 6
def _basic_transform(image, img_size):
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)

def get_transforms(mode='train'):
    def transform_fn(image):
        return _basic_transform(image, CFG.image_size)
    return transform_fn

class AlaskaDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.image_paths = df['image_path'].values
        self.labels = df['label'].values
        self.transforms = transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image)
        return image, label



## === cell 7
def train_fn(loader, model, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    return running_loss / len(loader)

def eval_fn(loader, model, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    val_loss = running_loss / len(loader)
    score = alaska_weighted_auc(all_labels, all_preds)
    return val_loss, score



## === cell 8
import torch
torch.cuda.empty_cache()
import gc
gc.collect()



## === cell 9
def run_training(fold):
    print(f"========== Starting Training for Fold {fold} ==========")
    
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)
    
    train_dataset = AlaskaDataset(train_df, transforms=get_transforms('train'))
    valid_dataset = AlaskaDataset(valid_df, transforms=get_transforms('valid'))
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    
    model = timm.create_model(CFG.model_name, pretrained=True, num_classes=CFG.num_classes).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG.T_0 * len(train_loader), eta_min=CFG.eta_min)
    criterion = nn.BCEWithLogitsLoss()
    
    best_score = 0.0
    start_epoch = 0
    
    if CFG.checkpoint_load_path and os.path.exists(CFG.checkpoint_load_path):
        print(f"Resuming from {CFG.checkpoint_load_path}")
        checkpoint = torch.load(CFG.checkpoint_load_path, map_location=CFG.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint.get('best_score', 0.0)
        print(f"Resumed at epoch {start_epoch}, best score {best_score:.4f}")
    else:
        print("No checkpoint found, training from scratch.")
    
    for epoch in range(start_epoch, CFG.epochs):
        print(f"\n--- Epoch {epoch+1}/{CFG.epochs} ---")
        train_loss = train_fn(train_loader, model, criterion, optimizer, scheduler, CFG.device)
        val_loss, val_score = eval_fn(valid_loader, model, criterion, CFG.device)
        print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}, Valid Weighted AUC: {val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), CFG.best_model_save_path)
            print(f"New best model saved with score {best_score:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_score': best_score,
        }, CFG.checkpoint_save_path)
    
    print(f"\n========== Finished Fold {fold}. Best Score: {best_score:.4f} ==========")

run_training(CFG.fold_to_train)



## === cell 10
def create_submission():
    print("\nStarting inference on the test set...")
    test_folder = os.path.join(CFG.data_path, 'Test')
    test_image_ids = [f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')]
    
    test_df = pd.DataFrame({'image_id': test_image_ids})
    test_df['image_path'] = test_df['image_id'].apply(lambda x: os.path.join(test_folder, x))
    test_df['label'] = 0  # dummy
    
    test_dataset = AlaskaDataset(test_df, transforms=get_transforms('test'))
    test_loader = DataLoader(test_dataset, batch_size=CFG.valid_batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    model = timm.create_model(CFG.model_name, pretrained=False, num_classes=CFG.num_classes)
    model.load_state_dict(torch.load(CFG.best_model_save_path, map_location=CFG.device))
    model.to(CFG.device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Predicting")
        for images, _ in pbar:
            images = images.to(CFG.device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions.extend(probs)
    
    submission_df = pd.DataFrame({'Id': test_image_ids, 'Label': predictions})
    submission_df = submission_df.sort_values('Id')
    submission_df.to_csv('submission.csv', index=False)
    print("\nSubmission file 'submission.csv' created successfully!")
    print(submission_df.head())

create_submission()
```
