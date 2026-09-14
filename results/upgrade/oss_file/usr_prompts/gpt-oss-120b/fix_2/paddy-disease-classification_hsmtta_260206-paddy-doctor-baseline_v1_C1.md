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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.14

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.8986175115207373

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will add a validation step after each training epoch, keep the model checkpoint with the highest validation accuracy, and extend training to 5 epochs so the model can learn enough to approach the target accuracy while preserving the original architecture and training logic.

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/2008062905.py", line 1
    I will add a validation step after each training epoch, keep the model checkpoint with the highest validation accuracy, and extend training to 5 epochs so the model can learn enough to approach the target accuracy while preserving the original architecture and training logic.
      ^
SyntaxError: invalid syntax


## === cell 1
import os
import pandas as pd

BASE_DIR = '/kaggle/input/paddy-disease-classification'
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

df = pd.read_csv(TRAIN_CSV)

print(f"data size: {len(df)}")

print(df.head())

print(df.info())



## === cell 2
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.countplot(y='label', data=df, order=df['label'].value_counts().index)
plt.title('Distribution of Diseases (Labels)')
plt.xlabel('Count')
plt.ylabel('Disease Name')

plt.subplot(1, 2, 2)
sns.countplot(y='variety', data=df, order=df['variety'].value_counts().index)
plt.title('Distribution of Varieties')
plt.xlabel('Count')
plt.ylabel('Variety Name')

plt.tight_layout()
plt.show()



## === cell 3
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['age'], kde=True, bins=20)
plt.title('Overall Age Distribution of Paddy Crops')
plt.xlabel('Age (days)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



## === cell 4
from PIL import Image

def get_image_path(row):
    return os.path.join(BASE_DIR, 'train_images', row['label'], row['image_id'])

df['path'] = df.apply(get_image_path, axis=1)

unique_labels = df['label'].unique()
plt.figure(figsize=(15, 12))

for i, label in enumerate(unique_labels):
    sample_row = df[df['label'] == label].sample(1).iloc[0]
    
    img = Image.open(sample_row['path'])
    
    plt.subplot(3, 4, i + 1) # 3行4列で表示
    plt.imshow(img)
    plt.title(f"{label}\n(Variety: {sample_row['variety']})")
    plt.axis('off')

plt.tight_layout()
plt.show()



## === cell 5
!pip install -q timm

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



## === cell 6
import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = '/kaggle/input/paddy-disease-classification'
train_df = pd.read_csv(f'{INPUT_DIR}/train.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')

train_df['path'] = train_df.apply(lambda x: f"{INPUT_DIR}/train_images/{x['label']}/{x['image_id']}", axis=1)
sample_sub['path'] = sample_sub['image_id'].apply(lambda x: f"{INPUT_DIR}/test_images/{x}")

labels = sorted(train_df['label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
train_df['label_id'] = train_df['label'].map(label2id)

train_data, valid_data = train_test_split(
    train_df, 
    test_size=0.2, 
    stratify=train_df['label'], 
    random_state=42
)

print(f"Train data: {len(train_data)}, Valid data: {len(valid_data)}")



## === cell 7
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PaddyDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return image
        else:
            return image, torch.tensor(row['label_id'], dtype=torch.long)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("Dataset and transform are ready")



## === cell 8
from torch.utils.data import DataLoader

train_ds = PaddyDataset(train_data, transform=train_transform)
valid_ds = PaddyDataset(valid_data, transform=val_transform)
test_ds = PaddyDataset(sample_sub, transform=val_transform, is_test=True)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

print("DataLoaders are ready.")



## === cell 9
import timm
import torch.nn as nn
import torch.optim as optim

model = timm.create_model('resnet34', pretrained=True, num_classes=len(labels))
model = model.to(device) # Block 1で定義した device を使用

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Model 'resnet34' loaded.")



## === cell 10
from tqdm.notebook import tqdm

epochs = 5  # extended from 2 to give the model more learning capacity

best_acc = 0.0
best_state = None

print("Start Training...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [valid]"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1} Validation Accuracy: {val_acc:.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_state = model.state_dict()
        print(f"  --> New best model saved (accuracy {best_acc:.4f})")

if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Best model (val acc {best_acc:.4f}) loaded for final prediction.")



## === cell 11
import torch
from tqdm.notebook import tqdm

model.eval()
preds = []

print("Predicting on test set...")
with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())

sample_sub['label'] = [id2label[i] for i in preds]
submission = sample_sub[['image_id', 'label']]
submission.to_csv('submission.csv', index=False)

print("Saved submission.csv!")
print(submission.head())
```

## --- ERROR in cell 11, traceback:
  File "/tmp/ipykernel_55/2475705621.py", line 21
    ```
    ^
SyntaxError: invalid syntax
