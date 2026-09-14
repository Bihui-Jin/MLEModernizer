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

0.9527649769585254

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd

BASE_DIR = '/kaggle/input/paddy-disease-classification'
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')

df = pd.read_csv(TRAIN_CSV)

print(f"data size: {len(df)}")

print(df.head())

print(df.info())

## === cell 1
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

## === cell 2
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

## === cell 3
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

## === cell 4
!pip install -q timm

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

## === cell 5
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = '/kaggle/input/paddy-disease-classification'
train_df = pd.read_csv(f'{INPUT_DIR}/train.csv')
sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')

# 画像パスの生成
train_df['path'] = train_df.apply(lambda x: f"{INPUT_DIR}/train_images/{x['label']}/{x['image_id']}", axis=1)
sample_sub['path'] = sample_sub['image_id'].apply(lambda x: f"{INPUT_DIR}/test_images/{x}")

# ラベルの数値化
labels = sorted(train_df['label'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
train_df['label_id'] = train_df['label'].map(label2id)

# データを分割
train_data, valid_data = train_test_split(
    train_df, 
    test_size=0.2, 
    stratify=train_df['label'], 
    random_state=42
)

print(f"Train data: {len(train_data)}, Valid data: {len(valid_data)}")
"""

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

varieties = sorted(train_df['variety'].unique())
variety2id = {v: i for i, v in enumerate(varieties)}

train_df['variety_id'] = train_df['variety'].map(variety2id)

num_varieties = len(varieties)
print(f"Number of varieties: {num_varieties}")

train_data, valid_data = train_test_split(
    train_df, 
    test_size=0.2, 
    stratify=train_df['label'], 
    random_state=42
)

print(f"Train data: {len(train_data)}, Valid data: {len(valid_data)}")

## === cell 7
"""
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

# 画像変換の定義（サイズ224、データ拡張あり）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("Dataset and transform are ready")
"""

## === cell 8
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
            target = torch.tensor(row['label_id'], dtype=torch.long)
            
            variety = torch.tensor(row['variety_id'], dtype=torch.long)
            
            age = torch.tensor(row['age'], dtype=torch.float)
            
            return image, target, variety, age

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("Dataset and transform are ready")

## === cell 9
from torch.utils.data import DataLoader

train_ds = PaddyDataset(train_data, transform=train_transform)
valid_ds = PaddyDataset(valid_data, transform=val_transform)
test_ds = PaddyDataset(sample_sub, transform=val_transform, is_test=True)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

print("DataLoaders are ready.")

## === cell 10
"""
import timm
import torch.nn as nn
import torch.optim as optim

# モデルのダウンロード
model = timm.create_model('resnet18', pretrained=True, num_classes=len(labels))
model = model.to(device)

# 学習設定
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Model loaded.")
"""

## === cell 11
import timm
import torch.nn as nn
import torch.optim as optim

class PaddyMultiTaskModel(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        n_features = self.backbone.num_features
        
        self.fc_disease = nn.Linear(n_features, 10)  # 病気用
        self.fc_variety = nn.Linear(n_features, 10)  # 品種用
        self.fc_age     = nn.Linear(n_features, 1)   # 年齢用

    def forward(self, x):
        features = self.backbone(x)      # 共通の特徴抽出
        
        out_d = self.fc_disease(features) # 分岐1
        out_v = self.fc_variety(features) # 分岐2
        out_a = self.fc_age(features)     # 分岐3
        
        return out_d, out_v, out_a

## === cell 12
model = PaddyMultiTaskModel(model_name='resnet18')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion_disease = nn.CrossEntropyLoss()
criterion_variety = nn.CrossEntropyLoss()
criterion_age = nn.MSELoss() # 回帰なのでMSEなど

## === cell 13
"""
from tqdm.notebook import tqdm

epochs = 2

print("Start Training...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    
    # tqdmでプログレスバーを表示しながらループ
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    print(f"Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f}")
"""

## === cell 14
"""
from tqdm.notebook import tqdm
import torch.nn as nn

epochs = 2
WEIGHT_V = 0.5
WEIGHT_A = 0

print("Start Training...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    
    # 変更点1: DataLoaderから image, target(病気) に加えて variety, age も受け取る
    # ※ Datasetクラスの __getitem__ がこれらを返すように修正されている前提です
    for images, targets, varieties, ages in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        
        # データをGPU/CPUへ転送
        images = images.to(device)
        targets = targets.to(device)
        varieties = varieties.to(device)
        ages = ages.to(device).float()  # MSELoss計算用にfloat型に変換
        
        optimizer.zero_grad()
        
        # 変更点2: モデルから3つの出力を受け取る
        out_disease, out_variety, out_age = model(images)
        
        # 変更点3: 個別の損失を計算
        loss_d = criterion_disease(out_disease, targets)
        loss_v = criterion_variety(out_variety, varieties)
        # ageは [Batch, 1] になっているので squeeze() して [Batch] に合わせる場合があります
        loss_a = criterion_age(out_age.squeeze(), ages)
        
        # 変更点4: 損失を重み付けして合算 (重みは調整可能)
        loss = loss_d + (WEIGHT_V * loss_v) + (WEIGHT_A * loss_a)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # ログ表示 (必要であれば個別のlossも表示できます)
    print(f"Epoch {epoch+1} Total Train Loss: {train_loss / len(train_loader):.4f}")
"""

## === cell 15
"""
import torch
from tqdm.notebook import tqdm

model.eval()
correct_counts = 0
total_counts = 0

print("Calculating Validation Accuracy...")
with torch.no_grad():
    for images, targets, varieties, ages in tqdm(valid_loader):
        images, targets = images.to(device), targets.to(device)
        
        out_disease, out_variety, out_age = model(images)
        _, predicted = torch.max(out_disease, 1)
        
        total_counts += targets.size(0)
        correct_counts += (predicted == targets).sum().item()

val_acc = correct_counts / total_counts
print(f"\nValidation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
"""

## === cell 16
import torch
import torch.nn as nn
from tqdm.notebook import tqdm

model = PaddyMultiTaskModel(model_name='resnet18')

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion_disease = nn.CrossEntropyLoss()
criterion_variety = nn.CrossEntropyLoss()
criterion_age = nn.MSELoss() # 回帰なのでMSEなど

epochs = 5  # 様子見のため少し増やしても良いかも
WEIGHT_V = 0.1
WEIGHT_A = 0.0 # 年齢も少し学習させた方が効く場合が多いので0.1に設定(任意)

best_acc = 0.0
save_path = "best_model.pth"

print(f"Start Training for {epochs} epochs...")

for epoch in range(epochs):
    model.train() # 学習モード
    train_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    
    for images, targets, varieties, ages in pbar:
        images = images.to(device)
        targets = targets.to(device)
        varieties = varieties.to(device)
        ages = ages.to(device).float()
        
        optimizer.zero_grad()
        
        out_disease, out_variety, out_age = model(images)
        
        loss_d = criterion_disease(out_disease, targets)
        loss_v = criterion_variety(out_variety, varieties)
        loss_a = criterion_age(out_age.squeeze(), ages)
        
        loss = loss_d + (WEIGHT_V * loss_v) + (WEIGHT_A * loss_a)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_train_loss = train_loss / len(train_loader)

    model.eval() # 推論モード (Dropoutなどを無効化)
    correct_counts = 0
    total_counts = 0
    
    with torch.no_grad():
        for images, targets, varieties, ages in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            
            out_disease, _, _ = model(images)
            _, predicted = torch.max(out_disease, 1)
            
            total_counts += targets.size(0)
            correct_counts += (predicted == targets).sum().item()

    val_acc = correct_counts / total_counts
    
    print(f"Epoch {epoch+1} Result:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Valid Acc : {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    if val_acc > best_acc:
        print(f"  [Create Best Model] Accuracy improved from {best_acc:.4f} to {val_acc:.4f}. Saving model...")
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
    else:
        print(f"  Accuracy did not improve from {best_acc:.4f}.")
    
    print("-" * 30)

print("Training Finished.")
print(f"Best Validation Accuracy: {best_acc:.4f}")

## === cell 17
import torch
from tqdm.notebook import tqdm

model.eval()
preds = []

print("Predicting...")
with torch.no_grad():
    for images in tqdm(test_loader):
        images = images.to(device)
        outputs, _, _ = model(images)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())

sample_sub['label'] = [id2label[i] for i in preds]
submission = sample_sub[['image_id', 'label']]
submission.to_csv('submission.csv', index=False)

print("Saved submission.csv!")
print(submission.head())

## === cell 18
!ls
