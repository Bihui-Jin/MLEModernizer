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

3.9

# 3. Installed packages

geopandas==0.14.4
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

0.8773043215472952

# 6. Current score

0.66293

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The original script fails because it relies on many custom modules (`utils`, `model`, a specific config file, etc.) that are not available in the current environment. To make the pipeline runnable and generate a valid `submission.csv`, the code is rewritten to avoid those missing dependencies. It simply uses the label distribution from the training set (choosing the most frequent class) as a baseline prediction for every test image. This guarantees a correctly‑formatted submission file with the required number of rows, allowing the notebook to finish without errors.'
- What this solution (achieved 0.63042) has done: 'I replace the dummy “most‑common‑class” baseline with a lightweight nearest‑neighbor classifier that uses ImageNet‑pretrained ResNet‑18 features. By extracting a 512‑dimensional embedding for each training image and then assigning each test image the label of its closest training embedding, we obtain a much stronger predictor while keeping the overall pipeline structure (load CSV, predict, write submission) unchanged. This should raise the validation accuracy toward the target 0.877 score.'
- What this solution (achieved 0.61099) has done: 'I keep the same ResNet‑18 feature extraction but replace the brute‑force nearest‑neighbour lookup with a much more stable class‑centroid classifier using cosine similarity. After encoding all training images I compute a normalized 512‑dim centroid for each of the five disease classes. During inference each test image is also normalized and assigned the label of the nearest centroid (highest dot‑product). This change preserves the original model and data pipeline while typically raising accuracy substantially, moving the score closer to the target.'
- What this solution (achieved 0.67227) has done: 'We switch to a lightweight k‑Nearest‑Neighbour classifier that uses the full set of ResNet‑18 embeddings instead of a single class‑centroid, which typically yields a noticeable accuracy gain when the current gap (>30 % of the target) allows a model change. The model and transforms stay the same, we simply move tensors to GPU when available and replace the centroid‑based similarity with cosine‑similarity + majority vote over the k=5 nearest neighbours. This keeps the original pipeline intact while improving predictions and therefore moving the score nearer to the target.'
- What this solution (achieved 0.68535) has done: 'I improve the k‑Nearest‑Neighbour classifier by increasing the neighbour count slightly and using similarity‑weighted voting instead of a simple majority vote. This keeps the same ResNet‑18 feature extraction and overall pipeline while providing a more informative decision rule, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.69768) has done: 'I keep the existing ResNet‑18 feature extractor but add a tiny linear classifier trained on the extracted training embeddings. This modest supervised step usually lifts accuracy beyond the weighted‑k‑NN baseline while preserving the core pipeline. The changes introduce a short training loop after feature encoding and replace the k‑NN voting with a forward pass through the learned linear layer for test predictions.'
- What this solution (achieved 0.61099) has done: 'I keep the existing pipeline but train the classifier longer and enforce a cosine‑style classifier by normalising its weights after each update. This modest change usually improves accuracy without altering the backbone model or overall logic, moving the validation score closer to the target.'
- What this solution (achieved 0.51682) has done: 'I replace the learned linear classifier with a simple class‑centroid based cosine classifier. After extracting and normalising the ResNet‑18 features for all training images, I compute a normalised mean vector (centroid) for each of the five disease classes. During inference the test image features are compared to these centroids via cosine similarity and the label of the most similar centroid is assigned. This change keeps the original feature‑extraction pipeline intact while usually improving validation accuracy, moving the score closer to the target.'
- What this solution (achieved 0.14051) has done: 'We replace the simple class‑centroid classifier with a lightweight linear layer trained on the normalized ResNet‑18 embeddings. The ResNet‑18 feature extractor stays unchanged; we just add a short supervised training loop (few epochs, Adam optimizer) on the already‑computed normalized train embeddings. At inference we use this trained linear head to predict test labels, which should raise accuracy toward the target while keeping the overall pipeline intact.'
- What this solution (achieved 0.72384) has done: 'The update forces all tensors to stay on CPU (avoiding CUDA initialization errors) and trains the linear head longer so the model gains better accuracy, moving the score closer to the target while preserving the original pipeline.'
- What this solution (achieved 0.61099) has done: 'The update keeps the same ResNet‑18 feature extractor and linear head but trains the linear classifier longer (30 epochs) and adds a cosine‑style weight normalisation after every optimiser step, with a modest learning‑rate drop halfway through. These tweaks improve the discriminative power of the head while preserving the original pipeline, moving the validation accuracy from 0.72384 closer to the target 0.8773.'
- What this solution (achieved 0.61099) has done: 'The changes increase training time (60 epochs) with a second learning‑rate drop, then compute per‑class centroids from the normalized embeddings and blend the linear‑head logits with cosine‑similarity scores to the centroids during inference. This keeps the original ResNet‑18 feature extractor and linear head while providing a modest accuracy gain, moving the validation score closer to the target.'
- What this solution (achieved 0.66293) has done: 'The model now use the available GPU if present and augment the existing linear + centroid prediction with a lightweight k‑Nearest‑Neighbour vote (k = 5) based on cosine similarity of the ResNet‑18 embeddings. This additional signal is inexpensive and expected to raise the validation accuracy, moving the score closer to the target while keeping the original architecture and training loop intact.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = Path("/kaggle/input/cassava-leaf-disease-classification")
TRAIN_CSV = BASE_PATH / "train.csv"
TRAIN_IMAGES_DIR = BASE_PATH / "train_images"
TEST_IMAGES_DIR = BASE_PATH / "test_images"
SUBMISSION_PATH = Path("./submission.csv")



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
assert "image_id" in train_df.columns and "label" in train_df.columns


class ImageLabelDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.paths = [img_dir / img_id for img_id in df["image_id"]]
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()  # output 512‑dim features
model = model.to(device)
model.eval()

train_dataset = ImageLabelDataset(train_df, TRAIN_IMAGES_DIR, transform)
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=False
)

train_features = []
train_labels = []

with torch.no_grad():
    for imgs, labs in tqdm(train_loader, desc="Encoding train set"):
        imgs = imgs.to(device)
        feats = model(imgs)  # [B, 512]
        train_features.append(feats.cpu())
        train_labels.extend(labs)

train_features = torch.cat(train_features, dim=0)  # [N_train, 512]
train_labels = torch.tensor(train_labels, dtype=torch.long)

train_features_norm = F.normalize(train_features, p=2, dim=1)




## === cell 2
linear = torch.nn.Linear(512, 5, bias=False).to(device)

optimizer = torch.optim.Adam(linear.parameters(), lr=0.01)

embed_dataset = TensorDataset(train_features_norm, train_labels)
embed_loader = DataLoader(
    embed_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=False
)

total_epochs = 60
for epoch in range(total_epochs):
    if epoch == 15:
        for g in optimizer.param_groups:
            g["lr"] = 0.005
    if epoch == 45:
        for g in optimizer.param_groups:
            g["lr"] = 0.001

    linear.train()
    epoch_loss = 0.0
    for feats_batch, labs_batch in embed_loader:
        feats_batch = feats_batch.to(device)
        labs_batch = labs_batch.to(device)

        optimizer.zero_grad()
        logits = linear(feats_batch)  # [B, 5]
        loss = F.cross_entropy(logits, labs_batch)  # CE loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            linear.weight.div_(linear.weight.norm(dim=1, keepdim=True))

        epoch_loss += loss.item() * feats_batch.size(0)

    epoch_loss /= len(embed_dataset)
    print(f"Epoch {epoch+1:02d}/{total_epochs} – loss: {epoch_loss:.4f}")

linear.eval()

with torch.no_grad():
    centroids = torch.zeros((5, 512), device=device)
    counts = torch.zeros(5, device=device)
    for feats_batch, labs_batch in embed_loader:
        feats_batch = feats_batch.to(device)  # already normalised
        labs_batch = labs_batch.to(device)
        for c in range(5):
            mask = labs_batch == c
            if mask.any():
                centroids[c] += feats_batch[mask].sum(dim=0)
                counts[c] += mask.sum()
    centroids = centroids / counts.unsqueeze(1)  # [5,512]
    centroids = F.normalize(centroids, p=2, dim=1)  # unit length

train_features_norm = train_features_norm.to(device)
train_labels = train_labels.to(device)




## === cell 3
test_filenames = sorted(
    [p.name for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() == ".jpg"]
)
assert len(test_filenames) > 0, "No test images found."

test_df = pd.DataFrame({"image_id": test_filenames})


class TestDataset(Dataset):
    def __init__(self, filenames, img_dir, transform):
        self.paths = [img_dir / fn for fn in filenames]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.paths[idx].name


test_dataset = TestDataset(test_filenames, TEST_IMAGES_DIR, transform)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=False
)

pred_labels = []

k = 5  # number of neighbours for k‑NN voting

with torch.no_grad():
    for imgs, fnames in tqdm(test_loader, desc="Predicting test set"):
        imgs = imgs.to(device)
        feats = model(imgs)  # [B, 512]
        feats_norm = F.normalize(feats, p=2, dim=1)  # [B, 512]

        logits_linear = linear(feats_norm)  # [B, 5]

        logits_centroid = feats_norm @ centroids.t()  # [B, 5]

        sim_matrix = feats_norm @ train_features_norm.t()  # [B, N_train]
        topk_vals, topk_idx = torch.topk(sim_matrix, k=k, dim=1)  # [B, k]

        neighbor_labels = train_labels[topk_idx]  # [B, k]
        one_hot = F.one_hot(neighbor_labels, num_classes=5).float()  # [B, k, 5]
        weighted = one_hot * topk_vals.unsqueeze(-1)  # weight by similarity
        logits_knn = weighted.sum(dim=1)  # [B, 5]

        logits = (logits_linear + logits_centroid + logits_knn) / 3.0

        preds = logits.argmax(dim=1)  # predicted class index
        pred_labels.extend(preds.cpu().tolist())

submission_df = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
submission_df.to_csv(SUBMISSION_PATH, index=False)




## === cell 4
print("First 5 rows of submission:")
print(submission_df.head())
