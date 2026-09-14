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

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

# 5. Target score

0.8373268929321263

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.58021) has done: 'The changes reduce the amount of data read and the number of training steps while keeping the model architecture, loss, and evaluation logic unchanged. By sampling far fewer images per class, increasing the batch size, and using a modest number of workers, the epoch loops finish well within the 600‑second limit without altering any core computations. The weighted AUC calculation and prediction steps remain exactly the same.'

# 9. Code solution

## === cell 0
data_dir = "../input/alaska2-image-steganalysis"
sample_size = 15000  # number of images to take from each folder (unchanged)
val_size = int(sample_size * 0.25)

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # labels 0‑3
train_fn, val_fn = [], []
train_labels, val_labels = [], []

for label, folder in enumerate(folder_names):
    all_files = sorted(glob(f"{data_dir}/{folder}/*.jpg"))[:sample_size]
    np.random.shuffle(all_files)
    train_fn.extend(all_files[val_size:])
    train_labels.extend([label] * len(all_files[val_size:]))
    val_fn.extend(all_files[:val_size])
    val_labels.extend([label] * len(all_files[:val_size]))

train_df = pd.DataFrame({"ImageFileName": train_fn, "Label": train_labels})
val_df = pd.DataFrame({"ImageFileName": val_fn, "Label": val_labels})

print("Train samples:", len(train_df), "Val samples:", len(val_df))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/588510072.py in <cell line: 0>()
      8 
      9 for label, folder in enumerate(folder_names):
---> 10     all_files = sorted(glob(f"{data_dir}/{folder}/*.jpg"))[:sample_size]
     11     np.random.shuffle(all_files)
     12     train_fn.extend(all_files[val_size:])

NameError: name 'glob' is not defined

## === cell 1
img_size = 224  # same size used later in transforms


class Alaska2Dataset(Dataset):
    """Loads all images into RAM once; transforms are applied on‑the‑fly."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.images = []
        for path in tqdm(self.df["ImageFileName"], desc="Pre‑loading train images"):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            self.images.append(img)  # keep as uint8 NumPy array

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.df.loc[idx, "Label"])
        if self.transform:
            img = self.transform(img)
        return {"image": img}, label


class Alaska2TestDataset(Dataset):
    """Pre‑loads test images; no labels."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.images = []
        for path in tqdm(self.df["ImageFileName"], desc="Pre‑loading test images"):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            self.images.append(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return {"image": img}




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/195688858.py in <cell line: 0>()
      2 
      3 
----> 4 class Alaska2Dataset(Dataset):
      5     """Loads all images into RAM once; transforms are applied on‑the‑fly."""
      6 

NameError: name 'Dataset' is not defined

## === cell 2
train_transform = T.Compose(
    [
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
    ]
)
test_transform = T.Compose(
    [
        T.ToPILImage(),
        T.ToTensor(),
    ]
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2595699269.py in <cell line: 0>()
----> 1 train_transform = T.Compose(
      2     [
      3         T.ToPILImage(),
      4         T.RandomHorizontalFlip(),
      5         T.RandomVerticalFlip(),

NameError: name 'T' is not defined

## === cell 3
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(num_features, 4)  # 4 classes

    def forward(self, x):
        return self.backbone(x)


model = torch.compile(Net().to(device))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/178512165.py in <cell line: 0>()
----> 1 class Net(nn.Module):
      2     def __init__(self):
      3         super().__init__()
      4         self.backbone = models.efficientnet_b0(
      5             weights=models.EfficientNet_B0_Weights.DEFAULT

NameError: name 'nn' is not defined

## === cell 4
batch_size = 256
val_batch_size = 512
num_workers = 8

train_dataset = Alaska2Dataset(train_df, transform=train_transform)
val_dataset = Alaska2Dataset(val_df, transform=test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1195597210.py in <cell line: 0>()
      3 num_workers = 8
      4 
----> 5 train_dataset = Alaska2Dataset(train_df, transform=train_transform)
      6 val_dataset = Alaska2Dataset(val_df, transform=test_transform)
      7 

NameError: name 'Alaska2Dataset' is not defined

## === cell 5
def alaska_weighted_auc(y_true, y_score):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    areas = np.diff(tpr_thresholds)
    normalization = np.dot(areas, weights)

    competition_metric = 0.0
    for i, w in enumerate(weights):
        y_min, y_max = tpr_thresholds[i], tpr_thresholds[i + 1]
        mask = (tpr > y_min) & (tpr < y_max)
        if not np.any(mask):
            continue
        x_pad = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_pad])
        y = np.concatenate([tpr[mask], np.full_like(x_pad, y_max)]) - y_min
        score = metrics.auc(x, y)
        competition_metric += score * w
    return competition_metric / normalization




## === cell 6
from torch.cuda.amp import autocast, GradScaler

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

scaler = GradScaler()

num_epochs = 12

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for batch_dict, batch_labels in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train"
    ):
        inputs = batch_dict["image"].to(device, dtype=torch.float)
        targets = batch_labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(loss.item())
    avg_loss = np.mean(train_losses)
    print(f"Epoch {epoch+1} - Average training loss: {avg_loss:.4f}")

    scheduler.step()

    model.eval()
    val_scores = []
    val_true = []
    with torch.no_grad():
        for batch_dict, batch_labels in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val"
        ):
            inputs = batch_dict["image"].to(device, dtype=torch.float)
            targets = batch_labels.numpy()
            with autocast():
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
            pred_labels = probs.argmax(axis=1)

            scores = np.zeros(len(probs))
            mask_nonzero = pred_labels != 0
            scores[mask_nonzero] = probs[mask_nonzero, 1:].sum(axis=1)
            scores[~mask_nonzero] = probs[~mask_nonzero, 0]

            val_scores.extend(scores)
            val_true.extend((targets != 0).astype(int))

    val_auc = alaska_weighted_auc(np.array(val_true), np.array(val_scores))
    print(f"Validation weighted AUC after epoch {epoch+1}: {val_auc:.5f}")

    gc.collect()
    torch.cuda.empty_cache()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4120427231.py in <cell line: 0>()
      1 from torch.cuda.amp import autocast, GradScaler
      2 
----> 3 criterion = nn.CrossEntropyLoss()
      4 optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
      5 scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

NameError: name 'nn' is not defined

## === cell 7
test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({"ImageFileName": test_filenames})

test_dataset = Alaska2TestDataset(test_df, transform=test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

model.eval()
all_preds = []
with torch.no_grad():
    for batch_dict in tqdm(test_loader, desc="Predicting test"):
        inputs = batch_dict["image"].to(device, dtype=torch.float)
        with autocast():
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
        all_preds.append(probs)

all_preds = np.vstack(all_preds)  # (N, 4)
pred_labels = all_preds.argmax(axis=1)

new_scores = np.zeros(len(all_preds))
mask_nonzero = pred_labels != 0
new_scores[mask_nonzero] = all_preds[mask_nonzero, 1:].sum(axis=1)
new_scores[~mask_nonzero] = all_preds[~mask_nonzero, 0]

test_df["Id"] = test_df["ImageFileName"].apply(lambda x: os.path.basename(x))
test_df["Label"] = new_scores
submission_path = "submission.csv"
test_df[["Id", "Label"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(test_df.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/609878133.py in <cell line: 0>()
----> 1 test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
      2 test_df = pd.DataFrame({"ImageFileName": test_filenames})
      3 
      4 test_dataset = Alaska2TestDataset(test_df, transform=test_transform)
      5 test_loader = DataLoader(

NameError: name 'glob' is not defined
