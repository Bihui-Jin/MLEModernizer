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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.818485687903971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = "../input/plant-models-v4"  # fallback if not present
else:
    DATA_PATH = "./data"
    MDLS_PATH = "./models_v4"

TEST = True  # generate predictions for test set
VER = "v4"
TH = 0.5  # threshold for label selection
TTAS = [0, 1, 2]  # test‑time augmentations (indices)
FOLDS = [3, 4]  # folds to ensemble
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2496739688.py in <cell line: 0>()
----> 1 if KAGGLE:
      2     DATA_PATH = "../input/plant-pathology-2021-fgvc8"
      3     MDLS_PATH = "../input/plant-models-v4"  # fallback if not present
      4 else:
      5     DATA_PATH = "./data"

NameError: name 'KAGGLE' is not defined

## === cell 1
params_path = os.path.join(MDLS_PATH, "params.json")
if os.path.isfile(params_path):
    with open(params_path) as f:
        params = json.load(f)
else:
    params = {
        "img_size": 224,
        "batch_size": 32,
        "dropout": 0.2,
        "backbone": "efficientnet_b0",
        "workers": 2,
        "labels_": None,  # will be filled below
        "labels": None,
    }

train_csv = os.path.join(DATA_PATH, "train.csv")
train_df = pd.read_csv(train_csv)
all_labels = set()
for lbls in train_df["labels"]:
    all_labels.update(lbls.split())
all_labels = sorted(all_labels)
label_to_idx = {lbl: idx for idx, lbl in enumerate(all_labels)}
idx_to_label = {str(idx): lbl for lbl, idx in label_to_idx.items()}

params["labels_"] = label_to_idx
params["labels"] = idx_to_label

LABELS_ = params["labels_"]  # mapping label -> idx
LABELS = params["labels"]  # mapping idx (str) -> label
WORKERS = 2 if KAGGLE else params.get("workers", 2)
print("Parameters loaded / defaulted.")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2024331748.py in <cell line: 0>()
----> 1 params_path = os.path.join(MDLS_PATH, "params.json")
      2 if os.path.isfile(params_path):
      3     with open(params_path) as f:
      4         params = json.load(f)
      5 else:

NameError: name 'os' is not defined

## === cell 2
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row["image"]
        img_path = os.path.join(IMGS_PATH, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if self.labels is not None:
            img = img.transpose(2, 0, 1)
            label_vec = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in row["labels"].split():
                label_vec[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label_vec)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2232936682.py in <cell line: 0>()
     10 
     11 
---> 12 class PlantDataset(data.Dataset):
     13     def __init__(self, df, size, labels, transform=None, tta=0):
     14         self.df = df.reset_index(drop=True)

NameError: name 'data' is not defined

## === cell 3
try:
    from efficientnet_pytorch import model as enet

    efficientnet_cls = lambda name: enet.EfficientNet.from_name(name)
except Exception:

    def efficientnet_cls(name):
        backbone_map = {
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
            "efficientnet_b3": models.efficientnet_b3,
            "efficientnet_b4": models.efficientnet_b4,
        }
        if name not in backbone_map:
            raise ValueError(f"Unsupported backbone: {name}")
        return backbone_map[name](pretrained=True)


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        self.enet = efficientnet_cls(params["backbone"])
        if hasattr(self.enet, "classifier"):
            nc = self.enet.classifier[1].in_features
            self.enet.classifier = nn.Identity()
        else:
            nc = self.enet._fc.in_features
            self.enet._fc = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, nc // 4),
            nn.Dropout(params["dropout"]),
            nn.Linear(nc // 4, out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


models = []
for n_fold in FOLDS:
    try:
        model = EffNet(params, out_dim=len(LABELS_))
        path = os.path.join(MDLS_PATH, f"model_best_{n_fold}.pth")
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(DEVICE).eval()
        models.append(model)
        print(f"Loaded model from {path}")
    except Exception as e:
        print(f"Could not load model {n_fold}: {e}")
if not models:
    print('No pretrained models loaded – predictions will default to "healthy".')
gc.collect()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/806638982.py in <cell line: 0>()
     18 
     19 
---> 20 class EffNet(nn.Module):
     21     def __init__(self, params, out_dim):
     22         super(EffNet, self).__init__()

NameError: name 'nn' is not defined

## === cell 4
test_files = os.listdir(IMGS_PATH)
df_sub = pd.DataFrame(test_files, columns=["image"])
df_sub["labels"] = "healthy"  # placeholder, will be overwritten if we have predictions
print("Submission template created with", len(df_sub), "records.")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2972702454.py in <cell line: 0>()
----> 1 test_files = os.listdir(IMGS_PATH)
      2 df_sub = pd.DataFrame(test_files, columns=["image"])
      3 df_sub["labels"] = "healthy"  # placeholder, will be overwritten if we have predictions
      4 print("Submission template created with", len(df_sub), "records.")
      5 

NameError: name 'os' is not defined

## === cell 5
datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
    )
    datasets.append(dataset)
    loader = data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=True,
    )
    loaders.append(loader)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4031995763.py in <cell line: 0>()
      1 datasets, loaders = [], []
----> 2 for tta in TTAS:
      3     dataset = PlantDataset(
      4         df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
      5     )

NameError: name 'TTAS' is not defined

## === cell 6
def get_labels(row, labels_map, th):
    try:
        idxs = [i for i, x in enumerate(row) if x > th]
        names = [labels_map[str(i)] for i in idxs]
        return "healthy" if ("healthy" in names or len(names) == 0) else " ".join(names)
    except Exception as e:
        print("Error in get_labels:", e, row)
        return "healthy"


if models:
    all_logits = []
    with torch.no_grad():
        for model in models:
            model_logits = []
            for loader in loaders:
                tta_logits = []
                for batch in loader:
                    batch = batch.to(DEVICE)
                    preds = torch.sigmoid(model(batch)).cpu().numpy()
                    tta_logits.append(preds)
                tta_mean = np.mean(np.vstack(tta_logits), axis=0)
                model_logits.append(tta_mean)
            model_logits = np.mean(np.stack(model_logits, axis=0), axis=0)
            all_logits.append(model_logits)
    logits = np.mean(
        np.stack(all_logits, axis=0), axis=0
    )  # shape (num_images, num_labels)
else:
    logits = np.zeros((len(df_sub), len(LABELS_)), dtype=np.float32)

df_sub["labels"] = [get_labels(row, LABELS, TH) for row in logits]




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2504133034.py in <cell line: 0>()
      9 
     10 
---> 11 if models:
     12     all_logits = []
     13     with torch.no_grad():

NameError: name 'models' is not defined

## === cell 7
print("Label distribution in submission:")
print(df_sub["labels"].value_counts())
out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)
print(f"Submission file written to {out_path}")
elapsed = time.time() - start_time
print(f"Total elapsed time: {int(elapsed//60)}m {int(elapsed%60)}s")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4096435584.py in <cell line: 0>()
      1 print("Label distribution in submission:")
----> 2 print(df_sub["labels"].value_counts())
      3 out_path = "submission.csv"
      4 df_sub.to_csv(out_path, index=False)
      5 print(f"Submission file written to {out_path}")

NameError: name 'df_sub' is not defined
