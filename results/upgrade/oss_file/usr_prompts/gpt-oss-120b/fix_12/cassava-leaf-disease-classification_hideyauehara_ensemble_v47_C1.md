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
seaborn==0.12.2
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

0.8942278634028408

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The fix adds robust path handling, safe imports for EfficientNet, correct Albumentations argument formats, proper dataframe creation, and safe aggregation of model probabilities. It also gracefully falls back to a simple baseline prediction when no pretrained weights are found, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.05531) has done: 'Implemented fixes to resolve Albumentations API mismatch and added a lightweight training fallback when no pretrained checkpoints are found.  
- Updated all `RandomResizedCrop` calls to use the `size` argument compatible with the installed Albumentations version.  
- Added a `TrainDataset` for loading training images and labels.  
- If pretrained models are missing, a small EfficientNet‑B0 (ImageNet pretrained) is fine‑tuned for a couple of epochs on the provided training data, then used for test‑time augmentation and prediction.  
- Wrapped the trained model to keep the existing `predict_model` interface.  
- Ensured a valid `submission.csv` is always written.'
- What this solution (achieved 0.05531) has done: 'The fix updates the Albumentations calls to use the correct `height`/`width` arguments, adds lightweight wrapper stubs for the custom model classes that were missing, and slightly extends the fine‑tuning epochs (from 2 to 5) to give a better‑trained baseline when no pretrained checkpoints are found. These changes remove the validation errors, ensure a valid `submission.csv` is always written, and modestly improve the accuracy toward the target score while keeping the original modelling approach unchanged.'
- What this solution (achieved 0.05531) has done: 'The fix updates the Albumentations augmentation calls to match the installed version (which requires a `size` argument instead of separate `height`/`width`). This resolves the validation errors that stopped the script, allowing the model (or fallback training) to run and produce a proper `submission.csv`. No other logic is altered, preserving the original modeling approach.'
- What this solution (achieved 0.05531) has done: 'The fixes correct the Albumentations API usage (providing height/width tuples instead of a single `size` argument) and adjust the training augmentation accordingly. This resolves the runtime errors, enables the model to train and predict properly, and the longer training (15 epochs) should raise accuracy toward the target score while keeping the original modeling approach unchanged.'
- What this solution (achieved 0.05531) has done: 'Implemented fixes for Albumentations RandomResizedCrop by using the correct `size` argument (instead of deprecated `height` / `width`). Updated both test‐time augmentations (cell 8) and the training augmentation (cell 13) to match the installed Albumentations v2 API, eliminating the validation errors that prevented model training/inference. These changes ensure the pipeline runs end‑to‑end and produces a valid `submission.csv` while preserving the original modeling logic.'

# 9. Code solution

## === cell 0
pretrained_models = (
    glob.glob("../input/densenet201-04-2019data/*.pth")
    + glob.glob("../input/resnet152-04-2019data/*.pth")
    + glob.glob("../input/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth")
)
print(f"{len(pretrained_models)} pretrained model(s) found.")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3449946667.py in <cell line: 0>()
      1 pretrained_models = (
----> 2     glob.glob("../input/densenet201-04-2019data/*.pth")
      3     + glob.glob("../input/resnet152-04-2019data/*.pth")
      4     + glob.glob("../input/eb7slseed70/efficientnet-b7sl_SEED70.best/*.pth")
      5 )

NameError: name 'glob' is not defined

## === cell 1
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # keep reproducibility for training
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(SEED)
torch.backends.cudnn.deterministic = False



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3696190493.py in <cell line: 0>()
     10 
     11 SEED = 42
---> 12 seed_everything(SEED)
     13 torch.backends.cudnn.deterministic = False
     14 

/tmp/ipykernel_55/3696190493.py in seed_everything(seed)
      1 def seed_everything(seed=42):
----> 2     random.seed(seed)
      3     os.environ["PYTHONHASHSEED"] = str(seed)
      4     np.random.seed(seed)
      5     torch.manual_seed(seed)

NameError: name 'random' is not defined

## === cell 2
try:
    sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
    from efficientnet_pytorch import EfficientNet
except Exception:
    EfficientNet = None  # fallback to torchvision later



## === cell 3
SIZE = 512  # image size
size_tuple = (SIZE, SIZE)  # Albumentations expects (height, width)
num_classes = 5



## === cell 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/492579752.py in <cell line: 0>()
----> 1 device = "cuda" if torch.cuda.is_available() else "cpu"
      2 print(f"Using device: {device}")
      3 

NameError: name 'torch' is not defined

## === cell 5
possible_bases = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "data/cassava-leaf-disease-classification",
    "data",
]
BASE_DIR = next((p for p in possible_bases if pathlib.Path(p).exists()), None)
if BASE_DIR is None:
    raise FileNotFoundError("Could not locate the dataset base directory.")
TEST_PATH = pathlib.Path(BASE_DIR) / "test_images"
test_files = sorted([f.name for f in TEST_PATH.iterdir() if f.is_file()])
print(f"Number of test images: {len(test_files)}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2387990567.py in <cell line: 0>()
      5     "data",
      6 ]
----> 7 BASE_DIR = next((p for p in possible_bases if pathlib.Path(p).exists()), None)
      8 if BASE_DIR is None:
      9     raise FileNotFoundError("Could not locate the dataset base directory.")

/tmp/ipykernel_55/2387990567.py in <genexpr>(.0)
      5     "data",
      6 ]
----> 7 BASE_DIR = next((p for p in possible_bases if pathlib.Path(p).exists()), None)
      8 if BASE_DIR is None:
      9     raise FileNotFoundError("Could not locate the dataset base directory.")

NameError: name 'pathlib' is not defined

## === cell 6
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 0  # placeholder; will be overwritten after prediction



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4090954434.py in <cell line: 0>()
----> 1 df_test = pd.DataFrame(test_files, columns=["image_id"])
      2 df_test["label"] = 0  # placeholder; will be overwritten after prediction
      3 

NameError: name 'pd' is not defined

## === cell 7
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}


class TestDataset(data.Dataset):
    """
    Loads test images on‑the‑fly and applies a single augmentation.
    """

    def __init__(self, df, transform=None):
        self.image_ids = df["image_id"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = pathlib.Path(TEST_PATH) / img_id
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, idx  # idx used only as dummy target placeholder


def predict_model(basename, net, dataloader):
    net.to(device)
    net.eval()
    with torch.inference_mode():
        probs = []
        for inputs, _ in tqdm(dataloader, desc=f"{basename}: "):
            inputs = inputs.to(device)
            outputs = net(inputs, False, "test")
            probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1272361055.py in <cell line: 0>()
      4 transform = {
      5     "test": [
----> 6         Compose(
      7             [
      8                 A.CenterCrop(height=SIZE, width=SIZE),

NameError: name 'Compose' is not defined

## === cell 8
class TrainDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = pathlib.Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def load_image(self, image_id):
        img_path = self.img_dir / image_id
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = int(row["label"])
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1395305778.py in <cell line: 0>()
----> 1 class TrainDataset(data.Dataset):
      2     def __init__(self, df, img_dir, transform=None):
      3         self.df = df
      4         self.img_dir = pathlib.Path(img_dir)
      5         self.transform = transform

NameError: name 'data' is not defined

## === cell 9
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha=0.0):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels=None, phase=None):
        return self.model(inputs)


class FinalLayerMixupModelDenseNet(FinalLayerMixupModel):
    pass


class FinalLayerMixupModelEN(FinalLayerMixupModel):
    pass




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3832339259.py in <cell line: 0>()
----> 1 class FinalLayerMixupModel(nn.Module):
      2     def __init__(self, model, criterion, num_classes, alpha=0.0):
      3         super().__init__()
      4         self.model = model
      5 

NameError: name 'nn' is not defined

## === cell 10
if not pretrained_models:
    print("No pretrained checkpoints found – training a simple model.")
    TRAIN_CSV = pathlib.Path(BASE_DIR) / "train.csv"
    TRAIN_IMG_DIR = pathlib.Path(BASE_DIR) / "train_images"

    df_train = pd.read_csv(TRAIN_CSV)

    train_aug = Compose(
        [
            A.RandomResizedCrop(size=(SIZE, SIZE)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    train_dataset = TrainDataset(df_train, TRAIN_IMG_DIR, transform=train_aug)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True
    )

    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 15  # extended training for better performance
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        print(f"Epoch {epoch+1} avg loss: {epoch_loss/len(train_loader.dataset):.4f}")

    class SimpleWrapper(nn.Module):
        def __init__(self, mdl):
            super().__init__()
            self.mdl = mdl

        def forward(self, inputs, labels=None, phase=None):
            return self.mdl(inputs)

    net = SimpleWrapper(model)

    all_probs = []
    for tid, aug_transform in enumerate(transform["test"]):
        print(f"Training model TTA {tid}")
        test_dataset = TestDataset(df_test, transform=aug_transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        probs = predict_model("effnet_b0_finetuned", net, test_loader)
        all_probs.append(probs)

    stacked = np.stack(all_probs, axis=0)
    mean_probs = stacked.mean(axis=0)
    df_test["label"] = mean_probs.argmax(axis=1)

else:
    all_probs = []
    start_time = time.time()
    for pretrained_path in pretrained_models:
        basename = pathlib.Path(pretrained_path).stem
        criterion = nn.CrossEntropyLoss()
        if "resnet18" in basename:
            net = models.resnet18(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0.0)
            batch_size = 64
        elif "resnet50" in basename:
            net = models.resnet50(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0.0)
            batch_size = 32
        elif "resnet152" in basename:
            net = models.resnet152(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0.0)
            batch_size = 16
        elif "resnext101" in basename:
            net = models.resnext101_32x8d(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, alpha=0.0)
            batch_size = 12
        elif "densenet201" in basename:
            net = models.densenet201(pretrained=False)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, alpha=0.0)
            batch_size = 12
        elif "efficientnet-b7" in basename:
            if EfficientNet is not None:
                net = EfficientNet.from_name("efficientnet-b7")
                net = FinalLayerMixupModelEN(net, criterion, num_classes, alpha=0.0)
            else:
                net = models.efficientnet_b7(pretrained=False)
                net = FinalLayerMixupModelEN(net, criterion, num_classes, alpha=0.0)
            batch_size = 10
        else:
            print(f"Unsupported model type in {basename}; skipping.")
            continue

        try:
            state = torch.load(pretrained_path, map_location="cpu")
            net.load_state_dict(state)
        except Exception as e:
            print(f"Failed to load weights for {basename}: {e}")
            continue

        for p in net.parameters():
            p.requires_grad = False

        for tid, aug_transform in enumerate(transform["test"]):
            print(f"Model {basename}, TTA {tid}")
            test_dataset = TestDataset(df_test, transform=aug_transform)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            probs = predict_model(basename, net, test_loader)
            all_probs.append(probs)
        del net
        torch.cuda.empty_cache()
    if all_probs:
        stacked = np.stack(
            all_probs, axis=0
        )  # (num_augmentations * models, n_samples, 5)
        mean_probs = stacked.mean(axis=0)  # (n_samples, 5)
        df_test["label"] = mean_probs.argmax(axis=1)
    else:
        df_test["label"] = 0  # safety fallback
    print(f"Total inference time: {time.time() - start_time:.2f}s")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3656684619.py in <cell line: 0>()
----> 1 if not pretrained_models:
      2     print("No pretrained checkpoints found – training a simple model.")
      3     TRAIN_CSV = pathlib.Path(BASE_DIR) / "train.csv"
      4     TRAIN_IMG_DIR = pathlib.Path(BASE_DIR) / "train_images"
      5 

NameError: name 'pretrained_models' is not defined

## === cell 11
submission_path = "submission.csv"
df_test[["image_id", "label"]].to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/670269318.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 df_test[["image_id", "label"]].to_csv(submission_path, index=False)
      3 print(f"Submission file written to {submission_path}")

NameError: name 'df_test' is not defined
