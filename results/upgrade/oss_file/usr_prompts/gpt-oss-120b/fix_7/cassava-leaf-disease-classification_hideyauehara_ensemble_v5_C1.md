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

0.8943789664551224

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the Albumentations RandomResizedCrop signature and ensure the test set is built from the provided sample_submission file (so the submission length matches Kaggle’s expectations). These changes resolve the runtime errors, allow the model loop to run, and produce a correctly‑sized submission .csv.'
- What this solution (achieved 0.0) has done: 'I correct the Albumentations RandomResizedCrop usage (it requires a height × width tuple) and restructure the model‑loading loop so that pretrained checkpoints are loaded into the base torchvision model before it is wrapped. This eliminates the `NameError` for `transform` and prevents state‑dict mismatches, allowing the pipeline to run end‑to‑end and produce a proper `submission.csv`. These fixes keep the original architecture and training logic unchanged while enabling the ensemble of test‑time augmentations to reach a score close to the target.'

# 9. Code solution

## === cell 0
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2045089245.py in <cell line: 0>()
     10 
     11 SEED = 42
---> 12 seed_everything(seed=SEED)
     13 

/tmp/ipykernel_55/2045089245.py in seed_everything(seed)
      1 def seed_everything(seed=42):
----> 2     random.seed(seed)
      3     os.environ["PYTHONHASHSEED"] = str(seed)
      4     np.random.seed(seed)
      5     torch.manual_seed(seed)

NameError: name 'random' is not defined

## === cell 1
try:
    sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
    from efficientnet_pytorch import EfficientNet
except Exception:
    EfficientNet = None
    print("EfficientNet not available; related models will be skipped.")



## === cell 2
SIZE = 512  # image size
num_classes = 5



## === cell 3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3274637083.py in <cell line: 0>()
----> 1 device = "cuda" if torch.cuda.is_available() else "cpu"
      2 print(f"使用デバイス: {device}")
      3 

NameError: name 'torch' is not defined

## === cell 4
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.isdir(BASE_DIR):
    BASE_DIR = "data"  # fallback for any local copy

sample_submission_path = os.path.join(BASE_DIR, "sample_submission.csv")
if not os.path.isfile(sample_submission_path):
    raise FileNotFoundError(
        f"sample_submission.csv not found at {sample_submission_path}"
    )

df_test = pd.read_csv(sample_submission_path)[["image_id"]].copy()
df_test["label"] = -1  # placeholder; will be overwritten later
print(f"Number of test images (from sample submission): {len(df_test)}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3048250781.py in <cell line: 0>()
      1 BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
----> 2 if not os.path.isdir(BASE_DIR):
      3     BASE_DIR = "data"  # fallback for any local copy
      4 
      5 sample_submission_path = os.path.join(BASE_DIR, "sample_submission.csv")

NameError: name 'os' is not defined

## === cell 5
if len(df_test) == 1:
    df_test = pd.concat([df_test, df_test], ignore_index=True)
    print(df_test)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3879634184.py in <cell line: 0>()
----> 1 if len(df_test) == 1:
      2     df_test = pd.concat([df_test, df_test], ignore_index=True)
      3     print(df_test)
      4 

NameError: name 'df_test' is not defined

## === cell 6
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop((SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop((SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop((SIZE, SIZE), scale=(0.08, 1.0)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop((SIZE, SIZE), scale=(0.08, 1.0)),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop((SIZE, SIZE), scale=(0.08, 1.0)),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1.0),
                A.RandomResizedCrop((SIZE, SIZE), scale=(0.08, 1.0)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1354895169.py in <cell line: 0>()
      4 transform = {
      5     "test": [
----> 6         Compose(
      7             [
      8                 A.CenterCrop((SIZE, SIZE)),

NameError: name 'Compose' is not defined

## === cell 7
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            return outputs
        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)
        labels_a = labels
        labels_b = labels[index]
        loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(
            outputs, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/220659496.py in <cell line: 0>()
----> 1 class FinalLayerMixupModel(nn.Module):
      2     def __init__(self, model, criterion, num_classes, alpha):
      3         super(FinalLayerMixupModel, self).__init__()
      4         self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
      5         num_ftrs = model.fc.in_features

NameError: name 'nn' is not defined

## === cell 8
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)
            return outputs
        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])
        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)
        labels_a = labels
        labels_b = labels[index]
        loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(
            outputs, labels_b
        )
        return outputs, loss, labels_a, labels_b, lam




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1063068879.py in <cell line: 0>()
----> 1 class FinalLayerMixupModelDenseNet(nn.Module):
      2     def __init__(self, model, criterion, num_classes, alpha):
      3         super(FinalLayerMixupModelDenseNet, self).__init__()
      4         self.convlayer = model.features
      5         self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

NameError: name 'nn' is not defined

## === cell 9
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)
        self.model = model
        self.criterion = criterion

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss
        if phase == "test":
            outputs = self.model(inputs)
            return outputs
        print("Unexpected path for EfficientNet mixup")
        sys.exit()




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1797601633.py in <cell line: 0>()
----> 1 class FinalLayerMixupModelEN(nn.Module):
      2     def __init__(self, model, criterion, num_classes, alpha):
      3         super(FinalLayerMixupModelEN, self).__init__()
      4         num_ftrs = model._fc.in_features
      5         model._fc = nn.Linear(num_ftrs, num_classes)

NameError: name 'nn' is not defined

## === cell 10
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = os.path.join(BASE_DIR, "test_images", image_id)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3478629139.py in <cell line: 0>()
----> 1 class TestDataset(data.Dataset):
      2     def __init__(self, df, transform=None):
      3         super().__init__()
      4         self.image_ids = df.image_id.tolist()
      5         self.transform = transform

NameError: name 'data' is not defined

## === cell 11
def predict_model(basename, net, dataloader):
    model_start_time = time.time()
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []
    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)
            outputs = net(inputs, False, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())
    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 12
pretrained_models = (
    glob.glob("/kaggle/input/densenet201-04-2019data/*.pth")
    + glob.glob("/kaggle/input/resnet152-04-2019data/*.pth")
    + glob.glob("/kaggle/input/eb7-00-baseline/*.pth")
)

if not pretrained_models:
    print(
        "No custom pretrained checkpoints found – using a default ImageNet‑pretrained ResNet50."
    )
    pretrained_models = [None]  # placeholder to trigger one iteration

probability = []  # will hold arrays of shape (n_samples, n_classes)

start_time = time.time()

for pretrained_model in pretrained_models:
    basename = (
        "resnet50_imagenet"
        if pretrained_model is None
        else os.path.splitext(os.path.basename(pretrained_model))[0]
    )
    criterion = nn.CrossEntropyLoss()

    if "resnet18" in basename:
        base_model = models.resnet18(pretrained=False)
        BATCH_SIZE = 64
    elif "resnet50" in basename:
        base_model = models.resnet50(pretrained=True)
        BATCH_SIZE = 32
    elif "resnet152" in basename:
        base_model = models.resnet152(pretrained=False)
        BATCH_SIZE = 16
    elif "resnext101" in basename:
        base_model = models.resnext101_32x8d(pretrained=False)
        BATCH_SIZE = 12
    elif "densenet201" in basename:
        base_model = models.densenet201(pretrained=False)
        BATCH_SIZE = 12
    elif "efficientnet-b7" in basename:
        if EfficientNet is None:
            print(f"EfficientNet unavailable, skipping model {basename}")
            continue
        base_model = EfficientNet.from_name("efficientnet-b7")
        BATCH_SIZE = 10
    else:
        base_model = models.resnet50(pretrained=True)
        BATCH_SIZE = 32

    if pretrained_model is not None:
        try:
            state_dict = torch.load(pretrained_model, map_location=device)
            if "efficientnet-b7" in basename:
                base_model.load_state_dict(state_dict)
            else:
                base_model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: could not load checkpoint {pretrained_model}: {e}")

    if "densenet201" in basename:
        net = FinalLayerMixupModelDenseNet(base_model, criterion, num_classes, False)
    elif "efficientnet-b7" in basename:
        net = FinalLayerMixupModelEN(base_model, criterion, num_classes, False)
    else:
        net = FinalLayerMixupModel(base_model, criterion, num_classes, False)

    for param in net.parameters():
        param.requires_grad = False

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        dataset = {"test": TestDataset(df_test, transform=transform_)}
        dataloader = {
            "test": torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
        }
        proba = predict_model(basename, net, dataloader)
        probability.append(proba)

    del net
    torch.cuda.empty_cache()

if probability:
    stacked = np.stack(probability, axis=0)  # (n_preds, n_samples, n_classes)
    mean_probs = stacked.mean(axis=0)  # (n_samples, n_classes)
    df_test["label"] = mean_probs.argmax(axis=1)
else:
    raise RuntimeError("No predictions were generated; check model loading paths.")

print(f"total time: {time.time() - start_time:.2f}[sec]")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1719228959.py in <cell line: 0>()
      1 pretrained_models = (
----> 2     glob.glob("/kaggle/input/densenet201-04-2019data/*.pth")
      3     + glob.glob("/kaggle/input/resnet152-04-2019data/*.pth")
      4     + glob.glob("/kaggle/input/eb7-00-baseline/*.pth")
      5 )

NameError: name 'glob' is not defined

## === cell 13
submission = df_test[["image_id", "label"]].copy()
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/455157575.py in <cell line: 0>()
----> 1 submission = df_test[["image_id", "label"]].copy()
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission saved to {submission_path}")
      5 

NameError: name 'df_test' is not defined

## === cell 14
print(submission.head())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2388794795.py in <cell line: 0>()
----> 1 print(submission.head())

NameError: name 'submission' is not defined
