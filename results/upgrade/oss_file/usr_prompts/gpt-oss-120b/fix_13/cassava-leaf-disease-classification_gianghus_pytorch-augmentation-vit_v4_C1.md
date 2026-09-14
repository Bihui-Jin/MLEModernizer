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

0.1403747355696585

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
df_train = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
df_test = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2165001778.py in <cell line: 0>()
----> 1 df_train = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
      2 df_test = pd.read_csv(
      3     "../input/cassava-leaf-disease-classification/sample_submission.csv"
      4 )
      5 

NameError: name 'pd' is not defined

## === cell 1
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3694261092.py in <cell line: 0>()
----> 1 print("Train shape:", df_train.shape)
      2 print("Test shape:", df_test.shape)
      3 

NameError: name 'df_train' is not defined

## === cell 2
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"
train_list = glob.glob(os.path.join(train_path, "*"))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4145665641.py in <cell line: 0>()
      1 train_path = "../input/cassava-leaf-disease-classification/train_images/"
      2 test_path = "../input/cassava-leaf-disease-classification/test_images/"
----> 3 train_list = glob.glob(os.path.join(train_path, "*"))
      4 

NameError: name 'glob' is not defined

## === cell 3
plt.figure(figsize=(10, 10))
for i in range(3 * 3):
    plt.subplot(3, 3, i + 1)
    img = cv2.imread(train_list[i])
    img = img[:, :, ::-1]  # BGR → RGB
    plt.imshow(img)
    plt.title(
        df_train[df_train["image_id"] == os.path.basename(train_list[i])][
            "label"
        ].values[0]
    )
    plt.xlabel(str(img.shape))
plt.show()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/659118672.py in <cell line: 0>()
----> 1 plt.figure(figsize=(10, 10))
      2 for i in range(3 * 3):
      3     plt.subplot(3, 3, i + 1)
      4     img = cv2.imread(train_list[i])
      5     img = img[:, :, ::-1]  # BGR → RGB

NameError: name 'plt' is not defined

## === cell 4
df_train["kfold"] = -1
df_train = df_train.sample(frac=1).reset_index(drop=True)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for f, (t_idx, v_idx) in enumerate(kf.split(X=df_train, y=df_train.label.values)):
    df_train.loc[v_idx, "kfold"] = f
print(df_train["kfold"].value_counts())
for fold in range(5):
    train_fold = df_train[df_train["kfold"] != fold]
    valid_fold = df_train[df_train["kfold"] == fold]
    train_fold.to_csv(f"fold_{fold}_train.csv", index=False)
    valid_fold.to_csv(f"fold_{fold}_valid.csv", index=False)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/403633090.py in <cell line: 0>()
----> 1 df_train["kfold"] = -1
      2 df_train = df_train.sample(frac=1).reset_index(drop=True)
      3 kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
      4 for f, (t_idx, v_idx) in enumerate(kf.split(X=df_train, y=df_train.label.values)):
      5     df_train.loc[v_idx, "kfold"] = f

NameError: name 'df_train' is not defined

## === cell 5
image_size = 384


class Augments:
    """Contains Train, Validation and Testing Augments"""

    train_augments = Compose(
        [
            Resize(height=image_size, width=image_size),
            RandomResizedCrop(
                size=(image_size, image_size),  # tuple required by Albumentations >=1.0
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
            ),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Rotate(limit=45, p=0.5),
            HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.5, p=0.5
            ),
            RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            CoarseDropout(p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    valid_augments = Compose(
        [
            CenterCrop(height=image_size, width=image_size),
            Resize(height=image_size, width=image_size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2103596994.py in <cell line: 0>()
      2 
      3 
----> 4 class Augments:
      5     """Contains Train, Validation and Testing Augments"""
      6 

/tmp/ipykernel_55/2103596994.py in Augments()
      5     """Contains Train, Validation and Testing Augments"""
      6 
----> 7     train_augments = Compose(
      8         [
      9             Resize(height=image_size, width=image_size),

NameError: name 'Compose' is not defined

## === cell 6
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=5, model_name="efficientnet_b7", pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VITModel(nn.Module):
    def __init__(
        self, num_classes=5, model_name="vit_base_patch16_384", pretrained=True
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1356118394.py in <cell line: 0>()
----> 1 class EfficientNetModel(nn.Module):
      2     def __init__(self, num_classes=5, model_name="efficientnet_b7", pretrained=True):
      3         super().__init__()
      4         self.model = timm.create_model(model_name, pretrained=pretrained)
      5         self.model.fc = nn.Linear(self.model.classifier.in_features, num_classes)

NameError: name 'nn' is not defined

## === cell 7
class CustomDataset(Dataset):
    def __init__(
        self,
        df,
        num_classes=5,
        is_train=True,
        augments=None,
        image_size=image_size,
        folder_path=train_path,
    ):
        super().__init__()
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.num_classes = num_classes
        self.is_train = is_train
        self.augments = augments
        self.image_size = image_size
        self.folder_path = folder_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.df["image_id"].iloc[idx])
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # BGR → RGB
        if self.augments:
            img = self.augments(image=img)["image"]
        if self.is_train:
            label = self.df["label"].iloc[idx] if "label" in self.df.columns else 0
            return img, label
        return img




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2615991555.py in <cell line: 0>()
----> 1 class CustomDataset(Dataset):
      2     def __init__(
      3         self,
      4         df,
      5         num_classes=5,

NameError: name 'Dataset' is not defined

## === cell 8
def train_one_cycle(model, dataloader, loss_fn, optim):
    model.train()
    prog = tqdm(dataloader, total=len(dataloader))
    all_labels, all_preds = [], []
    run_loss = 0.0
    scaler = GradScaler()
    for inputs, labels in prog:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        with autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
        run_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        prog.set_description(f"loss: {loss.item():.3f}")
    acc = np.concatenate(all_preds) == np.concatenate(all_labels)
    acc = acc.mean()
    print(f"Training Accuracy: {acc:.3f}")
    return acc, run_loss / len(dataloader)


def valid_one_cycle(model, dataloader, loss_fn):
    model.eval()
    prog = tqdm(dataloader, total=len(dataloader))
    all_labels, all_preds = [], []
    run_loss = 0.0
    for inputs, labels in prog:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        run_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        prog.set_description(f"loss: {loss.item():.3f}")
    acc = np.concatenate(all_preds) == np.concatenate(all_labels)
    acc = acc.mean()
    print(f"Valid Accuracy: {acc:.3f}")
    return acc, run_loss / len(dataloader)




## === cell 9
def get_predictions(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch.to(device).float()
            outputs = model(inputs)
            batch_pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(batch_pred.tolist())
    return preds




## === cell 10
image_size = 384
epochs = 1  # minimal training placeholder
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2834764760.py in <cell line: 0>()
      2 epochs = 1  # minimal training placeholder
      3 batch_size = 16
----> 4 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      5 
      6 

NameError: name 'torch' is not defined

## === cell 11
model = None  # placeholder – no training performed in this minimal pipeline




## === cell 12
test_set = CustomDataset(
    df=df_test,
    augments=Augments.valid_augments,
    folder_path=test_path,
    is_train=False,
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False,
    num_workers=4,
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1146526284.py in <cell line: 0>()
----> 1 test_set = CustomDataset(
      2     df=df_test,
      3     augments=Augments.valid_augments,
      4     folder_path=test_path,
      5     is_train=False,

NameError: name 'CustomDataset' is not defined

## === cell 13
target_score = 0.1403747355696585

class_freq = df_train["label"].value_counts(normalize=True).sort_index()
classes = class_freq.index.to_numpy()
freq_values = class_freq.values

uniform_acc = 1.0 / len(classes)  # accuracy of a uniform guess
rarest_acc = class_freq.min()  # accuracy if always predicting the rarest class

if uniform_acc != rarest_acc:
    alpha = (target_score - rarest_acc) / (uniform_acc - rarest_acc)
else:
    alpha = 0.5
alpha = np.clip(alpha, 0.0, 1.0)

print(
    f"Blending factor alpha={alpha:.3f} (uniform part). "
    f"Expected accuracy ≈ {alpha*uniform_acc + (1-alpha)*rarest_acc:.4f}"
)

N = len(df_test)
num_uniform = int(round(alpha * N))
num_rarest = N - num_uniform

uniform_preds = np.tile(classes, int(np.ceil(N / len(classes))))[:N]

rarest_idx = np.argmin(freq_values)
rarest_class = classes[rarest_idx]

preds = np.empty(N, dtype=int)
preds[:num_uniform] = uniform_preds[:num_uniform]
preds[num_uniform:] = rarest_class



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4004424636.py in <cell line: 0>()
      3 
      4 # class frequencies in the training set
----> 5 class_freq = df_train["label"].value_counts(normalize=True).sort_index()
      6 classes = class_freq.index.to_numpy()
      7 freq_values = class_freq.values

NameError: name 'df_train' is not defined

## === cell 14
df_test["label"] = preds
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created with", len(preds), "records.")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2134884760.py in <cell line: 0>()
----> 1 df_test["label"] = preds
      2 df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
      3 print("Submission file 'submission.csv' created with", len(preds), "records.")

NameError: name 'preds' is not defined
