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

No external packages required in the script and installed.

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

0.9028407373828952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.05531) has done: 'The script was missing all required imports, had undefined variables, and tried to use transforms that were not available. I added the necessary imports, defined missing globals (e.g., `IMG_SIZE`), simplified the dataset to return a tensor without external augmentations, and ensured the model‑loading/inference loop works even if weight files are absent. The final code now runs end‑to‑end and writes a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
def seed_torch(seed=1006):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_torch()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3272296166.py in <cell line: 0>()
      9 
     10 
---> 11 seed_torch()
     12 

/tmp/ipykernel_55/3272296166.py in seed_torch(seed)
      1 def seed_torch(seed=1006):
----> 2     random.seed(seed)
      3     os.environ["PYTHONHASHSEED"] = str(seed)
      4     np.random.seed(seed)
      5     torch.manual_seed(seed)

NameError: name 'random' is not defined

## === cell 1
OUTPUT_DIR = "./"
TEST_PATH = "../input/cassava-leaf-disease-classification/test_images"
TRAIN_PATH = "../input/cassava-leaf-disease-classification/train_images"
IMG_SIZE = 384  # default size used for transforms

test = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")

train_labels = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
majority_label = train_labels["label"].value_counts().idxmax()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4125307255.py in <cell line: 0>()
      4 IMG_SIZE = 384  # default size used for transforms
      5 
----> 6 test = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
      7 
      8 train_labels = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")

NameError: name 'pd' is not defined

## === cell 2
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df["image_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(TEST_PATH, file_name)
        image = cv2.imread(file_path)
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype(np.float32) / 255.0
        if self.transform:
            image = self.transform(image)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/600881146.py in <cell line: 0>()
----> 1 class TestDataset(Dataset):
      2     def __init__(self, df, transform=None):
      3         self.df = df
      4         self.file_names = df["image_id"].values
      5         self.transform = transform

NameError: name 'Dataset' is not defined

## === cell 3
def get_transforms(*, data, vit=False):
    return None




## === cell 4
class NetVit(nn.Module):
    def __init__(
        self, model_name, pretrained=False, n_class=5, att_activate=False, no_att=False
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            if not no_att:
                self.att_layer = nn.Linear(n_features, 1)
        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        output = self.head(x)
        return output




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3112142658.py in <cell line: 0>()
----> 1 class NetVit(nn.Module):
      2     def __init__(
      3         self, model_name, pretrained=False, n_class=5, att_activate=False, no_att=False
      4     ):
      5         super().__init__()

NameError: name 'nn' is not defined

## === cell 5
class NetVit4(nn.Module):
    def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            self.att_layer = nn.Linear(n_features, 1)
        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        l = x.shape[2] // 2
        h1 = self.model(x[:, :, :l, :l])
        h2 = self.model(x[:, :, :l, l:])
        h3 = self.model(x[:, :, l:, :l])
        h4 = self.model(x[:, :, l:, l:])

        a1 = self.att_layer(h1)
        a2 = self.att_layer(h2)
        a3 = self.att_layer(h3)
        a4 = self.att_layer(h4)

        w = F.softmax(torch.cat([a1, a2, a3, a4], dim=1), dim=1)

        h = (
            h1 * w[:, 0].unsqueeze(-1)
            + h2 * w[:, 1].unsqueeze(-1)
            + h3 * w[:, 2].unsqueeze(-1)
            + h4 * w[:, 3].unsqueeze(-1)
        )
        output = self.head(h)
        return output




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/857039745.py in <cell line: 0>()
----> 1 class NetVit4(nn.Module):
      2     def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False):
      3         super().__init__()
      4         self.model = timm.create_model(model_name, pretrained=pretrained)
      5         n_features = self.model.head.in_features

NameError: name 'nn' is not defined

## === cell 6
def inference_tta(model, state, test_loader, device, temp=1.0, tta=5):
    """
    Load a single checkpoint (state) once, run TTA passes, and return the
    averaged prediction for this fold.
    """
    if state:
        try:
            model.load_state_dict(state)
        except Exception:
            pass  # keep current weights if loading fails
    model.eval()
    model.to(device)

    agg = []
    with torch.no_grad():
        for _ in range(tta):
            batch_preds = []
            for image in test_loader:
                batch_preds.append((model(image.to(device)) * temp).softmax(1).cpu())
            agg.append(torch.cat(batch_preds, dim=0).numpy())
    return np.mean(agg, axis=0)




## === cell 7
from collections import OrderedDict


def multi2single(path):
    if not os.path.exists(path):
        return None
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module" in k:
            k = k.replace("se_module", "dummy")
            k = k.replace("module.", "")
            k = k.replace("dummy", "se_module")
        if "attention_linear" in k:
            k = k.replace("attention_linear", "att_layer")
        new_state_dict[k] = v
    return new_state_dict




## === cell 8
temp = 1.0



## === cell 9
MODEL_NAME = "vit_base_patch16_384"
MODEL_NUM = "No3001"
MODEL_DIR = "../input/cassavamodels/"
IMG_SIZE = 384
TTA = 5
BATCH = 32

model = NetVit(MODEL_NAME, pretrained=False, no_att=True)
states = [
    multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))
    for fold in range(5)
]

test_dataset = TestDataset(test)  # no augmentations
test_loader = DataLoader(
    test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
)

vit_predictions = np.zeros((len(test), 5))
num_folds = len(states)
for state in states:
    vit_predictions += (
        inference_tta(model, state, test_loader, device, temp, TTA) / num_folds
    )



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2251100632.py in <cell line: 0>()
      7 BATCH = 32
      8 
----> 9 model = NetVit(MODEL_NAME, pretrained=False, no_att=True)
     10 states = [
     11     multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))

NameError: name 'NetVit' is not defined

## === cell 10
MODEL_NAME = "vit_base_patch16_224"
MODEL_NUM = "vit4_ex"
MODEL_DIR = "../input/cassavamodels/"
IMG_SIZE = 448
TTA = 5
BATCH = 32
att_activate = False

model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
states = [
    multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))
    for fold in range(5)
]

test_dataset = TestDataset(test)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
)

vit4_predictions_a = np.zeros((len(test), 5))
num_folds = len(states)
for state in states:
    vit4_predictions_a += (
        inference_tta(model, state, test_loader, device, temp, TTA) / num_folds
    )



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2131373413.py in <cell line: 0>()
      8 att_activate = False
      9 
---> 10 model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
     11 states = [
     12     multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))

NameError: name 'NetVit4' is not defined

## === cell 11
MODEL_NAME = "vit_base_patch16_224"
MODEL_NUM = "vit4_ex_smooth001_att_act"
MODEL_DIR = "../input/cassavamymodels/"
IMG_SIZE = 448
TTA = 5
BATCH = 32
att_activate = True

model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
states = [
    multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))
    for fold in range(5)
]

test_dataset = TestDataset(test)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
)

vit4_predictions_b = np.zeros((len(test), 5))
num_folds = len(states)
for state in states:
    vit4_predictions_b += (
        inference_tta(model, state, test_loader, device, temp, TTA) / num_folds
    )



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3061536639.py in <cell line: 0>()
      8 att_activate = True
      9 
---> 10 model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
     11 states = [
     12     multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))

NameError: name 'NetVit4' is not defined

## === cell 12
predictions = (
    vit_predictions * 0.45 + vit4_predictions_a * 0.55
) / 9 * 10 + vit4_predictions_b * 0.08



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2958317445.py in <cell line: 0>()
      1 predictions = (
----> 2     vit_predictions * 0.45 + vit4_predictions_a * 0.55
      3 ) / 9 * 10 + vit4_predictions_b * 0.08
      4 

NameError: name 'vit_predictions' is not defined

## === cell 13
if predictions.sum() == 0:
    predictions = np.full((len(test), 5), -np.inf)
    predictions[:, majority_label] = 0  # make argmax return majority label

test["label"] = predictions.argmax(axis=1)
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
test[["image_id", "label"]].to_csv(submission_path, index=False)
print("Submission saved to", submission_path)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3137214836.py in <cell line: 0>()
----> 1 if predictions.sum() == 0:
      2     predictions = np.full((len(test), 5), -np.inf)
      3     predictions[:, majority_label] = 0  # make argmax return majority label
      4 
      5 test["label"] = predictions.argmax(axis=1)

NameError: name 'predictions' is not defined
