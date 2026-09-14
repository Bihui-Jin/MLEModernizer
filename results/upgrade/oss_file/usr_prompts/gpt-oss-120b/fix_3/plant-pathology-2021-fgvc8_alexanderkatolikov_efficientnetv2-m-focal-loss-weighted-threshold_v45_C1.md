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

0.7868144044321331

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
try:
    import sys

    sys.path.append("../input/torchcontrib/contrib-master/")
    import torchcontrib  # noqa: F401
    from torchcontrib.optim import SWA  # noqa: F401
    from torch.optim.swa_utils import AveragedModel, SWALR  # noqa: F401
except Exception:
    pass



## === cell 1
BATCH = 6
EPOCHS = 10

WEIGHT_DECAY = 0.000
LR = 0.000001
IM_SIZE = 640

trainnum = 14800
valnum = 3700
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/334942048.py in <cell line: 0>()
      8 trainnum = 14800
      9 valnum = 3700
---> 10 DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     11 
     12 TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"

NameError: name 'torch' is not defined

## === cell 2
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")

try:
    labeled_train_df = pd.read_csv("../input/train-labeled/train.csv")
except Exception:
    labeled_train_df = pd.DataFrame(columns=train_df.columns)  # empty placeholder



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4222953001.py in <cell line: 0>()
----> 1 train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
      2 
      3 try:
      4     labeled_train_df = pd.read_csv("../input/train-labeled/train.csv")
      5 except Exception:

NameError: name 'pd' is not defined

## === cell 3
train_df["labels"].value_counts()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1498449920.py in <cell line: 0>()
----> 1 train_df["labels"].value_counts()
      2 

NameError: name 'train_df' is not defined

## === cell 4
NUM_CL = len(train_df["labels"].value_counts())
NUM_CL



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/994583981.py in <cell line: 0>()
----> 1 NUM_CL = len(train_df["labels"].value_counts())
      2 NUM_CL
      3 

NameError: name 'train_df' is not defined

## === cell 5
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(train_df["labels"])
train_df["label_id"] = le.transform(train_df["labels"])
print(train_df)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1906625105.py in <cell line: 0>()
      2 
      3 le = preprocessing.LabelEncoder()
----> 4 le.fit(train_df["labels"])
      5 train_df["label_id"] = le.transform(train_df["labels"])
      6 print(train_df)

NameError: name 'train_df' is not defined

## === cell 6
my_dict = {}
if not labeled_train_df.empty:
    for i in range(NUM_CL):
        if not (labeled_train_df["label_id"] == i).any():
            continue
        equality = (
            labeled_train_df.loc[labeled_train_df["label_id"] == i].values[0][1]
            != train_df.loc[train_df["label_id"] == i].values[0][1]
        )
        if equality:
            for l in range(NUM_CL):
                if not (labeled_train_df["label_id"] == i).any():
                    continue
                equality = (
                    labeled_train_df.loc[labeled_train_df["label_id"] == i].values[0][1]
                    == train_df.loc[train_df["label_id"] == l].values[0][1]
                )
                if equality:
                    my_dict[l] = i
    if my_dict:
        train_df = train_df.replace({"label_id": my_dict})
print(train_df)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1179738215.py in <cell line: 0>()
      1 my_dict = {}
----> 2 if not labeled_train_df.empty:
      3     for i in range(NUM_CL):
      4         if not (labeled_train_df["label_id"] == i).any():
      5             continue

NameError: name 'labeled_train_df' is not defined

## === cell 7
class_map = dict(sorted(train_df[["label_id", "labels"]].values.tolist()))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2012094566.py in <cell line: 0>()
----> 1 class_map = dict(sorted(train_df[["label_id", "labels"]].values.tolist()))
      2 

NameError: name 'train_df' is not defined

## === cell 8
tr_df = train_df[:trainnum]
print(len(tr_df))
X_Train, Y_Train = tr_df["image"].values, tr_df["label_id"].values



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3649864049.py in <cell line: 0>()
----> 1 tr_df = train_df[:trainnum]
      2 print(len(tr_df))
      3 X_Train, Y_Train = tr_df["image"].values, tr_df["label_id"].values
      4 

NameError: name 'train_df' is not defined

## === cell 9
Transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.CenterCrop(int(IM_SIZE * 0.8)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1299510087.py in <cell line: 0>()
----> 1 Transform = transforms.Compose(
      2     [
      3         transforms.ToTensor(),
      4         transforms.Resize((IM_SIZE, IM_SIZE)),
      5         transforms.CenterCrop(int(IM_SIZE * 0.8)),

NameError: name 'transforms' is not defined

## === cell 10
Transformval = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.CenterCrop(int(IM_SIZE * 0.8)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3840593329.py in <cell line: 0>()
----> 1 Transformval = transforms.Compose(
      2     [
      3         transforms.ToTensor(),
      4         transforms.Resize((IM_SIZE, IM_SIZE)),
      5         transforms.CenterCrop(int(IM_SIZE * 0.8)),

NameError: name 'transforms' is not defined

## === cell 11
class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
        if "train" in self.dir:
            return self.transform(x), self.labels[index]
        elif "test" in self.dir:
            return self.transform(x), self.fnames[index]




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2109621533.py in <cell line: 0>()
----> 1 class GetData(Dataset):
      2     def __init__(self, Dir, FNames, Labels, Transform):
      3         self.dir = Dir
      4         self.fnames = FNames
      5         self.transform = Transform

NameError: name 'Dataset' is not defined

## === cell 12
trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2350421541.py in <cell line: 0>()
----> 1 trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
      2 trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
      3 

NameError: name 'GetData' is not defined

## === cell 13
val_df = train_df[-valnum:]
print(len(val_df))
X_val, Y_val = val_df["image"].values, val_df["label_id"].values
valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)
valloader = DataLoader(valset, batch_size=BATCH, shuffle=True)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/161322577.py in <cell line: 0>()
----> 1 val_df = train_df[-valnum:]
      2 print(len(val_df))
      3 X_val, Y_val = val_df["image"].values, val_df["label_id"].values
      4 valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)
      5 valloader = DataLoader(valset, batch_size=BATCH, shuffle=True)

NameError: name 'train_df' is not defined

## === cell 14
next(iter(trainloader))[0].shape




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3214837749.py in <cell line: 0>()
----> 1 next(iter(trainloader))[0].shape
      2 
      3 

NameError: name 'trainloader' is not defined

## === cell 15
def PREDS(l):
    word = train_df.loc[train_df["label_id"] == l].values[0][1]  # finds label's name
    words = word.split(" ")
    return words




## === cell 16
def metrics(preds, labels, tp, fn, fp):
    a = preds.tolist()
    b = labels.tolist()
    for i in range(len(preds)):
        tp += len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
        fn += len(PREDS(b[i])) - len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
        fp += len(PREDS(a[i])) - len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
    return tp, fn, fp




## === cell 17
model = torchvision.models.resnext101_32x8d(pretrained=True)
model.fc = nn.Linear(2048, NUM_CL, bias=True)

checkpoint_path = os.path.join("../input/resnet-model/ResNext16.pth")
if os.path.exists(checkpoint_path):
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("Custom checkpoint loaded.")
    except Exception as e:
        print(
            f"Failed to load custom checkpoint: {e}. Proceeding with pretrained weights."
        )
else:
    print("Custom checkpoint not found; using default pretrained weights.")

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1138486697.py in <cell line: 0>()
----> 1 model = torchvision.models.resnext101_32x8d(pretrained=True)
      2 model.fc = nn.Linear(2048, NUM_CL, bias=True)
      3 
      4 checkpoint_path = os.path.join("../input/resnet-model/ResNext16.pth")
      5 if os.path.exists(checkpoint_path):

NameError: name 'torchvision' is not defined

## === cell 18
testnum = 3700
test_df = train_df[-testnum:]
X_test, Y_test = test_df["image"].values, test_df["label_id"].values
testset = GetData(TRAIN_DIR, X_test, Y_test, Transformval)
testloader = DataLoader(testset, batch_size=BATCH, shuffle=True)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1415354671.py in <cell line: 0>()
      1 testnum = 3700
----> 2 test_df = train_df[-testnum:]
      3 X_test, Y_test = test_df["image"].values, test_df["label_id"].values
      4 testset = GetData(TRAIN_DIR, X_test, Y_test, Transformval)
      5 testloader = DataLoader(testset, batch_size=BATCH, shuffle=True)

NameError: name 'train_df' is not defined

## === cell 19
if os.path.isdir(os.path.join(TEST_DIR, "test_images")):
    TEST_DIR = os.path.join(TEST_DIR, "test_images")

X_Test = [
    f
    for f in os.listdir(TEST_DIR)
    if os.path.isfile(os.path.join(TEST_DIR, f))
    and f.lower().endswith((".jpg", ".jpeg", ".png"))
]



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4093037150.py in <cell line: 0>()
      1 # Ensure TEST_DIR points to the folder that directly contains the JPEG files.
      2 # Some extracts create a nested “test_images” folder; adjust if necessary.
----> 3 if os.path.isdir(os.path.join(TEST_DIR, "test_images")):
      4     TEST_DIR = os.path.join(TEST_DIR, "test_images")
      5 

NameError: name 'os' is not defined

## === cell 20
testset = GetData(TEST_DIR, X_Test, None, Transformval)
testloader = DataLoader(testset, batch_size=1, shuffle=False)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3438024505.py in <cell line: 0>()
----> 1 testset = GetData(TEST_DIR, X_Test, None, Transformval)
      2 testloader = DataLoader(testset, batch_size=1, shuffle=False)
      3 

NameError: name 'GetData' is not defined

## === cell 21
s_ls = []
model.eval()
with torch.no_grad():
    for image, fname in testloader:
        image = image.to(DEVICE)
        logits = model(image)
        _, top_class = logits.topk(1, dim=1)
        pred_id = top_class.item()
        if isinstance(fname, (list, tuple)):
            fname = fname[0]
        s_ls.append([fname, pred_id])



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4126312237.py in <cell line: 0>()
      1 s_ls = []
----> 2 model.eval()
      3 with torch.no_grad():
      4     for image, fname in testloader:
      5         image = image.to(DEVICE)

NameError: name 'model' is not defined

## === cell 22
pred_df = pd.DataFrame.from_records(s_ls, columns=["image", "label_id"])
pred_df



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2459329788.py in <cell line: 0>()
----> 1 pred_df = pd.DataFrame.from_records(s_ls, columns=["image", "label_id"])
      2 pred_df
      3 

NameError: name 'pd' is not defined

## === cell 23
pred_df["labels"] = pred_df["label_id"].map(class_map)
pred_df



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2676968713.py in <cell line: 0>()
----> 1 pred_df["labels"] = pred_df["label_id"].map(class_map)
      2 pred_df
      3 

NameError: name 'pred_df' is not defined

## === cell 24
sub = pred_df[["image", "labels"]]
sub.head()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3726815049.py in <cell line: 0>()
----> 1 sub = pred_df[["image", "labels"]]
      2 sub.head()
      3 

NameError: name 'pred_df' is not defined

## === cell 25
sub.to_csv("submission.csv", index=False)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1126610719.py in <cell line: 0>()
      1 # Write the submission file; the number of rows now matches the number of test images.
----> 2 sub.to_csv("submission.csv", index=False)

NameError: name 'sub' is not defined
