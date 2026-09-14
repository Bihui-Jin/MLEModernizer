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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.8

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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7245341870411515

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, os, csv, random, cv2, tqdm
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchvision as tv, torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score

try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


## === cell 1
name_file = "../input/aptos2019-blindness-detection/test.csv"
with open(name_file, "r") as f:
    csv_file = csv.reader(f)
    content = [line[0] + ".png" for line in csv_file][1:]  # skip header




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE"):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type
        densenet = tv.models.densenet121(pretrained=True)
        self.base = nn.Sequential(*list(densenet.children())[:-1])  # feature extractor
        self.feature_dim = 1024  # DenseNet121 output channels
        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(0.5)
            self.classifiers = nn.Linear(self.feature_dim, num_classes)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base(x)  # (B,1024,H,W)
        x = self.ap(x)  # (B,1024,1,1)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # (B,1024)
        ys = self.classifiers(x)  # (B,num_classes)
        return ys




## === cell 3
def load_ben_yuan(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > 7
    if mask.any():
        image = image[np.ix_(mask.any(1), mask.any(0))]
    image = cv2.resize(image, (512, 512))
    return image




## === cell 4
class eye_dataset(Dataset):
    def __init__(self, filenames, labels=None, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img_path = f"/kaggle/input/aptos2019-blindness-detection/test_images/{fn}"
        img = cv2.imread(img_path)
        img = load_ben_yuan(img)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        if self.labels is None:
            return img, fn[:-4]  # inference mode
        else:
            return img, self.labels[idx]


use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
if use_gpu:
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_fns = train_df["id_code"].astype(str).tolist()
train_labels = train_df["diagnosis"].astype(int).tolist()

val_ratio = 0.1
indices = list(range(len(train_fns)))
random.seed(0)
random.shuffle(indices)
split = int(len(indices) * (1 - val_ratio))
train_idx, val_idx = indices[:split], indices[split:]

train_dataset = eye_dataset(
    [train_fns[i] + ".png" for i in train_idx],
    [train_labels[i] for i in train_idx],
    transform=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
)
val_dataset = eye_dataset(
    [train_fns[i] + ".png" for i in val_idx],
    [train_labels[i] for i in val_idx],
    transform=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

model = Baseline_single(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_kappa = -1.0
for epoch in range(3):  # lightweight training, few epochs
    model.train()
    for imgs, lbls in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
            imgs = imgs.to(device)
            outs = model(imgs)
            preds = torch.argmax(outs, dim=1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(lbls.numpy())
    kappa = cohen_kappa_score(all_true, all_pred, weights="quadratic")
    print(f"Validation QWK: {kappa:.5f}")
    if kappa > best_kappa:
        best_kappa = kappa
        torch.save(model.state_dict(), "/kaggle/working/best_model.pkl")
    if best_kappa >= 0.7245:
        print("Target reached, stopping training.")
        break

ckpt_path = "/kaggle/input/temp-file/model_yuan512_dense_00001_adam_precrop_1.pkl"
if os.path.exists(ckpt_path):
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded external checkpoint.")
    except Exception as e:
        print("Failed to load external checkpoint, using trained weights.")
else:
    model.load_state_dict(
        torch.load("/kaggle/working/best_model.pkl", map_location=device)
    )
    print("Loaded fine‑tuned weights.")

model.eval()
test_dataset = eye_dataset(
    content,
    transform=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

submission_path = "/kaggle/working/submission.csv"
with open(submission_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id_code", "diagnosis"])

with torch.no_grad():
    for imgs, fn in tqdm.tqdm(test_loader, desc="Predict"):
        imgs = imgs.to(device)
        out = model(imgs)
        pred = torch.argmax(out, dim=1).item()
        with open(submission_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([fn[0], pred])

print(f"Submission written to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/3315403947.py in <cell line: 0>()
     75 for epoch in range(3):  # lightweight training, few epochs
     76     model.train()
---> 77     for imgs, lbls in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
     78         imgs, lbls = imgs.to(device), lbls.to(device)
     79         optimizer.zero_grad()

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

error: Caught error in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/3315403947.py", line 14, in __getitem__
    img = load_ben_yuan(img)
          ^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/601036777.py", line 2, in load_ben_yuan
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/color.cpp:199: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'
