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
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
google-ai-generativelanguage==0.6.15
google-api-core==2.28.1
google-api-python-client==2.177.0
google-auth==2.38.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==1.2.2
google-generativeai==0.8.5
googleapis-common-protos==1.70.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
pydata-google-auth==1.9.1
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
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.84508

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, locale, math, pathlib, cv2
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing, metrics

try:
    locale.setlocale(locale.LC_ALL, "")
except locale.Error:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {DEVICE}")

if os.path.isdir("/kaggle/input/plant-seedlings-classification/"):
    BASE_PATH = "/kaggle/input/"
elif os.path.isdir("../data/"):
    BASE_PATH = "../data/"
else:
    raise RuntimeError("Data directory not found.")

NUM_EPOCHS = 5  # modest epochs for quick run
my_seed = 123
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)



## === cell 1
path_train = pathlib.Path(
    os.path.join(BASE_PATH, "plant-seedlings-classification/train/")
)
labels = [d.name for d in path_train.iterdir() if d.is_dir()]

labels_arr, path_arr = [], []
for label in labels:
    dir_path = path_train / label
    imgs = list(dir_path.iterdir())
    labels_arr.extend([label] * len(imgs))
    path_arr.extend(imgs)

df_meta = pd.DataFrame({"path": path_arr, "label": labels_arr})
print(f"Total training images: {len(df_meta)}")



## === cell 2
SIZE = 70


def visualize_samples(df, n=4):
    fig, axes = plt.subplots(1, n, figsize=(12, 3))
    for i in range(n):
        idx = random.randrange(len(df))
        img = (
            torchvision.io.read_image(str(df.iloc[idx]["path"]))
            .permute(1, 2, 0)
            .numpy()
        )
        axes[i].imshow(img)
        axes[i].set_title(df.iloc[idx]["label"])
        axes[i].axis("off")





## === cell 3
shapes = np.zeros((len(df_meta), 2), dtype=np.uint16)
for i, p in enumerate(df_meta["path"]):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    shapes[i] = im.shape[:2]
df_meta["height"], df_meta["width"] = shapes[:, 0], shapes[:, 1]



## === cell 4
image_list = []
for p in df_meta["path"]:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (SIZE, SIZE))
    image_list.append(img_resized)
images = np.asarray(image_list)  # shape (N,70,70,3)




## === cell 5
def convert_image_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def create_mask_for_plant(image_hsv):
    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def mask_plant(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)




## === cell 6
masked_list = []
for img in images:
    hsv = convert_image_to_hsv(img)
    mask = create_mask_for_plant(hsv)
    masked = mask_plant(img, mask)
    sharpened = sharpen_image(masked)
    masked_list.append(sharpened)
masked_images = np.asarray(masked_list)  # (N,70,70,3)
normalized_images = masked_images.astype(np.float32) / 255.0
x = normalized_images



## === cell 7
label_encoder = preprocessing.LabelEncoder()
y = label_encoder.fit_transform(df_meta["label"])



## === cell 8
assert len(x) == len(y)
perm = np.random.permutation(len(x))
x_rnd, y_rnd = x[perm], y[perm]

num_train = int(0.85 * len(x))
x_train_arr, x_val_arr = x_rnd[:num_train], x_rnd[num_train:]
y_train_arr, y_val_arr = y_rnd[:num_train], y_rnd[num_train:]

x_train = torch.tensor(x_train_arr).to(DEVICE)
x_val = torch.tensor(x_val_arr).to(DEVICE)
y_train = torch.tensor(y_train_arr, dtype=torch.long).to(DEVICE)
y_val = torch.tensor(y_val_arr, dtype=torch.long).to(DEVICE)

BATCH_SIZE = 16
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)




## === cell 9
class CNNClassifier(nn.Module):
    def __init__(self, dropout_probability=0.3, num_labels=12):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_probability),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_probability),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.Dropout(dropout_probability),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9 * 9 * 128, 625)
        self.fc2 = nn.Linear(625, num_labels)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B,3,70,70]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # [B,128,9,9]
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)




## === cell 10
classifier = CNNClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", verbose=False)


def compute_metrics(model, loss_fn, xb, yb):
    logits = model(xb)
    loss = loss_fn(logits, yb).item()
    preds = logits.argmax(dim=1)
    acc = (preds == yb).float().mean().item()
    f1 = metrics.f1_score(yb.cpu(), preds.cpu(), average="micro")
    return loss, acc, f1


df_metrics = pd.DataFrame(
    columns=["loss_train", "acc_train", "f1_train", "loss_val", "acc_val", "f1_val"],
    index=range(NUM_EPOCHS),
    dtype=float,
)

for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
    classifier.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = classifier(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    classifier.eval()
    with torch.no_grad():
        loss_t, acc_t, f1_t = compute_metrics(classifier, criterion, x_train, y_train)
        loss_v, acc_v, f1_v = compute_metrics(classifier, criterion, x_val, y_val)
    df_metrics.iloc[epoch] = [loss_t, acc_t, f1_t, loss_v, acc_v, f1_v]
    scheduler.step(loss_v)

print(df_metrics.tail())



## === cell 11
path_test = pathlib.Path(
    os.path.join(BASE_PATH, "plant-seedlings-classification/test/")
)
test_images = []
filenames = []
for p in path_test.iterdir():
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (SIZE, SIZE))
    hsv = convert_image_to_hsv(img_resized)
    mask = create_mask_for_plant(hsv)
    masked = mask_plant(img_resized, mask)
    sharp = sharpen_image(masked)
    test_images.append(sharp)
    filenames.append(p.name)

test_arr = np.asarray(test_images).astype(np.float32) / 255.0
x_test = torch.tensor(test_arr).to(DEVICE)

classifier.eval()
with torch.no_grad():
    logits = classifier(x_test)
    preds = logits.argmax(dim=1).cpu().numpy()
pred_species = label_encoder.inverse_transform(preds)

df_submission = pd.DataFrame({"file": filenames, "species": pred_species})
output_path = "submission.csv"
df_submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
print(df_submission.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/502276078.py in <cell line: 0>()
      7 for p in path_test.iterdir():
      8     img = cv2.imread(str(p), cv2.IMREAD_COLOR)
----> 9     img_resized = cv2.resize(img, (SIZE, SIZE))
     10     hsv = convert_image_to_hsv(img_resized)
     11     mask = create_mask_for_plant(hsv)

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4208: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
