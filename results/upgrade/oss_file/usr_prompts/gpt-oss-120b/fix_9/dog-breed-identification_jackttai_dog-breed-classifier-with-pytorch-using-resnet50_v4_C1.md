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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.50039

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.63153) has done: 'I fixed the stratified split error by using the full training set and ensuring the validation split contains at least as many samples as there are classes (120). This resolves the `ValueError` and defines `train_loader`, `val_loader`, and `batch_size` so the rest of the pipeline can run and produce a proper `submission.csv`.'
- What this solution (achieved 0.95902) has done: 'I unfreeze the last block of the pretrained ResNet‑50 so the model can fine‑tune useful features, and I train a few more epochs (8 instead of 5). These minimal adjustments should improve validation loss and move the log‑loss closer to the target without altering the core architecture or training logic.'
- What this solution (achieved 0.83345) has done: 'Optimized data loading by enabling CUDA benchmark, using multiple worker processes, pinning memory, and persistent workers to speed up I/O and GPU utilization; these changes keep the model, training loop, and hyper‑parameters unchanged, preserving exact algorithmic behavior while considerably reducing runtime.'
- What this solution (achieved 0.62137) has done: 'The update unfreezes an additional ResNet block, lowers the learning rate for finer adjustments, and extends training to 20 epochs—small changes expected to lower the validation log‑loss and move the score toward the target while keeping the original architecture and pipeline intact.'
- What this solution (achieved 0.68795) has done: 'I add a small amount of label smoothing to the loss (helps regularize and usually lowers log‑loss) and apply a simple test‑time augmentation by averaging predictions on the original and horizontally‑flipped images. These tweaks keep the overall architecture and training loop unchanged while aiming to reduce the validation loss and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader




## === cell 1
comp_df = pd.read_csv("../input/dog-breed-identification/labels.csv")
test_df = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")
print("Training set rows:", comp_df.shape[0], "Test set rows:", test_df.shape[0])




## === cell 2
le = LabelEncoder()
comp_df["label"] = le.fit_transform(comp_df["breed"])

dict_df = comp_df[["label", "breed"]].drop_duplicates().set_index("label")
index_to_breed = dict_df["breed"].to_dict()




## === cell 3
train_dir = "../input/dog-breed-identification/train"
comp_df["id"] = comp_df["id"].apply(lambda x: os.path.join(train_dir, f"{x}.jpg"))
comp_df = comp_df.drop(columns=["breed"])




## === cell 4
def show_images(df, img_num):
    sample = df.sample(img_num)
    paths = sample["id"].tolist()
    for path in paths:
        plt.figure(figsize=(3, 3))
        img = plt.imread(path)
        plt.imshow(img)
        plt.axis("off")
        plt.show()




## === cell 5
class img_dataset(Dataset):
    def __init__(self, dataframe, transform=None, test=False):
        self.df = dataframe
        self.transform = transform
        self.test = test

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.test:
            return img
        else:
            label = self.df.iloc[idx, 1]
            return img, label

    def __len__(self):
        return len(self.df)




## === cell 6
train_transformer = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transformer = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True




## === cell 8
training_samples = comp_df.shape[0]  # use all rows
sample_df = comp_df.sample(training_samples, random_state=42)

val_frac = max(0.1, 120 / training_samples)  # at least 10% or enough rows
x_train, x_val = train_test_split(
    sample_df, test_size=val_frac, random_state=42, stratify=sample_df["label"]
)

train_set = img_dataset(x_train, transform=train_transformer)
val_set = img_dataset(x_val, transform=val_transformer)

batch_size = 64
num_workers = min(
    4, os.cpu_count() or 0
)  # moderate parallelism without oversubscription
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

print("Training subset:", x_train.shape[0], "Validation subset:", x_val.shape[0])




## === cell 9
class Net(torch.nn.Module):
    def __init__(self, base_model, num_classes):
        super(Net, self).__init__()
        self.base = torch.nn.Sequential(
            *list(base_model.children())[:-1]
        )  # remove original FC
        self.fc1 = torch.nn.Linear(base_model.fc.in_features, 512)
        self.out = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.out(x)
        return x


resnet = torchvision.models.resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.layer4.parameters():
    param.requires_grad = True
for param in resnet.layer3.parameters():
    param.requires_grad = True
for param in resnet.layer2.parameters():
    param.requires_grad = True

model_final = Net(resnet, num_classes=len(le.classes_)).to(device)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RemoteDisconnected                        Traceback (most recent call last)
/tmp/ipykernel_55/3492210198.py in <cell line: 0>()
     16 
     17 
---> 18 resnet = torchvision.models.resnet50(pretrained=True)
     19 for param in resnet.parameters():
     20     param.requires_grad = False

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in wrapper(*args, **kwargs)
    140             kwargs.update(keyword_only_kwargs)
    141 
--> 142         return fn(*args, **kwargs)
    143 
    144     return wrapper

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in inner_wrapper(*args, **kwargs)
    226                 kwargs[weights_param] = default_weights_arg
    227 
--> 228             return builder(*args, **kwargs)
    229 
    230         return inner_wrapper

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in resnet50(weights, progress, **kwargs)
    761     weights = ResNet50_Weights.verify(weights)
    762 
--> 763     return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
    764 
    765 

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in _resnet(block, layers, weights, progress, **kwargs)
    299 
    300     if weights is not None:
--> 301         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    302 
    303     return model

/usr/local/lib/python3.11/dist-packages/torchvision/models/_api.py in get_state_dict(self, *args, **kwargs)
     88 
     89     def get_state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
---> 90         return load_state_dict_from_url(self.url, *args, **kwargs)
     91 
     92     def __repr__(self) -> str:

/usr/local/lib/python3.11/dist-packages/torch/hub.py in load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name, weights_only)
    865             r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
    866             hash_prefix = r.group(1) if r else None
--> 867         download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    868 
    869     if _is_legacy_zip_format(cached_file):

/usr/local/lib/python3.11/dist-packages/torch/hub.py in download_url_to_file(url, dst, hash_prefix, progress)
    706     file_size = None
    707     req = Request(url, headers={"User-Agent": "torch.hub"})
--> 708     u = urlopen(req)
    709     meta = u.info()
    710     if hasattr(meta, "getheaders"):

/usr/lib/python3.11/urllib/request.py in urlopen(url, data, timeout, cafile, capath, cadefault, context)
    214     else:
    215         opener = _opener
--> 216     return opener.open(url, data, timeout)
    217 
    218 def install_opener(opener):

/usr/lib/python3.11/urllib/request.py in open(self, fullurl, data, timeout)
    517 
    518         sys.audit('urllib.Request', req.full_url, req.data, req.headers, req.get_method())
--> 519         response = self._open(req, data)
    520 
    521         # post-process response

/usr/lib/python3.11/urllib/request.py in _open(self, req, data)
    534 
    535         protocol = req.type
--> 536         result = self._call_chain(self.handle_open, protocol, protocol +
    537                                   '_open', req)
    538         if result:

/usr/lib/python3.11/urllib/request.py in _call_chain(self, chain, kind, meth_name, *args)
    494         for handler in handlers:
    495             func = getattr(handler, meth_name)
--> 496             result = func(*args)
    497             if result is not None:
    498                 return result

/usr/lib/python3.11/urllib/request.py in https_open(self, req)
   1389 
   1390         def https_open(self, req):
-> 1391             return self.do_open(http.client.HTTPSConnection, req,
   1392                 context=self._context, check_hostname=self._check_hostname)
   1393 

/usr/lib/python3.11/urllib/request.py in do_open(self, http_class, req, **http_conn_args)
   1350             except OSError as err: # timeout error
   1351                 raise URLError(err)
-> 1352             r = h.getresponse()
   1353         except:
   1354             h.close()

/usr/lib/python3.11/http/client.py in getresponse(self)
   1393         try:
   1394             try:
-> 1395                 response.begin()
   1396             except ConnectionError:
   1397                 self.close()

/usr/lib/python3.11/http/client.py in begin(self)
    323         # read until we get a non-100 response
    324         while True:
--> 325             version, status, reason = self._read_status()
    326             if status != CONTINUE:
    327                 break

/usr/lib/python3.11/http/client.py in _read_status(self)
    292             # Presumably, the server closed the connection before
    293             # sending a valid response.
--> 294             raise RemoteDisconnected("Remote end closed connection without"
    295                                      " response")
    296         try:

RemoteDisconnected: Remote end closed connection without response

## === cell 10
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_final.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1701600818.py in <cell line: 0>()
      4 criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
      5 optimizer = torch.optim.Adam(
----> 6     filter(lambda p: p.requires_grad, model_final.parameters()),
      7     lr=1e-4,
      8     weight_decay=1e-4,

NameError: name 'model_final' is not defined

## === cell 11
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_model(model, epochs, scheduler=None):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, device)

        train_losses.append(tr_loss)
        train_accuracies.append(tr_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"train loss: {tr_loss:.4f}, acc: {tr_acc:.4f} | "
            f"val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )
        if scheduler:
            scheduler.step()
    return train_losses, train_accuracies, val_losses, val_accuracies




## === cell 12
EPOCHS = 30
train_losses, train_acc, val_losses, val_acc = train_model(
    model_final, EPOCHS, scheduler
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/879929586.py in <cell line: 0>()
      2 EPOCHS = 30
      3 train_losses, train_acc, val_losses, val_acc = train_model(
----> 4     model_final, EPOCHS, scheduler
      5 )
      6 

NameError: name 'model_final' is not defined

## === cell 13
plt.style.use("ggplot")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(range(1, EPOCHS + 1), train_losses, label="train loss")
ax1.plot(range(1, EPOCHS + 1), val_losses, label="val loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(range(1, EPOCHS + 1), train_acc, label="train acc")
ax2.plot(range(1, EPOCHS + 1), val_acc, label="val acc")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
plt.show()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1755141271.py in <cell line: 0>()
      1 plt.style.use("ggplot")
      2 fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
----> 3 ax1.plot(range(1, EPOCHS + 1), train_losses, label="train loss")
      4 ax1.plot(range(1, EPOCHS + 1), val_losses, label="val loss")
      5 ax1.set_xlabel("Epoch")

NameError: name 'train_losses' is not defined

## === cell 14
test_dir = "../input/dog-breed-identification/test"
test_df_paths = test_df[["id"]].copy()
test_df_paths["id"] = test_df_paths["id"].apply(
    lambda x: os.path.join(test_dir, f"{x}.jpg")
)
test_set = img_dataset(test_df_paths, transform=val_transformer, test=True)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 15
model_final.eval()
all_preds = []
with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        outputs = model_final(imgs)
        imgs_flipped = torch.flip(imgs, dims=[3])  # flip width dimension
        outputs_flipped = model_final(imgs_flipped)
        outputs = (outputs + outputs_flipped) / 2
        all_preds.append(outputs.cpu())
pred_tensor = torch.cat(all_preds, dim=0)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3304986610.py in <cell line: 0>()
----> 1 model_final.eval()
      2 all_preds = []
      3 with torch.no_grad():
      4     for imgs in test_loader:
      5         imgs = imgs.to(device, non_blocking=True)

NameError: name 'model_final' is not defined

## === cell 16
probabilities = F.softmax(pred_tensor, dim=1).numpy()

answer_ids = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")[
    "id"
].tolist()
submission_df = pd.DataFrame(probabilities, index=answer_ids)
submission_df.columns = [index_to_breed[i] for i in range(len(index_to_breed))]
submission_df.index.name = "id"
submission_path = "submission.csv"
submission_df.to_csv(submission_path)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2815313959.py in <cell line: 0>()
----> 1 probabilities = F.softmax(pred_tensor, dim=1).numpy()
      2 
      3 answer_ids = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")[
      4     "id"
      5 ].tolist()

NameError: name 'pred_tensor' is not defined
