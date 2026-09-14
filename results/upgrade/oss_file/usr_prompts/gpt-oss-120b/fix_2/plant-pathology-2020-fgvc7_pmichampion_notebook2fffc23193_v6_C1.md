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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.12

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sentence-transformers==4.1.0
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
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.50635

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.special import softmax
import cv2
from transformers import get_cosine_schedule_with_warmup
from tqdm.notebook import tqdm

from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    RandomBrightnessContrast,
    OneOf,
    Sharpen,
    Blur,
    Resize,
    Normalize,
)
from albumentations.pytorch import ToTensorV2
import gc
import os



## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
Img_folder = "/kaggle/input/plant-pathology-2020-fgvc7/images/"


def get_path_of_img(filename):
    return Img_folder + filename + ".jpg"


train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
train.head()



## === cell 3
train["image_path"] = train["image_id"].apply(get_path_of_img)
test["image_path"] = test["image_id"].apply(get_path_of_img)



## === cell 4
from sklearn.model_selection import train_test_split

train_targets = train.loc[:, "healthy":"scab"]
train_paths = train.image_path
test_paths = test.image_path



## === cell 5
train_paths, valid_paths, train_targets, valid_targets = train_test_split(
    train_paths,
    train_targets,
    test_size=0.2,
    random_state=27,
    shuffle=True,
    stratify=train_targets,
)




## === cell 6
class Leaf_Dataset(Data.Dataset):
    def __init__(self, image_paths, labels=None, test=False, train=True):
        self.paths = image_paths.reset_index(drop=True)
        self.test = test
        self.train = train
        if not self.test:
            self.labels = labels.reset_index(drop=True)
        self.train_transform = Compose(
            [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(rotate_limit=25.0, p=0.7),
                RandomBrightnessContrast(
                    p=0.7, brightness_limit=0.2, contrast_limit=0.2
                ),
                OneOf([Sharpen(p=1), Blur(p=1)], p=0.5),
                Resize(height=224, width=224),
            ]
        )
        self.test_transform = Compose(
            [Resize(height=224, width=224)]  # ensure size matches model input
        )
        self.default_transform = Compose(
            [
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, item):
        img = cv2.imread(self.paths[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.test:
            label = torch.tensor(np.argmax(self.labels.loc[item].values))

        if self.train:
            img = self.train_transform(image=img)["image"]
        else:
            img = self.test_transform(image=img)["image"]

        img = self.default_transform(image=img)["image"]

        if not self.test:
            return img, label
        return img




## === cell 7
class CFG:
    batch_size = 8
    num_epochs = 30
    train_size = train_targets.shape[0]
    valid_size = valid_targets.shape[0]
    model_name = "ResNet18"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 8e-4




## === cell 8
train_targets.reset_index(drop=True, inplace=True)
train_paths = train_paths.reset_index(drop=True)
valid_targets.reset_index(drop=True, inplace=True)
valid_paths = valid_paths.reset_index(drop=True)



## === cell 9
train_dataset = Leaf_Dataset(train_paths, labels=train_targets, test=False, train=True)
train_loader = Data.DataLoader(
    train_dataset, shuffle=True, batch_size=CFG.batch_size, num_workers=2
)

valid_dataset = Leaf_Dataset(valid_paths, labels=valid_targets, test=False, train=False)
valid_loader = Data.DataLoader(
    valid_dataset, shuffle=False, batch_size=CFG.batch_size, num_workers=2
)

test_dataset = Leaf_Dataset(test_paths, test=True, train=False)
test_loader = Data.DataLoader(
    test_dataset, shuffle=False, batch_size=CFG.batch_size, num_workers=2
)



## === cell 10
from torchvision.models import resnet18

model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1000, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1000, 4, bias=True),
)
model.to(CFG.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=0.001)

num_train_steps = int(len(train_dataset) / CFG.batch_size * CFG.num_epochs)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(len(train_dataset) / CFG.batch_size * 5),
    num_training_steps=num_train_steps,
)

loss_fn = nn.CrossEntropyLoss()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
gaierror                                  Traceback (most recent call last)
/usr/lib/python3.11/urllib/request.py in do_open(self, http_class, req, **http_conn_args)
   1347             try:
-> 1348                 h.request(req.get_method(), req.selector, req.data, headers,
   1349                           encode_chunked=req.has_header('Transfer-encoding'))

/usr/lib/python3.11/http/client.py in request(self, method, url, body, headers, encode_chunked)
   1302         """Send a complete request to the server."""
-> 1303         self._send_request(method, url, body, headers, encode_chunked)
   1304 

/usr/lib/python3.11/http/client.py in _send_request(self, method, url, body, headers, encode_chunked)
   1348             body = _encode(body, 'body')
-> 1349         self.endheaders(body, encode_chunked=encode_chunked)
   1350 

/usr/lib/python3.11/http/client.py in endheaders(self, message_body, encode_chunked)
   1297             raise CannotSendHeader()
-> 1298         self._send_output(message_body, encode_chunked=encode_chunked)
   1299 

/usr/lib/python3.11/http/client.py in _send_output(self, message_body, encode_chunked)
   1057         del self._buffer[:]
-> 1058         self.send(msg)
   1059 

/usr/lib/python3.11/http/client.py in send(self, data)
    995             if self.auto_open:
--> 996                 self.connect()
    997             else:

/usr/lib/python3.11/http/client.py in connect(self)
   1467 
-> 1468             super().connect()
   1469 

/usr/lib/python3.11/http/client.py in connect(self)
    961         sys.audit("http.client.connect", self, self.host, self.port)
--> 962         self.sock = self._create_connection(
    963             (self.host,self.port), self.timeout, self.source_address)

/usr/lib/python3.11/socket.py in create_connection(address, timeout, source_address, all_errors)
    838     exceptions = []
--> 839     for res in getaddrinfo(host, port, 0, SOCK_STREAM):
    840         af, socktype, proto, canonname, sa = res

/usr/lib/python3.11/socket.py in getaddrinfo(host, port, family, type, proto, flags)
    973     addrlist = []
--> 974     for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
    975         af, socktype, proto, canonname, sa = res

gaierror: [Errno -3] Temporary failure in name resolution

During handling of the above exception, another exception occurred:

URLError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3519920412.py in <cell line: 0>()
      1 from torchvision.models import resnet18
      2 
----> 3 model = resnet18(pretrained=True)
      4 num_ftrs = model.fc.in_features
      5 model.fc = nn.Sequential(

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

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in resnet18(weights, progress, **kwargs)
    703     weights = ResNet18_Weights.verify(weights)
    704 
--> 705     return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
    706 
    707 

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
   1349                           encode_chunked=req.has_header('Transfer-encoding'))
   1350             except OSError as err: # timeout error
-> 1351                 raise URLError(err)
   1352             r = h.getresponse()
   1353         except:

URLError: <urlopen error [Errno -3] Temporary failure in name resolution>

## === cell 11
def train_fn(net, loader):
    running_loss = 0.0
    model_predictions = np.empty((0,))
    accuracy_labels = np.empty((0,))
    pbar = tqdm(total=len(loader), desc="Training")
    net.train()
    for _, (images, labels) in enumerate(loader):
        images, labels = images.to(CFG.device), labels.to(CFG.device)
        optimizer.zero_grad()
        preds = net(images)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * labels.size(0)
        accuracy_labels = np.concatenate((accuracy_labels, labels.cpu().numpy()), 0)
        model_predictions = np.concatenate(
            (model_predictions, np.argmax(preds.cpu().detach().numpy(), 1)), 0
        )
        pbar.update()
    pbar.close()
    accuracy = accuracy_score(accuracy_labels, model_predictions)
    return running_loss / CFG.train_size, accuracy


def valid_fn(net, loader):
    running_loss = 0.0
    model_predictions = np.empty((0,))
    accuracy_labels = np.empty((0,))
    pbar = tqdm(total=len(loader), desc="Validation")
    net.eval()
    with torch.no_grad():
        for _, (images, labels) in enumerate(loader):
            images, labels = images.to(CFG.device), labels.to(CFG.device)
            preds = net(images)
            loss = loss_fn(preds, labels)
            running_loss += loss.item() * labels.size(0)
            accuracy_labels = np.concatenate((accuracy_labels, labels.cpu().numpy()), 0)
            model_predictions = np.concatenate(
                (model_predictions, np.argmax(preds.cpu().detach().numpy(), 1)), 0
            )
            pbar.update()
    pbar.close()
    accuracy = accuracy_score(accuracy_labels, model_predictions)
    conf_matrix = confusion_matrix(accuracy_labels, model_predictions)
    return running_loss / CFG.valid_size, accuracy, conf_matrix


def test_fn(net, loader):
    preds_for_output = np.zeros((1, 4))
    net.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(loader), desc="Testing")
        for _, images in enumerate(loader):
            images = images.to(CFG.device)
            preds = net(images)
            preds_for_output = np.concatenate(
                (preds_for_output, preds.cpu().detach().numpy()), 0
            )
            pbar.update()
        pbar.close()
    return preds_for_output




## === cell 12
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []



## === cell 13
best_valid_loss = float("inf")
patience = 3
trigger_times = 0

for epoch in range(6):
    tl, ta = train_fn(model, loader=train_loader)
    vl, va, conf_matrix = valid_fn(model, loader=valid_loader)

    train_loss.append(tl)
    valid_loss.append(vl)
    train_acc.append(ta)
    valid_acc.append(va)

    print(
        f"Epoch {epoch+1}: Train loss {tl:.4f}, acc {ta:.4f} | Valid loss {vl:.4f}, acc {va:.4f}"
    )

    if vl < best_valid_loss:
        best_valid_loss = vl
        torch.save(model.state_dict(), "best_model.pt")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/642727248.py in <cell line: 0>()
      4 
      5 for epoch in range(6):
----> 6     tl, ta = train_fn(model, loader=train_loader)
      7     vl, va, conf_matrix = valid_fn(model, loader=valid_loader)
      8 

NameError: name 'model' is not defined

## === cell 14
plt.figure()
plt.ylim(0, 1.5)
sns.lineplot(x=list(range(len(train_loss))), y=train_loss, label="Train Loss")
sns.lineplot(x=list(range(len(valid_loss))), y=valid_loss, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



## === cell 15
model.load_state_dict(torch.load("best_model.pt", map_location=CFG.device))

out = test_fn(model, test_loader)
output = pd.DataFrame(
    softmax(out, axis=1), columns=["healthy", "multiple_diseases", "rust", "scab"]
)
output = output.drop(0).reset_index(drop=True)  # remove the initial zero row
output["image_id"] = test["image_id"].values
output = output[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]
output.to_csv("submission.csv", index=False)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1576090937.py in <cell line: 0>()
      1 # Load best model for inference
----> 2 model.load_state_dict(torch.load("best_model.pt", map_location=CFG.device))
      3 
      4 out = test_fn(model, test_loader)
      5 output = pd.DataFrame(

NameError: name 'model' is not defined
