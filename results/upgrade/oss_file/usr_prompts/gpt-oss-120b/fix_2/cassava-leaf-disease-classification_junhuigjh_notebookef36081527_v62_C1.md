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

3.13

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

0.8756421879721971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



## === cell 1
base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_img_dir = os.path.join(base_path, "train_images")
test_img_dir = os.path.join(base_path, "test_images")

train_df = pd.read_csv(train_csv_path)
train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(train_img_dir, x)
)


class CassavaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["image_path"].values
        self.labels = df["label"].values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label


train_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    stratify=train_df["label"],
    random_state=seed,
)

train_subset = train_df.iloc[train_idx].reset_index(drop=True)
val_subset = train_df.iloc[val_idx].reset_index(drop=True)

train_dataset = CassavaDataset(train_subset, transform=train_transform)
val_dataset = CassavaDataset(val_subset, transform=val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


def train_one_epoch(loader):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == lbls).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total




## --- ERROR in cell 2, traceback:
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
/tmp/ipykernel_55/1865400571.py in <cell line: 0>()
      1 # Model: pretrained ResNet18 modified for 5 classes
      2 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
----> 3 model = models.resnet18(pretrained=True)
      4 model.fc = nn.Linear(model.fc.in_features, 5)
      5 model = model.to(device)

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

## === cell 3
epochs = 3
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    print(
        f"Epoch {epoch}/{epochs} | "
        f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
    )



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2282874846.py in <cell line: 0>()
      2 epochs = 3
      3 for epoch in range(1, epochs + 1):
----> 4     train_loss, train_acc = train_one_epoch(train_loader)
      5     val_loss, val_acc = evaluate(val_loader)
      6     print(

NameError: name 'train_one_epoch' is not defined

## === cell 4
test_image_ids = sorted(os.listdir(test_img_dir))
test_paths = [os.path.join(test_img_dir, img_id) for img_id in test_image_ids]


class TestDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.paths[idx]  # return path for later ID extraction


test_dataset = TestDataset(test_paths, transform=val_transform)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

model.eval()
predictions = []
with torch.no_grad():
    for imgs, paths in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

submission = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/950885579.py in <cell line: 0>()
     24 )
     25 
---> 26 model.eval()
     27 predictions = []
     28 with torch.no_grad():

NameError: name 'model' is not defined
