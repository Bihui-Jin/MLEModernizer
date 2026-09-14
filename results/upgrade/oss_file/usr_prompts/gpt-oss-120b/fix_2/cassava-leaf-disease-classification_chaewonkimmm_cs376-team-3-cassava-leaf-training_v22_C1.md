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

0.7716832880024177

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm



## === cell 1
torch.cuda.is_available()



## === cell 2
df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.1, random_state=42, stratify=df.label.values
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3738301041.py in <cell line: 0>()
      1 # Load training metadata and create a stratified split
      2 df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
----> 3 train_df, valid_df = model_selection.train_test_split(
      4     df, test_size=0.1, random_state=42, stratify=df.label.values
      5 )

NameError: name 'model_selection' is not defined

## === cell 3
train_img_dir = "../input/cassava-leaf-disease-classification/train_images/"

train_paths = [os.path.join(train_img_dir, img) for img in train_df.image_id.values]
valid_paths = [os.path.join(train_img_dir, img) for img in valid_df.image_id.values]

train_labels = train_df.label.values
valid_labels = valid_df.label.values




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1117541033.py in <cell line: 0>()
      2 train_img_dir = "../input/cassava-leaf-disease-classification/train_images/"
      3 
----> 4 train_paths = [os.path.join(train_img_dir, img) for img in train_df.image_id.values]
      5 valid_paths = [os.path.join(train_img_dir, img) for img in valid_df.image_id.values]
      6 

NameError: name 'train_df' is not defined

## === cell 4
class CassavaDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
        self.input_size = 512
        self.imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop((self.input_size, self.input_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(*self.imagenet_stats),
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.targets[idx]
        return image, label




## === cell 5
train_dataset = CassavaDataset(train_paths, train_labels)
valid_dataset = CassavaDataset(valid_paths, valid_labels)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/631074473.py in <cell line: 0>()
      1 # Dataset instances (no extra external transforms)
----> 2 train_dataset = CassavaDataset(train_paths, train_labels)
      3 valid_dataset = CassavaDataset(valid_paths, valid_labels)
      4 

NameError: name 'train_paths' is not defined

## === cell 6
batch_size = 16
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/961726216.py in <cell line: 0>()
      1 batch_size = 16
      2 train_loader = DataLoader(
----> 3     train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
      4 )
      5 valid_loader = DataLoader(

NameError: name 'train_dataset' is not defined

## === cell 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## === cell 8
resnet = models.resnet34(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 5)  # 5 disease classes
resnet = resnet.to(device)



## --- ERROR in cell 8, traceback:
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
/tmp/ipykernel_55/723916779.py in <cell line: 0>()
      1 # Load a pretrained ResNet34 and adapt the final layer
----> 2 resnet = models.resnet34(pretrained=True)
      3 num_ftrs = resnet.fc.in_features
      4 resnet.fc = nn.Linear(num_ftrs, 5)  # 5 disease classes
      5 resnet = resnet.to(device)

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

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in resnet34(weights, progress, **kwargs)
    729     weights = ResNet34_Weights.verify(weights)
    730 
--> 731     return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)
    732 
    733 

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

## === cell 9
criterion = nn.CrossEntropyLoss()
resnet.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in tqdm(valid_loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f"Validation accuracy (untrained): {100 * correct / total:.2f}%")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2665437651.py in <cell line: 0>()
      1 # (Optional) quick validation to show that the model runs
      2 criterion = nn.CrossEntropyLoss()
----> 3 resnet.eval()
      4 correct, total = 0, 0
      5 with torch.no_grad():

NameError: name 'resnet' is not defined

## === cell 10
test_path = "../input/cassava-leaf-disease-classification/test_images/"
test_files = [f for f in os.listdir(test_path) if f.lower().endswith(".jpg")]
test_imgs = [os.path.join(test_path, f) for f in test_files]

resnet.eval()
test_preds = []
with torch.no_grad():
    for img_file in tqdm(test_imgs, desc="Test inference"):
        img = Image.open(img_file).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = transform(img).unsqueeze(0).to(device)
        out = resnet(img_tensor)
        _, pred = torch.max(out, 1)
        test_preds.append(int(pred.item()))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/271960514.py in <cell line: 0>()
      4 test_imgs = [os.path.join(test_path, f) for f in test_files]
      5 
----> 6 resnet.eval()
      7 test_preds = []
      8 with torch.no_grad():

NameError: name 'resnet' is not defined

## === cell 11
submission = pd.DataFrame({"image_id": test_files, "label": test_preds})
submission.to_csv("submission.csv", index=False)
print('Submission file "submission.csv" created with', len(submission), "rows.")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/813334084.py in <cell line: 0>()
      1 # Build and save the submission file
----> 2 submission = pd.DataFrame({"image_id": test_files, "label": test_preds})
      3 submission.to_csv("submission.csv", index=False)
      4 print('Submission file "submission.csv" created with', len(submission), "rows.")

NameError: name 'test_preds' is not defined
