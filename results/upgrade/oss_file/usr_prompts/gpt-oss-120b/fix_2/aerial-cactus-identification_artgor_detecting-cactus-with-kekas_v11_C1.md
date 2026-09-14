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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9996

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
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import subprocess, sys


def _install(pkg):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    except Exception:
        pass


for pkg in ["albumentations", "pretrainedmodels", "kekas", "adabound"]:
    _install(pkg)



## === cell 1
import albumentations
import pretrainedmodels
import adabound

from kekas import Keker, DataOwner, DataKek
from kekas.transformations import Transformer, to_torch, normalize
from kekas.metrics import accuracy
from kekas.modules import Flatten, AdaptiveConcatPool2d
from kekas.callbacks import Callback, Callbacks, DebuggerCallback
from kekas.utils import DotDict



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py in tf()
     41     try:
---> 42         from tensorboard.compat import notf  # noqa: F401
     43     except ImportError:

ImportError: cannot import name 'notf' from 'tensorboard.compat' (/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py)

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
labels = pd.read_csv("../input/train.csv")
fig = plt.figure(figsize=(25, 8))
train_imgs = os.listdir("../input/train/train")
for idx, img in enumerate(np.random.choice(train_imgs, 20)):
    ax = fig.add_subplot(4, 20 // 4, idx + 1, xticks=[], yticks=[])
    im = Image.open("../input/train/train/" + img)
    plt.imshow(im)
    lab = labels.loc[labels["id"] == img, "has_cactus"].values[0]
    ax.set_title(f"Label: {lab}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/729978083.py in <cell line: 0>()
      4 for idx, img in enumerate(np.random.choice(train_imgs, 20)):
      5     ax = fig.add_subplot(4, 20 // 4, idx + 1, xticks=[], yticks=[])
----> 6     im = Image.open("../input/train/train/" + img)
      7     plt.imshow(im)
      8     lab = labels.loc[labels["id"] == img, "has_cactus"].values[0]

NameError: name 'Image' is not defined

## === cell 3
test_img = os.listdir("../input/test/test")
test_df = pd.DataFrame(test_img, columns=["id"])
test_df["has_cactus"] = -1
test_df["data_type"] = "test"

labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"

labels.head()



## === cell 4
labels.loc[labels["data_type"] == "train", "has_cactus"].value_counts()



## === cell 5
train, valid = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=42
)




## === cell 6
def reader_fn(i, row):
    img_path = f"../input/{row['data_type']}/{row['data_type']}/{row['id']}"
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    image = image[:, :, ::-1]  # BGR -> RGB
    label = torch.Tensor([row["has_cactus"]])
    return {"image": image, "label": label}




## === cell 7
def augs(p=0.5):
    return albumentations.Compose(
        [
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.10, rotate_limit=15, p=0.75
            ),
            albumentations.HueSaturationValue(),
            albumentations.RandomBrightness(),
            albumentations.RandomContrast(),
        ],
        p=p,
    )




## === cell 8
def get_transforms(dataset_key, size, p):
    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    AUGS = Transformer(dataset_key, lambda x: augs()(image=x)["image"])
    NRM_TFMS = transforms.Compose(
        [Transformer(dataset_key, to_torch()), Transformer(dataset_key, normalize())]
    )
    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])
    return train_tfms, val_tfms




## === cell 9
train_tfms, val_tfms = get_transforms("image", 32, 0.5)



## === cell 10
train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 64
workers = 0

train_dl = DataLoader(
    train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True
)
val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)



## === cell 11
test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = DataLoader(test_dk, batch_size=batch_size, num_workers=workers, shuffle=False)




## === cell 12
class Net(nn.Module):
    def __init__(
        self,
        num_classes: int,
        p: float = 0.2,
        pooling_size: int = 2,
        last_conv_size: int = 1664,
        arch: str = "densenet169",
        pretrained: str = "imagenet",
    ) -> None:
        super().__init__()
        net = pretrainedmodels.__dict__[arch](pretrained=pretrained)
        modules = list(net.children())[:-1]  # remove final classifier
        modules += [
            nn.Sequential(
                Flatten(),
                nn.BatchNorm1d(last_conv_size),
                nn.Dropout(p),
                nn.Linear(last_conv_size, num_classes),
            )
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)




## === cell 13
dataowner = DataOwner(train_dl, val_dl, None)
model = Net(num_classes=1)
criterion = nn.BCEWithLogitsLoss()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
SSLCertVerificationError                  Traceback (most recent call last)
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
   1474 
-> 1475             self.sock = self._context.wrap_socket(self.sock,
   1476                                                   server_hostname=server_hostname)

/usr/lib/python3.11/ssl.py in wrap_socket(self, sock, server_side, do_handshake_on_connect, suppress_ragged_eofs, server_hostname, session)
    516         # ctx._wrap_socket()
--> 517         return self.sslsocket_class._create(
    518             sock=sock,

/usr/lib/python3.11/ssl.py in _create(cls, sock, server_side, do_handshake_on_connect, suppress_ragged_eofs, server_hostname, context, session)
   1103                         raise ValueError("do_handshake_on_connect should not be specified for non-blocking sockets")
-> 1104                     self.do_handshake()
   1105         except:

/usr/lib/python3.11/ssl.py in do_handshake(self, block)
   1381                 self.settimeout(None)
-> 1382             self._sslobj.do_handshake()
   1383         finally:

SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1016)

During handling of the above exception, another exception occurred:

URLError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3186603736.py in <cell line: 0>()
      1 dataowner = DataOwner(train_dl, val_dl, None)
----> 2 model = Net(num_classes=1)
      3 criterion = nn.BCEWithLogitsLoss()
      4 
      5 

/tmp/ipykernel_55/2846131373.py in __init__(self, num_classes, p, pooling_size, last_conv_size, arch, pretrained)
     10     ) -> None:
     11         super().__init__()
---> 12         net = pretrainedmodels.__dict__[arch](pretrained=pretrained)
     13         modules = list(net.children())[:-1]  # remove final classifier
     14         modules += [

/usr/local/lib/python3.11/dist-packages/pretrainedmodels/models/torchvision_models.py in densenet169(num_classes, pretrained)
    223     if pretrained is not None:
    224         settings = pretrained_settings['densenet169'][pretrained]
--> 225         model = load_pretrained(model, num_classes, settings)
    226     model = modify_densenets(model)
    227     return model

/usr/local/lib/python3.11/dist-packages/pretrainedmodels/models/torchvision_models.py in load_pretrained(model, num_classes, settings)
    116         "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    117 
--> 118     state_dict = model_zoo.load_url(settings['url'])
    119     state_dict = update_state_dict(state_dict)
    120     model.load_state_dict(state_dict)

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
    523         for processor in self.process_response.get(protocol, []):
    524             meth = getattr(processor, meth_name)
--> 525             response = meth(req, response)
    526 
    527         return response

/usr/lib/python3.11/urllib/request.py in http_response(self, request, response)
    632         # request was successfully received, understood, and accepted.
    633         if not (200 <= code < 300):
--> 634             response = self.parent.error(
    635                 'http', request, response, code, msg, hdrs)
    636 

/usr/lib/python3.11/urllib/request.py in error(self, proto, *args)
    555             http_err = 0
    556         args = (dict, proto, meth_name) + args
--> 557         result = self._call_chain(*args)
    558         if result:
    559             return result

/usr/lib/python3.11/urllib/request.py in _call_chain(self, chain, kind, meth_name, *args)
    494         for handler in handlers:
    495             func = getattr(handler, meth_name)
--> 496             result = func(*args)
    497             if result is not None:
    498                 return result

/usr/lib/python3.11/urllib/request.py in http_error_302(self, req, fp, code, msg, headers)
    747         fp.close()
    748 
--> 749         return self.parent.open(new, timeout=req.timeout)
    750 
    751     http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302

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

URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1016)>

## === cell 14
def step_fn(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    inp = batch["image"]
    return model(inp)




## === cell 15
def bce_accuracy(
    target: torch.Tensor, preds: torch.Tensor, thresh: bool = 0.5
) -> float:
    target_np = target.cpu().detach().numpy()
    pred_np = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)
    return accuracy_score(target_np, pred_np)


def roc_auc(target: torch.Tensor, preds: torch.Tensor) -> float:
    target_np = target.cpu().detach().numpy()
    pred_np = torch.sigmoid(preds).cpu().detach().numpy()
    return roc_auc_score(target_np, pred_np)




## === cell 16
keker = Keker(
    model=model,
    dataowner=dataowner,
    criterion=criterion,
    step_fn=step_fn,
    target_key="label",
    metrics={"acc": bce_accuracy, "auc": roc_auc},
    opt=torch.optim.SGD,
    opt_params={"momentum": 0.99},
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4229198289.py in <cell line: 0>()
      1 keker = Keker(
----> 2     model=model,
      3     dataowner=dataowner,
      4     criterion=criterion,
      5     step_fn=step_fn,

NameError: name 'model' is not defined

## === cell 17
keker.unfreeze(model_attr="net")
layer_num = -1
keker.freeze_to(layer_num, model_attr="net")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4222199681.py in <cell line: 0>()
----> 1 keker.unfreeze(model_attr="net")
      2 layer_num = -1
      3 keker.freeze_to(layer_num, model_attr="net")
      4 

NameError: name 'keker' is not defined

## === cell 18
keker.kek_one_cycle(
    max_lr=1e-2,
    cycle_len=4,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.3,
    logdir="train_logs",
)
keker.plot_kek("train_logs")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1947901615.py in <cell line: 0>()
----> 1 keker.kek_one_cycle(
      2     max_lr=1e-2,
      3     cycle_len=4,
      4     momentum_range=(0.95, 0.85),
      5     div_factor=25,

NameError: name 'keker' is not defined

## === cell 19
keker.kek_one_cycle(
    max_lr=1e-3,
    cycle_len=4,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.2,
    logdir="train_logs1",
)
keker.plot_kek("train_logs1")



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3157653000.py in <cell line: 0>()
----> 1 keker.kek_one_cycle(
      2     max_lr=1e-3,
      3     cycle_len=4,
      4     momentum_range=(0.95, 0.85),
      5     div_factor=25,

NameError: name 'keker' is not defined

## === cell 20
preds = keker.predict_loader(loader=test_dl)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3990093140.py in <cell line: 0>()
----> 1 preds = keker.predict_loader(loader=test_dl)
      2 

NameError: name 'keker' is not defined

## === cell 21
test_preds = pd.DataFrame(
    {
        "id": test_df.id.values,
        "has_cactus": preds.reshape(
            -1,
        ),
    }
)
test_preds.to_csv("sub.csv", index=False)
test_preds.head()

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2679691569.py in <cell line: 0>()
      2     {
      3         "id": test_df.id.values,
----> 4         "has_cactus": preds.reshape(
      5             -1,
      6         ),

NameError: name 'preds' is not defined
