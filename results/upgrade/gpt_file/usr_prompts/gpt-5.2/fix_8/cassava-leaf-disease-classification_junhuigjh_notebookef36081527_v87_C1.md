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

0.8869749168933212

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import vit_h_14, efficientnet_v2_l
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

SEED = 11
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

Image.MAX_IMAGE_PIXELS = None
try:
    Image.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass

try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass

torch.cuda.empty_cache()




## === cell 1
def invert_square_pad(img: Image.Image) -> Image.Image:
    w, h = img.size

    dx, dy = w // 2, h // 2
    if dx or dy:
        img2 = Image.new(img.mode, (w, h))
        img2.paste(img.crop((dx, dy, w, h)), (0, 0))
        img2.paste(img.crop((0, dy, dx, h)), (w - dx, 0))
        img2.paste(img.crop((dx, 0, w, dy)), (0, h - dy))
        img2.paste(img.crop((0, 0, dx, dy)), (w - dx, h - dy))
    else:
        img2 = img

    max_side = max(w, h)
    pad_left = (max_side - w) // 2
    pad_top = (max_side - h) // 2
    pad_right = (max_side - w) - pad_left
    pad_bottom = (max_side - h) - pad_top
    padding = (pad_left, pad_top, pad_right, pad_bottom)

    return transforms.functional.pad(img2, padding, padding_mode="reflect")


torch_transforms_VIT = transforms.Compose(
    [
        v2.Lambda(invert_square_pad),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((518, 518)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

torch_transforms_EfficientNet = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((480, 480)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)




## === cell 2
NUM_CLASSES = 5


class Head(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


model_vit = vit_h_14(weights="DEFAULT")
vit_in = model_vit.heads.head.in_features
model_vit.heads.head = Head(vit_in, NUM_CLASSES)
model_vit.to(device).eval()

model_eff = efficientnet_v2_l(weights="DEFAULT")
eff_in = model_eff.classifier[-1].in_features
model_eff.classifier[-1] = nn.Linear(eff_in, NUM_CLASSES)
model_eff.to(device).eval()

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    if hasattr(torch, "compile"):
        model_vit = torch.compile(model_vit, mode="reduce-overhead")
        model_eff = torch.compile(model_eff, mode="reduce-overhead")
except Exception:
    pass

print("Backbones ready:", type(model_vit).__name__, type(model_eff).__name__)




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
/tmp/ipykernel_55/2646014134.py in <cell line: 0>()
     11 
     12 
---> 13 model_vit = vit_h_14(weights="DEFAULT")
     14 vit_in = model_vit.heads.head.in_features
     15 model_vit.heads.head = Head(vit_in, NUM_CLASSES)

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

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in vit_h_14(weights, progress, **kwargs)
    775     weights = ViT_H_14_Weights.verify(weights)
    776 
--> 777     return _vision_transformer(
    778         patch_size=14,
    779         num_layers=32,

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in _vision_transformer(patch_size, num_layers, num_heads, hidden_dim, mlp_dim, weights, progress, **kwargs)
    333 
    334     if weights:
--> 335         model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    336 
    337     return model

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
def _load_rgb_pil(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class _PILCache:
    __slots__ = ("_cache",)

    def __init__(self):
        self._cache = {}

    def get(self, path: str) -> Image.Image:
        im = self._cache.get(path)
        if im is None:
            im = _load_rgb_pil(path)
            self._cache[path] = im
        return im


class CassavaTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, tfm_vit, tfm_eff):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm_vit = tfm_vit
        self.tfm_eff = tfm_eff

        self.image_ids = self.df["image_id"].to_list()
        self.labels = self.df["label"].to_numpy(dtype=np.int64)

        self._pil_cache = _PILCache()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        label = int(self.labels[i])
        path = os.path.join(self.img_dir, image_id)

        img = self._pil_cache.get(path)

        x_vit = self.tfm_vit(img)
        x_eff = self.tfm_eff(img)
        return x_vit, x_eff, label

    def __getitems__(self, indices):
        out = []
        for i in indices:
            out.append(self.__getitem__(i))
        return out


class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, img_dir: str, tfm_vit, tfm_eff):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir
        self.tfm_vit = tfm_vit
        self.tfm_eff = tfm_eff

        self._pil_cache = _PILCache()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        path = os.path.join(self.img_dir, image_id)

        img = self._pil_cache.get(path)

        x_vit = self.tfm_vit(img)
        x_eff = self.tfm_eff(img)
        return x_vit, x_eff, image_id

    def __getitems__(self, indices):
        out = []
        for i in indices:
            out.append(self.__getitem__(i))
        return out


@torch.inference_mode()
def extract_logits(dataloader: DataLoader, model_vit: nn.Module, model_eff: nn.Module):
    n = len(dataloader.dataset)
    feats = np.empty((n, 2 * NUM_CLASSES), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int64)

    offset = 0
    for x_vit, x_eff, y in dataloader:
        bs = x_vit.shape[0]

        x_vit = x_vit.to(device, non_blocking=True)
        x_eff = x_eff.to(device, non_blocking=True)

        if hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()
        out_v = model_vit(x_vit)

        if hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()
        out_e = model_eff(x_eff)

        comb = torch.cat([out_e, out_v], dim=1)  # (B, 10)

        comb_np = comb.detach().to("cpu", non_blocking=PIN).float().numpy()
        np.copyto(feats[offset : offset + bs], comb_np, casting="no")
        labels[offset : offset + bs] = np.asarray(y, dtype=np.int64)
        offset += bs

    return feats, labels


@torch.inference_mode()
def extract_logits_test(
    dataloader: DataLoader, model_vit: nn.Module, model_eff: nn.Module
):
    n = len(dataloader.dataset)
    feats = np.empty((n, 2 * NUM_CLASSES), dtype=np.float32)
    ids = [None] * n

    offset = 0
    for x_vit, x_eff, image_id in dataloader:
        bs = x_vit.shape[0]

        x_vit = x_vit.to(device, non_blocking=True)
        x_eff = x_eff.to(device, non_blocking=True)

        if hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()
        out_v = model_vit(x_vit)

        if hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()
        out_e = model_eff(x_eff)

        comb = torch.cat([out_e, out_v], dim=1)

        comb_np = comb.detach().to("cpu", non_blocking=PIN).float().numpy()
        np.copyto(feats[offset : offset + bs], comb_np, casting="no")
        ids[offset : offset + bs] = list(image_id)
        offset += bs

    return feats, ids




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
assert set(train_df.columns) == {"image_id", "label"}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_feats = np.zeros((len(train_df), 2 * NUM_CLASSES), dtype=np.float32)
oof_labels = train_df["label"].to_numpy(dtype=np.int64)

BATCH = 24 if torch.cuda.is_available() else 8

NUM_WORKERS = min(6, max(2, (os.cpu_count() or 2) - 2))
PIN = torch.cuda.is_available()

full_train_ds = CassavaTrainDataset(
    train_df, TRAIN_IMG_DIR, torch_transforms_VIT, torch_transforms_EfficientNet
)

loader_kwargs = dict(
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=(NUM_WORKERS > 0),
)
if NUM_WORKERS > 0:
    loader_kwargs["prefetch_factor"] = 4
    try:
        loader_kwargs["multiprocessing_context"] = "fork"
    except Exception:
        pass
if PIN:
    loader_kwargs["pin_memory_device"] = "cuda"

full_train_loader = DataLoader(full_train_ds, **loader_kwargs)

full_train_feats, full_train_y = extract_logits(full_train_loader, model_vit, model_eff)
assert np.array_equal(full_train_y, oof_labels), "Train feature/label order mismatch."

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(train_df["image_id"], train_df["label"]), start=1
):
    oof_feats[va_idx] = full_train_feats[va_idx]
    print(f"Fold {fold}: assigned {len(va_idx)} val rows from precomputed features")

decision_tree = RandomForestClassifier(
    n_estimators=30, criterion="gini", max_depth=8, random_state=SEED, n_jobs=-1
)
decision_tree.fit(oof_feats, oof_labels)
print("Meta-model trained on OOF features:", oof_feats.shape)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2990231468.py in <cell line: 0>()
     37 full_train_loader = DataLoader(full_train_ds, **loader_kwargs)
     38 
---> 39 full_train_feats, full_train_y = extract_logits(full_train_loader, model_vit, model_eff)
     40 assert np.array_equal(full_train_y, oof_labels), "Train feature/label order mismatch."
     41 

NameError: name 'model_vit' is not defined

## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sample_sub["image_id"].tolist()

test_ds = CassavaTestDataset(
    test_image_ids, TEST_IMG_DIR, torch_transforms_VIT, torch_transforms_EfficientNet
)
test_loader = DataLoader(test_ds, **loader_kwargs)

test_feats, out_ids = extract_logits_test(test_loader, model_vit, model_eff)
assert (
    out_ids == test_image_ids
), "Test ID order mismatch; submission would be misaligned."

prediction = decision_tree.predict(test_feats).astype(int)

submission = pd.DataFrame({"image_id": out_ids, "label": prediction})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote:", os.path.abspath("submission.csv"), "rows:", len(submission))
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0

_check = pd.read_csv("submission.csv")
assert list(_check.columns) == ["image_id", "label"]
assert len(_check) == len(sample_sub)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/47146036.py in <cell line: 0>()
      7 test_loader = DataLoader(test_ds, **loader_kwargs)
      8 
----> 9 test_feats, out_ids = extract_logits_test(test_loader, model_vit, model_eff)
     10 assert (
     11     out_ids == test_image_ids

NameError: name 'model_vit' is not defined
