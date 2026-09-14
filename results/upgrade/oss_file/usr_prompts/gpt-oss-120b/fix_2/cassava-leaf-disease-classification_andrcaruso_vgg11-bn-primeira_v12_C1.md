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

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.844061650045331

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'I fix the pipeline so it actually trains a model (instead of trying to load a missing checkpoint) and creates a proper validation split. Using a pretrained ResNet‑18 and a brief fine‑tune give a reasonable accuracy that moves the score toward the target, and the script now end with a valid submission.csv.'

# 9. Code solution

## === cell 0
path = "../input/cassava-leaf-disease-classification/"



## === cell 1
import numpy as np
import pandas as pd
from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback

df = pd.read_csv(path + "train.csv")
caminho = path + "train_images/"

dls = ImageDataLoaders.from_df(
    df, path=caminho, valid_pct=0.2, seed=42, item_tfms=Resize(224), bs=64
)



## === cell 2
import os

os.makedirs("/kaggle/working/models", exist_ok=True)



## === cell 3
learn = cnn_learner(
    dls,
    resnet18,
    metrics=[error_rate, accuracy],
    pretrained=True,
    path="/kaggle/working/models",
)

save_callback = SaveModelCallback(
    fname="best_model", monitor="valid_loss", with_opt=True, reset_on_fit=False
)

learn.fine_tune(3, cbs=[save_callback])



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
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
    862             if not all_errors:
--> 863                 raise exceptions[0]
    864             raise ExceptionGroup("create_connection failed", exceptions)

/usr/lib/python3.11/socket.py in create_connection(address, timeout, source_address, all_errors)
    847                 sock.bind(source_address)
--> 848             sock.connect(sa)
    849             # Break explicitly a reference cycle

OSError: [Errno 101] Network is unreachable

During handling of the above exception, another exception occurred:

URLError                                  Traceback (most recent call last)
/tmp/ipykernel_55/4067738067.py in <cell line: 0>()
      1 # build a learner with a pretrained ResNet‑18; use the models folder for checkpoints
----> 2 learn = cnn_learner(
      3     dls,
      4     resnet18,
      5     metrics=[error_rate, accuracy],

/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py in cnn_learner(*args, **kwargs)
    302     "Deprecated name for `vision_learner` -- do not use"
    303     warn("`cnn_learner` has been renamed to `vision_learner` -- please update your code")
--> 304     return vision_learner(*args, **kwargs)
    305 
    306 # %% ../../nbs/21_vision.learner.ipynb 62

/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py in vision_learner(dls, arch, normalize, n_out, pretrained, weights, loss_func, opt_func, lr, splitter, cbs, metrics, path, model_dir, wd, wd_bn_bias, train_bn, moms, cut, init, custom_head, concat_pool, pool, lin_ftrs, ps, first_bn, bn_final, lin_first, y_range, **kwargs)
    236     else:
    237         if normalize: _add_norm(dls, meta, pretrained, n_in)
--> 238         model = create_vision_model(arch, n_out, pretrained=pretrained, weights=weights, **model_args)
    239 
    240     splitter = ifnone(splitter, meta['split'])

/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py in create_vision_model(arch, n_out, pretrained, weights, cut, n_in, init, custom_head, concat_pool, pool, lin_ftrs, ps, first_bn, bn_final, lin_first, y_range)
    170         if weights is not None and not pretrained:
    171             warn(f'{pretrained=} but `weights` are set {weights=}. To randomly initialize set `pretrained=False` & `weights=None`')
--> 172         model = arch(weights=meta['weights'] if (weights is None and pretrained) else weights)
    173     else:
    174         model = arch(pretrained=pretrained)

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

URLError: <urlopen error [Errno 101] Network is unreachable>

## === cell 4
img = dls.train_ds[0][0]
learn.predict(img)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2722585898.py in <cell line: 0>()
      1 # sanity‑check prediction on a single training image
      2 img = dls.train_ds[0][0]
----> 3 learn.predict(img)
      4 

NameError: name 'learn' is not defined

## === cell 5
submission_df = pd.read_csv(path + "sample_submission.csv")
caminho_test = path + "test_images"

tst_dl = learn.dls.test_dl(
    submission_df["image_id"].apply(lambda x: os.path.join(caminho_test, x))
)

preds, _ = learn.tta(dl=tst_dl, n=10)

submission_df["label"] = np.argmax(preds, axis=1)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1326115931.py in <cell line: 0>()
      3 
      4 # build a test dataloader directly from the image filenames
----> 5 tst_dl = learn.dls.test_dl(
      6     submission_df["image_id"].apply(lambda x: os.path.join(caminho_test, x))
      7 )

NameError: name 'learn' is not defined

## === cell 6
submission_df.to_csv("submission.csv", index=False)
