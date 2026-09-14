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

3.12

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
sklearn-pandas==2.2.0
timm==1.0.19
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

0.8900593895168976

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I make the script robust by handling missing model checkpoint files: it fall back to ImageNet‑pretrained weights, skip any model that cannot be loaded, recompute the ensemble weights accordingly, and guard against empty output lists. This fixes the runtime errors and guarantees a valid `submission.csv` is written, while preserving the original ensemble logic as much as possible.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import warnings




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_paths = {
    "resnet18": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/resnet18(WD_1e-3)_aptos.pth",
    "efficientnet_b1": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b1.pth",
    "efficientnet_b2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b2.pth",
    "efficientnet_b3": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/efficentNet_b3.pth",
    "inception_resnet_v2": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/inception_resnet_v2.pth",
    "inception_v4": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/inception_v4.pth",
    "seresnext50_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext50_32x4d.pth",
    "seresnext101_32x4d": "/kaggle/input/aptos_ensamble-models/pytorch/ensamble_v2/2/seresnext101_32x4d.pth",
}
model_names = {
    "resnet18": "resnet18",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.9127,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
}
models_list = []
loaded_keys = []

for model_key, path in model_paths.items():
    model_name = model_names.get(model_key)
    if model_name is None:
        continue
    model = timm.create_model(model_name, pretrained=True, num_classes=5)
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        warnings.warn(
            f"Checkpoint not found for {model_key}. Using ImageNet pretrained weights."
        )
    except Exception as e:
        warnings.warn(
            f"Error loading {model_key}: {e}. Using ImageNet pretrained weights."
        )
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_keys.append(model_key)

total_score = sum(validation_scores[k] for k in loaded_keys)
weights = {k: validation_scores[k] / total_score for k in loaded_keys}



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
gaierror                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/urllib3/connection.py in _new_conn(self)
    197         try:
--> 198             sock = connection.create_connection(
    199                 (self._dns_host, self.port),

/usr/local/lib/python3.11/dist-packages/urllib3/util/connection.py in create_connection(address, timeout, source_address, socket_options)
     59 
---> 60     for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
     61         af, socktype, proto, canonname, sa = res

/usr/lib/python3.11/socket.py in getaddrinfo(host, port, family, type, proto, flags)
    973     addrlist = []
--> 974     for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
    975         af, socktype, proto, canonname, sa = res

gaierror: [Errno -3] Temporary failure in name resolution

The above exception was the direct cause of the following exception:

NameResolutionError                       Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, preload_content, decode_content, **response_kw)
    786             # Make the request on the HTTPConnection object
--> 787             response = self._make_request(
    788                 conn,

/usr/local/lib/python3.11/dist-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, body, headers, retries, timeout, chunked, response_conn, preload_content, decode_content, enforce_content_length)
    487                 new_e = _wrap_proxy_error(new_e, conn.proxy.scheme)
--> 488             raise new_e
    489 

/usr/local/lib/python3.11/dist-packages/urllib3/connectionpool.py in _make_request(self, conn, method, url, body, headers, retries, timeout, chunked, response_conn, preload_content, decode_content, enforce_content_length)
    463             try:
--> 464                 self._validate_conn(conn)
    465             except (SocketTimeout, BaseSSLError) as e:

/usr/local/lib/python3.11/dist-packages/urllib3/connectionpool.py in _validate_conn(self, conn)
   1092         if conn.is_closed:
-> 1093             conn.connect()
   1094 

/usr/local/lib/python3.11/dist-packages/urllib3/connection.py in connect(self)
    752             sock: socket.socket | ssl.SSLSocket
--> 753             self.sock = sock = self._new_conn()
    754             server_hostname: str = self.host

/usr/local/lib/python3.11/dist-packages/urllib3/connection.py in _new_conn(self)
    204         except socket.gaierror as e:
--> 205             raise NameResolutionError(self.host, self, e) from e
    206         except SocketTimeout as e:

NameResolutionError: <urllib3.connection.HTTPSConnection object at 0x7f46ae692350>: Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)

The above exception was the direct cause of the following exception:

MaxRetryError                             Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
    643         try:
--> 644             resp = conn.urlopen(
    645                 method=request.method,

/usr/local/lib/python3.11/dist-packages/urllib3/connectionpool.py in urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, preload_content, decode_content, **response_kw)
    840 
--> 841             retries = retries.increment(
    842                 method, url, error=new_e, _pool=self, _stacktrace=sys.exc_info()[2]

/usr/local/lib/python3.11/dist-packages/urllib3/util/retry.py in increment(self, method, url, response, error, _pool, _stacktrace)
    518             reason = error or ResponseError(cause)
--> 519             raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
    520 

MaxRetryError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /timm/resnet18.a1_in1k/resolve/main/pytorch_model.bin (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7f46ae692350>: Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)"))

During handling of the above exception, another exception occurred:

ConnectionError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _get_metadata_or_catch_error(repo_id, filename, repo_type, revision, endpoint, proxies, etag_timeout, headers, token, local_files_only, relative_filename, storage_folder)
   1542             try:
-> 1543                 metadata = get_hf_file_metadata(
   1544                     url=url, proxies=proxies, timeout=etag_timeout, headers=headers, token=token, endpoint=endpoint

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in get_hf_file_metadata(url, token, proxies, timeout, library_name, library_version, user_agent, headers, endpoint)
   1459     # Retrieve metadata
-> 1460     r = _request_wrapper(
   1461         method="HEAD",

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _request_wrapper(method, url, follow_relative_redirects, **params)
    282     if follow_relative_redirects:
--> 283         response = _request_wrapper(
    284             method=method,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _request_wrapper(method, url, follow_relative_redirects, **params)
    305     # Perform request and return if status_code is not in the retry list.
--> 306     response = http_backoff(method=method, url=url, **params)
    307     hf_raise_for_status(response)

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_http.py in http_backoff(method, url, max_retries, base_wait_time, max_wait_time, retry_on_exceptions, retry_on_status_codes, **kwargs)
    324             if nb_tries > max_retries:
--> 325                 raise err
    326 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_http.py in http_backoff(method, url, max_retries, base_wait_time, max_wait_time, retry_on_exceptions, retry_on_status_codes, **kwargs)
    305             # Perform request and return if status_code is not in the retry list.
--> 306             response = session.request(method=method, url=url, **kwargs)
    307             if response.status_code not in retry_on_status_codes:

/usr/local/lib/python3.11/dist-packages/requests/sessions.py in request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
    588         send_kwargs.update(settings)
--> 589         resp = self.send(prep, **send_kwargs)
    590 

/usr/local/lib/python3.11/dist-packages/requests/sessions.py in send(self, request, **kwargs)
    702         # Send the request
--> 703         r = adapter.send(request, **kwargs)
    704 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_http.py in send(self, request, *args, **kwargs)
     94         try:
---> 95             return super().send(request, *args, **kwargs)
     96         except requests.RequestException as e:

/usr/local/lib/python3.11/dist-packages/requests/adapters.py in send(self, request, stream, timeout, verify, cert, proxies)
    676 
--> 677             raise ConnectionError(e, request=request)
    678 

ConnectionError: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /timm/resnet18.a1_in1k/resolve/main/pytorch_model.bin (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7f46ae692350>: Failed to resolve \'huggingface.co\' ([Errno -3] Temporary failure in name resolution)"))'), '(Request ID: e0c76192-cdd6-4b5d-9da5-689d801edde3)')

The above exception was the direct cause of the following exception:

LocalEntryNotFoundError                   Traceback (most recent call last)
/tmp/ipykernel_55/1932616948.py in <cell line: 0>()
     38         continue
     39     # Use ImageNet‑pretrained weights as a safe default
---> 40     model = timm.create_model(model_name, pretrained=True, num_classes=5)
     41     try:
     42         state_dict = torch.load(path, map_location=device)

/usr/local/lib/python3.11/dist-packages/timm/models/_factory.py in create_model(model_name, pretrained, pretrained_cfg, pretrained_cfg_overlay, checkpoint_path, cache_dir, scriptable, exportable, no_jit, **kwargs)
    136     create_fn = model_entrypoint(model_name)
    137     with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
--> 138         model = create_fn(
    139             pretrained=pretrained,
    140             pretrained_cfg=pretrained_cfg,

/usr/local/lib/python3.11/dist-packages/timm/models/resnet.py in resnet18(pretrained, **kwargs)
   1442     """
   1443     model_args = dict(block=BasicBlock, layers=(2, 2, 2, 2))
-> 1444     return _create_resnet('resnet18', pretrained, **dict(model_args, **kwargs))
   1445 
   1446 

/usr/local/lib/python3.11/dist-packages/timm/models/resnet.py in _create_resnet(variant, pretrained, **kwargs)
    738         ResNet model instance.
    739     """
--> 740     return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)
    741 
    742 

/usr/local/lib/python3.11/dist-packages/timm/models/_builder.py in build_model_with_cfg(model_cls, variant, pretrained, pretrained_cfg, pretrained_cfg_overlay, model_cfg, feature_cfg, pretrained_strict, pretrained_filter_fn, cache_dir, kwargs_filter, **kwargs)
    455     num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    456     if pretrained:
--> 457         load_pretrained(
    458             model,
    459             pretrained_cfg=pretrained_cfg,

/usr/local/lib/python3.11/dist-packages/timm/models/_builder.py in load_pretrained(model, pretrained_cfg, num_classes, in_chans, filter_fn, strict, cache_dir)
    224                 state_dict = load_state_dict_from_hf(*pretrained_loc, cache_dir=cache_dir)
    225         else:
--> 226             state_dict = load_state_dict_from_hf(pretrained_loc, weights_only=True, cache_dir=cache_dir)
    227     elif load_from == 'local-dir':
    228         _logger.info(f'Loading pretrained weights from local directory ({pretrained_loc})')

/usr/local/lib/python3.11/dist-packages/timm/models/_hub.py in load_state_dict_from_hf(model_id, filename, weights_only, cache_dir)
    241 
    242     # Otherwise, load using pytorch.load
--> 243     cached_file = hf_hub_download(
    244         hf_model_id,
    245         filename=filename,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    112             kwargs = smoothly_deprecate_use_auth_token(fn_name=fn.__name__, has_token=has_token, kwargs=kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 
    116     return _inner_fn  # type: ignore

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1005         )
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination
   1009             cache_dir=cache_dir,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1112 
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 
   1116     # From now on, etag, commit_hash, url and size are not None.

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1656     else:
   1657         # Otherwise: most likely a connection issue or Hub downtime => let's warn the user
-> 1658         raise LocalEntryNotFoundError(
   1659             "An error happened while trying to locate the file on the Hub and we cannot find the requested files"
   1660             " in the local cache. Please check your connection and try again or make sure your Internet connection"

LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.

## === cell 5
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        model_outputs = []
        for model_key, model in zip(loaded_keys, models_list):
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=1)  # (B,5)
            weighted = weights[model_key] * probs
            model_outputs.append(weighted.unsqueeze(0))  # (1,B,5)
        if not model_outputs:
            uniform = torch.full((images.size(0), 5), 1.0 / 5, device=device)
            all_outputs.extend(uniform.cpu().numpy())
            continue
        stacked = torch.cat(model_outputs, dim=0)  # (M,B,5)
        ensemble = torch.sum(stacked, dim=0)  # (B,5)
        all_outputs.extend(ensemble.cpu().numpy())

all_outputs = np.array(all_outputs)
final_predictions = np.argmax(all_outputs, axis=1)



## === cell 6
submission_df = pd.DataFrame(
    {"id_code": pd.read_csv(test_csv_file)["id_code"], "diagnosis": final_predictions}
)
submission_df.to_csv("submission.csv", index=False)
