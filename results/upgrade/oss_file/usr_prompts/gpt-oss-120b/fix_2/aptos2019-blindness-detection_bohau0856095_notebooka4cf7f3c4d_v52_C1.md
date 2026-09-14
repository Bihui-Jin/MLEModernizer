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

3.9

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

0.9143358388025438

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
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2
from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    """Convert regression output to integer class using thresholds."""
    prediction = torch.zeros(out.size(0), device=out.device)
    for t in threshold:
        prediction += (out >= t).float()
    return prediction.long()


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = 1 - out[:, 0]
    pred_prob[:, 1] = out[:, 0] * (1 - out[:, 1])
    pred_prob[:, 2] = out[:, 1] * (1 - out[:, 2])
    pred_prob[:, 3] = out[:, 2] * (1 - out[:, 3])
    pred_prob[:, 4] = out[:, 3]
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i, l1] = 1 - (out[i] - l1)
            pred_prob[i, l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i, 4] = 1.0
    return pred_prob




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b5_ns", pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.backbone.num_features, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.backbone.num_features, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.backbone.num_features, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )

        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)

        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)

        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 3
class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            FT.adjust_brightness,
            FT.adjust_contrast,
            FT.adjust_saturation,
            FT.adjust_hue,
        ]
        random.shuffle(distortions)
        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    adjust_factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    adjust_factor = random.uniform(0.7, 1.3)
                image = d(image, adjust_factor)
        return image


class cropTo4_3(object):
    def __call__(self, image):
        w, h = image.size
        if (w / h) >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = left + new_w
        bottom = top + new_h
        return image.crop((left, top, right, bottom))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image




## === cell 4
test_ids_path = "../input/aptos2019-blindness-detection/test.csv"
test_ids = pd.read_csv(test_ids_path).squeeze().values  # shape (N,)

input_size = 380
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()
weight_path = "../input/weights/B4_3stage_51epoch_CLAHE.pkl"
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path, map_location=device))
    print("Loaded pretrained weights.")
else:
    print("Weights file not found – using model with ImageNet pretrained backbone.")

net = net.to(device)
net.eval()



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

NameResolutionError: <urllib3.connection.HTTPSConnection object at 0x7fae78350a50>: Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)

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

MaxRetryError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /timm/tf_efficientnet_b4.ns_jft_in1k/resolve/main/pytorch_model.bin (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7fae78350a50>: Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)"))

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

ConnectionError: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /timm/tf_efficientnet_b4.ns_jft_in1k/resolve/main/pytorch_model.bin (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7fae78350a50>: Failed to resolve \'huggingface.co\' ([Errno -3] Temporary failure in name resolution)"))'), '(Request ID: 3c4907e4-4bec-4e2f-91fe-3042721af25d)')

The above exception was the direct cause of the following exception:

LocalEntryNotFoundError                   Traceback (most recent call last)
/tmp/ipykernel_55/2545733781.py in <cell line: 0>()
     15 
     16 # Initialize model
---> 17 net = ThreeStage_Model()
     18 weight_path = "../input/weights/B4_3stage_51epoch_CLAHE.pkl"
     19 if os.path.exists(weight_path):

/tmp/ipykernel_55/373988703.py in __init__(self)
     38     def __init__(self):
     39         super(ThreeStage_Model, self).__init__()
---> 40         self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
     41         self.backbone.global_pool = GeM(flatten=True)
     42 

/usr/local/lib/python3.11/dist-packages/timm/models/_factory.py in create_model(model_name, pretrained, pretrained_cfg, pretrained_cfg_overlay, checkpoint_path, cache_dir, scriptable, exportable, no_jit, **kwargs)
    136     create_fn = model_entrypoint(model_name)
    137     with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
--> 138         model = create_fn(
    139             pretrained=pretrained,
    140             pretrained_cfg=pretrained_cfg,

/usr/local/lib/python3.11/dist-packages/timm/models/_registry.py in _fn(pretrained, **kwargs)
    143         warnings.warn(f'Mapping deprecated model name {deprecated_name} to current {current_name}.', stacklevel=2)
    144         pretrained_cfg = kwargs.pop('pretrained_cfg', None)
--> 145         return current_fn(pretrained=pretrained, pretrained_cfg=pretrained_cfg or current_tag, **kwargs)
    146     return _fn
    147 

/usr/local/lib/python3.11/dist-packages/timm/models/efficientnet.py in tf_efficientnet_b4(pretrained, **kwargs)
   2449     kwargs.setdefault('bn_eps', BN_EPS_TF_DEFAULT)
   2450     kwargs.setdefault('pad_type', 'same')
-> 2451     model = _gen_efficientnet(
   2452         'tf_efficientnet_b4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
   2453     return model

/usr/local/lib/python3.11/dist-packages/timm/models/efficientnet.py in _gen_efficientnet(variant, channel_multiplier, depth_multiplier, channel_divisor, group_size, pretrained, **kwargs)
    747         **kwargs,
    748     )
--> 749     model = _create_effnet(variant, pretrained, **model_kwargs)
    750     return model
    751 

/usr/local/lib/python3.11/dist-packages/timm/models/efficientnet.py in _create_effnet(variant, pretrained, **kwargs)
    448             features_mode = 'cls'
    449 
--> 450     model = build_model_with_cfg(
    451         model_cls,
    452         variant,

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
submission = []
for i, idx in enumerate(test_ids):
    print(f"Processing {i+1}/{len(test_ids)}: {idx}")
    image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    _, r_out, _ = net(img_tensor)
    pred_tensor = regress2class(r_out.squeeze(1))
    pred = int(pred_tensor.item())
    submission.append([idx, pred])

submission = np.array(submission)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/338595387.py in <cell line: 0>()
      6     img_tensor = transform(img).unsqueeze(0).to(device)
      7 
----> 8     _, r_out, _ = net(img_tensor)
      9     pred_tensor = regress2class(r_out.squeeze(1))
     10     pred = int(pred_tensor.item())

NameError: name 'net' is not defined

## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}, rows: {len(df)}")

## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame should not be empty
