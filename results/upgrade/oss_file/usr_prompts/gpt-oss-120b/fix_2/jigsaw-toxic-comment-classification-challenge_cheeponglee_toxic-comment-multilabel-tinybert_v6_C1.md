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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pyarrow==19.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.9711

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, time, datetime
import numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)



## === cell 1
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
sample_sub_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)

df_train = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

print("Train shape:", df_train.shape, "Test shape:", test_df.shape)



## === cell 2
checkpoint = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(categories)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



## --- ERROR in cell 2, traceback:
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

NameResolutionError: <urllib3.connection.HTTPSConnection object at 0x7f2ecec06a50>: Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)

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

MaxRetryError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /huawei-noah/TinyBERT_General_4L_312D/resolve/main/config.json (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7f2ecec06a50>: Failed to resolve 'huggingface.co' ([Errno -3] Temporary failure in name resolution)"))

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

ConnectionError: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /huawei-noah/TinyBERT_General_4L_312D/resolve/main/config.json (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7f2ecec06a50>: Failed to resolve \'huggingface.co\' ([Errno -3] Temporary failure in name resolution)"))'), '(Request ID: 13e2b1c0-0192-4d92-a5a9-185639ac8145)')

The above exception was the direct cause of the following exception:

LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1657         # Otherwise: most likely a connection issue or Hub downtime => let's warn the user
-> 1658         raise LocalEntryNotFoundError(
   1659             "An error happened while trying to locate the file on the Hub and we cannot find the requested files"

LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/1414277441.py in <cell line: 0>()
      1 # model checkpoint
      2 checkpoint = "huawei-noah/TinyBERT_General_4L_312D"
----> 3 tokenizer = AutoTokenizer.from_pretrained(checkpoint)
      4 
      5 # load pretrained model (no fine‑tuning to keep runtime short)

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
   1001                     config = AutoConfig.for_model(**config_dict)
   1002                 else:
-> 1003                     config = AutoConfig.from_pretrained(
   1004                         pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
   1005                     )

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1195         code_revision = kwargs.pop("code_revision", None)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
   1199         has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 3
test_enc = tokenizer.batch_encode_plus(
    test_df["comment_text"].tolist(),
    max_length=200,
    padding="max_length",  # modern argument replaces deprecated pad_to_max_length
    truncation=True,
    return_token_type_ids=False,
    return_attention_mask=True,
)

test_seq = torch.tensor(test_enc["input_ids"])
test_mask = torch.tensor(test_enc["attention_mask"])

test_dataset = TensorDataset(test_seq, test_mask)
test_sampler = SequentialSampler(test_dataset)

batch_size = 64
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1142556592.py in <cell line: 0>()
      1 # Tokenize test comments
----> 2 test_enc = tokenizer.batch_encode_plus(
      3     test_df["comment_text"].tolist(),
      4     max_length=200,
      5     padding="max_length",  # modern argument replaces deprecated pad_to_max_length

NameError: name 'tokenizer' is not defined

## === cell 4
all_predictions = []

with torch.no_grad():
    for step, batch in enumerate(test_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        outputs = model(b_input_ids, attention_mask=b_input_mask)
        probs = torch.sigmoid(outputs.logits)  # shape: (batch, 6)
        all_predictions.append(probs.cpu().numpy())

predictions = np.vstack(all_predictions)  # (num_test, 6)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3535427954.py in <cell line: 0>()
      3 
      4 with torch.no_grad():
----> 5     for step, batch in enumerate(test_dataloader):
      6         b_input_ids = batch[0].to(device)
      7         b_input_mask = batch[1].to(device)

NameError: name 'test_dataloader' is not defined

## === cell 5
submission = pd.DataFrame(predictions, columns=categories)
submission.insert(0, "id", test_df["id"])

print(submission.head())
print("Submission shape:", submission.shape)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3435424326.py in <cell line: 0>()
      1 # Build submission DataFrame
----> 2 submission = pd.DataFrame(predictions, columns=categories)
      3 submission.insert(0, "id", test_df["id"])
      4 
      5 # sanity check

NameError: name 'predictions' is not defined
