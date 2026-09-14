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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.58839

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random
import numpy as np, pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split

seed = 197
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

print("TensorFlow version:", tf.__version__)
print("Available files:", os.listdir("../input"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/train.tsv"
test_path = "../input/test.tsv"
sample_sub_path = "../input/sampleSubmission.csv"

train = pd.read_csv(train_path, sep="\t")
test = pd.read_csv(test_path, sep="\t")




## === cell 2
assert "Phrase" in train.columns, "train must contain a 'Phrase' column"
assert "Sentiment" in train.columns, "train must contain a 'Sentiment' column"
assert "Phrase" in test.columns, "test must contain a 'Phrase' column"
assert "PhraseId" in test.columns, "test must contain a 'PhraseId' column"




## === cell 3
train_df, val_df = train_test_split(
    train, test_size=0.1, random_state=seed, stratify=train["Sentiment"]
)

train_phrases = train_df["Phrase"].values
train_labels = train_df["Sentiment"].values

val_phrases = val_df["Phrase"].values
val_labels = val_df["Sentiment"].values




## === cell 4
embedding_url = "https://tfhub.dev/google/nnlm-en-dim50/1"
hub_layer = hub.KerasLayer(
    embedding_url, input_shape=[], dtype=tf.string, trainable=False
)

inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name="phrase_input")
x = hub_layer(inputs)
x = tf.keras.layers.Dense(500, activation="relu")(x)
x = tf.keras.layers.Dense(100, activation="relu")(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TimeoutError                              Traceback (most recent call last)
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

TimeoutError: [Errno 110] Connection timed out

During handling of the above exception, another exception occurred:

URLError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1040711122.py in <cell line: 0>()
      1 # Build Keras model with Hub text embedding and two hidden layers
      2 embedding_url = "https://tfhub.dev/google/nnlm-en-dim50/1"
----> 3 hub_layer = hub.KerasLayer(
      4     embedding_url, input_shape=[], dtype=tf.string, trainable=False
      5 )

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in __init__(self, handle, trainable, arguments, _sentinel, tags, signature, signature_outputs_as_dict, output_key, output_shape, load_options, **kwargs)
    163 
    164     self._load_options = load_options
--> 165     self._func = load_module(handle, tags, self._load_options)
    166     self._is_hub_module_v1 = getattr(self._func, "_is_hub_module_v1", False)
    167 

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in load_module(handle, tags, load_options)
    465         except ImportError:  # Expected before TF2.4.
    466           set_load_options = load_options
--> 467     return module_v2.load(handle, tags=tags, options=set_load_options)
    468 
    469 

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/module_v2.py in load(handle, tags, options)
     98   if not isinstance(handle, str):
     99     raise ValueError("Expected a string, got %s" % handle)
--> 100   module_path = resolve(handle)
    101   is_hub_module_v1 = tf.io.gfile.exists(_get_module_proto_path(module_path))
    102   if tags is None and is_hub_module_v1:

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/module_v2.py in resolve(handle)
     53     A string representing the Module path.
     54   """
---> 55   return registry.resolver(handle)
     56 
     57 

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/registry.py in __call__(self, *args, **kwargs)
     47     for impl in reversed(self._impls):
     48       if impl.is_supported(*args, **kwargs):
---> 49         return impl(*args, **kwargs)
     50       else:
     51         fails.append(type(impl).__name__)

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/compressed_module_resolver.py in __call__(self, handle)
     79           response, tmp_dir)
     80 
---> 81     return resolver.atomic_download(handle, download, module_dir,
     82                                     self._lock_file_timeout_sec())
     83 

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/resolver.py in atomic_download(handle, download_fn, module_dir, lock_file_timeout_sec)
    419     logging.info("Downloading TF-Hub Module '%s'.", handle)
    420     tf.compat.v1.gfile.MakeDirs(tmp_dir)
--> 421     download_fn(handle, tmp_dir)
    422     # Write module descriptor to capture information about which module was
    423     # downloaded by whom and when. The file stored at the same level as a

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/compressed_module_resolver.py in download(handle, tmp_dir)
     75       request = urllib.request.Request(
     76           self._append_compressed_format_query(handle))
---> 77       response = self._call_urlopen(request)
     78       return resolver.DownloadManager(handle).download_and_uncompress(
     79           response, tmp_dir)

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/resolver.py in _call_urlopen(self, request)
    526       return urllib.request.urlopen(request)
    527     else:
--> 528       return urllib.request.urlopen(request, context=self._context)
    529 
    530   def is_http_protocol(self, handle):

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

URLError: <urlopen error [Errno 110] Connection timed out>

## === cell 5
model.fit(
    x=train_phrases,
    y=train_labels,
    validation_data=(val_phrases, val_labels),
    epochs=5,
    batch_size=32,
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/419701026.py in <cell line: 0>()
      1 # Train the model
----> 2 model.fit(
      3     x=train_phrases,
      4     y=train_labels,
      5     validation_data=(val_phrases, val_labels),

NameError: name 'model' is not defined

## === cell 6
train_eval = model.evaluate(train_phrases, train_labels, verbose=0)
print(f"Training set accuracy: {train_eval[1]:.4f}")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3624601246.py in <cell line: 0>()
      1 # Evaluate on the whole training set for reference
----> 2 train_eval = model.evaluate(train_phrases, train_labels, verbose=0)
      3 print(f"Training set accuracy: {train_eval[1]:.4f}")
      4 
      5 

NameError: name 'model' is not defined

## === cell 7
test_phrases = test["Phrase"].values
test_pred_probs = model.predict(test_phrases, batch_size=32, verbose=0)
test_pred_classes = np.argmax(test_pred_probs, axis=1)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/747530953.py in <cell line: 0>()
      1 # Predict on the test set
      2 test_phrases = test["Phrase"].values
----> 3 test_pred_probs = model.predict(test_phrases, batch_size=32, verbose=0)
      4 test_pred_classes = np.argmax(test_pred_probs, axis=1)
      5 

NameError: name 'model' is not defined

## === cell 8
sub = pd.read_csv(sample_sub_path)
sub = sub.drop(columns=["Sentiment"], errors="ignore")
sub["Sentiment"] = test_pred_classes
sub.to_csv("sub_tfhub.csv", index=False)
print("Submission saved to sub_tfhub.csv")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2367134246.py in <cell line: 0>()
      2 sub = pd.read_csv(sample_sub_path)
      3 sub = sub.drop(columns=["Sentiment"], errors="ignore")
----> 4 sub["Sentiment"] = test_pred_classes
      5 sub.to_csv("sub_tfhub.csv", index=False)
      6 print("Submission saved to sub_tfhub.csv")

NameError: name 'test_pred_classes' is not defined
