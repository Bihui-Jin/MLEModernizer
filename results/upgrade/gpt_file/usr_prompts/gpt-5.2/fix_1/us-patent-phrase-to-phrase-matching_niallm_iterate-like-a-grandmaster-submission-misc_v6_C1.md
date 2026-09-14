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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8084755399348172

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 3
from fastai.imports import *

## === cell 5
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

## === cell 8
path = (Path('../input/us-patent-phrase-to-phrase-matching') if iskaggle
    else Path.home()/'data'/'us-patent-phrase-to-phrase-matching')
model_path = (Path('../input/download-pretrained-models/outputs/deberta-v3-small') if iskaggle
    else Path.home()/'data'/'deberta-v3-small')
path.ls()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4240160892.py in <cell line: 0>()
      3 model_path = (Path('../input/download-pretrained-models/outputs/deberta-v3-small') if iskaggle
      4     else Path.home()/'data'/'deberta-v3-small')
----> 5 path.ls()

/usr/local/lib/python3.11/dist-packages/fastcore/xtras.py in ls(self, n_max, file_type, file_exts)
    389     res = (o for o in self.iterdir() if has_extns or o.suffix in extns)
    390     if n_max is not None: res = itertools.islice(res, n_max)
--> 391     return L(res)
    392 
    393 # %% ../nbs/03_xtras.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __call__(cls, x, *args, **kwargs)
    103     def __call__(cls, x=None, *args, **kwargs):
    104         if not args and not kwargs and x is not None and isinstance(x,cls): return x
--> 105         return super().__call__(x, *args, **kwargs)
    106 
    107 # %% ../nbs/02_foundation.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __init__(self, items, use_list, match, *rest)
    111     def __init__(self, items=None, *rest, use_list=False, match=None):
    112         if (use_list is not None) or not is_array(items):
--> 113             items = listify(items, *rest, use_list=use_list, match=match)
    114         super().__init__(items)
    115 

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in listify(o, use_list, match, *rest)
     77     elif isinstance(o, list): res = o
     78     elif isinstance(o, str) or isinstance(o, bytes) or is_array(o): res = [o]
---> 79     elif is_iter(o): res = list(o)
     80     else: res = [o]
     81     if match is not None:

/usr/local/lib/python3.11/dist-packages/fastcore/xtras.py in <genexpr>(.0)
    387     if file_type: extns += L(k for k,v in mimetypes.types_map.items() if v.startswith(file_type+'/'))
    388     has_extns = len(extns)==0
--> 389     res = (o for o in self.iterdir() if has_extns or o.suffix in extns)
    390     if n_max is not None: res = itertools.islice(res, n_max)
    391     return L(res)

/usr/lib/python3.11/pathlib.py in iterdir(self)
    929         result for the special paths '.' and '..'.
    930         """
--> 931         for name in os.listdir(self):
    932             yield self._make_child_relpath(name)
    933 

FileNotFoundError: [Errno 2] No such file or directory: '/root/data/us-patent-phrase-to-phrase-matching'

## === cell 10
df = pd.read_csv(path/'train.csv')
df

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3201712262.py in <cell line: 0>()
----> 1 df = pd.read_csv(path/'train.csv')
      2 df

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/root/data/us-patent-phrase-to-phrase-matching/train.csv'

## === cell 12
eval_df = pd.read_csv(path/'test.csv')
len(eval_df)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2648827211.py in <cell line: 0>()
----> 1 eval_df = pd.read_csv(path/'test.csv')
      2 len(eval_df)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/root/data/us-patent-phrase-to-phrase-matching/test.csv'

## === cell 13
eval_df.head()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/735534184.py in <cell line: 0>()
----> 1 eval_df.head()

NameError: name 'eval_df' is not defined

## === cell 15
df.target.value_counts()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3118463302.py in <cell line: 0>()
----> 1 df.target.value_counts()

NameError: name 'df' is not defined

## === cell 17
df.anchor.value_counts()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2733761793.py in <cell line: 0>()
----> 1 df.anchor.value_counts()

NameError: name 'df' is not defined

## === cell 19
df.context.value_counts()

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/644616941.py in <cell line: 0>()
----> 1 df.context.value_counts()

NameError: name 'df' is not defined

## === cell 21
df['section'] = df.context.str[0]
df.section.value_counts()

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/451508367.py in <cell line: 0>()
----> 1 df['section'] = df.context.str[0]
      2 df.section.value_counts()

NameError: name 'df' is not defined

## === cell 23
df.score.hist();

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2147268738.py in <cell line: 0>()
----> 1 df.score.hist();

NameError: name 'df' is not defined

## === cell 25
df[df.score==1]

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3774886598.py in <cell line: 0>()
----> 1 df[df.score==1]

NameError: name 'df' is not defined

## === cell 29
from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification,AutoTokenizer

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 30
if iskaggle:
    !pip install -q datasets
import datasets
from datasets import load_dataset, Dataset, DatasetDict

## === cell 32
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

## === cell 34
model_nm = model_path

## === cell 36
model_nm.ls()

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3838333207.py in <cell line: 0>()
----> 1 model_nm.ls()

/usr/local/lib/python3.11/dist-packages/fastcore/xtras.py in ls(self, n_max, file_type, file_exts)
    389     res = (o for o in self.iterdir() if has_extns or o.suffix in extns)
    390     if n_max is not None: res = itertools.islice(res, n_max)
--> 391     return L(res)
    392 
    393 # %% ../nbs/03_xtras.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __call__(cls, x, *args, **kwargs)
    103     def __call__(cls, x=None, *args, **kwargs):
    104         if not args and not kwargs and x is not None and isinstance(x,cls): return x
--> 105         return super().__call__(x, *args, **kwargs)
    106 
    107 # %% ../nbs/02_foundation.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __init__(self, items, use_list, match, *rest)
    111     def __init__(self, items=None, *rest, use_list=False, match=None):
    112         if (use_list is not None) or not is_array(items):
--> 113             items = listify(items, *rest, use_list=use_list, match=match)
    114         super().__init__(items)
    115 

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in listify(o, use_list, match, *rest)
     77     elif isinstance(o, list): res = o
     78     elif isinstance(o, str) or isinstance(o, bytes) or is_array(o): res = [o]
---> 79     elif is_iter(o): res = list(o)
     80     else: res = [o]
     81     if match is not None:

/usr/local/lib/python3.11/dist-packages/fastcore/xtras.py in <genexpr>(.0)
    387     if file_type: extns += L(k for k,v in mimetypes.types_map.items() if v.startswith(file_type+'/'))
    388     has_extns = len(extns)==0
--> 389     res = (o for o in self.iterdir() if has_extns or o.suffix in extns)
    390     if n_max is not None: res = itertools.islice(res, n_max)
    391     return L(res)

/usr/lib/python3.11/pathlib.py in iterdir(self)
    929         result for the special paths '.' and '..'.
    930         """
--> 931         for name in os.listdir(self):
    932             yield self._make_child_relpath(name)
    933 

FileNotFoundError: [Errno 2] No such file or directory: '/root/data/deberta-v3-small'

## === cell 37
tokz = AutoTokenizer.from_pretrained(model_nm)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/root/data/deberta-v3-small'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/4217825891.py in <cell line: 0>()
----> 1 tokz = AutoTokenizer.from_pretrained(model_nm)
      2 # tokz = AutoTokenizer.from_pretrained('../inputs/deberta-v3-small')

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
    981 
    982         # Next, let's try to use the tokenizer_config file to get the tokenizer class.
--> 983         tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
    984         if "_commit_hash" in tokenizer_config:
    985             kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in get_tokenizer_config(pretrained_model_name_or_path, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, **kwargs)
    813 
    814     commit_hash = kwargs.get("_commit_hash", None)
--> 815     resolved_config_file = cached_file(
    816         pretrained_model_name_or_path,
    817         TOKENIZER_CONFIG_FILE,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    152 
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
    156             f" '{repo_id}'. Use `repo_type` argument if needed."

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/root/data/deberta-v3-small'. Use `repo_type` argument if needed.

## === cell 40
sep = tokz.sep_token
sep

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/53071346.py in <cell line: 0>()
----> 1 sep = tokz.sep_token
      2 sep

NameError: name 'tokz' is not defined

## === cell 42
df['inputs'] = df.context + sep + df.anchor + sep + df.target

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3164295909.py in <cell line: 0>()
----> 1 df['inputs'] = df.context + sep + df.anchor + sep + df.target

NameError: name 'df' is not defined

## === cell 44
ds = Dataset.from_pandas(df).rename_column('score', 'label')
eval_ds = Dataset.from_pandas(eval_df)

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/253954807.py in <cell line: 0>()
----> 1 ds = Dataset.from_pandas(df).rename_column('score', 'label')
      2 eval_ds = Dataset.from_pandas(eval_df)

NameError: name 'df' is not defined

## === cell 46
def tok_func(x): return tokz(x["inputs"])

## === cell 48
tok_func(ds[0])

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2428102215.py in <cell line: 0>()
----> 1 tok_func(ds[0])

NameError: name 'ds' is not defined

## === cell 50
tokz.all_special_tokens

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1518054995.py in <cell line: 0>()
----> 1 tokz.all_special_tokens

NameError: name 'tokz' is not defined

## === cell 52
inps = "anchor","target","context"
tok_ds = ds.map(tok_func, batched=True, remove_columns=inps+('inputs','id','section'))

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/523423207.py in <cell line: 0>()
      1 inps = "anchor","target","context"
----> 2 tok_ds = ds.map(tok_func, batched=True, remove_columns=inps+('inputs','id','section'))

NameError: name 'ds' is not defined

## === cell 54
tok_ds[0]

## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/760174941.py in <cell line: 0>()
----> 1 tok_ds[0]

NameError: name 'tok_ds' is not defined

## === cell 57
anchors = df.anchor.unique()
np.random.seed(42)
np.random.shuffle(anchors)
anchors[:5]

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3627589441.py in <cell line: 0>()
----> 1 anchors = df.anchor.unique()
      2 np.random.seed(42)
      3 np.random.shuffle(anchors)
      4 anchors[:5]

NameError: name 'df' is not defined

## === cell 59
val_prop = 0.25
val_sz = int(len(anchors)*val_prop)
val_anchors = anchors[:val_sz]

## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2476089575.py in <cell line: 0>()
      1 val_prop = 0.25
----> 2 val_sz = int(len(anchors)*val_prop)
      3 val_anchors = anchors[:val_sz]

NameError: name 'anchors' is not defined

## === cell 61
is_val = np.isin(df.anchor, val_anchors)
idxs = np.arange(len(df))
val_idxs = idxs[ is_val]
trn_idxs = idxs[~is_val]
len(val_idxs),len(trn_idxs)

## --- ERROR in cell 61, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/358907896.py in <cell line: 0>()
----> 1 is_val = np.isin(df.anchor, val_anchors)
      2 idxs = np.arange(len(df))
      3 val_idxs = idxs[ is_val]
      4 trn_idxs = idxs[~is_val]
      5 len(val_idxs),len(trn_idxs)

NameError: name 'df' is not defined

## === cell 63
dds = DatasetDict({"train":tok_ds.select(trn_idxs),
             "test": tok_ds.select(val_idxs)})

## --- ERROR in cell 63, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1197523405.py in <cell line: 0>()
----> 1 dds = DatasetDict({"train":tok_ds.select(trn_idxs),
      2              "test": tok_ds.select(val_idxs)})

NameError: name 'tok_ds' is not defined

## === cell 65
df.iloc[trn_idxs].score.mean(),df.iloc[val_idxs].score.mean()

## --- ERROR in cell 65, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4084557131.py in <cell line: 0>()
----> 1 df.iloc[trn_idxs].score.mean(),df.iloc[val_idxs].score.mean()

NameError: name 'df' is not defined

## === cell 68
def corr(eval_pred): return {'pearson': np.corrcoef(*eval_pred)[0][1]}

## === cell 70
lr,bs = 8e-5,128
wd,epochs = 0.01,4

## === cell 72
def get_trainer(dds):
    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs, weight_decay=wd, report_to='none')
    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokz, compute_metrics=corr)

## === cell 73
args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=wd, report_to='none')

## --- ERROR in cell 73, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3726411648.py in <cell line: 0>()
----> 1 args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
      2     evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
      3     num_train_epochs=epochs, weight_decay=wd, report_to='none')

TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'

## === cell 75
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
               tokenizer=tokz, compute_metrics=corr)

## --- ERROR in cell 75, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/root/data/deberta-v3-small'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/3349872590.py in <cell line: 0>()
----> 1 model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
      2 # model = AutoModelForSequenceClassification.from_pretrained('../inputs/deberta-v3-small', num_labels=1)
      3 trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
      4                tokenizer=tokz, compute_metrics=corr)

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/auto_factory.py in from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    506             if not isinstance(config, PretrainedConfig):
    507                 # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
--> 508                 resolved_config_file = cached_file(
    509                     pretrained_model_name_or_path,
    510                     CONFIG_NAME,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    152 
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
    156             f" '{repo_id}'. Use `repo_type` argument if needed."

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/root/data/deberta-v3-small'. Use `repo_type` argument if needed.

## === cell 78
trainer.train();

## --- ERROR in cell 78, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2912127331.py in <cell line: 0>()
----> 1 trainer.train();

NameError: name 'trainer' is not defined

## === cell 79
def prepare_eval_ds(eval_df):
    sep = tokz.sep_token
    eval_df['section'] = eval_df.context.str[0]
    eval_df['inputs'] = eval_df.context + sep + eval_df.anchor + sep + eval_df.target
    test_ds = Dataset.from_pandas(eval_df)
    return test_ds.map(tok_func, batched=True, remove_columns=inps+('inputs','id','section'))

def gen_submissions():
    test_ds = prepare_eval_ds(eval_df)
    prediction_results = trainer.predict(test_ds)
    submission_df = pd.DataFrame({'id': eval_df['id'], 'score': prediction_results[0].flatten()})
    submission_df.to_csv('submission.csv', index=False)
    
gen_submissions()

## --- ERROR in cell 79, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2590278486.py in <cell line: 0>()
     12     submission_df.to_csv('submission.csv', index=False)
     13 
---> 14 gen_submissions()

/tmp/ipykernel_11/2590278486.py in gen_submissions()
      7 
      8 def gen_submissions():
----> 9     test_ds = prepare_eval_ds(eval_df)
     10     prediction_results = trainer.predict(test_ds)
     11     submission_df = pd.DataFrame({'id': eval_df['id'], 'score': prediction_results[0].flatten()})

NameError: name 'eval_df' is not defined

## === cell 82
def get_dds(df):
    ds = Dataset.from_pandas(df).rename_column('score', 'label')
    tok_ds = ds.map(tok_func, batched=True, remove_columns=inps+('inputs','id','section'))
    return DatasetDict({"train":tok_ds.select(trn_idxs), "test": tok_ds.select(val_idxs)})

## === cell 84
def get_model():
    args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
        num_train_epochs=epochs, weight_decay=wd, report_to='none')
    return AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

def get_trainer(dds, model=None):
    if model is None: model = get_model()
    return Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                   tokenizer=tokz, compute_metrics=corr)

## === cell 106
n_folds = 4
