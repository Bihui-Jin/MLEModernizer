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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
sentence-transformers==4.1.0
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
tqdm==4.67.1
transformers==4.53.3

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

0.8100979449941955

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch
import tqdm
import math
import pandas as pd
import torch.nn as nn
from scipy import stats
from transformers import AdamW, AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/684817496.py in <cell line: 0>()
      5 import torch.nn as nn
      6 from scipy import stats
----> 7 from transformers import AdamW, AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

ImportError: cannot import name 'AdamW' from 'transformers' (/usr/local/lib/python3.11/dist-packages/transformers/__init__.py)

## === cell 1
import torch

model_n = "../input/roberta-pre/patent_pretrained"
max_len = 32
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0) 

warmup_ratio = 0.06
weight_decay=0.01
gradient_accumulation_steps = 1
num_train_epochs = 2
learning_rate = 2e-5
adam_epsilon = 1e-08

## === cell 2

class PhraseDataset:
    def __init__(self, anchor, target, context, score, tokenizer, max_len):
        self.anchor = anchor
        self.target = target
        self.context = context
        self.score = score
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, item):
        anchor = self.anchor[item] 
        context = self.context[item]
        target = self.target[item]
        score = self.score[item]

        encoded_text = self.tokenizer.encode_plus(
            context + " " + anchor,
            target,
            add_special_tokens = True,
            max_length=self.max_len,
            padding= 'max_length',
            truncation=True,
            return_attention_mask = True
        )

        input_ids = encoded_text.input_ids,
        attention_mask = encoded_text.attention_mask,

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "score": torch.tensor(score, dtype=torch.float),
        }

## === cell 3
class SimilarPhraseModel(nn.Module):
    def __init__(self, model_name, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name

        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "add_pooling_layer": True,
                "num_labels": 1,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output = nn.Linear(config.hidden_size, 1)
    
    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        outputs = outputs.cpu().detach().numpy().ravel()
        targets = targets.cpu().detach().numpy().ravel()
        pearsonr = stats.pearsonr(outputs, targets)
        return {"pearsonr": torch.tensor(pearsonr[0], device="cpu")}
    
    
    def forward(self, ids, mask, score):
        transformer_out= self.transformer(ids, mask)
        output = transformer_out.pooler_output
        output = self.dropout(output)
        output = self.output(output)
        loss = nn.MSELoss()(output.squeeze(), score.squeeze())
        metrics = self.monitor_metrics(output, score)
        return output, loss, metrics

## === cell 4
import pandas as pd
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
pearson_score= 0
for fold_ in range(10):
    df = pd.read_csv("../input/newdatawithfolds/train_folds.csv")

    context_mapping = {
        "A": "Human Necessities",
        "B": "Operations and Transport",
        "C": "Chemistry and Metallurgy",
        "D": "Textiles",
        "E": "Fixed Constructions",
        "F": "Mechanical Engineering",
        "G": "Physics",
        "H": "Electricity",
        "Y": "Emerging Cross-Sectional Technologies",
    }

    df.context = df.context.apply(lambda x: context_mapping[x[0]])

    train_df = df[df["kfold"] != fold_].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold_].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(model_n)
    train_dataset = PhraseDataset(
            anchor=train_df.anchor.values,
            target=train_df.target.values,
            context=train_df.context.values,
            score=train_df.score.values,
            tokenizer=tokenizer,
            max_len=max_len,
        )

    valid_dataset = PhraseDataset(
            anchor=valid_df.anchor.values,
            target=valid_df.target.values,
            context=valid_df.context.values,
            score=valid_df.score.values,
            tokenizer=tokenizer,
            max_len=max_len,
        )

    model = SimilarPhraseModel(
            model_name=model_n,
            learning_rate=0.001,
        )
    model.to(device)


    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

   
    t_total = len(train_loader) // gradient_accumulation_steps * num_train_epochs
    optimizer_grouped_parameters = []
    custom_parameter_names = set()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters.extend(
        [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    )

    warmup_steps = math.ceil(t_total * warmup_ratio)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    def get_inputs_dict(batch):
        inputs = {key: value.to(device) for key, value in batch.items()}
        return inputs

    for epoch in range(5):
        model.train()
        for batch in train_loader:
            batch = get_inputs_dict(batch)
            input_ids = batch['ids'].permute(0, 2, 1)[:, :, -1].to(device)
            attention_mask = batch['mask'].permute(0, 2, 1)[:, :, -1].to(device)
            labels = batch['score'].to(device)
            abc = model(input_ids,attention_mask,labels)
            abc[1].backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        print(f"Pearson correlation for epoch {epoch}  during training is {abc[2]}" )
    

        model.eval()
        
        with torch.no_grad():
            for batch in valid_loader:
                batch = get_inputs_dict(batch)
                input_ids = batch['ids'].permute(0, 2, 1)[:, :, -1].to(device)
                attention_mask = batch['mask'].permute(0, 2, 1)[:, :, -1].to(device)
                labels = batch['score'].to(device)
                outputs = model(input_ids,attention_mask,labels)
                
        print(f"Pearson correlation for epoch {epoch}  during validation is {outputs[2]}")

        if(outputs[2]['pearsonr'] > pearson_score):
            print(f"{outputs[2]['pearsonr']},{pearson_score}")
            torch.save(model.state_dict(), "my_best_model.pt")
            pearson_score = outputs[2]['pearsonr']



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/715627282.py in <cell line: 0>()
      6 pearson_score= 0
      7 for fold_ in range(10):
----> 8     df = pd.read_csv("../input/newdatawithfolds/train_folds.csv")
      9 
     10     context_mapping = {

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/newdatawithfolds/train_folds.csv'

## === cell 5
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from transformers import AutoModel, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

model_n ="../input/roberta-pre/patent_pretrained"
max_len = 32
batch_size = 32
epochs = 5
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0) 



df_test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")

context_mapping = {
    "A": "Human Necessities",
    "B": "Operations and Transport",
    "C": "Chemistry and Metallurgy",
    "D": "Textiles",
    "E": "Fixed Constructions",
    "F": "Mechanical Engineering",
    "G": "Physics",
    "H": "Electricity",
    "Y": "Emerging Cross-Sectional Technologies",
}
df_test.context = df_test.context.apply(lambda x: context_mapping[x[0]])


class PhraseTestDataset:
    def __init__(self, anchor, target, context, tokenizer, max_len):
        self.anchor = anchor
        self.target = target
        self.context = context
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, item):
        anchor = self.anchor[item]
        context = self.context[item]
        target = self.target[item]

        encoded_text = self.tokenizer.encode_plus(
            context + " " + anchor,
            target,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
        )
        input_ids = encoded_text["input_ids"]
        attention_mask = encoded_text["attention_mask"]
       

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long)
        }


class PhraseModelTest(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        config = AutoConfig.from_pretrained(model_name)
        config.update(
            {
                "output_hidden_states": True,
                "add_pooling_layer": True,
                "num_labels": 1,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output = nn.Linear(config.hidden_size, 1)

    def forward(self, ids, mask):
        transformer_out = self.transformer(ids, mask)
        output = transformer_out.pooler_output
        output = self.dropout(output)
        output = self.output(output)
        return output, 0, {}


tokenizer = AutoTokenizer.from_pretrained(model_n)
test_dataset = PhraseTestDataset(
        anchor=df_test.anchor.values,
        target=df_test.target.values,
        context=df_test.context.values,
        tokenizer=tokenizer,
        max_len=max_len,
    )

from torch.utils.data import DataLoader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def get_inputs_dict(batch):
    inputs = {key: value.to(device) for key, value in batch.items()}
    return inputs


def predict(model_paths):
    model_path = model_paths

    model_test = PhraseModelTest(model_n)
    model_test.load_state_dict(torch.load(model_path))
    model_test.to(device)

    model_test.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = get_inputs_dict(batch)
           
            input_ids = batch['ids'].to(device)
            attention_mask = batch['mask'].to(device)
            test_output,_,_ = model_test(input_ids,attention_mask)
            test_preds.append(test_output.detach().cpu().numpy())

    new_predctions = np.array(test_preds).ravel()
    return new_predctions


final_pred = predict("./my_best_model.pt")
print(final_pred)

submission = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/sample_submission.csv")
submission.score = final_pred
print(submission.head())
submission.to_csv("submission.csv", index = False)

## --- ERROR in cell 5, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/roberta-pre/patent_pretrained'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/127442579.py in <cell line: 0>()
     89 
     90 
---> 91 tokenizer = AutoTokenizer.from_pretrained(model_n)
     92 test_dataset = PhraseTestDataset(
     93         anchor=df_test.anchor.values,

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/roberta-pre/patent_pretrained'. Use `repo_type` argument if needed.
