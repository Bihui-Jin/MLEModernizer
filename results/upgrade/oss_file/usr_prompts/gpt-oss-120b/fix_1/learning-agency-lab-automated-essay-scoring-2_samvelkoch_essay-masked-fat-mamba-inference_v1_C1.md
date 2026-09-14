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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8174961363497553

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch
import pandas as pd
import os

## === cell 3
print(torch.cuda.device_count())

## === cell 4
test = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv')

## === cell 5
reverse_score_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

## === cell 11
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

local_model_path_part_1 = "/kaggle/input/hf-proxy-d-trick-part-1-masked-fat-mamba"
local_model_path_part_2 = "/kaggle/input/hf-proxy-d-trick-part-2-masked-fat-mamba"
local_tokenizer_path = "/kaggle/input/hf-proxy-d-trick-part-1-masked-fat-mamba"

combined_model_path = "/kaggle/working/combined_model"
os.makedirs(combined_model_path, exist_ok=True)

for file_name in os.listdir(local_model_path_part_1):
    if not file_name.endswith(('.safetensors', '.json', '.bin', '.py')):  # Ссылки только на нужные файлы
        continue
    full_file_name = os.path.join(local_model_path_part_1, file_name)
    symlink_name = os.path.join(combined_model_path, file_name)
    if os.path.isfile(full_file_name) and not os.path.exists(symlink_name):
        os.symlink(full_file_name, symlink_name)

for file_name in os.listdir(local_model_path_part_2):
    if not file_name.endswith('.safetensors'):  # Ссылки только на модельные файлы
        continue
    full_file_name = os.path.join(local_model_path_part_2, file_name)
    symlink_name = os.path.join(combined_model_path, file_name)
    if os.path.isfile(full_file_name) and not os.path.exists(symlink_name):
        os.symlink(full_file_name, symlink_name)

tokenizer = AutoTokenizer.from_pretrained(
    local_tokenizer_path,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    combined_model_path,
    torch_dtype=torch.float16,
    device_map="auto",  # Использование обеих GPU
    trust_remote_code=True,
    local_files_only=True
).cuda().eval()  # Перенос модели на GPU и установка в режим оценки

head_weights_path = os.path.join(local_model_path_part_1, "classification_head.pth")

head_weights = torch.load(head_weights_path, map_location="cuda")

head = torch.nn.Linear(1, 1, bias=False).to("cuda")
head.weight.data = head_weights

print('Hooray! We put it in the memory!!!')


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2248918959.py in <cell line: 0>()
     13 
     14 # Создание символических ссылок для файлов из первой части
---> 15 for file_name in os.listdir(local_model_path_part_1):
     16     if not file_name.endswith(('.safetensors', '.json', '.bin', '.py')):  # Ссылки только на нужные файлы
     17         continue

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hf-proxy-d-trick-part-1-masked-fat-mamba'

## === cell 17
for idx, row in test.iterrows():
    text = row['full_text']
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to("cuda")
    
    with torch.no_grad():
        out = model(**inputs).logits
        logits = head(out[:, -1])
        predicted_score = torch.argmax(logits, dim=1).item()
        predicted_label = reverse_score_mapping[predicted_score]
        
        test.at[idx, 'score'] = predicted_label
        
        del inputs, out, logits
        torch.cuda.empty_cache()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3478276476.py in <cell line: 0>()
      1 for idx, row in test.iterrows():
      2     text = row['full_text']
----> 3     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False).to("cuda")
      4 
      5     with torch.no_grad():

NameError: name 'tokenizer' is not defined

## === cell 18
test

## === cell 20
test['score'] = test['score'].astype('int')

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'score'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/951020435.py in <cell line: 0>()
----> 1 test['score'] = test['score'].astype('int')

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'score'

## === cell 21
test

## === cell 22
test[['essay_id', 'score']].to_csv('submission.csv', index=False)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/303917544.py in <cell line: 0>()
----> 1 test[['essay_id', 'score']].to_csv('submission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['score'] not in index"
