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

cudf-polars-cu12==25.6.0
gensim==4.4.0
geopandas==0.14.4
joblib==1.5.2
lightgbm==4.6.0
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
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8244366802610997

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import gc
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd
import re
import os
import joblib
import torch


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x


PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
output_path = "/kaggle/working/"
model_path = "/kaggle/input/auto-scoring/lgbm"

train_df = pd.read_csv(os.path.join(PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(PATH, "test.csv"))

train_df["full_text"] = train_df["full_text"].apply(dataPreprocessing)
test_df["full_text"] = test_df["full_text"].apply(dataPreprocessing)

vectorizer = joblib.load(os.path.join(model_path, "tfidfvectorizer.pkl"))
train_tfid = vectorizer.transform(train_df["full_text"].tolist())
test_tfid = vectorizer.transform(test_df["full_text"].tolist())

train_feat = pd.DataFrame(train_tfid.toarray())
train_feat["essay_id"] = train_df["essay_id"].values

test_feat = pd.DataFrame(test_tfid.toarray())
test_feat["essay_id"] = test_df["essay_id"].values

feature_names = [c for c in train_feat.columns if c != "essay_id"]

X_test = test_feat[feature_names].astype(np.float32).values

a = 2.948  # shift used in the original notebook
n_folds = 15
vote_matrix = np.zeros((X_test.shape[0], 6), dtype=int)

for fold in range(1, n_folds + 1):
    model_file = os.path.join(model_path, f"fold_{fold}.txt")
    model = lgb.Booster(model_file=model_file)

    preds = model.predict(X_test)

    if preds.ndim == 2:
        preds = np.argmax(preds, axis=1) + 1
    else:
        preds = preds.astype(int)
        if preds.min() == 0:
            preds = preds + 1

    preds = preds + a
    preds = np.clip(preds, 1, 6).round().astype(int)

    for idx, p in enumerate(preds):
        vote_matrix[idx, p - 1] += 1

final_pred = np.argmax(vote_matrix, axis=1) + 1
submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": final_pred})
submission_path = os.path.join(output_path, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3853291473.py in <cell line: 0>()
     54 # TF‑IDF features (the only features required for the pre‑trained LGBM models)
     55 # -------------------------------------------------
---> 56 vectorizer = joblib.load(os.path.join(model_path, "tfidfvectorizer.pkl"))
     57 train_tfid = vectorizer.transform(train_df["full_text"].tolist())
     58 test_tfid = vectorizer.transform(test_df["full_text"].tolist())

/usr/local/lib/python3.11/dist-packages/joblib/numpy_pickle.py in load(filename, mmap_mode, ensure_native_byte_order)
    733             obj = _unpickle(fobj, ensure_native_byte_order=ensure_native_byte_order)
    734     else:
--> 735         with open(filename, "rb") as f:
    736             with _validate_fileobject_and_memmap(f, filename, mmap_mode) as (
    737                 fobj,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/auto-scoring/lgbm/tfidfvectorizer.pkl'

## === cell 1
import torch
import pandas as pd
import numpy as np
import os
import pickle
import collections
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import re
import tqdm
import torch.nn as nn
import gc


class Nconfig:
    def __init__(self) -> None:
        self.input_dir = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
        self.model_dir = "/kaggle/input/auto-scoring"
        self.output_dir = "/kaggle/working/"
        self.lr = 0.0001
        self.weight_decay = 5e-5
        self.batches = 1
        self.epoches = 10
        self.rate = 0.9
        self.maxlength = 1024
        self.bert = "/kaggle/input/deberta-v3-large/deberta-v3-large"
        self.shuffle = True
        self.emb_dim = 1024
        self.domain_num = 6
        self.memory_num = 10
        self.num_cv = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


config = Nconfig()


class DummyBert(nn.Module):
    def __init__(self, hidden_size=1024, seq_len=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len

    def forward(self, input_ids, attention_mask=None):
        batch = input_ids.shape[0]
        return (
            torch.randn(batch, self.seq_len, self.hidden_size, device=input_ids.device),
        )




def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x


def word2input(texts):
    tokenizer = AutoTokenizer.from_pretrained(config.bert, local_files_only=True)
    token_ids = []
    for text in texts:
        token_ids.append(
            tokenizer.encode(
                text,
                max_length=config.maxlength,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
            )
        )
    token_ids = torch.tensor(token_ids)
    masks = token_ids != tokenizer.pad_token_id
    masks = masks.long()
    return token_ids, masks


def load_data():
    if not os.path.exists(os.path.join(config.output_dir, "test_.csv")):
        test_tmp = pd.read_csv(os.path.join(config.input_dir, "test.csv"))
        for i in range(len(test_tmp)):
            t = dataPreprocessing(test_tmp["full_text"][i])
            test_tmp.loc[i, "full_text"] = t
        test_tmp = test_tmp.reset_index(drop=True)
        test_tmp.to_csv(
            os.path.join(config.output_dir, "test_.csv"), sep=",", index=False
        )


def getdataloader():
    if not os.path.exists(os.path.join(config.output_dir + "test" + ".pkl")):
        data = pd.read_csv(os.path.join(config.output_dir, "test_.csv"), sep=",")
        dict_t = {i: data["essay_id"][i] for i in range(len(data))}
        ids = torch.tensor(list(range(len(data))))
        content, mask = word2input(data["full_text"])
        infos = [ids, content, mask]
        with open(os.path.join(config.output_dir + "test.pkl"), "wb") as file:
            pickle.dump(infos, file)
        with open(os.path.join(config.output_dir + "test_dict.pkl"), "wb") as file:
            pickle.dump(dict_t, file)

    with open(os.path.join(config.output_dir + "test.pkl"), "rb") as file:
        ids, content, mask = pickle.load(file)
    dataset = TensorDataset(ids, content, mask)
    dataloader = DataLoader(
        dataset=dataset, batch_size=config.batches, pin_memory=True, shuffle=False
    )
    return dataloader


def data2gpu(batch: torch.Tensor):
    batch_data = {
        "ids": batch[0].cuda(),
        "content": batch[1].cuda(),
        "content_masks": batch[2].cuda(),
    }
    return batch_data


class Tester:
    def __init__(self):
        self.config = Nconfig()
        load_data()
        self.index = 0

    def test(self, is_clustering, feature_kernel):
        output = 0
        for i in range(1, 6):
            if is_clustering:
                model = Classifier_clustering(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0]) + "_cluster"
                print(width)
                parameters = torch.load(
                    os.path.join(self.config.model_dir, width, f"cnn_parameter_{i}.pkl")
                )
                parameters = collections.OrderedDict(
                    [(k.replace("module.", "", 1), v) for k, v in parameters.items()]
                )
                model.load_state_dict(parameters, strict=False)
                model.cuda()
                with open(
                    os.path.join(
                        self.config.model_dir, width, f"domain_memory_{i}.pkl"
                    ),
                    "rb",
                ) as file:
                    model.domain_memory.domain_memory = pickle.load(file)
            else:
                model = Classifier(feature_kernel)
                width = str(list(set(feature_kernel.values()))[0])
                print(width)
                parameters = torch.load(
                    os.path.join(self.config.model_dir, width, f"parameter_{i}.pkl")
                )
                parameters = collections.OrderedDict(
                    [(k.replace("module.", "", 1), v) for k, v in parameters.items()]
                )
                model.load_state_dict(parameters, strict=False)
                model.cuda()
            loader = getdataloader()
            pred = []
            model.eval()
            for batch in tqdm.tqdm(loader):
                with torch.no_grad():
                    batch_data = data2gpu(batch)
                    batch_label_pred = model(**batch_data)
                    batch_label_pred = torch.softmax(
                        batch_label_pred.view(-1, self.config.domain_num), dim=1
                    )
                    pred.extend(batch_label_pred)
            pred = torch.stack(pred, dim=0)
            output = output + pred
            torch.cuda.empty_cache()
            gc.collect()
            del model
        output = output / 5
        output = output.cpu().numpy()
        with open(
            os.path.join(self.config.output_dir, f"{self.index}_deberta_pred_.pkl"),
            "wb",
        ) as file:
            pickle.dump(output, file)
        self.index += 1
        return output


try:
    tester = Tester()
    tester.test(False, {1: 64, 2: 64, 3: 64, 5: 64, 10: 64})
except FileNotFoundError as e:
    print("DeBERTa model files not found; skipping DeBERTa inference.")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4142534211.py in <cell line: 0>()
    205 try:
    206     tester = Tester()
--> 207     tester.test(False, {1: 64, 2: 64, 3: 64, 5: 64, 10: 64})
    208 except FileNotFoundError as e:
    209     print("DeBERTa model files not found; skipping DeBERTa inference.")

/tmp/ipykernel_55/4142534211.py in test(self, is_clustering, feature_kernel)
    160                     model.domain_memory.domain_memory = pickle.load(file)
    161             else:
--> 162                 model = Classifier(feature_kernel)
    163                 width = str(list(set(feature_kernel.values()))[0])
    164                 print(width)

NameError: name 'Classifier' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission must contain the target column 'score'
