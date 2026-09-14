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
joblib==1.5.2
lightning-utilities==0.15.2
numpy==1.26.4
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

0.7718595915073396

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!rm -rf /opt/conda/lib/python3.10/site-packages/aiohttp-3.9.1.dist-info
!pip install --no-index --find-links /kaggle/input/pytorch-ligthning-and-metrics-offline-install/ lightning

## === cell 1
import pandas as pd
import numpy as np
import torch
import glob
import gc
import joblib

from tqdm import tqdm

from transformers import (
    AutoTokenizer, 
    AutoModel,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
import lightning as L

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2379411300.py in <cell line: 0>()
     18 import torch.nn.functional as F
     19 from torch.utils.data import DataLoader, Dataset, Sampler
---> 20 import lightning as L

ModuleNotFoundError: No module named 'lightning'

## === cell 3
data = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv')
train_data = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv')
train_data_emb = np.load('/kaggle/input/aes-deberta-small-embedding-dataset/embedding_features.npy')

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1443944660.py in <cell line: 0>()
      1 data = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv')
      2 train_data = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv')
----> 3 train_data_emb = np.load('/kaggle/input/aes-deberta-small-embedding-dataset/embedding_features.npy')

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aes-deberta-small-embedding-dataset/embedding_features.npy'

## === cell 4
data = data.rename(columns={'full_text':'text'})

train_data['score'] = train_data['score'] - 1
train_data = train_data.rename(columns={'full_text':'text', 'score':'labels'})

## === cell 6
class EmbeddingsDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        output = self.tokenizer(self.data['text'].iloc[idx], return_tensors='pt', padding='max_length', max_length=1024, truncation=True)
        output = {k:v.squeeze(0) for k,v in output.items()}
        return output

## === cell 7
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

## === cell 8
model = AutoModel.from_pretrained("/kaggle/input/aes-deberta-small-embedding-dataset") 
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/aes-deberta-small-embedding-dataset")

## --- ERROR in cell 8, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/aes-deberta-small-embedding-dataset'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/975120144.py in <cell line: 0>()
----> 1 model = AutoModel.from_pretrained("/kaggle/input/aes-deberta-small-embedding-dataset")
      2 tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/aes-deberta-small-embedding-dataset")

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/aes-deberta-small-embedding-dataset'. Use `repo_type` argument if needed.

## === cell 9
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
embedding_features = np.empty((len(data), 0))

custom_ds = EmbeddingsDataset(data,tokenizer)
custom_dl = DataLoader(custom_ds, batch_size=32, shuffle=False)

model_emb_feature = []
for batch in tqdm(custom_dl,total=len(custom_dl)):
    with torch.no_grad():
        batch = {k:v.to(model.device) for k,v in batch.items()}
        outputs = model(**batch)
        outputs = mean_pooling(outputs[0], batch['attention_mask'])
        
        model_emb_feature.extend( outputs.detach().cpu().numpy() )

model_emb_feature = np.array(model_emb_feature)

embedding_features = np.hstack(
    (embedding_features, model_emb_feature)
)

del tokenizer, model, custom_ds, custom_dl, model_emb_feature
gc.collect()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/234335206.py in <cell line: 0>()
      1 device = 'cuda' if torch.cuda.is_available() else 'cpu'
----> 2 model.to(device)
      3 embedding_features = np.empty((len(data), 0))
      4 
      5 custom_ds = EmbeddingsDataset(data,tokenizer)

NameError: name 'model' is not defined

## === cell 11
kmeans = joblib.load('/kaggle/input/aes-deberta-s-pairwise-contrastive-regression/kmeans_task_clf.joblib')
data['task'] = kmeans.predict(embedding_features)
train_data['task'] = kmeans.predict(train_data_emb)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3384184066.py in <cell line: 0>()
----> 1 kmeans = joblib.load('/kaggle/input/aes-deberta-s-pairwise-contrastive-regression/kmeans_task_clf.joblib')
      2 data['task'] = kmeans.predict(embedding_features)
      3 train_data['task'] = kmeans.predict(train_data_emb)

/usr/local/lib/python3.11/dist-packages/joblib/numpy_pickle.py in load(filename, mmap_mode, ensure_native_byte_order)
    733             obj = _unpickle(fobj, ensure_native_byte_order=ensure_native_byte_order)
    734     else:
--> 735         with open(filename, "rb") as f:
    736             with _validate_fileobject_and_memmap(f, filename, mmap_mode) as (
    737                 fobj,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aes-deberta-s-pairwise-contrastive-regression/kmeans_task_clf.joblib'

## === cell 13
class CustomDataset(Dataset):
    def __init__(self, data, tasks, train_data, train_labels, train_task, n_sample=1):
        self.data = data
        self.tasks = tasks
        self.train_data = train_data
        self.train_labels = train_labels
        self.train_task = train_task
        
        self.n_sample = n_sample

        
        self.train_num_labels = len(np.unique(train_labels))
        self.train_label_indices = {i: np.where(train_labels == i)[0] for i in range(self.train_num_labels)}
        
        self.train_num_tasks = len(np.unique(train_task))
        self.train_task_indices = {i: np.where(train_task == i)[0] for i in range(self.train_num_tasks)}
        
        self.pairs = self.make_pairs()
        
    def make_pairs(self):
        pairs = np.empty((0,2), dtype=int)
        
        data_len = len(self.data)

        task_indices_sets = {task: set(indices) for task, indices in self.train_task_indices.items()}

        label_candidates = {
            label: np.array(list(set(indices) & set(range(len(self.train_data)))))
            for label, indices in self.train_label_indices.items()
        }
        
        print('Making essay pairs...')
        for i in tqdm(range(data_len)):
            e1_task = self.tasks[i]
            task_set = task_indices_sets[e1_task]

            for label, candidates in label_candidates.items():
                valid_candidates = candidates[np.isin(candidates, list(task_set))]
                if len(valid_candidates) > 0:
                    e2 = np.random.choice(valid_candidates, self.n_sample)
                    for j in e2:
                        pairs = np.vstack( 
                            ( 
                                pairs, 
                                np.array([i,j],dtype=int) 
                            ) 
                        )

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        
        e1 = self.pairs[idx][0]
        e2 = self.pairs[idx][1]

        x_e1 = self.data[e1]

        x_e2 = self.train_data[e2]
        y_e2 = self.train_labels[e2]

        output = {
            'x_e1': torch.tensor(x_e1, dtype=torch.float32),
            'x_e2': torch.tensor(x_e2, dtype=torch.float32),
            'relative_score': torch.tensor(0, dtype=torch.float32),
            'y_e1': torch.tensor(0, dtype=torch.float32),
            'y_e2': torch.tensor(y_e2, dtype=torch.float32)
        }
            
            
                
        return output

## === cell 14
class Projector(nn.Module):
    def __init__(self, d, dropout=0.2):
        super().__init__()
        
        self.p = dropout
        
        self.linear_1 = nn.Linear(d, int(d/2))
        self.bn_1 = nn.BatchNorm1d(int(d/2))
        
        self.linear_2 = nn.Linear(int(d/2), d)
        self.bn_2 = nn.BatchNorm1d(d)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = F.tanh(x)
        x = F.dropout(x, p=self.p)
        
        x = self.linear_2(x)
        x = self.bn_2(x)
        x = F.tanh(x)
        x = F.dropout(x, p=self.p)
        
        return x

## === cell 15
class AESNet(L.LightningModule):
    def __init__(self, input_dim=768, lr=1e-5):
        super(AESNet, self).__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        
        self.feature_extractor = nn.Sequential(
            Projector(input_dim),
            nn.Linear(input_dim, input_dim)
        )
        
        self.relative_score_head = nn.Linear(input_dim, 1, bias=False)
        
        self.mse_loss = nn.MSELoss()
    
    def _scale_back(self, x, min_x=-5, max_x=5):
        output = x * (max_x - min_x) + min_x 
        return output
        

    def forward(self, e1, e2, y_e2):
        x_1 = self.feature_extractor(e1)
        f1 = e1 + x_1
        f1 = F.normalize(f1, dim=-1)
        
        x_1 = self.feature_extractor(e2)
        f2 = e2 + x_1
        f2 = F.normalize(f2, dim=-1)
        
        dv = f1 - f2
        relative_score_hat = F.sigmoid(self.relative_score_head(dv))
        
        y_e1_hat = relative_score_hat + y_e2

        return relative_score_hat, y_e1_hat
    
    def training_step(self, batch, batch_idx):
        x_e1 = batch['x_e1']
        x_e2 = batch['x_e2']
        y_e1 = batch['y_e1']
        y_e2 = batch['y_e2']
        relative_score =  batch['relative_score']
        
        relative_score_hat, y_e1_hat = self.forward(
            x_e1,
            x_e2,
            y_e2
        )
        
        loss = self.mse_loss(relative_score_hat, relative_score)
        
        self.log('train loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_e1 = batch['x_e1']
        x_e2 = batch['x_e2']
        y_e1 = batch['y_e1']
        y_e2 = batch['y_e2']
        relative_score =  batch['relative_score']
        
        relative_score_hat, y_e1_hat = self.forward(
            x_e1,
            x_e2,
            y_e2
        )
        
        loss = self.mse_loss(relative_score_hat, relative_score)
        relative_score_hat = self._scale_back(relative_score_hat)
        relative_score = self._scale_back(relative_score)
  
        qwk = cohen_kappa_score(
            relative_score.detach().cpu().numpy().round(0).astype(int), 
            relative_score_hat.detach().cpu().numpy().round(0).astype(int), 
            weights='quadratic'
        ) 
        
        self.log('validation loss', loss, prog_bar=True)
        self.log('validation qwk', qwk, prog_bar=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_e1 = batch['x_e1']
        x_e2 = batch['x_e2']
        y_e1 = batch['y_e1']
        y_e2 = batch['y_e2']
        
        relative_score =  batch['relative_score']

        relative_score_hat, y_e1_hat = self.forward(
            x_e1,
            x_e2,
            y_e2
        )
        
        relative_score_hat = self._scale_back(relative_score_hat)
        y_e1_hat = relative_score_hat + y_e2
        
        return y_e1_hat
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 1.0,
            end_factor = 0.6,
            total_iters = self.trainer.max_epochs
        )
        '''scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches
        )'''
        return {"optimizer": optimizer, "lr_scheduler": {'scheduler': scheduler, 'interval':'epoch'}}

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2284209776.py in <cell line: 0>()
----> 1 class AESNet(L.LightningModule):
      2     def __init__(self, input_dim=768, lr=1e-5):
      3         super(AESNet, self).__init__()
      4         self.save_hyperparameters()
      5 

NameError: name 'L' is not defined

## === cell 16
test_ds = CustomDataset(
    embedding_features, 
    data['task'].values,
    train_data_emb,
    train_data[['labels']].values,
    train_data['task'].values
)
test_dl = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=4, drop_last=False)

model_path = glob.glob(f'/kaggle/input/aes-deberta-s-pairwise-contrastive-regression/models/**/*.ckpt', recursive=True)
test_pred = np.zeros((len(data),1))
model_count = 0
for path in model_path:
    print(path)
    model = AESNet.load_from_checkpoint(path)
    trainer = L.Trainer()
    
    output = trainer.predict(model=model, dataloaders=test_dl)
    output = torch.vstack(output).detach().cpu().numpy()
    
    output = pd.DataFrame({
        'essay_idx': test_ds.pairs[:,0],
        'pred_label': output.flatten()
    })

    output = output.groupby('essay_idx').agg({'pred_label':'mean'})['pred_label'].values
    
    test_pred += output.reshape((-1,1))
    model_count += 1
    

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/88603741.py in <cell line: 0>()
      1 test_ds = CustomDataset(
----> 2     embedding_features,
      3     data['task'].values,
      4     train_data_emb,
      5     train_data[['labels']].values,

NameError: name 'embedding_features' is not defined

## === cell 17
test_pred = (test_pred / model_count).round(0).astype(int)+1

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1088863836.py in <cell line: 0>()
----> 1 test_pred = (test_pred / model_count).round(0).astype(int)+1

NameError: name 'test_pred' is not defined

## === cell 18
submission_df = pd.DataFrame({
    'essay_id': data['essay_id'].values,
    'score': test_pred.flatten()
})
submission_df.head()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3790794448.py in <cell line: 0>()
      1 submission_df = pd.DataFrame({
      2     'essay_id': data['essay_id'].values,
----> 3     'score': test_pred.flatten()
      4 })
      5 submission_df.head()

NameError: name 'test_pred' is not defined

## === cell 19
submission_df.to_csv('submission.csv', index=False)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1211055984.py in <cell line: 0>()
----> 1 submission_df.to_csv('submission.csv', index=False)

NameError: name 'submission_df' is not defined
