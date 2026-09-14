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

0.8251491837546732

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
import numpy as np
import os
import pickle
import collections
from torch.utils.data import TensorDataset,DataLoader
from transformers import AutoTokenizer,AutoModel
import re
import tqdm
import torch.nn as nn
import torch.functional as F
import gc

class Nconfig():
    def __init__(self) -> None:
        self.input_dir='/kaggle/input/learning-agency-lab-automated-essay-scoring-2'
        self.model_dir='/kaggle/input/auto-scoring'
        self.output_dir='/kaggle/working/'
        self.lr=0.0001
        self.weight_decay=5e-5
        self.batches=1
        self.epoches=10
        self.rate=0.9
        self.maxlength=1024
        
        self.bert='/kaggle/input/deberta-v3-large/deberta-v3-large'
        self.shuffle=True
        
        self.emb_dim=1024
        self.domain_num=6
        self.memory_num=10

        self.num_cv=[0,0.2,0.4,0.6,0.8,1.0]
config=Nconfig()

class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])
        input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature
    

class MemoryNetwork(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num):
        super(MemoryNetwork, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.memory_num = memory_num
        self.tau = 32
        self.topic_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_fc = torch.nn.Linear(input_dim, emb_dim, bias=False)
        self.domain_memory = dict()

    def forward(self, feature):

        domain_memory = {}
        for i in range(self.domain_num):
            domain_memory[i]=self.domain_memory[i]

        sep_domain_embedding = []
        for i in range(self.domain_num):
            topic_att = torch.nn.functional.softmax(torch.mm(self.topic_fc(feature), domain_memory[i].T) * self.tau, dim=1)
            tmp_domain_embedding = torch.mm(topic_att, domain_memory[i]).unsqueeze(1)
            sep_domain_embedding.append(tmp_domain_embedding)
        sep_domain_embedding = torch.cat(sep_domain_embedding, 1)

        domain_att = torch.bmm(sep_domain_embedding, self.domain_fc(feature).unsqueeze(2)).squeeze()

        return domain_att
    

    def write(self, all_feature, category):
        domain_fea_dict = {}
        domain_set = set(category.cpu().detach().numpy().tolist())
        for i in domain_set:
            domain_fea_dict[i] = []
        for i in range(all_feature.size(0)):
            domain_fea_dict[category[i].item()].append(all_feature[i].view(1, -1))

        for i in domain_set:
            domain_fea_dict[i] = torch.cat(domain_fea_dict[i], 0)
            topic_att = torch.nn.functional.softmax(torch.mm(self.topic_fc(domain_fea_dict[i]), self.domain_memory[i].T) * self.tau, dim=1).unsqueeze(2)
            tmp_fea = domain_fea_dict[i].unsqueeze(1).repeat(1, self.memory_num, 1)
            new_mem = tmp_fea * topic_att
            new_mem = new_mem.mean(dim = 0)
            topic_att = torch.mean(topic_att, 0).view(-1, 1)
            self.domain_memory[i] = self.domain_memory[i] - 0.05 * topic_att * self.domain_memory[i] + 0.05 * new_mem

class Classifier_clustering(nn.Module):
    def __init__(self,feature_kernel):
        super(Classifier_clustering,self).__init__()
        config=Nconfig()
        self.bert = AutoModel.from_pretrained(config.bert).requires_grad_(False)
        
        mid_dim=sum(feature_kernel.values())
        self.sen_extractor = cnn_extractor(feature_kernel,config.emb_dim)
        self.domain_memory = MemoryNetwork(mid_dim,mid_dim,config.domain_num,config.memory_num)
        self.FFN=nn.Sequential(nn.Linear(in_features=mid_dim,out_features=mid_dim*2,bias=False),nn.ReLU(),nn.Linear(in_features=mid_dim*2,out_features=mid_dim,bias=False),nn.ReLU())
        self.memory_num=config.memory_num
        self.mid_dim=mid_dim
        self.all_feature={}

    def forward(self,**kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']
        content_feature = self.bert(content, attention_mask = content_masks)[0]
        T_feature= self.sen_extractor(content_feature)
        F_feature=self.FFN(T_feature)
        output = self.domain_memory(F_feature)
        return output

class Classifier(nn.Module):
    def __init__(self,feature_kernel):
        super(Classifier,self).__init__()
        config=Nconfig()
        self.bert = AutoModel.from_pretrained(config.bert).requires_grad_(False)
        if config.bert=='./english_roberta_base/':
            t=list(self.bert.children())
            t[-1].requires_grad_(True)
                
            t=list(list(t[1].children())[0].children())
            t[-1].requires_grad_(True)
            self.t=t
        elif config.bert=='./deberta-v3-base/':
            t=list(list(self.bert.children())[-1].children())
            t[-1].requires_grad_(True)
            t[-2].requires_grad_(True)
            t=list(t[0].children())
            t[-1].requires_grad_(True)
            self.t=t
        elif config.bert=='./deberta-v3-large/':
            t=list(list(self.bert.children())[-1].children())
            t[-1].requires_grad_(True)
            t[-2].requires_grad_(True)
            t=list(t[0].children())
            t[-1].requires_grad_(True)
            self.t=t
        mid_dim=sum(feature_kernel.values())
        self.sen_extractor = cnn_extractor(feature_kernel,config.emb_dim)
        self.FFN=nn.Sequential(nn.Linear(in_features=mid_dim,out_features=mid_dim*2,bias=False),nn.ReLU(),nn.Linear(in_features=mid_dim*2,out_features=mid_dim,bias=False),nn.ReLU())
        self.classihead=nn.Linear(in_features=mid_dim,out_features=config.domain_num,bias=False)
        self.memory_num=config.memory_num
        self.mid_dim=mid_dim
        self.all_feature={}

    def forward(self,**kwargs):
        content = kwargs['content']
        content_masks = kwargs['content_masks']
        content_feature = self.bert(content, attention_mask = content_masks)[0]
        T_feature= self.sen_extractor(content_feature)
        F_feature=self.FFN(T_feature)
        output=self.classihead(F_feature)
        return output

def removeHTML(x):
    html=re.compile(r'<.*?>')
    return html.sub(r'',x)
def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", '',x)
    x = re.sub("'\d+", '',x)
    x = re.sub("\d+", '',x)
    x = re.sub("http\w+", '',x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x

def word2input(texts):
    tokenizer = AutoTokenizer.from_pretrained(config.bert)
    token_ids = []
    length=list(map(lambda i:i.count(' '),texts))
    for i,text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=config.maxlength, add_special_tokens=True, padding='max_length',
                            truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)

    return token_ids, masks

def load_data():
    

    if not os.path.exists(os.path.join(config.output_dir,'test_.csv')):
        test_tmp=pd.read_csv(os.path.join(config.input_dir,'test.csv'))
        for i in range(len(test_tmp)):
            t=dataPreprocessing(test_tmp['full_text'][i])
            test_tmp.loc[i,'full_text']=t

        test_tmp=test_tmp.reset_index(drop=True)
        test_tmp.to_csv('test_.csv',sep=',',index=False)

def getdataloader():
    if not os.path.exists(os.path.join(config.output_dir+'test'+'.pkl')):
        data=pd.read_csv(config.output_dir+'test_.csv',sep=',')
        dict_t={i:data['essay_id'][i] for i in range(len(data))}
        ids=torch.tensor(list(range(len(data))))
        content,mask=word2input(data['full_text'])
        
        
        infos=[ids,content,mask]
        
        with open(os.path.join(config.output_dir+'test.pkl'),'wb') as file:
            pickle.dump(infos,file)
        with open(os.path.join(config.output_dir+'test_dict.pkl'),'wb') as file:
            pickle.dump(dict_t,file)
    
    
    with open(os.path.join(config.output_dir+'test.pkl'),'rb') as file:
        ids,content,mask=pickle.load(file)
    dataset=TensorDataset(
                        ids,
                        content,
                        mask
                        )

    dataloader=DataLoader(dataset=dataset,batch_size=config.batches,pin_memory=True,shuffle=False)
    return dataloader

def data2gpu(batch:torch.Tensor):
    batch_data={
        'ids':batch[0].cuda(),
        'content': batch[1].cuda(),
        'content_masks': batch[2].cuda()
    }
    return batch_data

class Tester():
    def __init__(self):
        
        self.config=Nconfig()
        load_data()
        self.index=0
    def test(self,is_clustering,feature_kernel):
        output=0
        
        for i in range(1,6):
            if is_clustering:
                model=Classifier_clustering(feature_kernel)
                width=str(list(set(feature_kernel.values()))[0])+'_cluster'
                print(width)
                parameters=torch.load(os.path.join(self.config.model_dir,width, 'cnn_parameter_'+str(i)+'.pkl'))
                parameters = collections.OrderedDict([(k.replace('module.','',1),v) for k, v in parameters .items()])

                model.load_state_dict(parameters,strict=False)
                model.cuda()
                with open(os.path.join(self.config.model_dir,width,'domain_memory_'+str(i)+'.pkl'),'rb') as file:
                    model.domain_memory.domain_memory = pickle.load(file)
            else:
                model=Classifier(feature_kernel)
                width=str(list(set(feature_kernel.values()))[0])
                print(width)
                parameters=torch.load(os.path.join(self.config.model_dir, width,'parameter_'+str(i)+'.pkl'))
                parameters = collections.OrderedDict([(k.replace('module.','',1),v) for k, v in parameters .items()])

                model.load_state_dict(parameters,strict=False)
                model.cuda()

            loader=getdataloader()
            test_dict=pd.read_pickle('./test_dict.pkl')
            pred = []
            model.eval()
            
            data_iter = tqdm.tqdm(loader)

            for step_n, batch in enumerate(data_iter):
                with torch.no_grad():
                    batch_data = data2gpu(batch)
                    batch_label_pred = model(**batch_data)
                    batch_label_pred=torch.softmax(batch_label_pred.view(-1,self.config.domain_num),dim=1)
                    pred.extend(batch_label_pred)
            pred=torch.stack(pred,dim=0)
            output=output+pred
            torch.cuda.empty_cache()
            gc.collect()
            del model

        output=output / 5
        output=output.cpu().numpy()
        with open(os.path.join(self.config.output_dir,str(self.index)+'_deberta_pred_.pkl'),'wb') as file:
            pickle.dump(output,file)
        self.index+=1
        return output

tester=Tester()
tester.test(False,{1: 64, 2: 64, 3: 64, 5: 64, 10: 64})
tester.test(True,{1: 256, 2: 256, 3: 256, 5: 256, 10: 256})

## --- ERROR in cell 0, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/deberta-v3-large/deberta-v3-large'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/1028782543.py in <cell line: 0>()
    316 
    317 tester=Tester()
--> 318 tester.test(False,{1: 64, 2: 64, 3: 64, 5: 64, 10: 64})
    319 tester.test(True,{1: 256, 2: 256, 3: 256, 5: 256, 10: 256})

/tmp/ipykernel_11/1028782543.py in test(self, is_clustering, feature_kernel)
    280                     model.domain_memory.domain_memory = pickle.load(file)
    281             else:
--> 282                 model=Classifier(feature_kernel)
    283                 width=str(list(set(feature_kernel.values()))[0])
    284                 print(width)

/tmp/ipykernel_11/1028782543.py in __init__(self, feature_kernel)
    125         super(Classifier,self).__init__()
    126         config=Nconfig()
--> 127         self.bert = AutoModel.from_pretrained(config.bert).requires_grad_(False)
    128         if config.bert=='./english_roberta_base/':
    129             t=list(self.bert.children())

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/deberta-v3-large/deberta-v3-large'. Use `repo_type` argument if needed.

## === cell 1
import gc
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import random
from sklearn.ensemble import VotingRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import cohen_kappa_score, f1_score
from lightgbm import log_evaluation, early_stopping
import polars as pl
import joblib
from gensim.models import Word2Vec
from sklearn.decomposition import LatentDirichletAllocation
import scipy
import os

columns = [  
    (
        pl.col("full_text").str.split(by="\n\n").alias("paragraph")
    ),
]
PATH = '/kaggle/input/learning-agency-lab-automated-essay-scoring-2'
output_path='/kaggle/working/'
model_path='/kaggle/input/auto-scoring/lgbm'
train = pl.read_csv(os.path.join(PATH,"test.csv")).with_columns(columns)
deberta_num=2

train.head(1)

def removeHTML(x):
    html=re.compile(r'<.*?>')
    return html.sub(r'',x)
def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", '',x)
    x = re.sub("'\d+", '',x)
    x = re.sub("\d+", '',x)
    x = re.sub("http\w+", '',x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = x.strip()
    return x

def Paragraph_Preprocess(tmp):
    tmp = tmp.explode('paragraph')
    tmp = tmp.with_columns(pl.col('paragraph').map_elements(dataPreprocessing))
    tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x)).alias("paragraph_len"))
    tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x.split('.'))).alias("paragraph_sentence_cnt"),
                    pl.col('paragraph').map_elements(lambda x: len(x.split(' '))).alias("paragraph_word_cnt"),)
    return tmp
paragraph_fea = ['paragraph_len','paragraph_sentence_cnt','paragraph_word_cnt']
def Paragraph_Eng(train_tmp):
    aggs = [
        *[pl.col('paragraph').filter(pl.col('paragraph_len') >= i).count().alias(f"paragraph_{i}_cnt") for i in [50,75,100,125,150,175,200,250,300,350,400,500,600,700] ], 
        *[pl.col('paragraph').filter(pl.col('paragraph_len') <= i).count().alias(f"paragraph_{i}_cnt") for i in [25,49]], 
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in paragraph_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in paragraph_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in paragraph_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in paragraph_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_fea],
        ]
    df = train_tmp.groupby(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df

tmp = Paragraph_Preprocess(train)
train_feats = Paragraph_Eng(tmp)

feature_names = list(filter(lambda x: x not in ['essay_id'], train_feats.columns))
print('Features Number: ',len(feature_names))
train_feats.head(3)

def Sentence_Preprocess(tmp):
    tmp = tmp.with_columns(pl.col('full_text').map_elements(dataPreprocessing).str.split(by=".").alias("sentence"))
    tmp = tmp.explode('sentence')
    tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x)).alias("sentence_len"))
    tmp = tmp.filter(pl.col('sentence_len')>=15)
    tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x.split(' '))).alias("sentence_word_cnt"))
    
    return tmp
sentence_fea = ['sentence_len','sentence_word_cnt']
def Sentence_Eng(train_tmp):
    aggs = [
        *[pl.col('sentence').filter(pl.col('sentence_len') >= i).count().alias(f"sentence_{i}_cnt") for i in [15,50,100,150,200,250,300] ], 
        *[pl.col(fea).max().alias(f"{fea}_max") for fea in sentence_fea],
        *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in sentence_fea],
        *[pl.col(fea).min().alias(f"{fea}_min") for fea in sentence_fea],
        *[pl.col(fea).first().alias(f"{fea}_first") for fea in sentence_fea],
        *[pl.col(fea).last().alias(f"{fea}_last") for fea in sentence_fea],
        ]
    df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df

tmp = Sentence_Preprocess(train)
train_feats = train_feats.merge(Sentence_Eng(tmp), on='essay_id', how='left')

feature_names = list(filter(lambda x: x not in ['essay_id'], train_feats.columns))
print('Features Number: ',len(feature_names))
train_feats.head(3)

def Word_Preprocess(tmp):
    tmp = tmp.with_columns(pl.col('full_text').map_elements(dataPreprocessing).str.split(by=" ").alias("word"))
    tmp = tmp.explode('word')
    tmp = tmp.with_columns(pl.col('word').map_elements(lambda x: len(x)).alias("word_len"))
    tmp = tmp.filter(pl.col('word_len')!=0)
    
    return tmp
def Word_Eng(train_tmp):
    aggs = [
        *[pl.col('word').filter(pl.col('word_len') >= i+1).count().alias(f"word_{i+1}_cnt") for i in range(15) ], 
        pl.col('word_len').max().alias(f"word_len_max"),
        pl.col('word_len').mean().alias(f"word_len_mean"),
        pl.col('word_len').std().alias(f"word_len_std"),
        pl.col('word_len').quantile(0.25).alias(f"word_len_q1"),
        pl.col('word_len').quantile(0.50).alias(f"word_len_q2"),
        pl.col('word_len').quantile(0.75).alias(f"word_len_q3"),
        ]
    df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
    df = df.to_pandas()
    return df

tmp = Word_Preprocess(train)
train_feats = train_feats.merge(Word_Eng(tmp), on='essay_id', how='left')

feature_names = list(filter(lambda x: x not in ['essay_id'], train_feats.columns))
print('Features Number: ',len(feature_names))
train_feats.head(3)


def op(x):
    return x
vectorizer=joblib.load(os.path.join(model_path,'tfidfvectorizer.pkl'))
train_tfid = vectorizer.transform([i for i in train['full_text']])
dense_matrix = train_tfid.toarray()
df = pd.DataFrame(dense_matrix)
tfid_columns = [ f'tfid_{i}' for i in range(len(df.columns))]
df.columns = tfid_columns
df['essay_id'] = train_feats['essay_id']
train_feats = train_feats.merge(df, on='essay_id', how='left')

feature_names = list(filter(lambda x: x not in ['essay_id'], train_feats.columns))
print('Features Number: ',len(feature_names))
train_feats.head(3)

for j in range(deberta_num):
    deberta_oof = joblib.load(os.path.join(output_path,str(j)+'_deberta_pred_.pkl'))
    
    for i in range(6):
        train_feats[f'{j}_deberta_oof_{i}'] = deberta_oof[:, i]



feature_names = list(filter(lambda x: x not in ['essay_id'], train_feats.columns))
print('Features Number: ', len(feature_names))    


def quadratic_weighted_kappa(y_true, y_pred):
    y_true = y_true + a
    y_pred = (y_pred + a).clip(1, 6).round()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return 'QWK', qwk, True
def qwk_obj(y_true, y_pred):
    labels = y_true + a
    preds = y_pred + a
    preds = preds.clip(1, 6)
    f = 1/2*np.sum((preds-labels)**2)
    g = 1/2*np.sum((preds-a)**2+b)
    df = preds - labels
    dg = preds - a
    grad = (df/g - f*dg/g**2)*len(labels)
    hess = np.ones(len(labels))
    return grad, hess
a = 2.948
b = 1.092

X = train_feats[feature_names].astype(np.float32).values

n_splits = 15 # Define the number of splits for cross-validation
models = []

for i in range(1,n_splits+1):
    models.append(lgb.Booster(model_file=os.path.join(model_path,'fold_'+str(i)+'.txt')))
output=np.zeros((X.shape[0],6))
output=0.0
for model in models:
    predictions = model.predict(X)
    predictions = predictions + a
    output=output+predictions
output=output/len(models)
output = output.clip(1, 6).round().astype(np.int32)

ids=pd.read_csv(os.path.join(PATH , "test.csv"))['essay_id'].to_numpy()
submission={'essay_id':ids,'score':output}
submission=pd.DataFrame(submission)
submission.to_csv(os.path.join(output_path,'submission.csv'), index=False)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1485297985.py in <cell line: 0>()
     93 
     94 tmp = Paragraph_Preprocess(train)
---> 95 train_feats = Paragraph_Eng(tmp)
     96 
     97 # Obtain feature names

/tmp/ipykernel_11/1485297985.py in Paragraph_Eng(train_tmp)
     88         *[pl.col(fea).last().alias(f"{fea}_last") for fea in paragraph_fea],
     89         ]
---> 90     df = train_tmp.groupby(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
     91     df = df.to_pandas()
     92     return df

AttributeError: 'DataFrame' object has no attribute 'groupby'

## --- ERROR in outputing the csv:
Invalid submission: Submission must contain the target column 'score'
