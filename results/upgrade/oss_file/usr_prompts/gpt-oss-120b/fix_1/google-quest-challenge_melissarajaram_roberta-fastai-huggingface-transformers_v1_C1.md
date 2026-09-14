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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

fastai==2.8.5
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
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.2415212217454294

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 9
!pip install ../input/sacremoses/sacremoses-master 
!pip install ../input/transformers/transformers-master 

## === cell 10
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path 

import os

import torch
import torch.optim as optim

import random 

from fastai import *
from fastai.text import *
from fastai.callbacks import *

from scipy.stats import spearmanr

from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1118920148.py in <cell line: 0>()
     13 from fastai import *
     14 from fastai.text import *
---> 15 from fastai.callbacks import *
     16 
     17 # classification metric

ModuleNotFoundError: No module named 'fastai.callbacks'

## === cell 11
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

## === cell 13
def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

## === cell 14
seed=42
seed_all(seed)

## === cell 15
DATA_ROOT = Path("../input/google-quest-challenge/")
MODEL_ROOT = Path("../input/robertabasepretrained")
train = pd.read_csv(DATA_ROOT / 'train.csv')
test = pd.read_csv(DATA_ROOT / 'test.csv')
sample_sub = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
print(train.shape,test.shape)

## === cell 17
train.head()

## === cell 19
labels = list(sample_sub.columns[1:].values)

## === cell 20
for label in labels: print(label) 

## === cell 23
MODEL_CLASSES = {
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
}

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3276611801.py in <cell line: 0>()
      1 MODEL_CLASSES = {
----> 2     'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
      3 }

NameError: name 'RobertaForSequenceClassification' is not defined

## === cell 25
seed = 42
use_fp16 = False
bs = 16

model_type = 'roberta'
pretrained_model_name = 'roberta-base' # 'roberta-base-openai-detector'

## === cell 26
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3542010558.py in <cell line: 0>()
----> 1 model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]

NameError: name 'MODEL_CLASSES' is not defined

## === cell 27
model_class.pretrained_model_archive_map.keys()

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4261325733.py in <cell line: 0>()
----> 1 model_class.pretrained_model_archive_map.keys()

NameError: name 'model_class' is not defined

## === cell 32
class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
        return [CLS] + tokens + [SEP]

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3555905996.py in <cell line: 0>()
----> 1 class TransformersBaseTokenizer(BaseTokenizer):
      2     """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
      3     def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
      4         self._pretrained_tokenizer = pretrained_tokenizer
      5         self.max_seq_len = pretrained_tokenizer.max_len

NameError: name 'BaseTokenizer' is not defined

## === cell 33
transformer_tokenizer = tokenizer_class.from_pretrained(MODEL_ROOT)
transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3334921815.py in <cell line: 0>()
----> 1 transformer_tokenizer = tokenizer_class.from_pretrained(MODEL_ROOT)
      2 transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
      3 fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])

NameError: name 'tokenizer_class' is not defined

## === cell 36
class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2630521868.py in <cell line: 0>()
----> 1 class TransformersVocab(Vocab):
      2     def __init__(self, tokenizer: PreTrainedTokenizer):
      3         super(TransformersVocab, self).__init__(itos = [])
      4         self.tokenizer = tokenizer
      5 

NameError: name 'Vocab' is not defined

## === cell 39
transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)

transformer_processor = [tokenize_processor, numericalize_processor]

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/638737798.py in <cell line: 0>()
----> 1 transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
      2 numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)
      3 
      4 tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)
      5 

NameError: name 'TransformersVocab' is not defined

## === cell 42
pad_first = bool(model_type in ['xlnet'])
pad_idx = transformer_tokenizer.pad_token_id

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2919105309.py in <cell line: 0>()
      1 pad_first = bool(model_type in ['xlnet'])
----> 2 pad_idx = transformer_tokenizer.pad_token_id

NameError: name 'transformer_tokenizer' is not defined

## === cell 44
databunch = (TextList.from_df(train, cols=['question_title','question_body','answer'], processor=transformer_processor)
             .split_by_rand_pct(0.1,seed=seed)
             .label_from_df(cols=labels)
             .add_test(test)
             .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/387656981.py in <cell line: 0>()
----> 1 databunch = (TextList.from_df(train, cols=['question_title','question_body','answer'], processor=transformer_processor)
      2              .split_by_rand_pct(0.1,seed=seed)
      3              .label_from_df(cols=labels)
      4              .add_test(test)
      5              .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))

NameError: name 'TextList' is not defined

## === cell 46
print('[CLS] token :', transformer_tokenizer.cls_token)
print('[SEP] token :', transformer_tokenizer.sep_token)
print('[PAD] token :', transformer_tokenizer.pad_token)
databunch.show_batch()

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2066084304.py in <cell line: 0>()
----> 1 print('[CLS] token :', transformer_tokenizer.cls_token)
      2 print('[SEP] token :', transformer_tokenizer.sep_token)
      3 print('[PAD] token :', transformer_tokenizer.pad_token)
      4 databunch.show_batch()

NameError: name 'transformer_tokenizer' is not defined

## === cell 48
print('[CLS] id :', transformer_tokenizer.cls_token_id)
print('[SEP] id :', transformer_tokenizer.sep_token_id)
print('[PAD] id :', pad_idx)
test_one_batch = databunch.one_batch()[0]
print('Batch shape : ',test_one_batch.shape)
print(test_one_batch)

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3427965070.py in <cell line: 0>()
----> 1 print('[CLS] id :', transformer_tokenizer.cls_token_id)
      2 print('[SEP] id :', transformer_tokenizer.sep_token_id)
      3 print('[PAD] id :', pad_idx)
      4 test_one_batch = databunch.one_batch()[0]
      5 print('Batch shape : ',test_one_batch.shape)

NameError: name 'transformer_tokenizer' is not defined

## === cell 51
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
            
        logits = self.transformer(input_ids,
                                attention_mask = attention_mask)[0]   
        return logits

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3488655454.py in <cell line: 0>()
      1 # defining our model architecture
----> 2 class CustomTransformerModel(nn.Module):
      3     def __init__(self, transformer_model: PreTrainedModel):
      4         super(CustomTransformerModel,self).__init__()
      5         self.transformer = transformer_model

NameError: name 'nn' is not defined

## === cell 53
config = config_class.from_pretrained(MODEL_ROOT)
config.num_labels = 30
config.use_bfloat16 = use_fp16

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2161047988.py in <cell line: 0>()
----> 1 config = config_class.from_pretrained(MODEL_ROOT)
      2 config.num_labels = 30
      3 config.use_bfloat16 = use_fp16

NameError: name 'config_class' is not defined

## === cell 54
transformer_model = model_class.from_pretrained(MODEL_ROOT, config = config)
custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)

## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3849176627.py in <cell line: 0>()
----> 1 transformer_model = model_class.from_pretrained(MODEL_ROOT, config = config)
      2 custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)

NameError: name 'model_class' is not defined

## === cell 57
from fastai.callbacks import *
from transformers import AdamW

learner = Learner(databunch, 
                  custom_transformer_model, 
                  opt_func = lambda input: AdamW(input,correct_bias=False), 
                  metrics=[accuracy])

learner.callbacks.append(ShowGraph(learner))

if use_fp16: learner = learner.to_fp16()

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2078990171.py in <cell line: 0>()
----> 1 from fastai.callbacks import *
      2 from transformers import AdamW
      3 
      4 learner = Learner(databunch, 
      5                   custom_transformer_model,

ModuleNotFoundError: No module named 'fastai.callbacks'

## === cell 60
print(learner.model)

## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2107500321.py in <cell line: 0>()
----> 1 print(learner.model)

NameError: name 'learner' is not defined

## === cell 62
num_groups = len(learner.layer_groups)
print('Learner split in',num_groups,'groups')

## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/15740580.py in <cell line: 0>()
----> 1 num_groups = len(learner.layer_groups)
      2 print('Learner split in',num_groups,'groups')

NameError: name 'learner' is not defined

## === cell 64
list_layers = [learner.model.transformer.roberta.embeddings,
              learner.model.transformer.roberta.encoder.layer[0],
              learner.model.transformer.roberta.encoder.layer[1],
              learner.model.transformer.roberta.encoder.layer[2],
              learner.model.transformer.roberta.encoder.layer[3],
              learner.model.transformer.roberta.encoder.layer[4],
              learner.model.transformer.roberta.encoder.layer[5],
              learner.model.transformer.roberta.encoder.layer[6],
              learner.model.transformer.roberta.encoder.layer[7],
              learner.model.transformer.roberta.encoder.layer[8],
              learner.model.transformer.roberta.encoder.layer[9],
              learner.model.transformer.roberta.encoder.layer[10],
              learner.model.transformer.roberta.encoder.layer[11],
              learner.model.transformer.roberta.pooler]

learner.split(list_layers);

## --- ERROR in cell 64, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4246511267.py in <cell line: 0>()
----> 1 list_layers = [learner.model.transformer.roberta.embeddings,
      2               learner.model.transformer.roberta.encoder.layer[0],
      3               learner.model.transformer.roberta.encoder.layer[1],
      4               learner.model.transformer.roberta.encoder.layer[2],
      5               learner.model.transformer.roberta.encoder.layer[3],

NameError: name 'learner' is not defined

## === cell 66
num_groups = len(learner.layer_groups)
print('Learner split in',num_groups,'groups')

## --- ERROR in cell 66, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/15740580.py in <cell line: 0>()
----> 1 num_groups = len(learner.layer_groups)
      2 print('Learner split in',num_groups,'groups')

NameError: name 'learner' is not defined

## === cell 70
seed_all(seed)
learner.freeze_to(-1)

## --- ERROR in cell 70, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3089612420.py in <cell line: 0>()
      1 seed_all(seed)
----> 2 learner.freeze_to(-1)

NameError: name 'learner' is not defined

## === cell 72
learner.lr_find()

## --- ERROR in cell 72, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1200345430.py in <cell line: 0>()
----> 1 learner.lr_find()

NameError: name 'learner' is not defined

## === cell 73
learner.recorder.plot(skip_end=7,suggestion=True)

## --- ERROR in cell 73, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1771708776.py in <cell line: 0>()
----> 1 learner.recorder.plot(skip_end=7,suggestion=True)

NameError: name 'learner' is not defined

## === cell 75
learner.fit_one_cycle(1,max_lr=2e-04,moms=(0.8,0.7))

## --- ERROR in cell 75, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/226574613.py in <cell line: 0>()
----> 1 learner.fit_one_cycle(1,max_lr=2e-04,moms=(0.8,0.7))

NameError: name 'learner' is not defined

## === cell 76
seed_all(seed)

## === cell 79
learner.freeze_to(-2)

## --- ERROR in cell 79, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3998215353.py in <cell line: 0>()
----> 1 learner.freeze_to(-2)

NameError: name 'learner' is not defined

## === cell 80
lr = 1e-5

## === cell 82
learner.fit_one_cycle(1, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))

## --- ERROR in cell 82, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1477130219.py in <cell line: 0>()
----> 1 learner.fit_one_cycle(1, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))

NameError: name 'learner' is not defined

## === cell 83
seed_all(seed)

## === cell 86
learner.freeze_to(-3)

## --- ERROR in cell 86, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3692651125.py in <cell line: 0>()
----> 1 learner.freeze_to(-3)

NameError: name 'learner' is not defined

## === cell 87
learner.fit_one_cycle(1, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))

## --- ERROR in cell 87, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1477130219.py in <cell line: 0>()
----> 1 learner.fit_one_cycle(1, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))

NameError: name 'learner' is not defined

## === cell 90
def get_preds_as_nparray(ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in databunch.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]

test_preds = get_preds_as_nparray(DatasetType.Test)

## --- ERROR in cell 90, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4196237364.py in <cell line: 0>()
      9     return preds[reverse_sampler, :]
     10 
---> 11 test_preds = get_preds_as_nparray(DatasetType.Test)

NameError: name 'DatasetType' is not defined

## === cell 91
test_preds,test_preds.shape

## --- ERROR in cell 91, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/318021998.py in <cell line: 0>()
----> 1 test_preds,test_preds.shape

NameError: name 'test_preds' is not defined

## === cell 92
sample_submission = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
sample_submission[labels] = test_preds
sample_submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 92, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3925071404.py in <cell line: 0>()
      1 sample_submission = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
----> 2 sample_submission[labels] = test_preds
      3 sample_submission.to_csv("submission.csv", index=False)

NameError: name 'test_preds' is not defined

## === cell 94
test.head()

## === cell 95
sample_submission.head()
