# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.12

# 2. Installed packages

catboost==1.2.8
cudf-polars-cu12==25.6.0
geopandas==0.14.4
imbalanced-learn==0.13.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import re
import polars as pl
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.ensemble import VotingClassifier,VotingRegressor
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm,trange
from lightgbm import log_evaluation, early_stopping
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE,BorderlineSMOTE
from collections import Counter
from sklearn.metrics import confusion_matrix


## --- ERROR in cell 0, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2965636326.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     17[0m [0;32mfrom[0m [0mcatboost[0m [0;32mimport[0m [0mCatBoostClassifier[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mmetrics[0m [0;32mimport[0m [0mprecision_score[0m[0;34m,[0m [0mrecall_score[0m[0;34m,[0m [0mf1_score[0m[0;34m,[0m [0mclassification_report[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 19[0;31m [0;32mfrom[0m [0mimblearn[0m[0;34m.[0m[0mover_sampling[0m [0;32mimport[0m [0mSMOTE[0m[0;34m,[0m[0mBorderlineSMOTE[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;32mfrom[0m [0mcollections[0m [0;32mimport[0m [0mCounter[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mmetrics[0m [0;32mimport[0m [0mconfusion_matrix[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/imblearn/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     50[0m     [0;31m# process, as it may not be compiled yet[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 52[0;31m     from . import (
[0m[1;32m     53[0m         [0mcombine[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     54[0m         [0mensemble[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/imblearn/combine/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m """
[1;32m      4[0m [0;34m[0m[0m
[0;32m----> 5[0;31m [0;32mfrom[0m [0;34m.[0m[0m_smote_enn[0m [0;32mimport[0m [0mSMOTEENN[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mfrom[0m [0;34m.[0m[0m_smote_tomek[0m [0;32mimport[0m [0mSMOTETomek[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/imblearn/combine/_smote_enn.py[0m in [0;36m<module>[0;34m[0m
[1;32m     10[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mcheck_X_y[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;34m[0m[0m
[0;32m---> 12[0;31m [0;32mfrom[0m [0;34m.[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mBaseSampler[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     13[0m [0;32mfrom[0m [0;34m.[0m[0;34m.[0m[0mover_sampling[0m [0;32mimport[0m [0mSMOTE[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m [0;32mfrom[0m [0;34m.[0m[0;34m.[0m[0mover_sampling[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mBaseOverSampler[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/imblearn/base.py[0m in [0;36m<module>[0;34m[0m
[1;32m     10[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mBaseEstimator[0m[0;34m,[0m [0mOneToOneFeatureMixin[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mpreprocessing[0m [0;32mimport[0m [0mlabel_binarize[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 12[0;31m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0m_metadata_requests[0m [0;32mimport[0m [0mMETHODS[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     13[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmulticlass[0m [0;32mimport[0m [0mcheck_classification_targets[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m [0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'sklearn.utils._metadata_requests'

## === cell 1
class FeatureEngineering():
    def __init__(self):
        self.columns = [
            (pl.col("full_text").str.split(by="\n\n").alias("paragraph"))
        ]
        self.train_dataset = pl.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv').with_columns(self.columns)
        self.test_dataset = pl.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv').with_columns(self.columns)
        self.sentence_fea = ['sentence_len','sentence_word_cnt']
        self.paragraph_fea = ['paragraph_len','paragraph_sentence_cnt','paragraph_word_cnt']
        self.vectorizer = TfidfVectorizer(tokenizer=lambda x: x,
                                          preprocessor=lambda x: x,
                                          token_pattern=None,
                                          strip_accents='unicode',
                                          analyzer = 'word',
                                          ngram_range=(2,3),
                                          min_df=0.05,
                                          max_df=0.9,
                                          sublinear_tf=True  
        )
    def removeHTML(self,x):
        html=re.compile(r'<.*?>')
        return html.sub(r'',x)
    def dataPreprocessing(self,x):
        x = x.lower()             # covert all letter to lower form
        x = self.removeHTML(x)
        x = re.sub("@\w+", '',x)
        x = re.sub("'\d+", '',x)
        x = re.sub("\d+", '',x)
        x = re.sub("http\w+", '',x)
        x = re.sub(r"\s+", " ",x) # replace any sequence of whitespace characters with a sigle whitespace
        x = re.sub(r"\.+", ".",x) # replace any sequence of periods with a sigle periods
        x = re.sub(r"\,+", ",",x) # replace any sequence of commas with a sigle comma
        x = x.strip()
        return x 
    def Paragraph_Preprocess(self,tmp):
        tmp = tmp.explode('paragraph')
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(self.dataPreprocessing))
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x:len(x)).alias("paragraph_len"))
        tmp = tmp.filter(pl.col('paragraph_len')>=25)
        tmp = tmp.with_columns(pl.col('paragraph').map_elements(lambda x: len(x.split("."))).alias("paragraph_sentence_cnt"),
                               pl.col('paragraph').map_elements(lambda x: len(x.split(" "))).alias("paragraph_word_cnt")
                              )
        return tmp
    def Paragraph_Eng(self,train_tmp):
        aggs = [
            *[pl.col('paragraph').filter(pl.col('paragraph_len')>=i)
            .count().alias(f'paragraph_{i}_cnt') for i in [25,100,200,300,400,500,600,700]],
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in self.paragraph_fea],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in self.paragraph_fea],
            *[pl.col(fea).min().alias(f"{fea}_min") for fea in self.paragraph_fea],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in self.paragraph_fea],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in self.paragraph_fea],
        ]
        df = train_tmp.group_by(["essay_id"], maintain_order=True).agg(aggs).sort("essay_id")
        df = df.to_pandas()
        print("done Paragraph_Eng +",len(df.columns),"features")
        return df
    def Sentence_Preprocess(self,tmp):
        tmp = tmp.with_columns(pl.col('full_text').map_elements(self.dataPreprocessing).str.split(by=".").alias("sentence"))
        tmp = tmp.explode('sentence')
        tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x)).alias("sentence_len"))
        tmp = tmp.filter(pl.col('sentence_len')>=15)
        tmp = tmp.with_columns(pl.col('sentence').map_elements(lambda x: len(x.split(' '))).alias("sentence_word_cnt"))

        return tmp
    def Sentence_Eng(self,train_tmp):
        aggs = [
            *[pl.col('sentence').filter(pl.col('sentence_len') >= i).count().alias(f"sentence_{i}_cnt") for i in [15,50,100,150,200,250,300] ], 
            *[pl.col(fea).max().alias(f"{fea}_max") for fea in self.sentence_fea],
            *[pl.col(fea).mean().alias(f"{fea}_mean") for fea in self.sentence_fea],
            *[pl.col(fea).min().alias(f"{fea}_min") for fea in self.sentence_fea],
            *[pl.col(fea).first().alias(f"{fea}_first") for fea in self.sentence_fea],
            *[pl.col(fea).last().alias(f"{fea}_last") for fea in self.sentence_fea],
            ]
        df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
        df = df.to_pandas()
        print("done Sentence_Eng +",len(df.columns),"features")
        return df
    def Word_Preprocess(self,tmp):
        tmp = tmp.with_columns(pl.col('full_text').map_elements(self.dataPreprocessing).str.split(by=" ").alias("word"))
        tmp = tmp.explode('word')
        tmp = tmp.with_columns(pl.col('word').map_elements(lambda x: len(x)).alias("word_len"))
        tmp = tmp.filter(pl.col('word_len')!=0)

        return tmp
    def Word_Eng(self,train_tmp):
        aggs = [
            *[pl.col('word').filter(pl.col('word_len') >= i+1)
              .count().alias(f"word_{i+1}_cnt") for i in range(15)], 
            pl.col('word_len').max().alias(f"word_len_max"),
            pl.col('word_len').mean().alias(f"word_len_mean"),
            pl.col('word_len').std().alias(f"word_len_std"),
            pl.col('word_len').quantile(0.25).alias(f"word_len_q1"),
            pl.col('word_len').quantile(0.50).alias(f"word_len_q2"),
            pl.col('word_len').quantile(0.75).alias(f"word_len_q3"),
        ]
        df = train_tmp.group_by(['essay_id'], maintain_order=True).agg(aggs).sort("essay_id")
        df = df.to_pandas()
        print("done Word_Eng +",len(df.columns),"features")
        return df
    def process(self):
        tmp = self.Paragraph_Preprocess(self.train_dataset)
        train_feats = self.Paragraph_Eng(tmp)
        train_feats['score'] = self.train_dataset['score']
        
        tmp = self.Sentence_Preprocess(self.train_dataset)
        train_feats = train_feats.merge(self.Sentence_Eng(tmp), on='essay_id', how='left')
        
        tmp = self.Word_Preprocess(self.train_dataset)
        train_feats = train_feats.merge(self.Word_Eng(tmp), on='essay_id', how='left')
        
        train_tfid = self.vectorizer.fit_transform([i for i in self.train_dataset['full_text']])
        dense_matrix = train_tfid.toarray()
        df = pd.DataFrame(dense_matrix)
        tfid_columns = [ f'tfid_{i}' for i in range(len(df.columns))]
        df.columns = tfid_columns
        df['essay_id'] = train_feats['essay_id']
        train_feats = train_feats.merge(df, on='essay_id', how='left')
        print('feature_num: ',len(train_feats.columns)-2)
        return train_feats
    def process_test(self):
        temp = self.Paragraph_Preprocess(self.test_dataset)
        test_feats = self.Paragraph_Eng(temp)
        
        temp = self.Sentence_Preprocess(self.test_dataset)
        test_feats = test_feats.merge(self.Sentence_Eng(temp), on='essay_id', how='left')
        
        temp = self.Word_Preprocess(self.test_dataset)
        test_feats = test_feats.merge(self.Word_Eng(temp), on='essay_id', how='left')
        
        test_tfid = self.vectorizer.transform([i for i in self.test_dataset['full_text']])
        dense_matrix = test_tfid.toarray()
        df = pd.DataFrame(dense_matrix)
        tfid_columns = [ f'tfid_{i}' for i in range(len(df.columns))]
        df.columns = tfid_columns
        df['essay_id'] = test_feats['essay_id']
        test_feats = test_feats.merge(df, on='essay_id', how='left')
        print('feature_num: ',len(test_feats.columns)-2)
        
        return test_feats
