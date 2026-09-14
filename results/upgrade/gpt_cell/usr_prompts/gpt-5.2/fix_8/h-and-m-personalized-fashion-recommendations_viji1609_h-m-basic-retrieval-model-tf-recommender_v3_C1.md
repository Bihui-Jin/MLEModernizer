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

3.10

# 2. Installed packages

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

# 3. Data file paths

```
/
    kaggle/
        data/
            articles.csv (105543 lines)
            articles.csv.zip (4.4 MB)
            customers.csv (1371981 lines)
            customers.csv.zip (102.4 MB)
            description.md (74 lines)
            images.zip (30.0 GB)
            sample_submission.csv (1371981 lines)
            sample_submission.csv.zip (53.3 MB)
            transactions_train.csv (31521961 lines)
            transactions_train.csv.zip (604.1 MB)
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
            images/
                010/
                    0108775015.jpg (154.6 kB)
                    0108775044.jpg (106.7 kB)
                    ... and 1 other files
                011/
                    0110065001.jpg (148.6 kB)
                    0110065002.jpg (85.7 kB)
                    ... and 18 other files
                ... and 84 other folders
        input/
            articles.csv (105543 lines)
            articles.csv.zip (4.4 MB)
            customers.csv (1371981 lines)
            customers.csv.zip (102.4 MB)
            description.md (74 lines)
            images.zip (30.0 GB)
            sample_submission.csv (1371981 lines)
            sample_submission.csv.zip (53.3 MB)
            transactions_train.csv (31521961 lines)
            transactions_train.csv.zip (604.1 MB)
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
            images/
                010/
                    0108775015.jpg (154.6 kB)
                    0108775044.jpg (106.7 kB)
                    ... and 1 other files
                011/
                    0110065001.jpg (148.6 kB)
                    0110065002.jpg (85.7 kB)
                    ... and 18 other files
                ... and 84 other folders
        working/
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
```

-> data/articles.csv has 105542 rows and 25 columns.
The columns are: article_id, product_code, prod_name, product_type_no, product_type_name, product_group_name, graphical_appearance_no, graphical_appearance_name, colour_group_code, colour_group_name, perceived_colour_value_id, perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name, department_no... and 10 more columns

-> data/customers.csv has 1371980 rows and 7 columns.
The columns are: customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code

-> data/h-and-m-personalized-fashion-recommendations/articles.csv has 105542 rows and 25 columns.
The columns are: article_id, product_code, prod_name, product_type_no, product_type_name, product_group_name, graphical_appearance_no, graphical_appearance_name, colour_group_code, colour_group_name, perceived_colour_value_id, perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name, department_no... and 10 more columns

-> data/h-and-m-personalized-fashion-recommendations/customers.csv has 1371980 rows and 7 columns.
The columns are: customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code

-> data/h-and-m-personalized-fashion-recommendations/sample_submission.csv has 1371980 rows and 2 columns.
The columns are: customer_id, prediction

-> data/h-and-m-personalized-fashion-recommendations/transactions_train.csv has 31521960 rows and 5 columns.
The columns are: t_dat, customer_id, article_id, price, sales_channel_id

-> data/sample_submission.csv has 1371980 rows and 2 columns.
The columns are: customer_id, prediction

-> data/transactions_train.csv has 31521960 rows and 5 columns.
The columns are: t_dat, customer_id, article_id, price, sales_channel_id

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
!pip install -q tensorflow-recommenders
!pip install -q scann


## === cell 1
import os
import sys
import subprocess  # kept for compatibility with the original cell, though unused now

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "--force-reinstall",
        "numpy==1.26.4",
    ]
)

import pandas as pd

os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "scann"]
    )
except Exception:
    pass

import tensorflow as tf
import tensorflow_recommenders as tfrs

from pathlib import Path
from typing import Dict, Text


## === cell 2
data_dir = Path('../input/h-and-m-personalized-fashion-recommendations')
train0 = pd.read_csv(data_dir/'transactions_train.csv')
train0 = train0[train0['t_dat'] >='2020-09-01']

train0['article_id'] = train0['article_id'].astype(str)
train0['article_id'] = train0['article_id'].apply(lambda x: x.zfill(10))
train0.head()


## === cell 3
customer_df = pd.read_csv(data_dir/'customers.csv')
customer_df.head()


## === cell 4
article_df = pd.read_csv(data_dir/'articles.csv')

article_df['article_id'] = article_df['article_id'].astype(str)
article_df['article_id'] = article_df['article_id'].apply(lambda x: x.zfill(10))
article_df.head()


## === cell 5

unique_customer_ids = customer_df.customer_id.unique()
unique_article_ids = article_df.article_id.unique()

article_ds = tf.data.Dataset.from_tensor_slices(dict(article_df[['article_id']]))
articles = article_ds.map(lambda x: x['article_id'])


## === cell 6
embedding_dimension = 64

customer_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_customer_ids, mask_token=None),  
  tf.keras.layers.Embedding(len(unique_customer_ids) + 1, embedding_dimension)
])


## === cell 7
article_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_article_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_article_ids) + 1, embedding_dimension)
])


## === cell 8

class HandMModel(tfrs.Model):
    
    def __init__(self, customer_model, article_model):
        super().__init__()
        self.article_model: tf.keras.Model = article_model
        self.customer_model: tf.keras.Model = customer_model
        self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=articles.batch(128).map(self.article_model),            
            ),
        )        

    def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
    
        customer_embeddings = self.customer_model(features["customer_id"])    
        article_embeddings = self.article_model(features["article_id"])

        return self.task(customer_embeddings, article_embeddings,compute_metrics=not training)


## === cell 9
model = HandMModel(customer_model, article_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


## === cell 10
train = train0[train0['t_dat']<='2020-09-15']
test = train0[train0['t_dat'] >='2020-09-15']

train_ds = tf.data.Dataset.from_tensor_slices(dict(train[['customer_id','article_id']])).shuffle(100_000).batch(256).cache()
test_ds = tf.data.Dataset.from_tensor_slices(dict(test[['customer_id','article_id']])).batch(256).cache()

num_epochs = 5

'''

history = model.fit(
    train_ds, 
    validation_data = test_ds,
    validation_freq=5,
    epochs=num_epochs,
    verbose=1)

'''


## === cell 11

train_ds = tf.data.Dataset.from_tensor_slices(dict(train0[['customer_id','article_id']])).shuffle(100_000).batch(256).cache()

num_epochs = 5

history = model.fit(
    train_ds,    
    epochs=num_epochs,
    verbose=1)


## === cell 12
scann_index = tfrs.layers.factorized_top_k.ScaNN(model.customer_model, k = 12 )
scann_index.index_from_dataset(
  tf.data.Dataset.zip((articles.batch(100), articles.batch(100).map(model.article_model)))
)


## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1841315678.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mscann_index[0m [0;34m=[0m [0mtfrs[0m[0;34m.[0m[0mlayers[0m[0;34m.[0m[0mfactorized_top_k[0m[0;34m.[0m[0mScaNN[0m[0;34m([0m[0mmodel[0m[0;34m.[0m[0mcustomer_model[0m[0;34m,[0m [0mk[0m [0;34m=[0m [0;36m12[0m [0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m scann_index.index_from_dataset(
[1;32m      3[0m   [0mtf[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mDataset[0m[0;34m.[0m[0mzip[0m[0;34m([0m[0;34m([0m[0marticles[0m[0;34m.[0m[0mbatch[0m[0;34m([0m[0;36m100[0m[0;34m)[0m[0;34m,[0m [0marticles[0m[0;34m.[0m[0mbatch[0m[0;34m([0m[0;36m100[0m[0;34m)[0m[0;34m.[0m[0mmap[0m[0;34m([0m[0mmodel[0m[0;34m.[0m[0marticle_model[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m )

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_recommenders/layers/factorized_top_k.py[0m in [0;36m__init__[0;34m(self, query_model, k, distance_measure, num_leaves, num_leaves_to_search, training_iterations, dimensions_per_block, num_reordering_candidates, parallelize_batch_searches, name)[0m
[1;32m    674[0m [0;34m[0m[0m
[1;32m    675[0m     [0;32mif[0m [0;32mnot[0m [0m_HAVE_SCANN[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 676[0;31m       raise ImportError(
[0m[1;32m    677[0m           [0;34m"The scann library is not present. Please install it using "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    678[0m           [0;34m"`pip install scann` to use the ScaNN layer."[0m[0;34m[0m[0;34m[0m[0m

[0;31mImportError[0m: The scann library is not present. Please install it using `pip install scann` to use the ScaNN layer.

## === cell 13
sub = pd.read_csv(data_dir/'sample_submission.csv')
_,articles = scann_index(sub.customer_id.values)
preds = articles.numpy().astype(str)
preds = pd.Series(map(' '.join, preds,))
sub['prediction'] = preds
sub.to_csv('submission.csv',index=False)
