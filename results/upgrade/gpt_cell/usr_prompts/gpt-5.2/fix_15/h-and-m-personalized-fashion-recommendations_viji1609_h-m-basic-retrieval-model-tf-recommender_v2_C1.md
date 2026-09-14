# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
The training data is the purchase history of customers across time. The task is to predict what articles each customer will purchase in the 7-day period immediately after the training data ends.

## Metric
Mean Average Precision @ 12 (MAP@12):

$$
\text{MAP@12}=\frac{1}{U} \sum_{u=1}^U \frac{1}{\min (m, 12)} \sum_{k=1}^{\min (n, 12)} P(k) \times \text{rel}(k)
$$

where $U$ is the number of customers, $P(k)$ is the precision at cutoff $k, n$ is the number predictions per customer, $m$ is the number of ground truth values per customer, and $\text{rel}(k)$ is an indicator function equaling 1 if the item at rank $k$ is a relevant (correct) label, zero otherwise.

You must make predictions for all `customer_id` values found in the sample submission. All customers who made purchases during the test period are scored, regardless of whether they had purchase history in the training data.

## Submission Format
For each `customer_id` observed in the training data, you may predict up to 12 labels for the `article_id`, which is the predicted items a customer will buy in the next 7-day period after the training time period. The file should contain a header and have the following format:

```
customer_id,prediction
00000dba,0706016001 0706016002 0372860001 ...
0000423b,0706016001 0706016002 0372860001 ...
...
```

## Dataset
- **images/** - a folder of images corresponding to each `article_id`; images are placed in subfolders starting with the first three digits of the `article_id`; note, not all `article_id` values have a corresponding image.
- **articles.csv** - detailed metadata for each `article_id` available for purchase
- **customers.csv** - metadata for each `customer_id` in dataset
- **sample_submission.csv** - a sample submission file in the correct format
- **transactions_train.csv** - the training data, consisting of the purchases each customer for each date, as well as additional information. Duplicate rows correspond to multiple purchases of the same item. Your task is to predict the `article_id`s each customer will purchase during the 7-day period immediately after the training data period.

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

# 5. Target score

0.01387

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys
import subprocess
import os
import importlib.metadata


def _ensure_numpy_compatible(target_version: str = "1.26.4") -> None:
    try:
        installed_version = importlib.metadata.version("numpy")
    except Exception:
        installed_version = None

    if installed_version != target_version:
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-cache-dir",
                "--force-reinstall",
                f"numpy=={target_version}",
            ]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)


def _ensure_tfrs_installed() -> None:
    try:
        importlib.metadata.version("tensorflow-recommenders")
    except Exception:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-cache-dir",
                "tensorflow-recommenders",
            ]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_numpy_compatible("1.26.4")
_ensure_tfrs_installed()

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from pathlib import Path
from typing import Dict

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 1
data_dir = Path("../input/h-and-m-personalized-fashion-recommendations")

train0 = pd.read_csv(
    data_dir / "transactions_train.csv",
    usecols=["t_dat", "customer_id", "article_id"],
    dtype={"customer_id": "string", "article_id": "int64"},
    parse_dates=["t_dat"],
)

train0 = train0[train0["t_dat"] >= pd.Timestamp("2020-09-01")].copy()

train0["article_id"] = train0["article_id"].astype(str).str.zfill(10)
train0["customer_id"] = train0["customer_id"].astype(str)

train0.head()



## === cell 2
customer_df = pd.read_csv(data_dir / "customers.csv", usecols=["customer_id"])
customer_df["customer_id"] = customer_df["customer_id"].astype(str)
customer_df.head()



## === cell 3
article_df = pd.read_csv(data_dir / "articles.csv", usecols=["article_id"])
article_df["article_id"] = article_df["article_id"].astype(str).str.zfill(10)
article_df.head()



## === cell 4
unique_customer_ids = train0["customer_id"].unique()
unique_article_ids = train0["article_id"].unique()

article_ds = tf.data.Dataset.from_tensor_slices({"article_id": unique_article_ids})
articles = article_ds.map(lambda x: x["article_id"])



## === cell 5
embedding_dimension = 64

customer_model = tf.keras.Sequential(
    [
        tf.keras.layers.StringLookup(vocabulary=unique_customer_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_customer_ids) + 1, embedding_dimension),
    ]
)



## === cell 6
article_model = tf.keras.Sequential(
    [
        tf.keras.layers.StringLookup(vocabulary=unique_article_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_article_ids) + 1, embedding_dimension),
    ]
)




## === cell 7
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
        return self.task(
            customer_embeddings, article_embeddings, compute_metrics=not training
        )




## === cell 8
model = HandMModel(customer_model, article_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))



## === cell 9
cutoff = pd.Timestamp("2020-09-15")
train = train0[train0["t_dat"] < cutoff].copy()
valid = train0[train0["t_dat"] >= cutoff].copy()

train_ds = (
    tf.data.Dataset.from_tensor_slices(dict(train[["customer_id", "article_id"]]))
    .shuffle(100_000, seed=42, reshuffle_each_iteration=True)
    .batch(256)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices(dict(valid[["customer_id", "article_id"]]))
    .batch(256)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

num_epochs = 5

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=num_epochs,
    verbose=1,
)



## === cell 10
scann_index = tfrs.layers.factorized_top_k.ScaNN(model.customer_model, k=12)
scann_index.index_from_dataset(
    tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(unique_article_ids).batch(100),
            tf.data.Dataset.from_tensor_slices(unique_article_ids)
            .batch(100)
            .map(model.article_model),
        )
    )
)



## === cell 11
sub = pd.read_csv(
    data_dir / "sample_submission.csv", usecols=["customer_id", "prediction"]
)
sub["customer_id"] = sub["customer_id"].astype(str)

customer_ids_tf = tf.constant(sub["customer_id"].values, dtype=tf.string)

_, rec_articles = scann_index(customer_ids_tf)  # (num_customers, 12), dtype: tf.string
rec_articles_np = rec_articles.numpy()  # bytes array

preds = [" ".join([a.decode("utf-8") for a in row]) for row in rec_articles_np]

sub["prediction"] = preds
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
