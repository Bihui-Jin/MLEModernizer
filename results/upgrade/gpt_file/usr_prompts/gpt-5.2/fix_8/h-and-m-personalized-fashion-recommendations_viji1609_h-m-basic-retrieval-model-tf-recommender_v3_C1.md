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

0.01388

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The initial crash comes from an incompatible `tensorflow_recommenders`/protobuf setup (the `MessageFactory.GetPrototype` error) and it prevents `data_dir` from being defined, cascading into all later `NameError`s. To make this run end-to-end in the Kaggle environment, I replace the TFRS/ScaNN dependency with an equivalent in-TensorFlow retrieval training and exact top‑K scoring, preserving the same two-tower architecture, dot-product retrieval loss, and overall semantics. I also fix submission formatting so `article_id` stays zero-padded strings and every customer in `sample_submission.csv` receives exactly 12 predictions. The result trains and writes a valid `submission.csv` without relying on unavailable/incompatible packages.'
- What this solution (achieved 0.0) has done: 'I remove the protobuf environment overrides that are triggering the `MessageFactory.GetPrototype` crash in this Kaggle TF2.18/protobuf6 environment, since this script does not use `tensorflow_recommenders` and should run fine with the default C++ protobuf runtime. I also fix the candidate-embedding build loop to use the same `unique_article_ids` tensor dataset you already created, ensuring consistent ordering and avoiding any subtle mismatch. Finally, I ensure predictions are always exactly 12 zero-padded `article_id`s per customer and write a valid `submission.csv` with the required columns, which should move the score from 0.0 to a reasonable (non-zero) MAP@12.'
- What this solution (achieved 0.0) has done: 'The crash in cell 1 happens before any of your logic runs because `tensorflow` import triggers a protobuf incompatibility (`MessageFactory.GetPrototype`) in this environment; the smallest fix is to force the pure‑Python protobuf implementation *before* importing TensorFlow. After that, the pipeline runs, but it is currently at high risk of OOM/timeout at inference because it computes a full `[batch_customers x num_articles]` score matrix; I keep the exact same two‑tower/dot‑product retrieval semantics but switch the top‑K retrieval to a streamed (chunked) exact top‑K that never materializes the full matrix. Finally, I ensure submission formatting is correct (always 12 zero‑padded `article_id`s per customer) and the script always writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the immediate crash caused by forcing the pure‑Python protobuf runtime, which is incompatible with TensorFlow 2.18 + protobuf 6 in this environment and triggers the `MessageFactory.GetPrototype` error during `import tensorflow`. Then I make the inference step more robust by ensuring candidate embeddings are built in the exact `unique_article_ids` order and that we always decode TF string tensors correctly to Python strings. Finally, I keep the existing two‑tower dot‑product retrieval training and streamed exact top‑K logic unchanged, and ensure the submission is written as a valid `submission.csv` with exactly 12 zero‑padded `article_id`s per customer, which should move the score from 0.0 to a non‑zero baseline.'
- What this solution (achieved 0.0) has done: 'The immediate blocker is the TensorFlow import crash caused by an incompatible protobuf runtime being used in this environment; the safest fix is to force the pure-Python protobuf implementation **before** importing TensorFlow so `MessageFactory.GetPrototype` is available. After that, the rest of your two-tower retrieval pipeline can run unchanged, but I also ensure that `article_id` values used for the candidate set are consistently **10-character zero-padded strings** (same as training) to avoid lookup misses and zero embeddings that can tank MAP@12. Finally, I simplify the TF-string decoding at inference to a robust path (TF string tensors always come back as bytes via `.numpy()`), and keep the exact streamed top‑K logic so it finishes within limits and always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf  # noqa: E402

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

DATA_CANDIDATES = [
    Path("/kaggle/input/h-and-m-personalized-fashion-recommendations"),
    Path("/kaggle/data/h-and-m-personalized-fashion-recommendations"),
    Path("../input/h-and-m-personalized-fashion-recommendations"),
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]
data_dir = next(
    (p for p in DATA_CANDIDATES if (p / "transactions_train.csv").exists()), None
)
if data_dir is None:
    raise FileNotFoundError(
        f"Could not find dataset directory containing transactions_train.csv in: {DATA_CANDIDATES}"
    )

print("Using data_dir:", data_dir)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train0 = pd.read_csv(data_dir / "transactions_train.csv")
train0 = train0[train0["t_dat"] >= "2020-09-01"].copy()

train0["article_id"] = train0["article_id"].astype(str).str.zfill(10)
train0["customer_id"] = train0["customer_id"].astype(str)

print("Filtered transactions:", train0.shape)
train0.head()



## === cell 2
customer_df = pd.read_csv(data_dir / "customers.csv")
customer_df["customer_id"] = customer_df["customer_id"].astype(str)
customer_df.head()



## === cell 3
article_df = pd.read_csv(data_dir / "articles.csv")
article_df["article_id"] = article_df["article_id"].astype(str).str.zfill(10)
article_df.head()



## === cell 4
unique_customer_ids = customer_df["customer_id"].astype(str).unique()
unique_article_ids = article_df["article_id"].astype(str).unique()

print("Unique customers:", len(unique_customer_ids))
print("Unique articles:", len(unique_article_ids))

article_ids_tensor = tf.constant(unique_article_ids.astype(str), dtype=tf.string)
article_id_ds = tf.data.Dataset.from_tensor_slices(article_ids_tensor)



## === cell 5
embedding_dimension = 64

customer_model = tf.keras.Sequential(
    [
        tf.keras.layers.StringLookup(vocabulary=unique_customer_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_customer_ids) + 1, embedding_dimension),
    ]
)

article_model = tf.keras.Sequential(
    [
        tf.keras.layers.StringLookup(vocabulary=unique_article_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_article_ids) + 1, embedding_dimension),
    ]
)




## === cell 6
class HandMModelTF(tf.keras.Model):
    def __init__(self, customer_model: tf.keras.Model, article_model: tf.keras.Model):
        super().__init__()
        self.customer_model = customer_model
        self.article_model = article_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=False):
        cust = self.customer_model(inputs["customer_id"])
        art = self.article_model(inputs["article_id"])
        return cust, art

    def train_step(self, data):
        with tf.GradientTape() as tape:
            cust_emb, art_emb = self(data, training=True)
            logits = tf.linalg.matmul(cust_emb, art_emb, transpose_b=True)  # [B, B]
            labels = tf.range(tf.shape(logits)[0])
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits, from_logits=True
                )
            )

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def make_interaction_dataset(
    df: pd.DataFrame, shuffle: bool, batch_size: int
) -> tf.data.Dataset:
    cust = tf.constant(df["customer_id"].values.astype(str), dtype=tf.string)
    art = tf.constant(df["article_id"].values.astype(str), dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices({"customer_id": cust, "article_id": art})
    if shuffle:
        ds = ds.shuffle(100_000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 7
model = HandMModelTF(customer_model, article_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))



## === cell 8
train = train0[train0["t_dat"] <= "2020-09-15"]
test = train0[train0["t_dat"] >= "2020-09-15"]

train_ds_split = make_interaction_dataset(train, shuffle=True, batch_size=256)
test_ds_split = make_interaction_dataset(test, shuffle=False, batch_size=256)

num_epochs = 5

"""
history = model.fit(
    train_ds_split,
    validation_data=test_ds_split,
    validation_freq=5,
    epochs=num_epochs,
    verbose=1,
)
"""



## === cell 9
train_ds = make_interaction_dataset(train0, shuffle=True, batch_size=256)

num_epochs = 5
history = model.fit(train_ds, epochs=num_epochs, verbose=1)



## === cell 10
BATCH_CAND = 2048

all_article_ids = tf.constant(unique_article_ids.astype(str), dtype=tf.string)

cand_emb_list = []
for ids in tf.data.Dataset.from_tensor_slices(all_article_ids).batch(BATCH_CAND):
    cand_emb_list.append(article_model(ids))
candidate_embeddings = tf.concat(cand_emb_list, axis=0)  # [num_articles, dim]

print("Candidate embedding matrix:", candidate_embeddings.shape)


@tf.function
def topk_articles_for_customers_streaming(
    customer_ids: tf.Tensor, k: int = 12, chunk_size: int = 4096
) -> tf.Tensor:
    cust_emb = customer_model(customer_ids)  # [N, dim]
    n_customers = tf.shape(cust_emb)[0]
    n_articles = tf.shape(candidate_embeddings)[0]

    best_scores = tf.fill([n_customers, k], tf.constant(-1e9, dtype=tf.float32))
    best_indices = tf.zeros([n_customers, k], dtype=tf.int32)

    i = tf.constant(0, dtype=tf.int32)

    def cond(i, bs, bi):
        return i < n_articles

    def body(i, bs, bi):
        j = tf.minimum(i + tf.cast(chunk_size, tf.int32), n_articles)
        chunk_emb = candidate_embeddings[i:j]  # [C, dim]
        scores = tf.linalg.matmul(cust_emb, chunk_emb, transpose_b=True)  # [N, C]

        local_k = tf.minimum(k, tf.shape(scores)[1])
        local = tf.math.top_k(scores, k=local_k)
        local_scores = local.values  # [N, k']
        local_idx = local.indices + i  # global indices [N, k']

        merged_scores = tf.concat([bs, local_scores], axis=1)  # [N, k+k']
        merged_idx = tf.concat([bi, tf.cast(local_idx, tf.int32)], axis=1)
        merged = tf.math.top_k(merged_scores, k=k)
        take = merged.indices  # indices into merged_* arrays

        new_bs = tf.gather(merged_scores, take, batch_dims=1)
        new_bi = tf.gather(merged_idx, take, batch_dims=1)
        return j, new_bs, new_bi

    _, best_scores, best_indices = tf.while_loop(
        cond,
        body,
        loop_vars=[i, best_scores, best_indices],
        parallel_iterations=1,
        maximum_iterations=1000000,
    )

    return tf.gather(all_article_ids, best_indices)




## === cell 11
sub = pd.read_csv(data_dir / "sample_submission.csv")
sub["customer_id"] = sub["customer_id"].astype(str)

cust_tensor = tf.constant(sub["customer_id"].values.astype(str), dtype=tf.string)

K = 12
BATCH_PRED = 4096
pred_rows = []

for i in range(0, len(sub), BATCH_PRED):
    batch_cust = cust_tensor[i : i + BATCH_PRED]
    batch_pred_ids = topk_articles_for_customers_streaming(
        batch_cust, k=K, chunk_size=4096
    ).numpy()

    if batch_pred_ids.dtype.kind in ("S", "O"):
        batch_pred_ids = np.char.decode(batch_pred_ids.astype("S"), "utf-8")
    else:
        batch_pred_ids = batch_pred_ids.astype(str)

    if batch_pred_ids.shape[1] != K:
        raise ValueError(
            f"Expected {K} predictions per customer, got shape {batch_pred_ids.shape}"
        )

    batch_pred_ids = np.char.zfill(batch_pred_ids.astype(str), 10)
    pred_rows.extend([" ".join(row.tolist()) for row in batch_pred_ids])

sub["prediction"] = pred_rows
sub = sub[["customer_id", "prediction"]]
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
