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

0.01387

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The initial crash is caused by an incompatibility between `tensorflow_recommenders` and the installed `protobuf==6.x` (it triggers the `MessageFactory.GetPrototype` AttributeError before `data_dir` is defined, cascading into all later NameErrors). The minimal robust fix is to avoid importing TFRS entirely and instead implement the same core two-tower retrieval training objective directly in Keras (customer/article embeddings with sampled-softmax over article IDs), which preserves the model architecture and training semantics closely while running in this environment. We then build a brute-force top‑K scorer over all article embeddings to generate 12 recommendations per customer, and write a valid `submission.csv` with the required columns and formatting. This produces an end-to-end runnable pipeline and should yield a non-trivial MAP@12 score (better than “no submission”), while keeping the overall approach (two-tower retrieval with embeddings trained from transaction pairs) intact.'
- What this solution (achieved 0.0) has done: 'I remove the protobuf-triggering import that crashes before `data_dir` is set, since this solution already uses pure Keras and doesn’t need `tensorflow_recommenders`. Then I fix the embedding-weight access bug by explicitly building the two towers (so the Embedding layer has variables) and by referencing the embedding matrix via `.embeddings` after build, avoiding the empty `.weights` list. Finally, I fix the inference shape issue by ensuring customer/article towers always output 2D `[B, D]` tensors (including for single examples) and keep the submission formatting exactly as required (`customer_id`, `prediction`) written to `submission.csv`.'
- What this solution (achieved 0.0) has done: 'The crash comes from a `protobuf`/TensorFlow internal incompatibility triggered at import-time; the safest minimal fix is to force the pure‑Python protobuf implementation *before* importing TensorFlow so the notebook can run. Then, to avoid producing an all‑zero score due to recommending arbitrary items, we keep your two‑tower sampled-softmax training but add a minimal, standard post-processing blend: fill each customer’s 12 predictions with their most recent purchases (from the last weeks of training) and backfill with global popular items, which is score-positive for MAP@12 while preserving the core model. Finally, we ensure article IDs are always 10‑digit strings and that the submission has the exact required columns and `.csv` suffix.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd
import numpy as np
import tensorflow as tf

from pathlib import Path
from typing import Dict

tf.keras.utils.set_random_seed(42)

DATA_DIR_CANDIDATES = [
    Path("../input/h-and-m-personalized-fashion-recommendations"),
    Path("/kaggle/input/h-and-m-personalized-fashion-recommendations"),
    Path("/kaggle/data/h-and-m-personalized-fashion-recommendations"),
    Path("/kaggle/data"),
    Path("/kaggle/input"),
]
data_dir = next(
    (p for p in DATA_DIR_CANDIDATES if (p / "transactions_train.csv").exists()), None
)
if data_dir is None:
    raise FileNotFoundError(
        "Could not find transactions_train.csv in expected Kaggle locations. "
        f"Tried: {DATA_DIR_CANDIDATES}"
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
unique_customer_ids = customer_df.customer_id.unique()
unique_article_ids = article_df.article_id.unique()

print(
    "unique customers:",
    len(unique_customer_ids),
    "unique articles:",
    len(unique_article_ids),
)

customer_lookup = tf.keras.layers.StringLookup(
    vocabulary=unique_customer_ids, mask_token=None, num_oov_indices=1
)
article_lookup = tf.keras.layers.StringLookup(
    vocabulary=unique_article_ids, mask_token=None, num_oov_indices=1
)



## === cell 5
embedding_dimension = 64

customer_model = tf.keras.Sequential(
    [
        customer_lookup,
        tf.keras.layers.Embedding(
            customer_lookup.vocabulary_size(), embedding_dimension
        ),
    ],
    name="customer_tower",
)

article_model = tf.keras.Sequential(
    [
        article_lookup,
        tf.keras.layers.Embedding(
            article_lookup.vocabulary_size(), embedding_dimension
        ),
    ],
    name="article_tower",
)

_ = customer_model(tf.constant(["__dummy_customer__"]))
_ = article_model(tf.constant(["__dummy_article__"]))




## === cell 6
class HandMModel(tf.keras.Model):
    def __init__(
        self,
        customer_model: tf.keras.Model,
        article_model: tf.keras.Model,
        num_articles: int,
        embedding_dim: int,
        num_sampled: int = 100,
    ):
        super().__init__()
        self.customer_model = customer_model
        self.article_model = article_model
        self.num_articles = int(num_articles)
        self.embedding_dim = int(embedding_dim)
        self.num_sampled = int(num_sampled)

        emb_layer = None
        for layer in self.article_model.layers:
            if isinstance(layer, tf.keras.layers.Embedding):
                emb_layer = layer
                break
        if emb_layer is None:
            raise ValueError("Could not find Embedding layer inside article_model.")
        self.article_embedding_layer = emb_layer

        self.softmax_b = self.add_weight(
            name="softmax_b",
            shape=(self.num_articles,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        return self.customer_model(inputs["customer_id"])

    def train_step(self, data: Dict[str, tf.Tensor]):
        x = data
        labels = tf.cast(x["article_id"], tf.int64)

        if labels.shape.rank == 1:
            labels = tf.expand_dims(labels, axis=1)

        labels = tf.clip_by_value(labels, 0, self.num_articles - 1)

        with tf.GradientTape() as tape:
            user_emb = self.customer_model(x["customer_id"])  # [B, D]
            user_emb = tf.cast(user_emb, tf.float32)
            if user_emb.shape.rank == 1:
                user_emb = tf.expand_dims(user_emb, axis=0)

            weights = tf.cast(
                self.article_embedding_layer.embeddings, tf.float32
            )  # [V, D]

            loss_vec = tf.nn.sampled_softmax_loss(
                weights=weights,
                biases=self.softmax_b,
                labels=labels,
                inputs=user_emb,
                num_sampled=min(self.num_sampled, self.num_articles - 1),
                num_classes=self.num_articles,
            )
            loss = tf.reduce_mean(loss_vec)
            loss += tf.add_n(self.losses) if self.losses else 0.0

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}




## === cell 7
num_articles = article_lookup.vocabulary_size()
model = HandMModel(
    customer_model=customer_model,
    article_model=article_model,
    num_articles=num_articles,
    embedding_dim=embedding_dimension,
    num_sampled=100,
)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))



## === cell 8
train_pairs = train0[["customer_id", "article_id"]].copy()

train_ds = tf.data.Dataset.from_tensor_slices(
    {
        "customer_id": train_pairs["customer_id"].values.astype("str"),
        "article_id": train_pairs["article_id"].values.astype("str"),
    }
)

train_ds = train_ds.map(
    lambda x: {
        "customer_id": x["customer_id"],
        "article_id": article_lookup(x["article_id"]),
    },
    num_parallel_calls=tf.data.AUTOTUNE,
)

train_ds = (
    train_ds.shuffle(100_000, seed=42, reshuffle_each_iteration=True)
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)

num_epochs = 5
history = model.fit(train_ds, epochs=num_epochs, verbose=1)



## === cell 9
all_article_ids = unique_article_ids.astype(str)
article_id_tensor = tf.constant(all_article_ids)

cand_emb = article_model(article_id_tensor)
cand_emb = tf.cast(cand_emb, tf.float32)
if cand_emb.shape.rank == 1:
    cand_emb = tf.expand_dims(cand_emb, axis=0)
cand_emb = tf.math.l2_normalize(cand_emb, axis=1)

cand_ids_np = all_article_ids

print("Candidate embedding matrix:", cand_emb.shape)



## === cell 10

train0["t_dat"] = pd.to_datetime(train0["t_dat"], errors="coerce")

pop_window_start = train0["t_dat"].max() - pd.Timedelta(days=30)
popular_articles = (
    train0.loc[train0["t_dat"] >= pop_window_start, "article_id"]
    .value_counts()
    .head(200)
    .index.astype(str)
    .tolist()
)

cust_window_start = train0["t_dat"].max() - pd.Timedelta(days=60)
tmp = train0.loc[
    train0["t_dat"] >= cust_window_start, ["customer_id", "t_dat", "article_id"]
].copy()
tmp = tmp.sort_values(["customer_id", "t_dat"], ascending=[True, False])

cust_recent = {}
for cid, g in tmp.groupby("customer_id", sort=False):
    seen = set()
    recs = []
    for a in g["article_id"].astype(str).tolist():
        if a not in seen:
            seen.add(a)
            recs.append(a)
        if len(recs) >= 12:
            break
    cust_recent[cid] = recs

print("Popular list size:", len(popular_articles), "Example:", popular_articles[:5])
print("Customers with recent history (60d):", len(cust_recent))



## === cell 11
sub = pd.read_csv(data_dir / "sample_submission.csv")
sub["customer_id"] = sub["customer_id"].astype(str)

batch_size = 4096  # keep memory bounded
cust_values = sub["customer_id"].values.astype(str)

pred_strings = np.empty(len(cust_values), dtype=object)

for start in range(0, len(cust_values), batch_size):
    batch = cust_values[start : start + batch_size]
    batch_tensor = tf.constant(batch)

    user_emb = customer_model(batch_tensor)
    user_emb = tf.cast(user_emb, tf.float32)
    if user_emb.shape.rank == 1:
        user_emb = tf.expand_dims(user_emb, axis=0)
    user_emb = tf.math.l2_normalize(user_emb, axis=1)  # [B, D]

    scores = tf.linalg.matmul(user_emb, cand_emb, transpose_b=True)  # [B, num_articles]
    topk = tf.math.top_k(scores, k=12)
    topk_idx = topk.indices.numpy()  # [B, 12]
    model_recs = cand_ids_np[topk_idx]  # [B, 12] of strings

    out_rows = []
    for i, cid in enumerate(batch):
        recs = []
        seen = set()

        for a in cust_recent.get(cid, []):
            if a not in seen:
                seen.add(a)
                recs.append(a)
            if len(recs) >= 12:
                break

        if len(recs) < 12:
            for a in model_recs[i].tolist():
                if a not in seen:
                    seen.add(a)
                    recs.append(a)
                if len(recs) >= 12:
                    break

        if len(recs) < 12:
            for a in popular_articles:
                if a not in seen:
                    seen.add(a)
                    recs.append(a)
                if len(recs) >= 12:
                    break

        out_rows.append(" ".join(recs[:12]))

    pred_strings[start : start + len(batch)] = out_rows

sub["prediction"] = pred_strings
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
