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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.1664010301247311

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = "/kaggle/input/google-quest-challenge"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

_non_letters = re.compile(r"[^a-zA-Z]+")

WORKDIR = "/kaggle/working"
CACHE_DIR = os.path.join(WORKDIR, "gq_cache_rnn_glove")
os.makedirs(CACHE_DIR, exist_ok=True)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def loadWordVectors(tokens, filepath, dimensions=50):
    """Read pretrained GloVe vectors if available; otherwise raise FileNotFoundError."""
    wordVectors = np.random.randn(len(tokens), dimensions).astype(np.float32)
    tok_get = tokens.get
    with open(filepath, "r", encoding="utf-8", errors="ignore") as ifs:
        for line in ifs:
            if not line:
                continue
            row = line.rstrip().split(" ")
            if len(row) != dimensions + 1:
                continue
            idx = tok_get(row[0])
            if idx is None:
                continue
            wordVectors[idx] = np.asarray(row[1:], dtype=np.float32)
    return wordVectors




## === cell 2
def df_process(
    df=None,
    train=False,
    model="rnn_glove",
    *,
    host_categories=None,
    category_categories=None
):
    """
    Speed + correctness: apply the same fitted categories to test; avoid re-fitting categories.
    """
    if df is None:
        df = pd.read_csv(TRAIN_PATH, index_col="qa_id")

    if model != "rnn_glove":
        return df

    df = df.copy()

    qpage = df["question_user_page"].fillna("").astype(str)
    apage = df["answer_user_page"].fillna("").astype(str)
    df["question_user_page"] = qpage.str.rsplit("/", n=1).str[-1]
    df["answer_user_page"] = apage.str.rsplit("/", n=1).str[-1]

    cat = df["category"].fillna("").astype(str)
    df["category"] = cat.str.replace(_non_letters, " ", regex=True).str.lower()

    host = df["host"].fillna("").astype(str)
    df["host"] = host.str.split(".", n=1).str[0]

    if host_categories is None:
        host_cat = pd.Categorical(df["host"])
        host_categories = host_cat.categories
        df["host"] = host_cat.codes
    else:
        df["host"] = pd.Categorical(df["host"], categories=host_categories).codes

    if category_categories is None:
        cat_cat = pd.Categorical(df["category"])
        category_categories = cat_cat.categories
        df["category"] = cat_cat.codes
    else:
        df["category"] = pd.Categorical(
            df["category"], categories=category_categories
        ).codes

    df = df.drop(["question_user_name", "answer_user_name", "url"], axis=1)

    qb = df["question_body"].fillna("").astype(str)
    qt = df["question_title"].fillna("").astype(str)
    ans = df["answer"].fillna("").astype(str)
    df["q_len"] = qb.str.len()
    df["t_len"] = qt.str.len()
    df["ans_len"] = ans.str.len()

    return df, host_categories, category_categories




## === cell 3
class wordbank:
    def __init__(self, df=None, train=True):
        if df is None:
            df = pd.read_csv(TRAIN_PATH, index_col="qa_id")
        self.df = df
        self.train = train

    @staticmethod
    def _normalized_text_series(texts: pd.Series) -> pd.Series:
        s = texts.fillna("").astype(str)
        s = s.str.replace(_non_letters, " ", regex=True).str.strip().str.lower()
        return s

    def tokens(self):
        """
        Speed: Tokenizer works on list-of-strings; avoid building nested python token lists.
        Preserves semantics via lower=False, filters="", split=" " with our own normalization.
        """
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        q = self._normalized_text_series(self.df["question_body"])
        a = self._normalized_text_series(self.df["answer"])
        t = self._normalized_text_series(self.df["question_title"])
        texts = pd.concat([q, a, t], axis=0).tolist()

        tok = tf.keras.preprocessing.text.Tokenizer(
            lower=False,
            filters="",
            split=" ",
            oov_token="unknown",
        )
        tok.fit_on_texts(texts)

        word_index = dict(tok.word_index)
        word_index["<pad>"] = 0

        if word_index.get("unknown", None) != 1:
            items = sorted(
                [(w, i) for w, i in word_index.items() if w != "<pad>"],
                key=lambda x: x[1],
            )
            new_index = {"<pad>": 0, "unknown": 1}
            next_i = 2
            for w, _ in items:
                if w in ("unknown", "<pad>"):
                    continue
                new_index[w] = next_i
                next_i += 1
            word_index = new_index

        self._tokens = word_index
        tok.word_index = word_index
        tok.index_word = None  # save RAM

        self._titles = tok.texts_to_sequences(
            self._normalized_text_series(self.df["question_title"]).tolist()
        )
        self._questions = tok.texts_to_sequences(
            self._normalized_text_series(self.df["question_body"]).tolist()
        )
        self._answers = tok.texts_to_sequences(
            self._normalized_text_series(self.df["answer"]).tolist()
        )
        return self._tokens

    def test(self):
        """
        Speed + correctness: for test, use the same Keras Tokenizer path as train.
        This avoids slow Python loops; mapping is exactly tokenizer-based with same word_index.
        """
        if not (hasattr(self, "_tokens") and self._tokens):
            raise ValueError("tokens must be set before calling test()")

        tok = tf.keras.preprocessing.text.Tokenizer(
            lower=False, filters="", split=" ", oov_token="unknown"
        )
        tok.word_index = self._tokens
        tok.index_word = None

        self._titles = tok.texts_to_sequences(
            self._normalized_text_series(self.df["question_title"]).tolist()
        )
        self._questions = tok.texts_to_sequences(
            self._normalized_text_series(self.df["question_body"]).tolist()
        )
        self._answers = tok.texts_to_sequences(
            self._normalized_text_series(self.df["answer"]).tolist()
        )




## === cell 4
class rnnGoogleQuest:
    def __init__(self, data, vocab, embedding_matrix=None, max_length=0):
        self._data = data
        self._vocab = vocab
        self._embedding_matrix = embedding_matrix
        if self._embedding_matrix is None:
            self._embedding_dim = 50
            self._embedding_matrix = np.random.normal(
                loc=0.0, scale=1.0, size=(len(self._vocab), self._embedding_dim)
            ).astype(np.float32)
        else:
            self._embedding_dim = self._embedding_matrix.shape[1]
        self._num_vocab = len(self._vocab)
        self._max_length = max_length

    def embeddings(self, input_layer):
        embedding_layer = tf.keras.layers.Embedding(
            self._num_vocab,
            self._embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(
                self._embedding_matrix
            ),
            trainable=True,
            mask_zero=True,
        )(input_layer)
        lstm_layer = tf.keras.layers.LSTM(64, return_state=True)(embedding_layer)
        return lstm_layer

    def rnn_model(self):
        title_input = tf.keras.Input(
            shape=(self._data["title"].shape[1],), name="title"
        )
        question_input = tf.keras.Input(
            shape=(self._data["question"].shape[1],), name="question"
        )
        answer_input = tf.keras.Input(
            shape=(self._data["answer"].shape[1],), name="answer"
        )
        category_input = tf.keras.Input(shape=(5,), name="category")
        host_input = tf.keras.Input(shape=(59,), name="host")
        stats_input = tf.keras.Input(shape=(3,), name="stats")

        _, title_state_h, _ = self.embeddings(title_input)
        _, question_state_h, _ = self.embeddings(question_input)
        _, answer_state_h, _ = self.embeddings(answer_input)

        features = tf.keras.layers.concatenate(
            [
                title_state_h,
                question_state_h,
                answer_state_h,
                category_input,
                host_input,
                stats_input,
            ]
        )
        hidden = tf.keras.layers.Dense(512, activation="relu", name="hidden")(features)
        drop1 = tf.keras.layers.Dropout(0.4)(hidden)
        hidden2 = tf.keras.layers.Dense(128, activation="relu", name="hidden2")(drop1)
        drop2 = tf.keras.layers.Dropout(0.3)(hidden2)
        pred = tf.keras.layers.Dense(30, activation="sigmoid", name="prediction")(drop2)

        model = tf.keras.Model(
            inputs=[
                title_input,
                question_input,
                answer_input,
                category_input,
                host_input,
                stats_input,
            ],
            outputs=pred,
        )
        self._model = model

    def train(self):
        self.rnn_model()
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                name="Adam",
            ),
            loss="mean_absolute_error",
            metrics=["mean_squared_error"],
        )

        options = tf.data.Options()
        options.experimental_deterministic = True

        ds = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "title": self._data["title"],
                    "question": self._data["question"],
                    "answer": self._data["answer"],
                    "category": self._data["category"],
                    "host": self._data["host"],
                    "stats": self._data["stats"],
                },
                self._data["output"],
            )
        )
        ds = (
            ds.with_options(options)
            .cache()
            .batch(128, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        self._model.fit(ds, epochs=25, verbose=2)
        self._model.save("rnn_glove_model.keras")




## === cell 5
def normalize(df, fit_scaler=None):
    x = df.values
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    if fit_scaler is None:
        x_scaled = min_max_scaler.fit_transform(x)
        return x_scaled, min_max_scaler
    else:
        return fit_scaler.transform(x), fit_scaler


def one_hot_from_codes(codes: np.ndarray, depth: int) -> tf.Tensor:
    codes = np.asarray(codes, dtype=np.int32)
    return tf.one_hot(codes, depth=depth, dtype=tf.float32)


def _np_save(path, arr):
    tmp = path + ".tmp"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _load_or_build_train_arrays():
    paths = {
        "title_pad": os.path.join(CACHE_DIR, "train_title_pad.npy"),
        "question_pad": os.path.join(CACHE_DIR, "train_question_pad.npy"),
        "answer_pad": os.path.join(CACHE_DIR, "train_answer_pad.npy"),
        "host_oh": os.path.join(CACHE_DIR, "train_host_oh.npy"),
        "cat_oh": os.path.join(CACHE_DIR, "train_cat_oh.npy"),
        "stats": os.path.join(CACHE_DIR, "train_stats.npy"),
        "y": os.path.join(CACHE_DIR, "train_y.npy"),
        "host_categories": os.path.join(CACHE_DIR, "host_categories.npy"),
        "category_categories": os.path.join(CACHE_DIR, "category_categories.npy"),
        "tokens": os.path.join(CACHE_DIR, "tokens.npy"),
        "stats_min": os.path.join(CACHE_DIR, "stats_min.npy"),
        "stats_scale": os.path.join(CACHE_DIR, "stats_scale.npy"),
    }

    sample_sub = pd.read_csv(SAMPLE_PATH)
    target_cols = [c for c in sample_sub.columns if c != "qa_id"]

    all_exist = all(
        os.path.exists(p)
        for p in paths.values()
        if not p.endswith(
            ("host_categories.npy", "category_categories.npy", "tokens.npy")
        )
    )
    all_exist = (
        all_exist
        and os.path.exists(paths["host_categories"])
        and os.path.exists(paths["category_categories"])
        and os.path.exists(paths["tokens"])
    )

    if all_exist:
        title_pad = np.load(paths["title_pad"], mmap_mode="r")
        question_pad = np.load(paths["question_pad"], mmap_mode="r")
        answer_pad = np.load(paths["answer_pad"], mmap_mode="r")
        host_oh = np.load(paths["host_oh"], mmap_mode="r")
        cat_oh = np.load(paths["cat_oh"], mmap_mode="r")
        stats = np.load(paths["stats"], mmap_mode="r")
        y = np.load(paths["y"], mmap_mode="r")

        host_categories = np.load(paths["host_categories"], allow_pickle=True).tolist()
        category_categories = np.load(
            paths["category_categories"], allow_pickle=True
        ).tolist()
        tokens = np.load(paths["tokens"], allow_pickle=True).item()

        stats_min = np.load(paths["stats_min"])
        stats_scale = np.load(paths["stats_scale"])
        stats_scaler = sklearn.preprocessing.MinMaxScaler()
        stats_scaler.min_ = stats_min
        stats_scaler.scale_ = stats_scale
        stats_scaler.data_min_ = np.zeros_like(stats_min)  # unused by transform
        stats_scaler.data_max_ = np.zeros_like(stats_min)  # unused by transform
        stats_scaler.data_range_ = np.zeros_like(stats_min)  # unused by transform
        stats_scaler.n_features_in_ = 3

        return (
            title_pad,
            question_pad,
            answer_pad,
            host_oh,
            cat_oh,
            stats,
            y,
            tokens,
            host_categories,
            category_categories,
            stats_scaler,
            target_cols,
        )

    train_raw = pd.read_csv(TRAIN_PATH, index_col="qa_id")
    train, host_categories, category_categories = df_process(train_raw, True)

    Gq = wordbank(train)
    tokens = Gq.tokens()

    titles_vec = Gq._titles
    questions_vec = Gq._questions
    answers_vec = Gq._answers

    title_pad = tf.keras.preprocessing.sequence.pad_sequences(
        titles_vec, padding="post", dtype="int32"
    )
    question_pad = tf.keras.preprocessing.sequence.pad_sequences(
        questions_vec, padding="post", dtype="int32"
    )
    answer_pad = tf.keras.preprocessing.sequence.pad_sequences(
        answers_vec, padding="post", dtype="int32"
    )

    HOST_DEPTH = 59
    CAT_DEPTH = 5

    host_codes = train["host"].to_numpy(np.int32)
    cat_codes = train["category"].to_numpy(np.int32)
    host_codes = np.where(host_codes < 0, 0, host_codes)
    cat_codes = np.where(cat_codes < 0, 0, cat_codes)

    host_oh = np.eye(HOST_DEPTH, dtype=np.float32)[host_codes]
    cat_oh = np.eye(CAT_DEPTH, dtype=np.float32)[cat_codes]

    stats_scaled, stats_scaler = normalize(train[["q_len", "t_len", "ans_len"]])
    stats_scaled = stats_scaled.astype(np.float32)

    y = train[target_cols].values.astype(np.float32)

    _np_save(paths["title_pad"], np.asarray(title_pad, dtype=np.int32))
    _np_save(paths["question_pad"], np.asarray(question_pad, dtype=np.int32))
    _np_save(paths["answer_pad"], np.asarray(answer_pad, dtype=np.int32))
    _np_save(paths["host_oh"], np.asarray(host_oh, dtype=np.float32))
    _np_save(paths["cat_oh"], np.asarray(cat_oh, dtype=np.float32))
    _np_save(paths["stats"], np.asarray(stats_scaled, dtype=np.float32))
    _np_save(paths["y"], np.asarray(y, dtype=np.float32))
    _np_save(paths["host_categories"], np.array(host_categories, dtype=object))
    _np_save(paths["category_categories"], np.array(category_categories, dtype=object))
    _np_save(paths["tokens"], np.array(tokens, dtype=object))
    _np_save(paths["stats_min"], stats_scaler.min_.astype(np.float64))
    _np_save(paths["stats_scale"], stats_scaler.scale_.astype(np.float64))

    return (
        title_pad,
        question_pad,
        answer_pad,
        host_oh,
        cat_oh,
        stats_scaled,
        y,
        tokens,
        host_categories,
        category_categories,
        stats_scaler,
        target_cols,
    )


(
    title_pad,
    question_pad,
    answer_pad,
    host_oh,
    cat_oh,
    stats_scaled,
    y,
    tokens,
    host_categories,
    category_categories,
    stats_scaler,
    target_cols,
) = _load_or_build_train_arrays()

glove_candidates = [
    "/kaggle/input/filepython/glove.6B.50d.txt",
    os.path.join("/kaggle/input", "glove.6B.50d.txt"),
    os.path.join(DATA_DIR, "glove.6B.50d.txt"),
]
embedding_matrix = None
for fp in glove_candidates:
    if os.path.exists(fp):
        embedding_matrix = loadWordVectors(tokens, filepath=fp, dimensions=50)
        break

data = {
    "question": tf.convert_to_tensor(question_pad, dtype=tf.int32),
    "answer": tf.convert_to_tensor(answer_pad, dtype=tf.int32),
    "title": tf.convert_to_tensor(title_pad, dtype=tf.int32),
    "host": tf.convert_to_tensor(host_oh, dtype=tf.float32),
    "category": tf.convert_to_tensor(cat_oh, dtype=tf.float32),
    "stats": tf.convert_to_tensor(stats_scaled, dtype=tf.float32),
    "output": tf.convert_to_tensor(y, dtype=tf.float32),
}

model = rnnGoogleQuest(data, tokens, embedding_matrix)
model.train()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3558467294.py in <cell line: 0>()
    176     stats_scaler,
    177     target_cols,
--> 178 ) = _load_or_build_train_arrays()
    179 
    180 glove_candidates = [

/tmp/ipykernel_11/3558467294.py in _load_or_build_train_arrays()
    134     y = train[target_cols].values.astype(np.float32)
    135 
--> 136     _np_save(paths["title_pad"], np.asarray(title_pad, dtype=np.int32))
    137     _np_save(paths["question_pad"], np.asarray(question_pad, dtype=np.int32))
    138     _np_save(paths["answer_pad"], np.asarray(answer_pad, dtype=np.int32))

/tmp/ipykernel_11/3558467294.py in _np_save(path, arr)
     19     tmp = path + ".tmp"
     20     np.save(tmp, arr)
---> 21     os.replace(tmp, path)
     22 
     23 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/gq_cache_rnn_glove/train_title_pad.npy.tmp' -> '/kaggle/working/gq_cache_rnn_glove/train_title_pad.npy'

## === cell 6
pred_model = model._model




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/948415034.py in <cell line: 0>()
----> 1 pred_model = model._model
      2 
      3 

NameError: name 'model' is not defined

## === cell 7
def _load_or_build_test_arrays(train_title_len, train_question_len, train_answer_len):
    paths = {
        "title_pad": os.path.join(CACHE_DIR, "test_title_pad.npy"),
        "question_pad": os.path.join(CACHE_DIR, "test_question_pad.npy"),
        "answer_pad": os.path.join(CACHE_DIR, "test_answer_pad.npy"),
        "host_oh": os.path.join(CACHE_DIR, "test_host_oh.npy"),
        "cat_oh": os.path.join(CACHE_DIR, "test_cat_oh.npy"),
        "stats": os.path.join(CACHE_DIR, "test_stats.npy"),
        "qa_id": os.path.join(CACHE_DIR, "test_qa_id.npy"),
    }
    if all(os.path.exists(p) for p in paths.values()):
        return (
            np.load(paths["title_pad"], mmap_mode="r"),
            np.load(paths["question_pad"], mmap_mode="r"),
            np.load(paths["answer_pad"], mmap_mode="r"),
            np.load(paths["host_oh"], mmap_mode="r"),
            np.load(paths["cat_oh"], mmap_mode="r"),
            np.load(paths["stats"], mmap_mode="r"),
            np.load(paths["qa_id"], allow_pickle=True),
        )

    test_raw = pd.read_csv(TEST_PATH, index_col="qa_id")
    test_proc, _, _ = df_process(
        test_raw,
        False,
        host_categories=host_categories,
        category_categories=category_categories,
    )

    GqTest = wordbank(test_proc, train=False)
    GqTest._tokens = tokens
    GqTest.test()

    title_padT = tf.keras.preprocessing.sequence.pad_sequences(
        GqTest._titles, padding="post", maxlen=train_title_len, dtype="int32"
    )
    question_padT = tf.keras.preprocessing.sequence.pad_sequences(
        GqTest._questions, padding="post", maxlen=train_question_len, dtype="int32"
    )
    answer_padT = tf.keras.preprocessing.sequence.pad_sequences(
        GqTest._answers, padding="post", maxlen=train_answer_len, dtype="int32"
    )

    host_codes_t = test_proc["host"].to_numpy(np.int32)
    cat_codes_t = test_proc["category"].to_numpy(np.int32)
    host_codes_t = np.where(host_codes_t < 0, 0, host_codes_t)
    cat_codes_t = np.where(cat_codes_t < 0, 0, cat_codes_t)

    HOST_DEPTH = 59
    CAT_DEPTH = 5
    host_oh_t = np.eye(HOST_DEPTH, dtype=np.float32)[host_codes_t]
    cat_oh_t = np.eye(CAT_DEPTH, dtype=np.float32)[cat_codes_t]

    statsT_scaled, _ = normalize(
        test_proc[["q_len", "t_len", "ans_len"]], fit_scaler=stats_scaler
    )
    statsT_scaled = statsT_scaled.astype(np.float32)

    _np_save(paths["title_pad"], np.asarray(title_padT, dtype=np.int32))
    _np_save(paths["question_pad"], np.asarray(question_padT, dtype=np.int32))
    _np_save(paths["answer_pad"], np.asarray(answer_padT, dtype=np.int32))
    _np_save(paths["host_oh"], np.asarray(host_oh_t, dtype=np.float32))
    _np_save(paths["cat_oh"], np.asarray(cat_oh_t, dtype=np.float32))
    _np_save(paths["stats"], np.asarray(statsT_scaled, dtype=np.float32))
    _np_save(paths["qa_id"], np.asarray(test_raw.index.values, dtype=object))

    return (
        np.asarray(title_padT, dtype=np.int32),
        np.asarray(question_padT, dtype=np.int32),
        np.asarray(answer_padT, dtype=np.int32),
        np.asarray(host_oh_t, dtype=np.float32),
        np.asarray(cat_oh_t, dtype=np.float32),
        np.asarray(statsT_scaled, dtype=np.float32),
        np.asarray(test_raw.index.values, dtype=object),
    )


train_title_len = int(data["title"].shape[1])
train_question_len = int(data["question"].shape[1])
train_answer_len = int(data["answer"].shape[1])

title_padT, question_padT, answer_padT, host_oh_t, cat_oh_t, statsT_scaled, test_ids = (
    _load_or_build_test_arrays(train_title_len, train_question_len, train_answer_len)
)

dataT = {
    "title": tf.convert_to_tensor(title_padT, dtype=tf.int32),
    "question": tf.convert_to_tensor(question_padT, dtype=tf.int32),
    "answer": tf.convert_to_tensor(answer_padT, dtype=tf.int32),
    "category": tf.convert_to_tensor(cat_oh_t, dtype=tf.float32),
    "host": tf.convert_to_tensor(host_oh_t, dtype=tf.float32),
    "stats": tf.convert_to_tensor(statsT_scaled, dtype=tf.float32),
}

preds = pred_model.predict(
    {
        "title": dataT["title"],
        "question": dataT["question"],
        "answer": dataT["answer"],
        "category": dataT["category"],
        "host": dataT["host"],
        "stats": dataT["stats"],
    },
    batch_size=256,
    verbose=1,
).astype(np.float32)

preds = np.clip(preds, 0.0, 1.0)

submission = pd.DataFrame(
    preds, columns=target_cols, index=pd.Index(test_ids, name="qa_id")
)
submission.insert(0, "qa_id", submission.index)
submission.to_csv("submission.csv", index=False)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2418044309.py in <cell line: 0>()
     77 
     78 
---> 79 train_title_len = int(data["title"].shape[1])
     80 train_question_len = int(data["question"].shape[1])
     81 train_answer_len = int(data["answer"].shape[1])

NameError: name 'data' is not defined

## === cell 8
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["qa_id"] + target_cols
assert len(check) == len(test_ids)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1148043287.py in <cell line: 0>()
----> 1 check = pd.read_csv("submission.csv")
      2 assert list(check.columns) == ["qa_id"] + target_cols
      3 assert len(check) == len(test_ids)
      4 
      5 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

## === cell 9
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2863848987.py in <cell line: 0>()
----> 1 print("Wrote submission.csv with shape:", submission.shape)
      2 print(submission.head())

NameError: name 'submission' is not defined
