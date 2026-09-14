# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.preprocessing

np.random.seed(42)
tf.random.set_seed(42)

INPUT_DIR = "/kaggle/input/google-quest-challenge"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

DEFAULT_FILE_PATH = "/kaggle/input/filepython/glove.6B.50d.txt"

print("Files in input dir:", INPUT_DIR)
print("train exists:", os.path.exists(TRAIN_PATH), TRAIN_PATH)
print("test exists:", os.path.exists(TEST_PATH), TEST_PATH)
print("sample exists:", os.path.exists(SAMPLE_SUB_PATH), SAMPLE_SUB_PATH)
print("glove exists:", os.path.exists(DEFAULT_FILE_PATH), DEFAULT_FILE_PATH)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
def df_process(df=None, train=False, model="rnn_glove"):
    if df is None:
        df = pd.read_csv(TRAIN_PATH, index_col="qa_id")

    if model == "rnn_glove":
        df["question_user_page"] = (
            df["question_user_page"].fillna("").astype(str).str.split("/").str[-1]
        )
        df["answer_user_page"] = (
            df["answer_user_page"].fillna("").astype(str).str.split("/").str[-1]
        )
        df["category"] = (
            df["category"]
            .fillna("")
            .astype(str)
            .str.replace("[^a-zA-Z]", " ", regex=True)
            .str.lower()
        )
        df["host"] = df["host"].fillna("").astype(str).str.split(".").str[0]

        df.host = pd.Categorical(df.host)
        df["host"] = df.host.cat.codes
        df.category = pd.Categorical(df.category)
        df["category"] = df.category.cat.codes

        drop_cols = [
            c
            for c in ["question_user_name", "answer_user_name", "url"]
            if c in df.columns
        ]
        df = df.drop(drop_cols, axis=1)

        df["q_len"] = df["question_body"].fillna("").astype(str).str.len()
        df["t_len"] = df["question_title"].fillna("").astype(str).str.len()
        df["ans_len"] = df["answer"].fillna("").astype(str).str.len()

        return df

    return df




## === cell 2
def loadWordVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
    """Read pretrained GloVe vectors if available; otherwise return random embeddings.
    Speed-up: use a set of target tokens and avoid repeated dict lookups; parse floats via numpy.
    Core logic preserved: returns an embedding matrix aligned to `tokens` indices.
    """
    wordVectors = np.random.randn(len(tokens), dimensions).astype(np.float32)

    if not os.path.exists(filepath):
        print(f"[WARN] GloVe file not found at {filepath}. Using random embeddings.")
        return wordVectors

    token_set = set(tokens.keys())
    with open(filepath, encoding="utf-8") as ifs:
        for line in ifs:
            parts = line.rstrip().split(" ")
            if not parts:
                continue
            token = parts[0]
            if token not in token_set:
                continue
            vec = np.fromstring(" ".join(parts[1:]), sep=" ", dtype=np.float32)
            if vec.size != dimensions:
                raise RuntimeError("wrong number of dimensions")
            wordVectors[tokens[token]] = vec
    return wordVectors




## === cell 3
_TOKEN_RE = re.compile(r"[^a-zA-Z]")


def _tokenize_series_to_list(s: pd.Series):
    s = s.fillna("").astype(str).str.replace(_TOKEN_RE, " ", regex=True).str.lower()
    return s.str.split().tolist()


class wordbank:
    def __init__(self, df=None, train=True):
        if df is None:
            df = pd.read_csv(TRAIN_PATH, index_col="qa_id")
        self.df = df
        self.train = train

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        idx = 0

        tokens["<pad>"] = idx
        tokenfreq["<pad>"] = 0
        wordcount += 1
        idx += 1

        tokens["unknown"] = idx
        tokenfreq["unknown"] = 0
        wordcount += 1
        idx += 1

        self._titles_words = self.titles()
        self._questions_words = self.questions()
        self._answers_words = self.answers()
        self._sentences = (
            self._questions_words + self._answers_words + self._titles_words
        )

        for sentence in self._sentences:
            for w in sentence:
                wordcount += 1
                if w not in tokens:
                    tokens[w] = idx
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount

        self._titles = [[tokens[word] for word in sent] for sent in self._titles_words]
        self._questions = [
            [tokens[word] for word in sent] for sent in self._questions_words
        ]
        self._answers = [
            [tokens[word] for word in sent] for sent in self._answers_words
        ]
        return self._tokens

    def answers(self):
        if (
            hasattr(self, "_answers")
            and self._answers
            and isinstance(self._answers[0], list)
            and (len(self._answers) == len(self.df))
        ):
            return self._answers
        self._answers = _tokenize_series_to_list(self.df["answer"])
        return self._answers

    def titles(self):
        if (
            hasattr(self, "_titles")
            and self._titles
            and isinstance(self._titles[0], list)
            and (len(self._titles) == len(self.df))
        ):
            return self._titles
        self._titles = _tokenize_series_to_list(self.df["question_title"])
        return self._titles

    def questions(self):
        if (
            hasattr(self, "_questions")
            and self._questions
            and isinstance(self._questions[0], list)
            and (len(self._questions) == len(self.df))
        ):
            return self._questions
        self._questions = _tokenize_series_to_list(self.df["question_body"])
        return self._questions

    def test(self):
        ind_unk = self._tokens.get("unknown", 1)
        titles_words = self.titles()
        questions_words = self.questions()
        answers_words = self.answers()
        tok = self._tokens
        self._titles = [
            [tok.get(word, ind_unk) for word in sent] for sent in titles_words
        ]
        self._questions = [
            [tok.get(word, ind_unk) for word in sent] for sent in questions_words
        ]
        self._answers = [
            [tok.get(word, ind_unk) for word in sent] for sent in answers_words
        ]




## === cell 4
class NNGoogleQuest:
    def __init__(
        self, data, vocab, embedding_matrix=None, max_length=[0, 0, 0], method="lstm"
    ):
        self._data = data
        self._vocab = vocab
        self._embedding_matrix = embedding_matrix
        self._embedding_dim = self._embedding_matrix.shape[1]
        self._num_vocab = len(self._vocab)
        self._max_length = max_length
        self._method = method

    def embeddings(self, input, ind):
        embedding_layer = tf.keras.layers.Embedding(
            self._num_vocab,
            self._embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(
                self._embedding_matrix
            ),
            trainable=True,
            mask_zero=True,
            input_length=self._max_length[ind],
        )(input)

        if self._method == "lstm":
            lstm_layer = tf.keras.layers.LSTM(64, return_state=True)(embedding_layer)
            return lstm_layer
        else:
            cov_layer = tf.keras.layers.Conv1D(64, 3, activation="relu")(
                embedding_layer
            )
            flat = tf.keras.layers.GlobalMaxPooling1D()(cov_layer)
            return flat

    def NN_model(self):
        title_input = tf.keras.Input(shape=(None,), name="title")
        question_input = tf.keras.Input(shape=(None,), name="question")
        answer_input = tf.keras.Input(shape=(None,), name="answer")
        category_input = tf.keras.Input(shape=(5,), name="category")
        stats_input = tf.keras.Input(shape=(3,), name="stats")

        if self._method == "lstm":
            encoder_outputs, title_state_h, state_c = self.embeddings(title_input, 0)
            encoder_outputs, question_state_h, state_c = self.embeddings(
                question_input, 1
            )
            encoder_outputs, answer_state_h, state_c = self.embeddings(answer_input, 2)
        else:
            title_state_h = self.embeddings(title_input, 0)
            question_state_h = self.embeddings(question_input, 1)
            answer_state_h = self.embeddings(answer_input, 2)

        features = tf.keras.layers.concatenate(
            [
                title_state_h,
                question_state_h,
                answer_state_h,
                category_input,
                stats_input,
            ]
        )
        hidden = tf.keras.layers.Dense(128, activation="relu", name="hidden")(features)
        drop1 = tf.keras.layers.Dropout(0.4)(hidden)
        pred = tf.keras.layers.Dense(30, activation="sigmoid", name="prediction")(drop1)

        model = tf.keras.Model(
            inputs=[
                title_input,
                question_input,
                answer_input,
                category_input,
                stats_input,
            ],
            outputs=pred,
        )
        self._model = model

    def train(self):
        self.NN_model()

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
        )

        x = {
            "title": self._data["title"],
            "question": self._data["question"],
            "answer": self._data["answer"],
            "category": self._data["category"],
            "stats": self._data["stats"],
        }
        y = self._data["output"]

        n = int(y.shape[0])
        val_n = int(round(n * 0.1))
        train_n = n - val_n

        def _slice_x(xdict, sl):
            return {k: v[sl] for k, v in xdict.items()}

        x_train = _slice_x(x, slice(0, train_n))
        y_train = y[:train_n]
        x_val = _slice_x(x, slice(train_n, n))
        y_val = y[train_n:]

        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_train = ds_train.batch(128, drop_remainder=False)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        ds_val = ds_val.batch(128, drop_remainder=False)
        ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

        self._model.fit(
            ds_train,
            epochs=20,
            validation_data=ds_val,
            verbose=2,
        )
        if self._method == "lstm":
            self._model.save("rnn_glove_model.keras")
        else:
            self._model.save("cnn_glove_model.keras")




## === cell 5
def multi_hot_enc_from_train(train_array, test_array):
    """Fit on train codes and transform both train and test; output fixed 5-dim as expected by model."""
    lb = sklearn.preprocessing.LabelBinarizer()
    classes = np.unique(train_array)
    lb.fit(classes)
    train_oh = lb.transform(train_array)
    test_oh = lb.transform(test_array)

    if train_oh.ndim == 1:
        train_oh = train_oh.reshape(-1, 1)
    if test_oh.ndim == 1:
        test_oh = test_oh.reshape(-1, 1)

    def fix_dim(x, dim=5):
        if x.shape[1] == dim:
            return x
        if x.shape[1] > dim:
            return x[:, :dim]
        pad = np.zeros((x.shape[0], dim - x.shape[1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)

    return fix_dim(train_oh, 5).astype(np.float32), fix_dim(test_oh, 5).astype(
        np.float32
    )


def normalize_train_test(train_df, test_df):
    scaler = sklearn.preprocessing.MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_df.values)
    return train_scaled.astype(np.float32), test_scaled.astype(np.float32)




## === cell 6
sample = pd.read_csv(SAMPLE_SUB_PATH)
TARGET_COLS = [c for c in sample.columns if c != "qa_id"]

train = pd.read_csv(TRAIN_PATH, index_col="qa_id")
train = df_process(train, True)

Gq = wordbank(train)
tokens = Gq.tokens()

word_vectors = loadWordVectors(tokens, dimensions=50)

titles_vec = Gq._titles
questions_vec = Gq._questions
answers_vec = Gq._answers

title_pad = tf.keras.preprocessing.sequence.pad_sequences(
    titles_vec, padding="post"
).astype(np.int32)
question_pad = tf.keras.preprocessing.sequence.pad_sequences(
    questions_vec, padding="post"
).astype(np.int32)
answer_pad = tf.keras.preprocessing.sequence.pad_sequences(
    answers_vec, padding="post"
).astype(np.int32)

test = pd.read_csv(TEST_PATH, index_col="qa_id")
test2 = df_process(test.copy(), False)

cat_train, cat_test = multi_hot_enc_from_train(
    train["category"].to_numpy(), test2["category"].to_numpy()
)
stats_train, stats_test = normalize_train_test(
    train[["q_len", "t_len", "ans_len"]], test2[["q_len", "t_len", "ans_len"]]
)

data = {}
data["question"] = question_pad
data["answer"] = answer_pad
data["title"] = title_pad
data["category"] = cat_train.astype(np.float32)
data["stats"] = stats_train.astype(np.float32)
data["output"] = train[TARGET_COLS].values.astype(np.float32)

print(
    "Train arrays:",
    data["title"].shape,
    data["question"].shape,
    data["answer"].shape,
    data["category"].shape,
    data["stats"].shape,
    data["output"].shape,
)



## === cell 7
print("Training using LSTM model\n")
rnn_trainer = NNGoogleQuest(
    data,
    tokens,
    word_vectors,
    method="lstm",
    max_length=[
        data["title"].shape[1],
        data["question"].shape[1],
        data["answer"].shape[1],
    ],
)
rnn_trainer.train()



## === cell 8
print("Training using CNN model\n")
cnn_trainer = NNGoogleQuest(
    data,
    tokens,
    word_vectors,
    method="cnn",
    max_length=[
        data["title"].shape[1],
        data["question"].shape[1],
        data["answer"].shape[1],
    ],
)
cnn_trainer.train()



## === cell 9
rnn_model = None
cnn_model = None

if os.path.exists("rnn_glove_model.keras"):
    try:
        rnn_model = tf.keras.models.load_model("rnn_glove_model.keras", compile=False)
    except Exception as e:
        print(
            "[WARN] Could not load rnn_glove_model.keras, will use in-memory model if available:",
            e,
        )

if os.path.exists("cnn_glove_model.keras"):
    try:
        cnn_model = tf.keras.models.load_model("cnn_glove_model.keras", compile=False)
    except Exception as e:
        print(
            "[WARN] Could not load cnn_glove_model.keras, will use in-memory model if available:",
            e,
        )

if rnn_model is None:
    rnn_model = getattr(rnn_trainer, "_model", None)
if cnn_model is None:
    cnn_model = getattr(cnn_trainer, "_model", None)

print(
    "Models ready:",
    "rnn_model is None?",
    rnn_model is None,
    "| cnn_model is None?",
    cnn_model is None,
)



## === cell 10
GqTest = wordbank(test2, train=False)
GqTest._tokens = tokens  # use train vocab
GqTest.test()

titles_vecT = GqTest._titles
questions_vecT = GqTest._questions
answers_vecT = GqTest._answers

title_padT = tf.keras.preprocessing.sequence.pad_sequences(
    titles_vecT, padding="post"
).astype(np.int32)
question_padT = tf.keras.preprocessing.sequence.pad_sequences(
    questions_vecT, padding="post"
).astype(np.int32)
answer_padT = tf.keras.preprocessing.sequence.pad_sequences(
    answers_vecT, padding="post"
).astype(np.int32)

dataT = {}
dataT["question"] = question_padT
dataT["answer"] = answer_padT
dataT["title"] = title_padT
dataT["category"] = cat_test.astype(np.float32)
dataT["stats"] = stats_test.astype(np.float32)

print(
    "Test arrays:",
    dataT["title"].shape,
    dataT["question"].shape,
    dataT["answer"].shape,
    dataT["category"].shape,
    dataT["stats"].shape,
)



## === cell 11
if cnn_model is None:
    raise RuntimeError("cnn_model is not available; training/loading failed.")

pred = cnn_model.predict(
    {
        "title": dataT["title"],
        "question": dataT["question"],
        "answer": dataT["answer"],
        "category": dataT["category"],
        "stats": dataT["stats"],
    },
    batch_size=256,
    verbose=1,
).astype(np.float32)

pred = np.clip(pred, 0.0, 1.0)

sub = pd.DataFrame(pred, columns=TARGET_COLS, index=test.index)
sub.insert(0, "qa_id", sub.index)

sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("Submission columns ok?", list(sub.columns) == list(sample.columns))
