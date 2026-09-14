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




## === cell 1
_non_letters = re.compile(r"[^a-zA-Z]+")


def df_process(df=None, train=False, model="rnn_glove"):
    if df is None:
        df = pd.read_csv(TRAIN_PATH, index_col="qa_id")
    if model == "rnn_glove":
        df = df.copy()

        qpage = df["question_user_page"].fillna("").astype(str)
        apage = df["answer_user_page"].fillna("").astype(str)
        df["question_user_page"] = qpage.str.rsplit("/", n=1).str[-1]
        df["answer_user_page"] = apage.str.rsplit("/", n=1).str[-1]

        cat = df["category"].fillna("").astype(str)
        df["category"] = cat.str.replace(_non_letters, " ", regex=True).str.lower()

        host = df["host"].fillna("").astype(str)
        df["host"] = host.str.split(".", n=1).str[0]

        df.host = pd.Categorical(df.host)
        df["host"] = df.host.cat.codes
        df.category = pd.Categorical(df.category)
        df["category"] = df.category.cat.codes

        df = df.drop(["question_user_name", "answer_user_name", "url"], axis=1)

        df["q_len"] = df["question_body"].fillna("").astype(str).str.len()
        df["t_len"] = df["question_title"].fillna("").astype(str).str.len()
        df["ans_len"] = df["answer"].fillna("").astype(str).str.len()
        return df




## === cell 2
def loadWordVectors(tokens, filepath, dimensions=50):
    """Read pretrained GloVe vectors if available; otherwise raise FileNotFoundError."""
    wordVectors = np.random.randn(len(tokens), dimensions).astype(np.float32)
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            token = row[0]
            if token not in tokens:
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions")
            wordVectors[tokens[token]] = np.asarray(data, dtype=np.float32)
    return wordVectors




## === cell 3
class wordbank:
    def __init__(self, df=None, train=True):
        if df is None:
            df = pd.read_csv(TRAIN_PATH, index_col="qa_id")
        self.df = df
        self.train = train

    @staticmethod
    def _texts_to_token_lists(texts: pd.Series):
        s = texts.fillna("").astype(str)
        s = s.str.replace(_non_letters, " ", regex=True).str.strip().str.lower()
        return s.str.split().tolist()

    def answers(self):
        if hasattr(self, "_answers") and self._answers:
            return self._answers
        self._answers = self._texts_to_token_lists(self.df["answer"])
        return self._answers

    def titles(self):
        if hasattr(self, "_titles") and self._titles:
            return self._titles
        self._titles = self._texts_to_token_lists(self.df["question_title"])
        return self._titles

    def questions(self):
        if hasattr(self, "_questions") and self._questions:
            return self._questions
        self._questions = self._texts_to_token_lists(self.df["question_body"])
        return self._questions

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        self._sentences = self.questions() + self.answers() + self.titles()
        texts = [" ".join(ws) for ws in self._sentences]

        tok = tf.keras.preprocessing.text.Tokenizer(
            lower=False,  # already lowered
            filters="",  # already filtered
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
        self._tokenfreq = None
        self._wordcount = None

        def to_ids(token_lists):
            unk = word_index.get("unknown", 1)
            get = word_index.get
            return [[get(w, unk) for w in sent] for sent in token_lists]

        self._titles = to_ids(self.titles())
        self._questions = to_ids(self.questions())
        self._answers = to_ids(self.answers())
        return self._tokens

    def test(self):
        ind_unk = self._tokens.get("unknown", 1)
        get = self._tokens.get
        if not (
            hasattr(self, "_titles")
            and self._titles
            and isinstance(self._titles[0], list)
            and (len(self._titles[0]) == 0 or isinstance(self._titles[0][0], str))
        ):
            self._titles = self.titles()
        if not (
            hasattr(self, "_questions")
            and self._questions
            and isinstance(self._questions[0], list)
            and (len(self._questions[0]) == 0 or isinstance(self._questions[0][0], str))
        ):
            self._questions = self.questions()
        if not (
            hasattr(self, "_answers")
            and self._answers
            and isinstance(self._answers[0], list)
            and (len(self._answers[0]) == 0 or isinstance(self._answers[0][0], str))
        ):
            self._answers = self.answers()

        self._titles = [[get(word, ind_unk) for word in sent] for sent in self._titles]
        self._questions = [
            [get(word, ind_unk) for word in sent] for sent in self._questions
        ]
        self._answers = [
            [get(word, ind_unk) for word in sent] for sent in self._answers
        ]




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
        title_input = tf.keras.Input(shape=(None,), name="title")
        question_input = tf.keras.Input(shape=(None,), name="question")
        answer_input = tf.keras.Input(shape=(None,), name="answer")
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

        ds = (
            tf.data.Dataset.from_tensor_slices(
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
            .batch(128, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        self._model.fit(
            ds,
            epochs=25,
            verbose=2,
        )
        self._model.save("rnn_glove_model.keras")




## === cell 5
def fit_multi_hot_encoder(train_df, label):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(range(int(train_df[label].max()) + 1))
    return lb


def multi_hot_transform(lb, array):
    return lb.transform(array)


def normalize(df, fit_scaler=None):
    x = df.values
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    if fit_scaler is None:
        x_scaled = min_max_scaler.fit_transform(x)
        return x_scaled, min_max_scaler
    else:
        return fit_scaler.transform(x), fit_scaler


train = pd.read_csv(TRAIN_PATH, index_col="qa_id")
train = df_process(train, True)

Gq = wordbank(train)
tokens = Gq.tokens()

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

titles_vec = Gq._titles
questions_vec = Gq._questions
answers_vec = Gq._answers

title_pad = tf.keras.preprocessing.sequence.pad_sequences(titles_vec, padding="post")
question_pad = tf.keras.preprocessing.sequence.pad_sequences(
    questions_vec, padding="post"
)
answer_pad = tf.keras.preprocessing.sequence.pad_sequences(answers_vec, padding="post")

host_lb = fit_multi_hot_encoder(train, "host")
cat_lb = fit_multi_hot_encoder(train, "category")

data = {}
data["question"] = tf.convert_to_tensor(question_pad)
data["answer"] = tf.convert_to_tensor(answer_pad)
data["title"] = tf.convert_to_tensor(title_pad)

data["host"] = tf.convert_to_tensor(
    multi_hot_transform(host_lb, train["host"].to_numpy()).astype(np.float32)
)
data["category"] = tf.convert_to_tensor(
    multi_hot_transform(cat_lb, train["category"].to_numpy()).astype(np.float32)
)

stats_scaled, stats_scaler = normalize(train[["q_len", "t_len", "ans_len"]])
data["stats"] = tf.convert_to_tensor(stats_scaled.astype(np.float32))

sample_sub = pd.read_csv(SAMPLE_PATH)
target_cols = [c for c in sample_sub.columns if c != "qa_id"]
data["output"] = tf.convert_to_tensor(train[target_cols].values.astype(np.float32))

model = rnnGoogleQuest(data, tokens, embedding_matrix)
model.train()




## === cell 6
pred_model = model._model




## === cell 7
test = pd.read_csv(TEST_PATH, index_col="qa_id")
test_proc = df_process(test.copy(), False)

GqTest = wordbank(test_proc, train=False)
GqTest._tokens = tokens  # use training tokens
GqTest.test()

titles_vecT = GqTest._titles
questions_vecT = GqTest._questions
answers_vecT = GqTest._answers

title_padT = tf.keras.preprocessing.sequence.pad_sequences(titles_vecT, padding="post")
question_padT = tf.keras.preprocessing.sequence.pad_sequences(
    questions_vecT, padding="post"
)
answer_padT = tf.keras.preprocessing.sequence.pad_sequences(
    answers_vecT, padding="post"
)

dataT = {}
dataT["question"] = tf.convert_to_tensor(question_padT)
dataT["answer"] = tf.convert_to_tensor(answer_padT)
dataT["title"] = tf.convert_to_tensor(title_padT)

dataT["host"] = tf.convert_to_tensor(
    multi_hot_transform(host_lb, test_proc["host"].to_numpy()).astype(np.float32)
)
dataT["category"] = tf.convert_to_tensor(
    multi_hot_transform(cat_lb, test_proc["category"].to_numpy()).astype(np.float32)
)

statsT_scaled, _ = normalize(
    test_proc[["q_len", "t_len", "ans_len"]], fit_scaler=stats_scaler
)
dataT["stats"] = tf.convert_to_tensor(statsT_scaled.astype(np.float32))

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

submission = pd.DataFrame(preds, columns=target_cols, index=test.index)
submission.insert(0, "qa_id", submission.index)

submission.to_csv("submission.csv", index=False)




## === cell 8
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["qa_id"] + target_cols
assert len(check) == len(test)




## === cell 9
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
