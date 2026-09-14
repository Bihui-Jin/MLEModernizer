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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os, re, sklearn.preprocessing

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DEFAULT_FILE_PATH = "/kaggle/input/filepython/glove.6B.50d.txt"  # may not exist




## === cell 2
def df_process(df=None, train=False, model="rnn_glove"):
    if df is None:
        df = pd.read_csv("google-quest-challenge/train.csv", index_col="qa_id")
    if model == "rnn_glove":
        df["question_user_page"] = df["question_user_page"].apply(
            lambda x: x.split("/")[-1]
        )
        df["answer_user_page"] = df["answer_user_page"].apply(
            lambda x: x.split("/")[-1]
        )
        df["category"] = df["category"].apply(
            lambda x: re.sub("[^a-zA-Z]", " ", x).lower()
        )
        df["host"] = df["host"].apply(lambda x: x.split(".")[0])
        df.host = pd.Categorical(df.host)
        df["host"] = df.host.cat.codes
        df.category = pd.Categorical(df.category)
        df["category"] = df.category.cat.codes
        df = df.drop(["question_user_name", "answer_user_name", "url"], axis=1)
        df["q_len"] = df["question_body"].str.len()
        df["t_len"] = df["question_title"].str.len()
        df["ans_len"] = df["answer"].str.len()
        return df




## === cell 3
def loadWordVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=50):
    """Read pretrained GloVe vectors; fall back to random vectors if file missing."""
    wordVectors = np.random.randn(len(tokens), dimensions).astype(np.float32)
    try:
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
    except FileNotFoundError:
        print(f"Warning: GloVe file {filepath} not found. Using random embeddings.")
    return wordVectors




## === cell 4
class wordbank:
    def __init__(self, df=None, train=True):
        if df is None:
            df = pd.read_csv("google-quest-challenge/train.csv", index_col="qa_id")
        self.df = df
        self.train = train

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = {"<pad>": 0}
        tokenfreq = {"<pad>": 0}
        wordcount = 1
        idx = 1

        self._sentences = self.questions() + self.answers() + self.titles()
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
        self._titles = [[tokens[word] for word in sent] for sent in self.titles()]
        self._questions = [[tokens[word] for word in sent] for sent in self.questions()]
        self._answers = [[tokens[word] for word in sent] for sent in self.answers()]
        return self._tokens

    def answers(self):
        if hasattr(self, "_answers") and self._answers:
            return self._answers
        ans_df = self.df["answer"].apply(
            lambda x: re.sub("[^a-zA-Z]", " ", x).strip().split()
        )
        ans_df = ans_df.apply(lambda x: [w.lower() for w in x])
        self._answers = ans_df.tolist()
        return self._answers

    def titles(self):
        if hasattr(self, "_titles") and self._titles:
            return self._titles
        titles_df = self.df["question_title"].apply(
            lambda x: re.sub("[^a-zA-Z]", " ", x).strip().split()
        )
        titles_df = titles_df.apply(lambda x: [w.lower() for w in x])
        self._titles = titles_df.tolist()
        return self._titles

    def questions(self):
        if hasattr(self, "_questions") and self._questions:
            return self._questions
        q_df = self.df["question_body"].apply(
            lambda x: re.sub("[^a-zA-Z]", " ", x).strip().split()
        )
        q_df = q_df.apply(lambda x: [w.lower() for w in x])
        self._questions = q_df.tolist()
        return self._questions

    def test(self):
        unk_idx = self._tokens.get("unknown", 0)
        self._titles = [
            [self._tokens.get(word, unk_idx) for word in sent] for sent in self.titles()
        ]
        self._questions = [
            [self._tokens.get(word, unk_idx) for word in sent]
            for sent in self.questions()
        ]
        self._answers = [
            [self._tokens.get(word, unk_idx) for word in sent]
            for sent in self.answers()
        ]




## === cell 5
class rnnGoogleQuest:
    def __init__(self, data, vocab, embedding_matrix=None, max_length=0):
        self._data = data
        self._vocab = vocab
        self._embedding_matrix = embedding_matrix
        self._embedding_dim = (
            self._embedding_matrix.shape[1]
            if self._embedding_matrix is not None
            else 50
        )
        self._num_vocab = len(self._vocab)
        self._max_length = max_length

    def embeddings(self, inp):
        embed_layer = tf.keras.layers.Embedding(
            self._num_vocab,
            self._embedding_dim,
            embeddings_initializer=(
                tf.keras.initializers.Constant(self._embedding_matrix)
                if self._embedding_matrix is not None
                else "uniform"
            ),
            trainable=True,
            mask_zero=True,
        )(inp)
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(64, return_state=True)(
            embed_layer
        )
        return lstm_out, state_h, state_c

    def rnn_model(self):
        title_in = tf.keras.Input(shape=(None,), name="title")
        ques_in = tf.keras.Input(shape=(None,), name="question")
        ans_in = tf.keras.Input(shape=(None,), name="answer")
        cat_in = tf.keras.Input(shape=(5,), name="category")
        host_in = tf.keras.Input(shape=(59,), name="host")
        stats_in = tf.keras.Input(shape=(3,), name="stats")

        _, title_h, _ = self.embeddings(title_in)
        _, ques_h, _ = self.embeddings(ques_in)
        _, ans_h, _ = self.embeddings(ans_in)

        concat = tf.keras.layers.concatenate(
            [title_h, ques_h, ans_h, cat_in, host_in, stats_in]
        )
        hidden = tf.keras.layers.Dense(512, activation="relu")(concat)
        drop1 = tf.keras.layers.Dropout(0.4)(hidden)
        hidden2 = tf.keras.layers.Dense(128, activation="relu")(drop1)
        drop2 = tf.keras.layers.Dropout(0.3)(hidden2)
        pred = tf.keras.layers.Dense(30, activation="sigmoid")(drop2)

        self._model = tf.keras.Model(
            inputs=[title_in, ques_in, ans_in, cat_in, host_in, stats_in], outputs=pred
        )
        tf.keras.utils.plot_model(self._model, "rnn_model.png", show_shapes=True)

    def train(self, epochs=5):
        self.rnn_model()
        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_absolute_error",
            metrics=["mean_squared_error"],
        )
        self._model.fit(
            {
                "title": self._data["title"],
                "question": self._data["question"],
                "answer": self._data["answer"],
                "category": self._data["category"],
                "host": self._data["host"],
                "stats": self._data["stats"],
            },
            self._data["output"],
            epochs=epochs,
            batch_size=128,
            verbose=2,
        )
        self._model.save("rnn_glove_model.keras")




## === cell 6
def multi_hot_enc(array):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(range(int(array.max()) + 1))
    return lb.transform(array)


def normalize(df):
    scaler = sklearn.preprocessing.MinMaxScaler()
    return scaler.fit_transform(df.values)




## === cell 7
train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv", index_col="qa_id")
train = df_process(train, train=True)

Gq = wordbank(train)
tokens = Gq.tokens()
word_vectors = loadWordVectors(tokens)

titles_vec = Gq._titles
questions_vec = Gq._questions
answers_vec = Gq._answers

data = {}
data["title"] = tf.convert_to_tensor(
    tf.keras.preprocessing.sequence.pad_sequences(titles_vec, padding="post")
)
data["question"] = tf.convert_to_tensor(
    tf.keras.preprocessing.sequence.pad_sequences(questions_vec, padding="post")
)
data["answer"] = tf.convert_to_tensor(
    tf.keras.preprocessing.sequence.pad_sequences(answers_vec, padding="post")
)
data["host"] = tf.convert_to_tensor(multi_hot_enc(train["host"].to_numpy()))
data["category"] = tf.convert_to_tensor(multi_hot_enc(train["category"].to_numpy()))
data["stats"] = tf.convert_to_tensor(normalize(train[["q_len", "t_len", "ans_len"]]))
data["output"] = tf.convert_to_tensor(train.iloc[:, -30:].values.astype(np.float32))

model = rnnGoogleQuest(data, tokens, embedding_matrix=word_vectors)
model.train(epochs=5)

trained_model = tf.keras.models.load_model("rnn_glove_model.keras")




## === cell 8
test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv", index_col="qa_id")
test_processed = df_process(test.copy(), train=False)

Gq_test = wordbank(test_processed, train=False)
Gq_test._tokens = tokens  # reuse training vocab
Gq_test.test()

titles_vecT = Gq_test._titles
questions_vecT = Gq_test._questions
answers_vecT = Gq_test._answers

dataT = {}
dataT["title"] = tf.convert_to_tensor(
    tf.keras.preprocessing.sequence.pad_sequences(titles_vecT, padding="post")
)
dataT["question"] = tf.convert_to_tensor(
    tf.keras.preprocessing.sequence.pad_sequences(questions_vecT, padding="post")
)
dataT["answer"] = tf.convert_to_tensor(
    tf.keras.preprocessing.sequence.pad_sequences(answers_vecT, padding="post")
)
dataT["host"] = tf.convert_to_tensor(multi_hot_enc(test_processed["host"].to_numpy()))
dataT["category"] = tf.convert_to_tensor(
    multi_hot_enc(test_processed["category"].to_numpy())
)
dataT["stats"] = tf.convert_to_tensor(
    normalize(test_processed[["q_len", "t_len", "ans_len"]])
)

preds = trained_model.predict(
    {
        "title": dataT["title"],
        "question": dataT["question"],
        "answer": dataT["answer"],
        "category": dataT["category"],
        "host": dataT["host"],
        "stats": dataT["stats"],
    },
    batch_size=128,
    verbose=0,
)

target_cols = train.columns[-30:]  # correct 30 target columns
submission = pd.DataFrame(preds, index=test.index, columns=target_cols)
submission.reset_index(inplace=True)  # bring qa_id back as a column
submission.to_csv("submission.csv", index=False)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2354598502.py in <cell line: 0>()
     28 )
     29 
---> 30 preds = trained_model.predict(
     31     {
     32         "title": dataT["title"],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    243                 if spec_dim is not None and dim is not None:
    244                     if spec_dim != dim:
--> 245                         raise ValueError(
    246                             f'Input {input_index} of layer "{layer_name}" is '
    247                             "incompatible with the layer: "

ValueError: Input 4 of layer "functional" is incompatible with the layer: expected shape=(None, 59), found shape=(128, 56)

## === cell 9
print("Submission file created:", os.path.abspath("submission.csv"))
