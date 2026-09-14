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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
nltk==3.9.2
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.6536479592323303

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from __future__ import unicode_literals, print_function

import os
import re
import string
import random
import datetime
from pathlib import Path

import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from tqdm import tqdm

LABEL = "SELECTEDTEXT"


def read_data(datadir):
    train = pd.read_csv(os.path.join(datadir, "train.csv"))
    test = pd.read_csv(os.path.join(datadir, "test.csv"))
    sample_submission = pd.read_csv(os.path.join(datadir, "sample_submission.csv"))
    return (train, test, sample_submission)


def jaccard_similarity(string1, string2):
    wordset = lambda x: set(str(x).lower().split())
    a, b = wordset(string1), wordset(string2)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


def training_data(dataframe):
    data = dataframe.dropna(subset=["text", "selected_text", "sentiment"]).copy()
    data["text"] = data["text"].astype(str).str.lower()
    data["selected_text"] = data["selected_text"].astype(str).str.lower()

    texts = data["text"].to_numpy()
    sels = data["selected_text"].to_numpy()
    starts = [t.find(s) for t, s in zip(texts, sels)]
    data["start"] = starts
    data["end"] = data["start"] + data["selected_text"].str.len()

    valid = (data["start"] >= 0) & (data["end"] > data["start"])
    data = data.loc[valid, ["text", "sentiment", "start", "end"]]

    pos_df = data[data["sentiment"] == "positive"]
    neg_df = data[data["sentiment"] == "negative"]

    positive = list(
        zip(
            pos_df["text"].tolist(),
            [
                {"entities": [(int(s), int(e), LABEL)]}
                for s, e in zip(pos_df["start"].to_numpy(), pos_df["end"].to_numpy())
            ],
        )
    )
    negative = list(
        zip(
            neg_df["text"].tolist(),
            [
                {"entities": [(int(s), int(e), LABEL)]}
                for s, e in zip(neg_df["start"].to_numpy(), neg_df["end"].to_numpy())
            ],
        )
    )

    print(f"Positive data size: {len(positive):,}")
    print(f"Negative data size: {len(negative):,}")

    return positive, negative


def run_model(data, positivemodel, negativemodel, outputpath=None):
    print("Loading models...")
    positivemodel = str(Path(positivemodel).resolve())
    negativemodel = str(Path(negativemodel).resolve())
    positive_nlp = spacy.load(positivemodel)
    negative_nlp = spacy.load(negativemodel)

    data = data.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

    input_train = "selected_text" in data.columns

    text_series = data["text"].astype(str)
    sent_series = data["sentiment"].astype(str)

    short_mask = text_series.str.split().str.len().le(2)

    selected_texts = [""] * len(data)

    neutral_mask = sent_series.eq("neutral")
    for i in data.index[neutral_mask]:
        selected_texts[i] = text_series.iat[i]

    pos_mask = sent_series.eq("positive")
    neg_mask = sent_series.eq("negative")
    for i in data.index[pos_mask & short_mask]:
        selected_texts[i] = text_series.iat[i]
    for i in data.index[neg_mask & short_mask]:
        selected_texts[i] = text_series.iat[i]

    pos_long_idx = data.index[pos_mask & (~short_mask)]
    if len(pos_long_idx) > 0:
        pos_text_gen = (text_series.iat[i] for i in pos_long_idx)
        for i, doc in zip(
            pos_long_idx, positive_nlp.pipe(pos_text_gen, batch_size=128)
        ):
            selected_texts[i] = (
                doc.ents[0].text if len(doc.ents) > 0 else text_series.iat[i]
            )

    neg_long_idx = data.index[neg_mask & (~short_mask)]
    if len(neg_long_idx) > 0:
        neg_text_gen = (text_series.iat[i] for i in neg_long_idx)
        for i, doc in zip(
            neg_long_idx, negative_nlp.pipe(neg_text_gen, batch_size=128)
        ):
            selected_texts[i] = (
                doc.ents[0].text if len(doc.ents) > 0 else text_series.iat[i]
            )

    output_df = pd.DataFrame(
        {"textID": data["textID"].tolist(), "selected_text": selected_texts}
    )

    if not input_train:
        if not outputpath:
            suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs("submissions", exist_ok=True)
            outputpath = os.path.join("submissions", "submission_" + suffix + ".csv")

        print(f"Saving submission to {outputpath} ...")
        output_df.to_csv(outputpath, index=False)

    if input_train:
        jaccards = {"positive": [], "negative": [], "neutral": []}
        for txt, sent, sel in zip(
            text_series.tolist(), sent_series.tolist(), selected_texts
        ):
            jaccards[sent].append(jaccard_similarity(txt, sel))

        nums, dens = [], []
        for key in ("positive", "negative", "neutral"):
            num = sum(jaccards[key])
            den = len(jaccards[key]) if len(jaccards[key]) > 0 else 1
            print(f"Jaccard score for {key}: {num / den:.3f}")
            nums.append(num)
            dens.append(den)
        print(f"Jaccard score for overall: {sum(nums) / sum(dens):.3f}")


def train_model(traindata, new_model_name, model=None, output_dir=None, n_iter=30):
    """Train a spaCy NER model to extract selected text spans."""
    random.seed(0)

    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)

    move_names = list(ner.move_names)
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    if model is None:
        init_tuples = traindata[: min(100, len(traindata))]

        def init_examples():
            texts = [t for t, _ in init_tuples]
            anns = [a for _, a in init_tuples]
            docs = list(nlp.pipe(texts, batch_size=64))
            return [Example.from_dict(doc, ann) for doc, ann in zip(docs, anns)]

        optimizer = nlp.initialize(get_examples=init_examples)
    else:
        optimizer = nlp.resume_training()

    with nlp.disable_pipes(*other_pipes):
        sizes = compounding(1.0, 4.0, 1.001)
        for itn in range(n_iter):
            random.shuffle(traindata)
            losses = {}
            for batch in minibatch(traindata, size=sizes):
                texts = [t for t, _ in batch]
                anns = [a for _, a in batch]
                docs = list(nlp.pipe(texts, batch_size=len(texts)))
                examples = [Example.from_dict(doc, ann) for doc, ann in zip(docs, anns)]
                nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    test_text = "i`d have responded, if i were going"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        print("Loading from", output_dir)
        nlp2 = spacy.load(str(output_dir))
        assert list(nlp2.get_pipe("ner").move_names) == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)




## === cell 1
spacy.prefer_gpu()

n_iter = 20

datadir = "/kaggle/input/tweet-sentiment-extraction"
models_root = Path("models")
pos_dir = models_root / "positive"
neg_dir = models_root / "negative"

train, test, _ = read_data(datadir)
positive, negative = training_data(train)

models_root.mkdir(parents=True, exist_ok=True)

if not pos_dir.exists():
    print(f"Positive model not found at {pos_dir}. Training and saving...")
    train_model(
        positive,
        new_model_name="positive",
        model=None,
        output_dir=str(pos_dir),
        n_iter=n_iter,
    )

if not neg_dir.exists():
    print(f"Negative model not found at {neg_dir}. Training and saving...")
    train_model(
        negative,
        new_model_name="negative",
        model=None,
        output_dir=str(neg_dir),
        n_iter=n_iter,
    )



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1826143462.py in <cell line: 0>()
     15 if not pos_dir.exists():
     16     print(f"Positive model not found at {pos_dir}. Training and saving...")
---> 17     train_model(
     18         positive,
     19         new_model_name="positive",

/tmp/ipykernel_11/1041434197.py in train_model(traindata, new_model_name, model, output_dir, n_iter)
    203             return [Example.from_dict(doc, ann) for doc, ann in zip(docs, anns)]
    204 
--> 205         optimizer = nlp.initialize(get_examples=init_examples)
    206     else:
    207         optimizer = nlp.resume_training()

/usr/local/lib/python3.11/dist-packages/spacy/language.py in initialize(self, get_examples, sgd)
   1351                     proc.initialize, p_settings, section="components", name=name
   1352                 )
-> 1353                 proc.initialize(get_examples, nlp=self, **p_settings)
   1354         pretrain_cfg = config.get("pretraining")
   1355         if pretrain_cfg:

/usr/local/lib/python3.11/dist-packages/spacy/pipeline/transition_parser.pyx in spacy.pipeline.transition_parser.Parser.initialize()

/usr/local/lib/python3.11/dist-packages/spacy/training/example.pyx in spacy.training.example.validate_get_examples()

/tmp/ipykernel_11/1041434197.py in init_examples()
    200             texts = [t for t, _ in init_tuples]
    201             anns = [a for _, a in init_tuples]
--> 202             docs = list(nlp.pipe(texts, batch_size=64))
    203             return [Example.from_dict(doc, ann) for doc, ann in zip(docs, anns)]
    204 

/usr/local/lib/python3.11/dist-packages/spacy/language.py in pipe(self, texts, as_tuples, batch_size, disable, component_cfg, n_process)
   1620             for pipe in pipes:
   1621                 docs = pipe(docs)
-> 1622         for doc in docs:
   1623             yield doc
   1624 

/usr/local/lib/python3.11/dist-packages/spacy/util.py in _pipe(docs, proc, name, default_error_handler, kwargs)
   1712 ) -> Iterator["Doc"]:
   1713     if hasattr(proc, "pipe"):
-> 1714         yield from proc.pipe(docs, **kwargs)
   1715     else:
   1716         # We added some args for pipe that __call__ doesn't expect.

/usr/local/lib/python3.11/dist-packages/spacy/pipeline/transition_parser.pyx in pipe()

/usr/local/lib/python3.11/dist-packages/spacy/util.py in raise_error(proc_name, proc, docs, e)
   1731 
   1732 def raise_error(proc_name, proc, docs, e):
-> 1733     raise e
   1734 
   1735 

/usr/local/lib/python3.11/dist-packages/spacy/pipeline/transition_parser.pyx in spacy.pipeline.transition_parser.Parser.pipe()

/usr/local/lib/python3.11/dist-packages/spacy/pipeline/transition_parser.pyx in spacy.pipeline.transition_parser.Parser.predict()

/usr/local/lib/python3.11/dist-packages/spacy/pipeline/transition_parser.pyx in spacy.pipeline.transition_parser.Parser.greedy_parse()

/usr/local/lib/python3.11/dist-packages/thinc/model.py in predict(self, X)
    332         only the output, instead of the `(output, callback)` tuple.
    333         """
--> 334         return self._func(self, X, is_train=False)[0]
    335 
    336     def finish_update(self, optimizer: Optimizer) -> None:

/usr/local/lib/python3.11/dist-packages/spacy/ml/tb_framework.py in forward(model, X, is_train)
     31 
     32 def forward(model, X, is_train):
---> 33     step_model = ParserStepModel(
     34         X,
     35         model.layers,

/usr/local/lib/python3.11/dist-packages/spacy/ml/parser_model.pyx in spacy.ml.parser_model.ParserStepModel.__init__()

/usr/local/lib/python3.11/dist-packages/thinc/model.py in __call__(self, X, is_train)
    308         """Call the model's `forward` function, returning the output and a
    309         callback to compute the gradients via backpropagation."""
--> 310         return self._func(self, X, is_train=is_train)
    311 
    312     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":

/usr/local/lib/python3.11/dist-packages/thinc/layers/chain.py in forward(model, X, is_train)
     52     callbacks = []
     53     for layer in model.layers:
---> 54         Y, inc_layer_grad = layer(X, is_train=is_train)
     55         callbacks.append(inc_layer_grad)
     56         X = Y

/usr/local/lib/python3.11/dist-packages/thinc/model.py in __call__(self, X, is_train)
    308         """Call the model's `forward` function, returning the output and a
    309         callback to compute the gradients via backpropagation."""
--> 310         return self._func(self, X, is_train=is_train)
    311 
    312     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":

/usr/local/lib/python3.11/dist-packages/thinc/layers/chain.py in forward(model, X, is_train)
     52     callbacks = []
     53     for layer in model.layers:
---> 54         Y, inc_layer_grad = layer(X, is_train=is_train)
     55         callbacks.append(inc_layer_grad)
     56         X = Y

/usr/local/lib/python3.11/dist-packages/thinc/model.py in __call__(self, X, is_train)
    308         """Call the model's `forward` function, returning the output and a
    309         callback to compute the gradients via backpropagation."""
--> 310         return self._func(self, X, is_train=is_train)
    311 
    312     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":

/usr/local/lib/python3.11/dist-packages/thinc/layers/chain.py in forward(model, X, is_train)
     52     callbacks = []
     53     for layer in model.layers:
---> 54         Y, inc_layer_grad = layer(X, is_train=is_train)
     55         callbacks.append(inc_layer_grad)
     56         X = Y

/usr/local/lib/python3.11/dist-packages/thinc/model.py in __call__(self, X, is_train)
    308         """Call the model's `forward` function, returning the output and a
    309         callback to compute the gradients via backpropagation."""
--> 310         return self._func(self, X, is_train=is_train)
    311 
    312     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":

/usr/local/lib/python3.11/dist-packages/thinc/layers/with_array.py in forward(model, Xseq, is_train)
     34 ) -> Tuple[SeqT, Callable]:
     35     if isinstance(Xseq, Ragged):
---> 36         return cast(Tuple[SeqT, Callable], _ragged_forward(model, Xseq, is_train))
     37     elif isinstance(Xseq, Padded):
     38         return cast(Tuple[SeqT, Callable], _padded_forward(model, Xseq, is_train))

/usr/local/lib/python3.11/dist-packages/thinc/layers/with_array.py in _ragged_forward(model, Xr, is_train)
     89 ) -> Tuple[Ragged, Callable]:
     90     layer: Model[ArrayXd, ArrayXd] = model.layers[0]
---> 91     Y, get_dX = layer(Xr.dataXd, is_train)
     92 
     93     def backprop(dYr: Ragged) -> Ragged:

/usr/local/lib/python3.11/dist-packages/thinc/model.py in __call__(self, X, is_train)
    308         """Call the model's `forward` function, returning the output and a
    309         callback to compute the gradients via backpropagation."""
--> 310         return self._func(self, X, is_train=is_train)
    311 
    312     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":

/usr/local/lib/python3.11/dist-packages/thinc/layers/concatenate.py in forward(model, X, is_train)
     55 
     56 def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
---> 57     Ys, callbacks = zip(*[layer(X, is_train=is_train) for layer in model.layers])
     58     if isinstance(Ys[0], list):
     59         data_l, backprop = _list_forward(model, X, Ys, callbacks, is_train)

/usr/local/lib/python3.11/dist-packages/thinc/layers/concatenate.py in <listcomp>(.0)
     55 
     56 def forward(model: Model[InT, OutT], X: InT, is_train: bool) -> Tuple[OutT, Callable]:
---> 57     Ys, callbacks = zip(*[layer(X, is_train=is_train) for layer in model.layers])
     58     if isinstance(Ys[0], list):
     59         data_l, backprop = _list_forward(model, X, Ys, callbacks, is_train)

/usr/local/lib/python3.11/dist-packages/thinc/model.py in __call__(self, X, is_train)
    308         """Call the model's `forward` function, returning the output and a
    309         callback to compute the gradients via backpropagation."""
--> 310         return self._func(self, X, is_train=is_train)
    311 
    312     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":

/usr/local/lib/python3.11/dist-packages/thinc/layers/chain.py in forward(model, X, is_train)
     52     callbacks = []
     53     for layer in model.layers:
---> 54         Y, inc_layer_grad = layer(X, is_train=is_train)
     55         callbacks.append(inc_layer_grad)
     56         X = Y

/usr/local/lib/python3.11/dist-packages/thinc/model.py in __call__(self, X, is_train)
    308         """Call the model's `forward` function, returning the output and a
    309         callback to compute the gradients via backpropagation."""
--> 310         return self._func(self, X, is_train=is_train)
    311 
    312     def initialize(self, X: Optional[InT] = None, Y: Optional[OutT] = None) -> "Model":

/usr/local/lib/python3.11/dist-packages/thinc/layers/hashembed.py in forward(model, ids, is_train)
     60     model: Model[Ints1d, OutT], ids: Ints1d, is_train: bool
     61 ) -> Tuple[OutT, Callable]:
---> 62     vectors = cast(Floats2d, model.get_param("E"))
     63     nV = vectors.shape[0]
     64     nO = vectors.shape[1]

/usr/local/lib/python3.11/dist-packages/thinc/model.py in get_param(self, name)
    233             raise KeyError(f"Unknown param: '{name}' for model '{self.name}'.")
    234         if not self._params.has_param(self.id, name):
--> 235             raise KeyError(
    236                 f"Parameter '{name}' for model '{self.name}' has not been allocated yet."
    237             )

KeyError: "Parameter 'E' for model 'hashembed' has not been allocated yet."

## === cell 2
run_model(
    test,
    str(pos_dir),
    str(neg_dir),
    outputpath="submission.csv",
)

sub = pd.read_csv("submission.csv")
print("Wrote submission.csv; head:")
print(sub.head())
print("submission.csv shape:", sub.shape)
print("submission.csv columns:", list(sub.columns))
assert list(sub.columns) == ["textID", "selected_text"]
assert sub.shape[0] == test.shape[0]
assert str(Path("submission.csv")).endswith(".csv")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/3406245061.py in <cell line: 0>()
----> 1 run_model(
      2     test,
      3     str(pos_dir),
      4     str(neg_dir),
      5     outputpath="submission.csv",

/tmp/ipykernel_11/1041434197.py in run_model(data, positivemodel, negativemodel, outputpath)
     93     positivemodel = str(Path(positivemodel).resolve())
     94     negativemodel = str(Path(negativemodel).resolve())
---> 95     positive_nlp = spacy.load(positivemodel)
     96     negative_nlp = spacy.load(negativemodel)
     97 

/usr/local/lib/python3.11/dist-packages/spacy/__init__.py in load(name, vocab, disable, enable, exclude, config)
     50     RETURNS (Language): The loaded nlp object.
     51     """
---> 52     return util.load_model(
     53         name,
     54         vocab=vocab,

/usr/local/lib/python3.11/dist-packages/spacy/util.py in load_model(name, vocab, disable, enable, exclude, config)
    482     if name in OLD_MODEL_SHORTCUTS:
    483         raise IOError(Errors.E941.format(name=name, full=OLD_MODEL_SHORTCUTS[name]))  # type: ignore[index]
--> 484     raise IOError(Errors.E050.format(name=name))
    485 
    486 

OSError: [E050] Can't find model '/kaggle/working/models/positive'. It doesn't seem to be a Python package or a valid path to a data directory.
