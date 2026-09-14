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

fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tokenizers==0.21.2
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3

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

0.7094413042068481

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61626) has done: 'I fixed the import errors, removed the unusable fastai components, and replaced the whole training pipeline with a lightweight rule‑based predictor that runs end‑to‑end. The script now loads the data, applies a simple sentiment‑aware heuristic to choose a span, and writes a correctly formatted `submission.csv` file.'
- What this solution (achieved 0.59995) has done: 'I keep the overall structure but improve the span‑extraction heuristic: instead of returning just the first sentiment word, the new rule expands to a surrounding phrase up to punctuation marks. This modest change keeps the original logic while providing longer, more context‑rich predictions, which should raise the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
from pathlib import Path



## === cell 1
base_dir = Path("/kaggle/input/tweet-sentiment-extraction")

train_df = pd.read_csv(base_dir / "train.csv")
test_df = pd.read_csv(base_dir / "test.csv")

train_df["selected_len"] = train_df["selected_text"].str.split().apply(len)
avg_len_pos = train_df.loc[train_df["sentiment"] == "positive", "selected_len"].mean()
avg_len_neg = train_df.loc[train_df["sentiment"] == "negative", "selected_len"].mean()
AVG_LEN = {"positive": avg_len_pos, "negative": avg_len_neg}



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1956058215.py in <cell line: 0>()
      5 
      6 # compute average token length of the gold selected_text for each sentiment
----> 7 train_df["selected_len"] = train_df["selected_text"].str.split().apply(len)
      8 avg_len_pos = train_df.loc[train_df["sentiment"] == "positive", "selected_len"].mean()
      9 avg_len_neg = train_df.loc[train_df["sentiment"] == "negative", "selected_len"].mean()

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

TypeError: object of type 'float' has no len()

## === cell 2
positive_words = {
    "good",
    "great",
    "love",
    "nice",
    "excellent",
    "happy",
    "awesome",
    "fantastic",
    "amazing",
    "wonderful",
}
negative_words = {
    "bad",
    "hate",
    "terrible",
    "worst",
    "sad",
    "awful",
    "poor",
    "horrible",
    "angry",
    "disappointed",
}


def _expand_to_target_len(tokens, left, right, target_len):
    """Grow the window [left, right] while keeping it within punctuation
    boundaries until it reaches roughly target_len tokens."""
    target_len = int(round(target_len))
    while (right - left + 1) < target_len:
        expanded = False
        if left > 0 and not any(p in tokens[left - 1] for p in ".,!?"):
            left -= 1
            expanded = True
        if (right - left + 1) >= target_len:
            break
        if right + 1 < len(tokens) and not any(p in tokens[right + 1] for p in ".,!?"):
            right += 1
            expanded = True
        if not expanded:
            break
    return " ".join(tokens[left : right + 1])


def _extract_phrase(text, target_set, sentiment):
    """Return a phrase containing the first matching sentiment word,
    expanded to an approximate average gold span length for the given sentiment."""
    tokens = text.split()
    for i, tok in enumerate(tokens):
        cleaned = tok.lower().strip(".,!?\"'")
        if cleaned in target_set:
            l = i
            while l > 0 and not any(p in tokens[l - 1] for p in ".,!?"):
                l -= 1
            r = i
            while r + 1 < len(tokens) and not any(p in tokens[r + 1] for p in ".,!?"):
                r += 1
            target_len = AVG_LEN.get(sentiment, len(tokens))
            return _expand_to_target_len(tokens, l, r, target_len)
    return text


def predict_span(row):
    text = row["text"]
    sentiment = row["sentiment"].lower()

    if sentiment == "neutral" or len(text) < 3:
        return text

    if sentiment == "positive":
        return _extract_phrase(text, positive_words, "positive")
    elif sentiment == "negative":
        return _extract_phrase(text, negative_words, "negative")
    else:
        return text




## === cell 3
test_df["selected_text"] = test_df.apply(predict_span, axis=1)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1588852001.py in <cell line: 0>()
----> 1 test_df["selected_text"] = test_df.apply(predict_span, axis=1)
      2 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_55/3714578434.py in predict_span(row)
     78 
     79     if sentiment == "positive":
---> 80         return _extract_phrase(text, positive_words, "positive")
     81     elif sentiment == "negative":
     82         return _extract_phrase(text, negative_words, "negative")

/tmp/ipykernel_55/3714578434.py in _extract_phrase(text, target_set, sentiment)
     64                 r += 1
     65             # expand toward average length for this sentiment
---> 66             target_len = AVG_LEN.get(sentiment, len(tokens))
     67             return _expand_to_target_len(tokens, l, r, target_len)
     68     # fallback: return whole text if no sentiment word found

NameError: name 'AVG_LEN' is not defined

## === cell 4
submission = test_df[["textID", "selected_text"]].copy()
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3304247509.py in <cell line: 0>()
----> 1 submission = test_df[["textID", "selected_text"]].copy()
      2 submission.to_csv("submission.csv", index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['selected_text'] not in index"
