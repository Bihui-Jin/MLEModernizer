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

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1
wordcloud==1.9.4

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

0.5741774439811707

# 6. Current score

0.24191

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23796) has done: 'I tighten the prediction heuristics to better match the evaluation metric.  
- Increase the pool of frequent selected_text candidates to 500 per sentiment.  
- Match candidates case‑insensitively against the original tweet (without aggressive cleaning) so punctuation is respected.  
- For neutral sentiment, default to the whole tweet, which aligns with many neutral examples.  
These modest tweaks keep the overall pipeline unchanged while aiming for a higher Jaccard score closer to the target.'
- What this solution (achieved 0.2367) has done: 'I added the missing spacy import, expanded the pool of frequent selected_text candidates from 500 to 2000, and added a safe fallback that returns the whole tweet when no candidate is found. These changes resolve the NameError, give the heuristic more material to work with, and should lift the Jaccard score toward the target while keeping the original pipeline intact.'
- What this solution (achieved 0.25078) has done: 'I increase the candidate pool and replace the cleaned‑text Jaccard with a token‑based Jaccard that keeps punctuation, then always pick the highest‑similarity candidate when an exact match is not found. This small change should raise the Jaccard score toward the target while keeping the original pipeline intact.'
- What this solution (achieved 0.27175) has done: 'I keep the overall pipeline unchanged but improve the prediction logic: increase the candidate pool to 10 000, collect all candidates that appear as substrings in the tweet, and choose the one with the highest token‑Jaccard similarity (instead of the first match). This modest change should raise the Jaccard score toward the target while preserving the original model‑free approach.'
- What this solution (achieved 0.24191) has done: 'I keep the overall pipeline unchanged but improve the similarity computation used to pick candidate substrings. By stripping punctuation before tokenising, the token‑Jaccard similarity becomes more tolerant of punctuation differences between the candidate and the tweet, which should raise the Jaccard score toward the target. The change is limited to the similarity helper and its use in cell 41.'

# 9. Code solution

## === cell 0
import os, re, string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import spacy  # added import for spacy used later

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df_train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")



## === cell 2
df_train.shape



## === cell 3
df_train.info



## === cell 4
df_train.head()



## === cell 5
df_train["sentiment"].value_counts()



## === cell 6
df_train["sentiment"].value_counts().plot.bar()



## === cell 7
df_train.isna().sum()



## === cell 8
df_train.dropna(inplace=True)



## === cell 9
df_train.isna().sum()



## === cell 10
df_train["Num_of_words_text"] = df_train["text"].apply(lambda x: len(str(x).split()))
df_train["Num_of_words_ST"] = df_train["selected_text"].apply(
    lambda x: len(str(x).split())
)
df_train["Difference"] = df_train["Num_of_words_text"] - df_train["Num_of_words_ST"]
df_train.head()




## === cell 11
def jaccard_similarity(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c) / (len(a) + len(b) - len(c)))




## === cell 12
jaccard_sim = []
for index, rows in df_train.iterrows():
    st1 = rows.text
    st2 = rows.selected_text
    jaccard_sim.append([st1, st2, jaccard_similarity(st1, st2)])

df_jaccard = pd.DataFrame(
    jaccard_sim, columns=["text", "selected_text", "jaccard_similarity"]
)
df_train = df_train.merge(df_jaccard, how="left")
df_train.head()



## === cell 13
plt.figure(figsize=(16, 6))
p1 = sns.kdeplot(
    df_train[df_train["sentiment"] == "positive"]["Difference"], shade=True, color="y"
).set_title("Kernel Distribution of Difference in Number of Words(Pos/Neg)")
p2 = sns.kdeplot(
    df_train[df_train["sentiment"] == "negative"]["Difference"], shade=True, color="c"
)



## === cell 14
plt.figure(figsize=(16, 6))
p3 = sns.kdeplot(
    df_train[df_train["sentiment"] == "neutral"]["Difference"], shade=True, color="r"
).set_title("Kernel Distribution of Difference in Number of Words(Neutral)")



## === cell 15
plt.figure(figsize=(16, 6))
p1 = sns.kdeplot(
    df_train[df_train["sentiment"] == "positive"]["jaccard_similarity"],
    shade=True,
    color="y",
).set_title("Kernel Distribution of Jaccard Similarity(Pos/Neg)")
p2 = sns.kdeplot(
    df_train[df_train["sentiment"] == "negative"]["jaccard_similarity"],
    shade=True,
    color="g",
)



## === cell 16
plt.figure(figsize=(16, 6))
p1 = sns.kdeplot(
    df_train[df_train["sentiment"] == "neutral"]["jaccard_similarity"],
    shade=True,
    color="r",
).set_title("Kernel Distribution of Jaccard Similarity(Neutral)")




## === cell 17
def clean_text(text):
    """Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers."""
    text = str(text).lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("http?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text




## === cell 18
df_train["text"] = df_train["text"].apply(lambda x: clean_text(x))
df_train["selected_text"] = df_train["selected_text"].apply(lambda x: clean_text(x))
df_train.head(10)



## === cell 19
nltk.download("stopwords", quiet=True)

df_train["st_list"] = df_train["selected_text"].apply(lambda x: str(x).split())
df_train["text_list"] = df_train["text"].apply(lambda x: str(x).split())


def remove_stopwords(x):
    return [y for y in x if y not in stopwords.words("english")]


df_train["st_list"] = df_train["st_list"].apply(lambda x: remove_stopwords(x))
df_train["text_list"] = df_train["text_list"].apply(lambda x: remove_stopwords(x))
df_train.head()



## === cell 20
"""most common words in positive sentiment selected text"""
top = Counter(
    [
        item
        for sublist in df_train[df_train["sentiment"] == "positive"]["st_list"]
        for item in sublist
    ]
)
top_pos = pd.DataFrame(top.most_common(20), columns=["Common Words", "Count"])
top_pos.style.background_gradient(cmap="Greens")



## === cell 21
"""most common words in negative sentiment selected text"""
top = Counter(
    [
        item
        for sublist in df_train[df_train["sentiment"] == "negative"]["st_list"]
        for item in sublist
    ]
)
top_neg = pd.DataFrame(top.most_common(20), columns=["Common Words", "Count"])
top_neg.style.background_gradient(cmap="Oranges")



## === cell 22
"""most common words in neutral sentiment selected text"""
top = Counter(
    [
        item
        for sublist in df_train[df_train["sentiment"] == "neutral"]["st_list"]
        for item in sublist
    ]
)
top_neu = pd.DataFrame(top.most_common(20), columns=["Common Words", "Count"])
top_neu.style.background_gradient(cmap="Blues")




## === cell 23
def unique_words(sentiment, num):
    all_other = []
    for sublist in df_train[df_train["sentiment"] != sentiment]["st_list"]:
        for word in sublist:
            all_other.append(word)
    unique = Counter(
        [
            word
            for sublist in df_train[df_train["sentiment"] == sentiment]["st_list"]
            for word in sublist
            if word not in all_other
        ]
    )
    return pd.DataFrame(unique.most_common(num), columns=["Words", "Count"])




## === cell 24
unique_pos = unique_words("positive", 20)
print("20 unique postive words:")
unique_pos.style.background_gradient(cmap="Greens")



## === cell 25
unique_neg = unique_words("negative", 20)
print("20 unique negative words:")
unique_neg.style.background_gradient(cmap="Oranges")



## === cell 26
unique_neu = unique_words("neutral", 20)
print("20 unique neutral words:")
unique_neu.style.background_gradient(cmap="Blues")




## === cell 27
def create_wordcloud(text):
    stopwords = set(STOPWORDS)
    more_stopwords = {"u", "im"}
    stopwords = stopwords.union(more_stopwords)
    wordcloud = WordCloud(
        background_color="white", stopwords=stopwords, max_words=50, max_font_size=40
    )
    wordcloud.generate(str(text))
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")




## === cell 28
create_wordcloud(df_train[df_train["sentiment"] == "positive"]["text"])



## === cell 29
create_wordcloud(df_train[df_train["sentiment"] == "negative"]["text"])



## === cell 30
create_wordcloud(df_train[df_train["sentiment"] == "neutral"]["text"])



## === cell 31
df_train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_train = df_train.dropna()



## === cell 32
"""return train data in a format needed for spacy NER"""


def get_training_data(sentiment):
    train_data = []
    for index, row in df_train.iterrows():
        text = row.text
        selected_text = row.selected_text
        start = text.find(selected_text)
        end = start + len(selected_text)
        train_data.append((text, {"entities": [[start, end, "selected_text"]]}))
    return train_data




## === cell 33
"""return model output path"""


def get_model_out_path(sentiment):
    model_out_path = None
    if sentiment == "positive":
        model_out_path = "models/model_pos"
    elif sentiment == "negative":
        model_out_path = "models/model_neg"
    elif sentiment == "neutral":
        model_out_path = "models/model_neu"
    return model_out_path




## === cell 34
def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans."""
    invalid_span_tokens = re.compile(r"\s")

    cleaned_data = []
    for text, annotations in data:
        entities = annotations["entities"]
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                text[valid_start]
            ):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {"entities": valid_entities}])
    return cleaned_data




## === cell 35
def train(train_data, output_dir, n_iter=20, model=None):
    print("Training step skipped (baseline model does not require NER).")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        nlp = spacy.blank("en")
        nlp.to_disk(output_dir)
        print("Saved placeholder model to", output_dir)




## === cell 36
def save_model(output_dir, nlp, new_model_name):
    """This Function Saves model to given output directory"""
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = new_model_name
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)




## === cell 37
sentiment = "positive"
train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train(train_data, model_path, n_iter=0, model=None)



## === cell 38
sentiment = "negative"
train_data = get_training_data(sentiment)
model_path = get_model_out_path(sentiment)
train(train_data, model_path, n_iter=0, model=None)




## === cell 39
def predict_entities(text, model):
    """Return the selected text predicted by the NER model.
    Falls back to the whole text if no entity is detected."""
    doc = model(text)
    selected_text = ""
    for ent in doc.ents:
        selected_text = ent.text
        break
    if not selected_text:
        selected_text = text
    return selected_text




## === cell 40
positive_words = set(top_pos["Common Words"].astype(str).str.lower())
negative_words = set(top_neg["Common Words"].astype(str).str.lower())




## === cell 41
def token_jaccard_clean(str1, str2):
    translator = str.maketrans("", "", string.punctuation)
    clean1 = str1.lower().translate(translator)
    clean2 = str2.lower().translate(translator)
    set1 = set(clean1.split())
    set2 = set(clean2.split())
    if not set1 and not set2:
        return 1.0
    inter = set1 & set2
    return len(inter) / (len(set1) + len(set2) - len(inter))


max_candidates = 10000
top_selected = {}
for sentiment in ["positive", "negative", "neutral"]:
    freq_series = df_train[df_train["sentiment"] == sentiment][
        "selected_text"
    ].value_counts()
    top_selected[sentiment] = [
        c.strip() for c in list(freq_series.index)[:max_candidates]
    ]

selected_texts = []
for idx, row in df_test.iterrows():
    text = row["text"]
    sentiment = row["sentiment"]
    norm_text_lower = text.lower()

    matching_candidates = [
        cand
        for cand in top_selected.get(sentiment, [])
        if cand.lower() in norm_text_lower
    ]

    if matching_candidates:
        best_cand = max(matching_candidates, key=lambda c: token_jaccard_clean(c, text))
        chosen = best_cand
    else:
        best_sim = 0.0
        best_cand = None
        for cand in top_selected.get(sentiment, []):
            sim = token_jaccard_clean(cand, text)
            if sim > best_sim:
                best_sim = sim
                best_cand = cand
        if best_sim > 0.0:
            chosen = best_cand
        else:
            chosen = None
            if sentiment == "positive":
                for w in positive_words:
                    if re.search(r"\b" + re.escape(w) + r"\b", norm_text_lower):
                        chosen = w
                        break
            elif sentiment == "negative":
                for w in negative_words:
                    if re.search(r"\b" + re.escape(w) + r"\b", norm_text_lower):
                        chosen = w
                        break

            if chosen is None:
                chosen = text

    selected_texts.append(chosen)

df_test["selected_text"] = selected_texts



## === cell 42
df_submission = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)



## === cell 43
df_submission.shape



## === cell 44
df_test.shape



## === cell 45
df_test.head()



## === cell 46
df_submission["selected_text"] = df_test["selected_text"]
df_submission.to_csv("submission.csv", index=False)
display(df_submission.head(10))
