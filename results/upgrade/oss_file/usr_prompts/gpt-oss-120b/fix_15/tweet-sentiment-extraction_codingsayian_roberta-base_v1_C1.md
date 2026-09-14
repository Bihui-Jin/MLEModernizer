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

datasets==4.4.1
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
transformers==4.53.3
vega-datasets==0.9.0

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

0.7191751003265381

# 6. Current score

0.50525

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61768) has done: 'I replace the failing transformer loading and downstream processing with a lightweight, rule‑based prediction that works entirely offline. The new code keeps the original data loading, defines a simple function that selects a sentiment‑related word (or the whole tweet if none is found), fills the `selected_text` column, and writes a valid `submission.csv`. This removes the HF validation errors and ensures the notebook runs end‑to‑end while producing a properly formatted submission file.'
- What this solution (achieved 0.58666) has done: 'I expand the sentiment word lists and replace the single‑word extraction with a small regex‑based phrase extraction: after finding a sentiment word, the code now returns the substring from that word up to the next punctuation (or the end of the tweet). This modest heuristic usually captures a larger, more relevant portion of the tweet and should raise the Jaccard score toward the target while keeping the original simple rule‑based structure unchanged.'
- What this solution (achieved 0.58694) has done: 'I expand the sentiment word lists slightly, add a small set of preceding modifiers (e.g., “not”, “very”), and revise `select_text` to consider **all** occurrences of sentiment words, optionally include a preceding modifier, and choose the longest resulting phrase up to the next punctuation. This keeps the rule‑based core unchanged while giving a richer extracted substring, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.5859) has done: 'I expand the sentiment word lists with more common positive and negative terms and broaden the characters that end a phrase (adding `;`, `:`, `-`, `—`). This keeps the rule‑based core unchanged but lets the extractor capture longer, more relevant substrings, which should raise the Jaccard score toward the target. The rest of the pipeline remains identical, and the script still writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.58952) has done: 'I expand the sentiment word lists slightly and improve the phrase‑boundary logic: instead of stopping only at the first punctuation after the sentiment word, the extractor now looks back to the previous punctuation (or start) and forward to the next punctuation (or end) so the whole meaningful clause is captured. This modestly widens the extracted text, which should raise the Jaccard score toward the target while keeping the overall rule‑based approach unchanged.'
- What this solution (achieved 0.58434) has done: 'I expand the word and modifier lists used by the rule‑based extractor so it can capture more sentiment‑related phrases (e.g., common contractions, informal words, and emojis). This slightly broadens the candidate substrings while keeping the original logic unchanged, which should move the Jaccard score closer to the target without altering the core pipeline.'
- What this solution (achieved 0.58488) has done: 'I modify the `select_text` function so it picks the shortest matching phrase (instead of the longest) because the gold “selected_text” is usually the minimal relevant snippet, which should raise the Jaccard score toward the target. The rest of the pipeline stays unchanged.'
- What this solution (achieved 0.58434) has done: 'I adjust the rule‑based extractor to choose the **longest** sentiment‑containing phrase (instead of the shortest) bounded by punctuation, which historically raises the Jaccard score toward the target region. The change is limited to the comparison logic inside `select_text`, keeping all other data handling and output steps unchanged, so the script still runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.58503) has done: 'I load the training data to compute the median length of the gold `selected_text` strings and use that statistic to pick, among all candidate phrases that contain a sentiment word, the one whose length is closest to the median (instead of always taking the longest). This small calibration keeps the rule‑based core unchanged while aligning the extracted snippets more closely with typical gold snippets, which should move the Jaccard score toward the target.'
- What this solution (achieved 0.53956) has done: 'I extend the sentiment word lists with the most frequent words that appear in the training `selected_text` for each sentiment, and modify the selector to favor candidate phrases that contain more sentiment words (while still staying close to the median length). This keeps the original rule‑based pipeline intact but gives it a richer vocabulary and a slightly smarter tie‑breaker, which should raise the Jaccard score toward the target without large architectural changes.'
- What this solution (achieved 0.54505) has done: 'I keep the overall rule‑based pipeline unchanged but adjust the candidate‑selection logic to favor phrases that contain more sentiment‑word hits before considering length closeness. This slight re‑ranking is expected to raise the Jaccard score toward the target while preserving the core extractor. The rest of the code, data handling and CSV output remain identical.'
- What this solution (achieved 0.5419) has done: 'I increase the number of sentiment‑word candidates extracted from the training data (using a larger `top_n` so the word lists cover more useful terms) and simplify the ranking of candidate phrases by removing the “longer‑is‑better” tie‑breaker – we now rank only by hit count and closeness to the median gold length. These small, targeted tweaks keep the overall rule‑based pipeline unchanged while giving the selector more relevant vocabulary and a more appropriate length preference, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.5347) has done: 'I expand the sentiment‑word vocab by taking **all** words that appear in the training `selected_text` for each sentiment (instead of only the top N). This gives the regex matcher many more legitimate hits, allowing the rule‑based extractor to find longer and more accurate phrases and thus raise the Jaccard score toward the target. The rest of the pipeline—including boundary detection, modifier handling, and CSV output—remains unchanged.'
- What this solution (achieved 0.50525) has done: 'I add a lightweight nearest‑neighbor lookup based on TF‑IDF vectors of the training tweets (grouped by sentiment) and combine its prediction with the existing rule‑based extractor. The nearest‑neighbor candidate is chosen when it is closer to the typical gold length, which usually improves the Jaccard score while keeping the original rule‑based logic untouched. I also renumber the cells to start from 1 as required.'

# 9. Code solution

## === cell 0
import os, re, string
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

train_path = "../input/tweet-sentiment-extraction/train.csv"
df_train = pd.read_csv(train_path)

MEDIAN_SEL_LEN = int(df_train["selected_text"].str.len().median())


def extract_top_words(df, sentiment_label, top_n=None):
    """
    Return a set of words that appear in the selected_text of the given sentiment.
    If top_n is None (default), return **all** words; otherwise return the most
    frequent `top_n` words.
    """
    texts = df.loc[df["sentiment"] == sentiment_label, "selected_text"]
    words = (
        texts.str.lower()
        .str.replace(r"[{}]".format(re.escape(string.punctuation)), " ", regex=True)
        .str.split()
        .explode()
        .value_counts()
    )
    if top_n is None:
        return set(words.index)
    return set(words.head(top_n).index)


extra_positive = extract_top_words(df_train, "positive")  # all words
extra_negative = extract_top_words(df_train, "negative")  # all words
print("Extra positive words added:", len(extra_positive))
print("Extra negative words added:", len(extra_negative))

pos_mask = df_train["sentiment"] == "positive"
neg_mask = df_train["sentiment"] == "negative"

tfidf_pos = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
tfidf_neg = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

vecs_pos = tfidf_pos.fit_transform(df_train.loc[pos_mask, "text"])
vecs_neg = tfidf_neg.fit_transform(df_train.loc[neg_mask, "text"])

selected_pos = df_train.loc[pos_mask, "selected_text"].values
selected_neg = df_train.loc[neg_mask, "selected_text"].values




## === cell 1
df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")




## === cell 2
positive_words = {
    "good",
    "great",
    "love",
    "awesome",
    "best",
    "fantastic",
    "nice",
    "perfect",
    "happy",
    "excellent",
    "wonderful",
    "amazing",
    "cool",
    "liked",
    "enjoy",
    "enjoyed",
    "pleasant",
    "delight",
    "sweet",
    "brilliant",
    "like",
    "better",
    "well",
    "fine",
    "positive",
    "superb",
    "glad",
    "joyful",
    "delighted",
    "satisfied",
    "thrilled",
    "fabulous",
    "marvelous",
    "okay",
    "ok",
    "pleased",
    "content",
    "charming",
    "lovely",
    "lol",
    "haha",
    "yay",
    "yes",
    "😊",
    "😍",
    "😁",
} | extra_positive  # extended with all training‑derived words

negative_words = {
    "bad",
    "hate",
    "worst",
    "terrible",
    "awful",
    "sad",
    "poor",
    "disappointed",
    "angry",
    "negative",
    "horrible",
    "dislike",
    "sucks",
    "unhappy",
    "lousy",
    "dreadful",
    "painful",
    "ugly",
    "annoyed",
    "disgusting",
    "regret",
    "bored",
    "annoying",
    "depressing",
    "miserable",
    "unpleasant",
    "cry",
    "upset",
    "horrid",
    "gross",
    "filthy",
    "nasty",
    "despair",
    "distressed",
    "ugh",
    "no",
    "😞",
    "😡",
    "😭",
    "💔",
} | extra_negative  # extended with all training‑derived words

preceding_modifiers = {
    "not",
    "no",
    "never",
    "n't",
    "very",
    "so",
    "really",
    "quite",
    "much",
    "still",
    "hardly",
    "barely",
    "don't",
    "doesn't",
    "didn't",
    "can't",
    "couldn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "haven't",
    "hasn't",
    "hadn't",
    "mustn't",
    "needn't",
    "mightn't",
    "mayn't",
}

punctuation_chars = set(string.punctuation) | {"—"}  # include em‑dash


def find_boundary(text, pos, direction):
    """
    Return the index of the nearest punctuation (or string end) in the given
    direction from position `pos`. `direction` should be -1 (search backwards)
    or +1 (search forwards). The returned index is the first character after
    the punctuation when searching backwards, and the position of the punctuation
    when searching forwards (or the end of the string).
    """
    if direction == -1:
        for i in range(pos - 1, -1, -1):
            if text[i] in punctuation_chars:
                return i + 1
        return 0
    else:
        for i in range(pos, len(text)):
            if text[i] in punctuation_chars:
                return i
        return len(text)


def count_sentiment_hits(phrase, target_set):
    """Count how many times words from target_set appear in the phrase."""
    tokens = re.findall(r"\b\w+\b", phrase.lower())
    return sum(tok in target_set for tok in tokens)


def get_nn_selected(text, sentiment):
    """Return the selected_text of the most TF‑IDF‑similar training tweet."""
    if sentiment == "positive":
        vec = tfidf_pos.transform([text])
        sims = cosine_similarity(vec, vecs_pos).flatten()
        best_idx = int(sims.argmax())
        return selected_pos[best_idx]
    elif sentiment == "negative":
        vec = tfidf_neg.transform([text])
        sims = cosine_similarity(vec, vecs_neg).flatten()
        best_idx = int(sims.argmax())
        return selected_neg[best_idx]
    else:
        return text


def select_text(row):
    """
    Combine rule‑based extraction with a TF‑IDF nearest‑neighbor fallback.
    The phrase whose length is closer to the median gold length is returned.
    """
    text = row["text"]
    sentiment = row["sentiment"]
    target_set = (
        positive_words
        if sentiment == "positive"
        else negative_words if sentiment == "negative" else None
    )
    if target_set is None:
        return text

    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, target_set)) + r")\b", re.IGNORECASE
    )
    candidates = []  # (phrase, length, hit_count)

    for match in pattern.finditer(text):
        start_idx = find_boundary(text, match.start(), -1)

        prefix = text[start_idx : match.start()].rstrip()
        if prefix:
            last_token = prefix.split()[-1]
            if last_token.lower() in preceding_modifiers:
                mod_start = text.rfind(last_token, start_idx, match.start())
                if mod_start != -1:
                    start_idx = mod_start

        end_idx = find_boundary(text, match.end(), 1)

        phrase = text[start_idx:end_idx].strip()
        if phrase:
            hit_cnt = count_sentiment_hits(phrase, target_set)
            candidates.append((phrase, len(phrase), hit_cnt))

    if candidates:
        rule_phrase = max(
            candidates, key=lambda x: (x[2], -abs(x[1] - MEDIAN_SEL_LEN))
        )[0]
    else:
        rule_phrase = text

    nn_phrase = get_nn_selected(text, sentiment)

    chosen = min([rule_phrase, nn_phrase], key=lambda p: abs(len(p) - MEDIAN_SEL_LEN))
    return chosen


sub_df["selected_text"] = df_test.apply(select_text, axis=1)




## === cell 3
sub_df.to_csv("submission.csv", index=False)
print("submission.csv written successfully, shape:", sub_df.shape)
