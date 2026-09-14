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

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scipy==1.15.3
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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

0.717913031578064

# 6. Current score

0.58988

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59357) has done: 'I replace the failing transformer‑based loading and model inference with a lightweight, rule‑based fallback that extracts a short phrase containing the sentiment word (or the whole tweet if not found). This removes the protobuf import error, avoids missing checkpoint files, and guarantees a valid `submission.csv` is written while keeping the overall pipeline structure.'
- What this solution (achieved 0.60166) has done: 'I replace the very simple token‑match heuristic with a slightly richer rule‑based extractor:  
- Define small positive / negative word lists.  
- Locate the first word from the appropriate list that appears in the tweet (case‑insensitive).  
- Return a short phrase surrounding that word (two tokens before and after) so the selected text is more likely to match the true excerpt.  
- If no cue word is found (or the sentiment is neutral) fall back to the whole tweet.  
This change keeps the overall pipeline unchanged while providing a more informed prediction, which should move the Jaccard score upward toward the target.'
- What this solution (achieved 0.55298) has done: 'I add a lightweight training‑data‑driven cue list and expand the extracted phrase to the surrounding punctuation instead of a fixed two‑token window. This keeps the rule‑based approach while giving the heuristic more relevant words and a more natural phrase length, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.55906) has done: 'I added a richer extraction routine that looks for every cue‑word occurrence, expands the match to nearby punctuation **or** up to a limited number of surrounding tokens (whichever yields a shorter, more natural excerpt).  This keeps the original rule‑based pipeline but gives the selector more flexibility, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.55298) has done: 'I enlarge the token window used when expanding a cue‑word match (from 5 to 10 tokens) and simplify the heuristic to return the first expanded snippet rather than the shortest among all candidates. This modest change should capture more of the true sentiment phrase, moving the Jaccard score upward toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.55298) has done: 'I slightly tighten the heuristic by reducing the token window from 10 to 5 when expanding a cue‑word match. This keeps the overall rule‑based approach unchanged while producing a shorter, more precise excerpt that should raise the Jaccard score toward the target. I also ensure the extracted snippet is stripped of surrounding whitespace before creating the submission.'
- What this solution (achieved 0.59173) has done: 'I make the heuristic a bit more generous:  
- `_expand_to_window` now returns the **longer** of the punctuation‑based or fixed‑window expansion (so the snippet can cover more of the true phrase).  
- `_extract_selected_text` scans **all** cue‑word positions and picks the longest expanded snippet instead of just the first match.  
These small changes keep the overall rule‑based pipeline intact but should capture larger, more accurate excerpts, moving the Jaccard score upward toward the target.'
- What this solution (achieved 0.59135) has done: 'I add a lightweight sentence‑transformer embedder and use cosine similarity between each cue‑based candidate snippet and the full tweet (or the sentiment word) to pick the most representative excerpt. This keeps the original rule‑based pipeline but refines the final choice, which should raise the Jaccard score toward the target.'
- What this solution (achieved 0.58988) has done: 'Implemented missing imports, fixed undefined names, and adjusted the heuristic to choose the longest candidate when no embedder is available (helps improve Jaccard score). The script now runs end‑to‑end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import string
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class CFG:
    DEBUG = False
    TRAIN = True
    N_FOLDS = 5
    TRAIN_FOLDS = [i for i in range(N_FOLDS)]
    SEED = 42
    TEST_BATCHSIZE = 100
    MAX_LENGTH = 128
    MODEL_NAME = "microsoft/deberta-v3-base"
    FC_DROPOUT = [0.1, 0.2, 0.3, 0.4, 0.5]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(CFG.SEED)




## === cell 1
test_path = "/kaggle/input/tweet-sentiment-extraction/test.csv"
test_df = pd.read_csv(test_path)

train_path = "/kaggle/input/tweet-sentiment-extraction/train.csv"
train_df = pd.read_csv(train_path)


def _tokens_from_text(txt):
    return [t.strip(string.punctuation).lower() for t in str(txt).split() if t]


pos_cues_train = set()
neg_cues_train = set()
for _, row in train_df.iterrows():
    sentiment = str(row["sentiment"]).lower()
    toks = _tokens_from_text(row["selected_text"])
    if sentiment == "positive":
        pos_cues_train.update(toks)
    elif sentiment == "negative":
        neg_cues_train.update(toks)

pos_cues_manual = {
    "good",
    "great",
    "awesome",
    "happy",
    "love",
    "excellent",
    "nice",
    "best",
    "amazing",
    "fantastic",
    "wonderful",
    "delight",
    "joy",
    "pleased",
    "awesome",
}
neg_cues_manual = {
    "bad",
    "terrible",
    "sad",
    "hate",
    "worst",
    "awful",
    "poor",
    "disappointed",
    "angry",
    "horrible",
    "painful",
    "sadness",
    "annoyed",
    "disgust",
    "unsatisfied",
}

POS_CUES = pos_cues_manual.union(pos_cues_train)
NEG_CUES = neg_cues_manual.union(neg_cues_train)

EMBEDDER_AVAILABLE = False
embedder = None
try:
    from sentence_transformers import SentenceTransformer, util

    embedder = SentenceTransformer(
        "paraphrase-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    EMBEDDER_AVAILABLE = True
except Exception as e:
    print(f"SentenceTransformer loading failed ({e}); falling back to heuristic only.")
    EMBEDDER_AVAILABLE = False
    embedder = None




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def _expand_to_punct(tokens, idx):
    """
    Expand left/right from idx until a token ends with a punctuation mark
    (.,!?) or the list boundaries are reached.
    """
    left = idx
    while left > 0 and not tokens[left - 1].endswith((".", ",", "!", "?", ";", ":")):
        left -= 1
    right = idx
    while right + 1 < len(tokens) and not tokens[right + 1].endswith(
        (".", ",", "!", "?", ";", ":")
    ):
        right += 1
    return " ".join(tokens[left : right + 1])


def _expand_to_window(tokens, idx, window=3):
    """
    Expand a fixed number of tokens on each side (up to `window`) while
    respecting punctuation boundaries. Return the **longer** of the
    punctuation‑based or window‑based expansion so the snippet can
    include more of the true selected text.
    """
    punct_exp = _expand_to_punct(tokens, idx)

    left = max(0, idx - window)
    right = min(len(tokens) - 1, idx + window)

    while left > 0 and not tokens[left - 1].endswith((".", ",", "!", "?", ";", ":")):
        left -= 1
    while right + 1 < len(tokens) and not tokens[right + 1].endswith(
        (".", ",", "!", "?", ";", ":")
    ):
        right += 1

    window_exp = " ".join(tokens[left : right + 1])

    return window_exp if len(window_exp) > len(punct_exp) else punct_exp


def _extract_selected_text(text: str, sentiment: str) -> str:
    """
    Heuristic extractor enhanced with optional semantic similarity:
    - Find all cue‑word positions for the given sentiment.
    - For each position compute an expanded snippet (window=3, longer of two strategies).
    - If the sentence‑transformer embedder is available, rank candidates by cosine
      similarity to the full tweet (the most representative snippet).
    - When the embedder is unavailable, choose the **longest** candidate
      (empirically better for this task).
    - Return the top‑ranked snippet; if no cue is found, fall back to the whole tweet.
    """
    tokens = text.split()
    lowered = [t.strip(string.punctuation).lower() for t in tokens]

    cue_set = (
        POS_CUES
        if sentiment.lower() == "positive"
        else NEG_CUES if sentiment.lower() == "negative" else set()
    )

    candidates = []
    if cue_set:
        for i, w in enumerate(lowered):
            if w in cue_set:
                snippet = _expand_to_window(tokens, i, window=3).strip()
                candidates.append(snippet)

    if not candidates:
        return text.strip()

    if EMBEDDER_AVAILABLE and embedder is not None:
        try:
            cand_emb = embedder.encode(
                candidates, convert_to_tensor=True, show_progress_bar=False
            )
            txt_emb = embedder.encode(
                [text], convert_to_tensor=True, show_progress_bar=False
            )
            sims = util.cos_sim(cand_emb, txt_emb).cpu().numpy().flatten()
            best_idx = int(sims.argmax())
            return candidates[best_idx]
        except Exception as e:
            print(f"Embedding fallback ({e}); using longest heuristic.")
            return max(candidates, key=lambda s: len(s))
    else:
        return max(candidates, key=lambda s: len(s))




## === cell 3
final_outputs = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    selected = _extract_selected_text(row["text"], row["sentiment"])
    final_outputs.append(selected)




## === cell 4
submission = pd.DataFrame({"textID": test_df["textID"], "selected_text": final_outputs})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape: {submission.shape}")
