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

0.49288

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.25096) has done: 'I fix the immediate import/runtime crash by pinning protobuf to the compatible pure-Python implementation and deferring `transformers` imports until after that environment fix. Then I remove the broken dependencies on missing Kaggle Dataset paths (`/kaggle/input/k/...`) by loading the tokenizer/model directly from `CFG.MODEL_NAME`, which is available in the Kaggle environment, so inference can run end-to-end. Finally, I correct a few inference-time logic/shape issues (softmax over the correct dimension, safe truncation, and consistent mask handling) so `final_outputs` is produced with the right length and a valid `submission.csv` is written.'
- What this solution (achieved 0.25096) has done: 'I fix the runtime crash coming from an incompatible protobuf implementation used by `transformers` by forcing the pure-Python protobuf backend *before* any transformers-related import and by guarding the import order. Then I correct two inference-time logic bugs that strongly hurt Jaccard: applying softmax over the wrong dimension (should be over sequence length, not batch) and using a proper tweet+sentiment separator (tokenizer’s `sep_token`) instead of the literal string `"[SEP]"`. These changes preserve the core model/inference approach (same backbone, same start/end argmax decoding), but should move the score substantially upward toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.25096) has done: 'I fix the `transformers` import/runtime crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf backend *and* disabling C++ protobuf before any `transformers` import, which is the typical cause in recent Python 3.11 Kaggle images. I also switch the model loading to use the actual fine-tuned QA head weights if they exist locally (common in Kaggle notebooks via a `pytorch_model.bin`/`model.safetensors`), otherwise fall back to the current base model so the notebook still runs end-to-end. Finally, I keep your decoding logic intact but ensure softmax is applied after masking and that the model is put in eval mode with deterministic settings preserved, producing a valid `submission.csv`.'
- What this solution (achieved 0.25096) has done: 'I fix the protobuf/transformers runtime crash by forcing the pure-Python protobuf backend *before* any `transformers` import and by avoiding the problematic `MessageFactory.GetPrototype` path via a safe fallback if needed. Then I ensure the model and tokenizer load correctly in the Kaggle environment and that inference runs end-to-end to produce `submission.csv`. Finally, I keep your start/end argmax decoding logic intact but make a minimal correction to use the tokenizer’s special tokens consistently when filtering output tokens (preventing accidental removal mismatches that severely hurt Jaccard). These changes are directly aimed at unblocking execution and improving the score from the random-head baseline toward the target.'
- What this solution (achieved 0.25096) has done: 'I fix the protobuf/transformers crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf backend early and also downgrading `protobuf` to a compatible 3.20.x inside the notebook before importing `transformers`, which is the most reliable fix on Kaggle Python 3.11 images. Then I ensure we don’t accidentally hit DataLoader multiprocessing/import edge cases by setting `num_workers=0` (score-neutral, stability-only). Finally, I keep your model/inference/decoding logic intact so the output format stays correct and a valid `submission.csv` is always written.'
- What this solution (achieved 0.50507) has done: 'Your current score is low because the QA head is effectively random unless a fine-tuned checkpoint is found, and your decoding is based on token strings rather than using the tokenizer’s offset mapping to extract an exact substring from the original tweet (which is crucial for word-level Jaccard). To move toward the target with minimal core-logic changes, I (1) load a known-good fine-tuned model if it exists in the local Kaggle dataset folder (common in this competition), and (2) keep the same start/end argmax approach but convert predicted token spans back to text using `offset_mapping` so punctuation/spacing match the original tweet. I also add the standard “neutral => full text” fallback that is typical for this task and improves Jaccard without changing the modeling approach. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.4859) has done: 'Your current gap to the target is large (0.50507 → 0.7179), and the biggest limiting factor is that you’re using a base DeBERTa backbone with an essentially random QA head, so span predictions are weak even with offset decoding. With minimal changes that preserve your inference-only QA span approach, I (1) properly use the tokenizer’s pair encoding so offsets map to the tweet text (instead of concatenating with `[SEP]` into one string), (2) restrict start/end selection to only tokens belonging to the tweet segment (exclude sentiment and special tokens), and (3) add the standard “positive/negative: if extraction looks bad, fall back to best matching contiguous word-span by Jaccard against the tweet” heuristic that improves word-level Jaccard without changing the model or training. These changes are directly aimed at improving span-to-text fidelity and avoiding selecting sentiment/special-token regions, which should move the score upward toward your target while still producing a valid `submission.csv`. No training is added and all paths/output format remain unchanged.'
- What this solution (achieved 0.47714) has done: 'Your current score (0.4859) is far below the target (0.7179), and the biggest low-risk gain without changing your model/training is to decode the predicted span correctly and to choose a coherent (start,end) pair. I (1) fix the segment masking so we only consider tokens that actually belong to the tweet text (for DeBERTa token_type_ids are often all zeros, so your current “ttype==0” mask mistakenly includes sentiment/special tokens), (2) replace independent argmax start/end with a minimal constrained joint selection that maximizes start_prob[i]*end_prob[j] with j>=i (same logits, same QA decoding idea, just a correct pairing), and (3) use `offset_mapping` against the original tweet (not `text_clean`) to preserve exact punctuation/spacing for Jaccard. These are inference-only, keep your architecture intact, and are directly targeted at increasing Jaccard toward the target.'
- What this solution (achieved 0.47643) has done: 'Your current score is far below target, and the biggest low-risk gain without changing your model/training is to ensure the start/end span selection and decoding only consider tokens that truly map to the tweet text (not the sentiment segment or special tokens). I keep your same DeBERTa backbone + QA head inference and the same offset-based substring decoding, but I (1) compute a “tweet-only” validity mask using `offset_mapping` plus special-token filtering, (2) prevent selecting offsets that point to whitespace-only spans by tightening the valid-offset criteria, and (3) add a minimal length-1 fallback for non-neutral when the chosen span is empty. These are inference-time masking/decoding fixes (not architecture or training changes) and are directly aimed at improving word-level Jaccard toward your target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.47588) has done: 'Your gap to the target is large (0.476 → 0.718), so we need a meaningful but still inference-only improvement that preserves the core QA-span approach. The biggest current issue is that your “valid token” mask still allows selecting tokens from the *sentiment* segment (pair encoding), because offset mappings for the second sequence can be non-zero and pass your filters. I minimally tighten the mask to only allow offsets that fall within the tweet text length **and** are not special tokens, then (still using the same logits) select the best (start,end) pair on that constrained region. This should move the Jaccard upward without changing your model architecture, loss, or training approach, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.47809) has done: 'Your current score is far below the target, so we should make a small but meaningful inference-time fix that improves span localization without changing the model or adding training. The biggest controllable issue here is that you tokenize a whitespace-normalized `text_clean` but then decode offsets against the original `orig_text`, which makes offset mappings misaligned and hurts Jaccard; we tokenize the original text (keep it as-is) so offsets map correctly. Then we add a minimal “strip outer punctuation” post-process for non-neutral predictions to better match typical ground-truth spans while keeping the same start/end decoding logic. These changes are inference-only, preserve the core QA-span approach, and should move the score upward toward your target.'
- What this solution (achieved 0.49256) has done: 'Your current score (0.478) is far below the target (0.718), so we should make a small but meaningful inference-time fix that improves span localization while keeping your exact DeBERTa QA-span approach. The biggest bug hurting you is that your “within_tweet” mask still allows tokens from the *sentiment* (second) sequence, because offset mappings for the second sequence can also be within `[0, len(text)]`; we should instead restrict to tokens belonging to sequence-0 using `token_type_ids==0` and `offset!= (0,0)`. Then we should decode spans only from those tweet tokens and avoid stripping important punctuation too aggressively by only trimming whitespace and surrounding quotes (not commas/periods), which typically improves word-level Jaccard. These are minimal, inference-only changes: same model, same softmax, same joint start/end selection, same CSV output.'
- What this solution (achieved 0.49256) has done: 'Your current score (0.49256) is well below the target (0.7179), so we should make small inference-only fixes that improve span extraction fidelity without changing the model or adding training. The biggest issue is the tweet/sentiment masking: for DeBERTa, `token_type_ids` are often all zeros, so your `ttype==0` mask accidentally allows selecting tokens from the *sentiment* segment; we instead use `sequence_ids()` from the fast tokenizer encoding to restrict valid tokens to sequence-0 (tweet) only. We also stop re-tokenizing each sample inside the decoding loop (which can desync) and instead carry `input_ids` and `sequence_ids` out of the DataLoader for consistent, faster, correct masking. These changes keep the same backbone, the same logits, the same softmax and joint (start,end) selection, and the same CSV output, but should move the Jaccard upward toward the target.'
- What this solution (achieved 0.49288) has done: 'Your current gap to the target is large (0.49256 → 0.71791), but we can still improve meaningfully with inference-only changes that preserve the same DeBERTa QA span approach. The main fix is to correctly restrict candidate start/end tokens to the tweet segment using `offset_mapping` + `special_tokens_mask` (instead of `sequence_ids`, which is unreliable for DeBERTa fast tokenizers and can silently allow sentiment tokens). Then we make the span selection slightly more robust by adding a small maximum answer length constraint (typical for extractive QA) while still using the same start/end probabilities and joint selection idea. Finally, we keep your existing neutral/full-text fallback and keep the submission formatting unchanged.'
- What this solution (achieved 0.49288) has done: 'Your score is being held back mainly by the fact that you’re using a base DeBERTa backbone with a random (untrained) QA head most of the time, because your checkpoint search doesn’t actually load a fine-tuned *QA span* model for this competition. To move toward the target with minimal changes and without changing the core “extractive QA span from start/end logits” logic, I (1) load a known-good fine-tuned Tweet Sentiment Extraction extractive model from Hugging Face (a DeBERTa-based span model) as a fallback when no local checkpoint is found, and (2) make the tokenizer/model name consistent so the head matches the backbone. Everything else (dataset creation, logits->softmax, joint (start,end) selection, offset-based decoding, neutral fallback, and submission writing) remains the same.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import random
import time
import math
import re
import string
import sys
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
class CFG:
    DEBUG = False
    TRAIN = False  # inference-only
    N_FOLDS = 5
    TRAIN_FOLDS = [i for i in range(N_FOLDS)]
    SEED = 42
    TEST_BATCHSIZE = 100
    MAX_LENGTH = 128

    MODEL_NAME = "microsoft/deberta-v3-base"

    FC_DROPOUT = [0.1, 0.2, 0.3, 0.4, 0.5]
    MAX_ANSWER_LEN = 30




## === cell 2
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed=CFG.SEED)




## === cell 3
def _ensure_compatible_protobuf():
    import importlib

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"],
        check=False,
    )

    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf") or m == "protobuf":
            sys.modules.pop(m, None)

    importlib.invalidate_caches()


_ensure_compatible_protobuf()

try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
except Exception as e:
    raise RuntimeError(
        "Transformers import failed. This is usually a protobuf runtime mismatch. "
        "We attempted to pin protobuf==3.20.3 and force the pure-Python backend."
    ) from e

test_path_candidates = [
    "/kaggle/input/tweet-sentiment-extraction/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/tweet-sentiment-extraction/test.csv",
    "/kaggle/data/test.csv",
]
test_path = None
for p in test_path_candidates:
    if os.path.exists(p):
        test_path = p
        break
if test_path is None:
    raise FileNotFoundError(
        f"Could not find test.csv in any of: {test_path_candidates}"
    )

test_df = pd.read_csv(test_path)
test_df.head()




## === cell 4
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, use_fast=True)
CFG.TOKENIZER = tokenizer




## === cell 5
class QADataset:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = str(self.df.text.iloc[item])
        text_for_tokenize = text
        text_clean = " ".join(text.split())
        sent = str(self.df.sentiment.iloc[item])

        enc = CFG.TOKENIZER(
            text_for_tokenize,
            sent,
            add_special_tokens=True,
            max_length=CFG.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )

        seq_ids = enc.sequence_ids()  # list of {0,1,None}
        seq_ids = [(-1 if s is None else int(s)) for s in seq_ids]

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(enc["token_type_ids"], dtype=torch.long)
        offset_mapping = torch.tensor(enc["offset_mapping"], dtype=torch.long)
        sequence_ids = torch.tensor(seq_ids, dtype=torch.long)
        special_tokens_mask = torch.tensor(enc["special_tokens_mask"], dtype=torch.long)

        tok_text_tokens = CFG.TOKENIZER.convert_ids_to_tokens(enc["input_ids"])

        return {
            "input_ids": input_ids,
            "mask": attention_mask,
            "token_type_ids": token_type_ids,
            "sequence_ids": sequence_ids,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "text_clean": text_clean,  # keep for heuristic fallback only
            "text_tokens": " ".join(tok_text_tokens),
            "orig_text": text,
            "orig_sentiment": sent,
        }




## === cell 6
class QAModel(nn.Module):
    def __init__(self, config_path=None, pretrained=True):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                CFG.MODEL_NAME, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                CFG.MODEL_NAME, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        self.fc_dropout = nn.ModuleList([nn.Dropout(val) for val in CFG.FC_DROPOUT])
        self.fc = nn.Linear(self.config.hidden_size, 2)

    def forward(self, input_ids, mask, token_type_ids=None):
        embeddings = self.backbone(
            input_ids=input_ids, attention_mask=mask
        ).last_hidden_state
        logits = None
        for dropout_layer in self.fc_dropout:
            out = self.fc(dropout_layer(embeddings))
            logits = out if logits is None else (logits + out)

        logits = logits / len(CFG.FC_DROPOUT)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 7
def test_fn(dataloader, model):
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_mask = []

    fin_text_tokens = []
    fin_orig_text = []
    fin_orig_sentiment = []

    fin_offset_mapping = []
    fin_text_clean = []
    fin_token_type_ids = []
    fin_input_ids = []
    fin_sequence_ids = []
    fin_special_tokens_mask = []

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["mask"].to(device)

            start_logits, end_logits = model(input_ids, mask)

            fin_output_start.append(start_logits.detach().cpu())
            fin_output_end.append(end_logits.detach().cpu())
            fin_mask.append(mask.detach().cpu())

            fin_text_tokens.extend(data["text_tokens"])
            fin_orig_text.extend(data["orig_text"])
            fin_orig_sentiment.extend(data["orig_sentiment"])

            fin_offset_mapping.append(data["offset_mapping"].detach().cpu())
            fin_text_clean.extend(data["text_clean"])
            fin_token_type_ids.append(data["token_type_ids"].detach().cpu())

            fin_input_ids.append(data["input_ids"].detach().cpu())
            fin_sequence_ids.append(data["sequence_ids"].detach().cpu())
            fin_special_tokens_mask.append(data["special_tokens_mask"].detach().cpu())

    fin_output_start = torch.vstack(fin_output_start)
    fin_output_end = torch.vstack(fin_output_end)
    fin_mask = torch.vstack(fin_mask)
    fin_offset_mapping = torch.vstack(fin_offset_mapping)
    fin_token_type_ids = torch.vstack(fin_token_type_ids)
    fin_input_ids = torch.vstack(fin_input_ids)
    fin_sequence_ids = torch.vstack(fin_sequence_ids)
    fin_special_tokens_mask = torch.vstack(fin_special_tokens_mask)

    return (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_text_tokens,
        fin_orig_text,
        fin_orig_sentiment,
        fin_offset_mapping,
        fin_text_clean,
        fin_token_type_ids,
        fin_input_ids,
        fin_sequence_ids,
        fin_special_tokens_mask,
    )




## === cell 8
test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

model = QAModel(config_path=None, pretrained=True).to(device)

ckpt_candidates = [
    "/kaggle/input/tweet-sentiment-extraction",
    "/kaggle/data/tweet-sentiment-extraction",
    "/kaggle/input/tweet-sentiment-extraction/tweet-sentiment-extraction",
    "/kaggle/data/tweet-sentiment-extraction/tweet-sentiment-extraction",
]

loaded_any = False
for ckpt_dir in ckpt_candidates:
    if os.path.isdir(ckpt_dir) and any(
        os.path.exists(os.path.join(ckpt_dir, fn))
        for fn in ("pytorch_model.bin", "model.safetensors", "config.json")
    ):
        try:
            model.backbone = AutoModel.from_pretrained(ckpt_dir).to(device)
            print(f"Loaded backbone from directory: {ckpt_dir}")
            loaded_any = True
            break
        except Exception:
            pass

if not loaded_any:
    ckpt_file_candidates = [
        "/kaggle/input/tweet-sentiment-extraction/pytorch_model.bin",
        "/kaggle/input/tweet-sentiment-extraction/model.bin",
        "/kaggle/input/tweet-sentiment-extraction/model.pth",
        "/kaggle/data/tweet-sentiment-extraction/pytorch_model.bin",
        "/kaggle/data/tweet-sentiment-extraction/model.bin",
        "/kaggle/data/tweet-sentiment-extraction/model.pth",
        "/kaggle/working/pytorch_model.bin",
        "/kaggle/working/model.bin",
        "/kaggle/working/model.pth",
    ]
    ckpt_path = None
    for p in ckpt_file_candidates:
        if os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        if (
            isinstance(state, dict)
            and "model" in state
            and isinstance(state["model"], dict)
        ):
            state = state["model"]

        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k
                for prefix in ("module.", "model."):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix) :]
                new_state[nk] = v
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            print(f"Loaded checkpoint: {ckpt_path}")
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            loaded_any = True

if not loaded_any:
    hf_finetuned_candidates = [
        "twmkn9/deberta-v3-base-tweet-sentiment-extraction",
        "twmkn9/deberta-v3-large-tweet-sentiment-extraction",
        "mrm8488/deberta-v3-base-finetuned-tweet-sentiment-extraction",
    ]

    loaded_hf = False
    for name in hf_finetuned_candidates:
        try:
            sd = None
            try:
                from transformers import (
                    AutoModelForQuestionAnswering,
                )  # local import after protobuf fix

                qa_model = AutoModelForQuestionAnswering.from_pretrained(name).to(
                    device
                )
                model.backbone = qa_model.base_model.to(device)
                if hasattr(qa_model, "qa_outputs") and isinstance(
                    qa_model.qa_outputs, nn.Linear
                ):
                    if qa_model.qa_outputs.weight.shape == model.fc.weight.shape:
                        model.fc.weight.data.copy_(qa_model.qa_outputs.weight.data)
                        model.fc.bias.data.copy_(qa_model.qa_outputs.bias.data)
                loaded_hf = True
                print(f"Loaded Hugging Face fine-tuned QA model: {name}")
                break
            except Exception:
                model.backbone = AutoModel.from_pretrained(name).to(device)
                loaded_hf = True
                print(f"Loaded Hugging Face fine-tuned backbone: {name}")
                break
        except Exception:
            continue

    if not loaded_hf:
        print(
            "No fine-tuned checkpoint found locally or via fallback; using base pretrained backbone + random QA head (score will be low)."
        )

(
    fin_output_start,
    fin_output_end,
    fin_mask,
    fin_text_tokens,
    fin_orig_text,
    fin_orig_sentiment,
    fin_offset_mapping,
    fin_text_clean,
    fin_token_type_ids,
    fin_input_ids,
    fin_sequence_ids,
    fin_special_tokens_mask,
) = test_fn(test_loader, model)




## === cell 9
fin_mask = fin_mask.float()
pad_mask = fin_mask == 0

fin_output_start = fin_output_start.masked_fill(pad_mask, -1e9)
fin_output_end = fin_output_end.masked_fill(pad_mask, -1e9)

s = torch.nn.Softmax(dim=-1)
fin_output_start = s(fin_output_start)
fin_output_end = s(fin_output_end)

(
    fin_output_start.shape,
    fin_output_end.shape,
    fin_mask.shape,
    fin_offset_mapping.shape,
    fin_token_type_ids.shape,
    fin_input_ids.shape,
    fin_sequence_ids.shape,
    fin_special_tokens_mask.shape,
    len(fin_text_tokens),
    len(fin_orig_text),
)




## === cell 10
def _jaccard(a: str, b: str) -> float:
    a = str(a).lower().split()
    b = str(b).lower().split()
    sa, sb = set(a), set(b)
    if len(sa) == 0 and len(sb) == 0:
        return 1.0
    if len(sa) == 0 or len(sb) == 0:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _best_contiguous_span_by_self_jaccard(text_clean: str, sentiment: str) -> str:
    words = text_clean.split()
    if not words:
        return text_clean.strip()

    if str(sentiment).lower() == "positive":
        cues = {
            "good",
            "great",
            "love",
            "best",
            "happy",
            "amazing",
            "awesome",
            "nice",
            "thank",
            "thanks",
        }
    elif str(sentiment).lower() == "negative":
        cues = {
            "bad",
            "hate",
            "worst",
            "sad",
            "angry",
            "terrible",
            "awful",
            "sucks",
            "sorry",
            "pain",
        }
    else:
        return text_clean.strip()

    best = words[0]
    best_score = 0.0
    n = len(words)
    for i in range(n):
        for j in range(i, min(n, i + 10)):  # cap span length for stability
            span = " ".join(words[i : j + 1])
            score = len(set(span.lower().split()) & cues) / max(
                1, len(set(span.lower().split()) | cues)
            )
            if score > best_score:
                best_score = score
                best = span
    return best.strip() if best.strip() else text_clean.strip()


def _postprocess_pred_span(pred: str, sentiment: str) -> str:
    p = str(pred).strip()
    return p.strip(" \t\r\n\"'`")


def _decode_span_from_offsets(
    orig_text: str,
    sentiment: str,
    offsets: np.ndarray,
    start_idx: int,
    end_idx: int,
):
    if str(sentiment).lower() == "neutral":
        return str(orig_text).strip()

    text = str(orig_text)
    tweet_end_char = len(text)

    start_idx = int(max(0, min(start_idx, offsets.shape[0] - 1)))
    end_idx = int(max(0, min(end_idx, offsets.shape[0] - 1)))
    if end_idx < start_idx:
        end_idx = start_idx

    char_starts = []
    char_ends = []
    for i in range(start_idx, end_idx + 1):
        s0, e0 = int(offsets[i, 0]), int(offsets[i, 1])
        if e0 <= s0:
            continue
        if s0 >= tweet_end_char:
            continue
        s0 = max(0, min(s0, tweet_end_char))
        e0 = max(0, min(e0, tweet_end_char))
        if e0 > s0:
            char_starts.append(s0)
            char_ends.append(e0)

    if not char_starts:
        return text.strip()

    span = text[min(char_starts) : max(char_ends)].strip()
    if span == "":
        span = text.strip()
    return span


def _select_best_span(
    start_prob: np.ndarray,
    end_prob: np.ndarray,
    valid_mask: np.ndarray,
    max_answer_len: int = 30,
):
    sp = start_prob * valid_mask
    ep = end_prob * valid_mask

    if float(sp.max()) <= 0 and float(ep.max()) <= 0:
        i = int(np.argmax(start_prob))
        j = int(np.argmax(end_prob))
        if j < i:
            j = i
        return i, j

    score = np.outer(sp, ep)
    score = np.triu(score)  # enforce end >= start

    if max_answer_len is not None and max_answer_len > 0:
        n = score.shape[0]
        for i in range(n):
            j_max = min(n, i + max_answer_len)
            if j_max < n:
                score[i, j_max:] = 0.0

    flat = int(np.argmax(score))
    n = score.shape[0]
    i, j = divmod(flat, n)
    return int(i), int(j)


special_id_set = set(CFG.TOKENIZER.all_special_ids)

final_outputs = []

for j in range(len(fin_text_tokens)):
    mask_vec = fin_mask[j].numpy()
    offsets = fin_offset_mapping[j].numpy()
    orig_text = fin_orig_text[j]
    sentiment = fin_orig_sentiment[j]
    text_clean = fin_text_clean[j]

    input_ids_j = fin_input_ids[j].numpy().astype(np.int64)
    special_mask = fin_special_tokens_mask[j].numpy().astype(np.int64)

    text = str(orig_text)
    text_len = len(text)

    valid_offsets = ((offsets[:, 1] - offsets[:, 0]) > 0).astype(np.float32)
    within_text = ((offsets[:, 0] >= 0) & (offsets[:, 1] <= text_len)).astype(
        np.float32
    )

    not_special = 1.0 - (special_mask > 0).astype(np.float32)

    ws_ok = np.zeros_like(valid_offsets, dtype=np.float32)
    for i in range(len(valid_offsets)):
        if valid_offsets[i] == 0:
            continue
        s0, e0 = int(offsets[i, 0]), int(offsets[i, 1])
        s0 = max(0, min(s0, text_len))
        e0 = max(0, min(e0, text_len))
        if e0 > s0 and text[s0:e0].strip() != "":
            ws_ok[i] = 1.0

    valid_mask = mask_vec * valid_offsets * within_text * not_special * ws_ok

    start_prob = fin_output_start[j].numpy()
    end_prob = fin_output_end[j].numpy()

    idx_start, idx_end = _select_best_span(
        start_prob, end_prob, valid_mask, max_answer_len=CFG.MAX_ANSWER_LEN
    )

    pred = _decode_span_from_offsets(
        orig_text=orig_text,
        sentiment=sentiment,
        offsets=offsets,
        start_idx=idx_start,
        end_idx=idx_end,
    )

    if str(sentiment).lower() != "neutral" and pred.strip() == "":
        best_i = int(np.argmax(start_prob * valid_mask))
        pred = _decode_span_from_offsets(
            orig_text=orig_text,
            sentiment=sentiment,
            offsets=offsets,
            start_idx=best_i,
            end_idx=best_i,
        )

    pred = _postprocess_pred_span(pred, sentiment)

    if str(sentiment).lower() != "neutral":
        if pred.strip() == "" or len(pred.split()) > max(12, len(text_clean.split())):
            pred = _best_contiguous_span_by_self_jaccard(text_clean, sentiment)

    if pred == "":
        pred = str(orig_text).strip()

    final_outputs.append(pred)

len(final_outputs), final_outputs[:5]




## === cell 11
test_ids = test_df["textID"].astype(str).values
sub = pd.DataFrame({"textID": test_ids, "selected_text": final_outputs})

assert len(sub) == len(test_df)
sub.head()




## === cell 12
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.iloc[0])
