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

0.5516448616981506

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.14602) has done: 'I fix the environment/runtime failures by (1) removing the protobuf-triggering import stack that breaks at startup, (2) ensuring `DataCollatorWithPadding` is imported, and (3) replacing the invalid private Kaggle input paths for tokenizer/config/weights with public Hugging Face model loading so the notebook can run without missing files. To preserve the core inference logic (start/end span logits over tokens), I keep the same `QAModel` architecture and decoding approach, but load a compatible backbone and run a single deterministic forward pass. Finally, I guarantee we always generate exactly 2749 predictions and write a valid `submission.csv` with the required columns and quoting handled by pandas.'
- What this solution (achieved 0.14602) has done: 'I fix the startup crash caused by an incompatible protobuf/transformers import stack by forcing the pure-Python protobuf implementation before importing `transformers`. Then I correct a key decoding logic bug: applying `Softmax(dim=1)` to a 1D logit vector is wrong (it normalizes across samples when batched and errors/behaves badly per-row); instead I use `dim=-1` so probabilities are computed across token positions. These two changes keep your model/inference approach the same (start/end span logits over tokens) but make it run end-to-end and materially improve span selection toward the target Jaccard score. Finally, I keep the submission writing as-is to ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 0.16526) has done: 'I fix the tensor stacking crash by padding the variable-length logits/masks to a common sequence length before `vstack`, which preserves your core inference logic but makes batching robust. Then I ensure downstream cells run by producing `fin_output_start/end/mask` deterministically even when batches have different padded lengths. Finally, I keep your existing softmax-over-tokens and span decoding, and guarantee a correctly formatted `submission.csv` is written with exactly 2749 rows.'
- What this solution (achieved 0.06911) has done: 'Your current inference is using a randomly initialized DeBERTa QA head (because there is no fine-tuning), so the span predictions are essentially noise and the Jaccard score stays very low. To move the score substantially toward the target while preserving the same core span-logit decoding semantics, the minimal legitimate improvement is to switch the backbone to a sentiment-extraction model that already has the same start/end span head trained for this competition. I keep your dataset/offset mapping, start/end softmax over tokens, and argmax span decoding unchanged; the main change is loading `AutoModelForQuestionAnswering` weights and reading `start_logits/end_logits` directly. This should move the score much closer to the target band without changing the evaluation semantics or adding new training loops.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import subprocess


def _ensure_protobuf_compat():
    """
    Bug fix: some Kaggle images ship protobuf>=5 which can break certain transformers stacks.
    Force protobuf<5 if needed. If pip install fails (offline), continue with existing version.
    """
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version: {pb_ver}")
    except Exception:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf>=4.21.0,<5"]
            )
            import importlib

            importlib.invalidate_caches()
        except Exception:
            pass


_ensure_protobuf_compat()

import gc
import random
import math
import re
import string

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForQuestionAnswering,
)

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

    FINETUNED_QA_MODEL = "/kaggle/input/tweet-sentiment-extraction"

    FC_DROPOUT = [0.1, 0.2, 0.3, 0.4, 0.5]




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
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
test_df.head()



## === cell 4
tokenizer = AutoTokenizer.from_pretrained(CFG.FINETUNED_QA_MODEL, use_fast=True)
CFG.TOKENIZER = tokenizer

collate_fn = DataCollatorWithPadding(
    CFG.TOKENIZER, padding="longest", return_tensors="pt"
)


class QADataset:
    """
    DataCollatorWithPadding can only pad/tokenize numeric fields.
    Return only tokenizer outputs + meta fields collated manually.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        orig_text = str(self.df.text.iloc[item])
        text = " ".join(orig_text.split())
        sentiment_str = self.df.sentiment.iloc[item]

        question = f"extract {sentiment_str}"

        inputs = CFG.TOKENIZER(
            question,
            text,
            add_special_tokens=True,
            max_length=CFG.MAX_LENGTH,
            padding=False,  # let collator pad dynamically
            truncation=True,
            return_offsets_mapping=True,
        )

        sentiment = [1, 0, 0]
        if sentiment_str == "positive":
            sentiment = [0, 0, 1]
        if sentiment_str == "negative":
            sentiment = [0, 1, 0]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "offset_mapping": inputs["offset_mapping"],
            "orig_text": orig_text,
            "orig_sentiment": sentiment_str,
            "sentiment": sentiment,  # unused, but kept
        }


def qa_collate(batch):
    token_features = [
        {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]}
        for x in batch
    ]
    padded = collate_fn(token_features)

    padded["offset_mapping"] = [x["offset_mapping"] for x in batch]
    padded["orig_text"] = [x["orig_text"] for x in batch]
    padded["orig_sentiment"] = [x["orig_sentiment"] for x in batch]
    padded["sentiment"] = torch.tensor(
        [x["sentiment"] for x in batch], dtype=torch.long
    )
    return padded




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1645844348.py in <cell line: 0>()
      1 # Bug fix: avoid gated/unauthorized HF downloads by loading tokenizer from local path.
----> 2 tokenizer = AutoTokenizer.from_pretrained(CFG.FINETUNED_QA_MODEL, use_fast=True)
      3 CFG.TOKENIZER = tokenizer
      4 
      5 collate_fn = DataCollatorWithPadding(

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
   1001                     config = AutoConfig.for_model(**config_dict)
   1002                 else:
-> 1003                     config = AutoConfig.from_pretrained(
   1004                         pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
   1005                     )

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1236                     return CONFIG_MAPPING[pattern].from_dict(config_dict, **unused_kwargs)
   1237 
-> 1238         raise ValueError(
   1239             f"Unrecognized model in {pretrained_model_name_or_path}. "
   1240             f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "

ValueError: Unrecognized model in /kaggle/input/tweet-sentiment-extraction. Should have a `model_type` key in its config.json, or contain one of the following strings in its name: albert, align, altclip, arcee, aria, aria_text, audio-spectrogram-transformer, autoformer, aya_vision, bamba, bark, bart, beit, bert, bert-generation, big_bird, bigbird_pegasus, biogpt, bit, bitnet, blenderbot, blenderbot-small, blip, blip-2, blip_2_qformer, bloom, bridgetower, bros, camembert, canine, chameleon, chinese_clip, chinese_clip_vision_model, clap, clip, clip_text_model, clip_vision_model, clipseg, clvp, code_llama, codegen, cohere, cohere2, colpali, colqwen2, conditional_detr, convbert, convnext, convnextv2, cpmant, csm, ctrl, cvt, d_fine, dab-detr, dac, data2vec-audio, data2vec-text, data2vec-vision, dbrx, deberta, deberta-v2, decision_transformer, deepseek_v3, deformable_detr, deit, depth_anything, depth_pro, deta, detr, dia, diffllama, dinat, dinov2, dinov2_with_registers, distilbert, donut-swin, dots1, dpr, dpt, efficientformer, efficientnet, electra, emu3, encodec, encoder-decoder, ernie, ernie_m, esm, falcon, falcon_h1, falcon_mamba, fastspeech2_conformer, flaubert, flava, fnet, focalnet, fsmt, funnel, fuyu, gemma, gemma2, gemma3, gemma3_text, gemma3n, gemma3n_audio, gemma3n_text, gemma3n_vision, git, glm, glm4, glm4v, glm4v_text, glpn, got_ocr2, gpt-sw3, gpt2, gpt_bigcode, gpt_neo, gpt_neox, gpt_neox_japanese, gptj, gptsan-japanese, granite, granite_speech, granitemoe, granitemoehybrid, granitemoeshared, granitevision, graphormer, grounding-dino, groupvit, helium, hgnet_v2, hiera, hubert, ibert, idefics, idefics2, idefics3, idefics3_vision, ijepa, imagegpt, informer, instructblip, instructblipvideo, internvl, internvl_vision, jamba, janus, jetmoe, jukebox, kosmos-2, kyutai_speech_to_text, layoutlm, layoutlmv2, layoutlmv3, led, levit, lightglue, lilt, llama, llama4, llama4_text, llava, llava_next, llava_next_video, llava_onevision, longformer, longt5, luke, lxmert, m2m_100, mamba, mamba2, marian, markuplm, mask2former, maskformer, maskformer-swin, mbart, mctct, mega, megatron-bert, mgp-str, mimi, minimax, mistral, mistral3, mixtral, mlcd, mllama, mobilebert, mobilenet_v1, mobilenet_v2, mobilevit, mobilevitv2, modernbert, moonshine, moshi, mpnet, mpt, mra, mt5, musicgen, musicgen_melody, mvp, nat, nemotron, nezha, nllb-moe, nougat, nystromformer, olmo, olmo2, olmoe, omdet-turbo, oneformer, open-llama, openai-gpt, opt, owlv2, owlvit, paligemma, patchtsmixer, patchtst, pegasus, pegasus_x, perceiver, persimmon, phi, phi3, phi4_multimodal, phimoe, pix2struct, pixtral, plbart, poolformer, pop2piano, prompt_depth_anything, prophetnet, pvt, pvt_v2, qdqbert, qwen2, qwen2_5_omni, qwen2_5_vl, qwen2_5_vl_text, qwen2_audio, qwen2_audio_encoder, qwen2_moe, qwen2_vl, qwen2_vl_text, qwen3, qwen3_moe, rag, realm, recurrent_gemma, reformer, regnet, rembert, resnet, retribert, roberta, roberta-prelayernorm, roc_bert, roformer, rt_detr, rt_detr_resnet, rt_detr_v2, rwkv, sam, sam_hq, sam_hq_vision_model, sam_vision_model, seamless_m4t, seamless_m4t_v2, segformer, seggpt, sew, sew-d, shieldgemma2, siglip, siglip2, siglip_vision_model, smollm3, smolvlm, smolvlm_vision, speech-encoder-decoder, speech_to_text, speech_to_text_2, speecht5, splinter, squeezebert, stablelm, starcoder2, superglue, superpoint, swiftformer, swin, swin2sr, swinv2, switch_transformers, t5, t5gemma, table-transformer, tapas, textnet, time_series_transformer, timesfm, timesformer, timm_backbone, timm_wrapper, trajectory_transformer, transfo-xl, trocr, tvlt, tvp, udop, umt5, unispeech, unispeech-sat, univnet, upernet, van, video_llava, videomae, vilt, vipllava, vision-encoder-decoder, vision-text-dual-encoder, visual_bert, vit, vit_hybrid, vit_mae, vit_msn, vitdet, vitmatte, vitpose, vitpose_backbone, vits, vivit, vjepa2, wav2vec2, wav2vec2-bert, wav2vec2-conformer, wavlm, whisper, xclip, xglm, xlm, xlm-prophetnet, xlm-roberta, xlm-roberta-xl, xlnet, xmod, yolos, yoso, zamba, zamba2, zoedepth

## === cell 5
class QAModel(nn.Module):
    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        self.qa = AutoModelForQuestionAnswering.from_pretrained(CFG.FINETUNED_QA_MODEL)

    def forward(self, input_ids, mask):
        out = self.qa(input_ids=input_ids, attention_mask=mask)
        return out.start_logits, out.end_logits




## === cell 6
def _pad_to_length(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    """
    Bug fix for vstack size mismatch:
    Different batches can have different sequence lengths due to dynamic padding.
    We pad each batch output to the global max length before stacking.
    """
    if x.size(1) == length:
        return x
    pad_len = length - x.size(1)
    if pad_len < 0:
        return x[:, :length]
    return torch.nn.functional.pad(x, (0, pad_len), value=value)


def test_fn(dataloader, model):
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_mask = []
    fin_offsets = []
    fin_orig_text = []
    fin_orig_sentiment = []

    max_len = 0

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, mask)

            max_len = max(
                max_len, start_logits.size(1), end_logits.size(1), mask.size(1)
            )

            fin_output_start.append(start_logits.cpu())
            fin_output_end.append(end_logits.cpu())
            fin_mask.append(mask.cpu())

            fin_offsets.extend(data["offset_mapping"])
            fin_orig_text.extend(data["orig_text"])
            fin_orig_sentiment.extend(data["orig_sentiment"])

    fin_output_start = torch.vstack(
        [_pad_to_length(t, max_len, value=0.0) for t in fin_output_start]
    )
    fin_output_end = torch.vstack(
        [_pad_to_length(t, max_len, value=0.0) for t in fin_output_end]
    )
    fin_mask = torch.vstack([_pad_to_length(t, max_len, value=0.0) for t in fin_mask])

    return (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_offsets,
        fin_orig_text,
        fin_orig_sentiment,
    )




## === cell 7
test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    collate_fn=qa_collate,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4103579393.py in <cell line: 0>()
----> 1 test_dataset = QADataset(test_df)
      2 test_loader = DataLoader(
      3     test_dataset,
      4     batch_size=CFG.TEST_BATCHSIZE,
      5     shuffle=False,

NameError: name 'QADataset' is not defined

## === cell 8
model = QAModel(config_path=None, pretrained=True).to(device)

(
    fin_output_start,
    fin_output_end,
    fin_mask,
    fin_offsets,
    fin_orig_text,
    fin_orig_sentiment,
) = test_fn(test_loader, model)

(
    fin_output_start.shape,
    fin_output_end.shape,
    fin_mask.shape,
    len(fin_offsets),
    len(fin_orig_text),
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3437608911.py in <cell line: 0>()
----> 1 model = QAModel(config_path=None, pretrained=True).to(device)
      2 
      3 (
      4     fin_output_start,
      5     fin_output_end,

/tmp/ipykernel_55/3405695193.py in __init__(self, config_path, pretrained)
      4         # Bug fix: load model locally (no HF hub access needed).
      5         # Core logic preserved: start/end span logits over tokens.
----> 6         self.qa = AutoModelForQuestionAnswering.from_pretrained(CFG.FINETUNED_QA_MODEL)
      7 
      8     def forward(self, input_ids, mask):

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/auto_factory.py in from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    545                 _ = kwargs.pop("quantization_config")
    546 
--> 547             config, kwargs = AutoConfig.from_pretrained(
    548                 pretrained_model_name_or_path,
    549                 return_unused_kwargs=True,

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1236                     return CONFIG_MAPPING[pattern].from_dict(config_dict, **unused_kwargs)
   1237 
-> 1238         raise ValueError(
   1239             f"Unrecognized model in {pretrained_model_name_or_path}. "
   1240             f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "

ValueError: Unrecognized model in /kaggle/input/tweet-sentiment-extraction. Should have a `model_type` key in its config.json, or contain one of the following strings in its name: albert, align, altclip, arcee, aria, aria_text, audio-spectrogram-transformer, autoformer, aya_vision, bamba, bark, bart, beit, bert, bert-generation, big_bird, bigbird_pegasus, biogpt, bit, bitnet, blenderbot, blenderbot-small, blip, blip-2, blip_2_qformer, bloom, bridgetower, bros, camembert, canine, chameleon, chinese_clip, chinese_clip_vision_model, clap, clip, clip_text_model, clip_vision_model, clipseg, clvp, code_llama, codegen, cohere, cohere2, colpali, colqwen2, conditional_detr, convbert, convnext, convnextv2, cpmant, csm, ctrl, cvt, d_fine, dab-detr, dac, data2vec-audio, data2vec-text, data2vec-vision, dbrx, deberta, deberta-v2, decision_transformer, deepseek_v3, deformable_detr, deit, depth_anything, depth_pro, deta, detr, dia, diffllama, dinat, dinov2, dinov2_with_registers, distilbert, donut-swin, dots1, dpr, dpt, efficientformer, efficientnet, electra, emu3, encodec, encoder-decoder, ernie, ernie_m, esm, falcon, falcon_h1, falcon_mamba, fastspeech2_conformer, flaubert, flava, fnet, focalnet, fsmt, funnel, fuyu, gemma, gemma2, gemma3, gemma3_text, gemma3n, gemma3n_audio, gemma3n_text, gemma3n_vision, git, glm, glm4, glm4v, glm4v_text, glpn, got_ocr2, gpt-sw3, gpt2, gpt_bigcode, gpt_neo, gpt_neox, gpt_neox_japanese, gptj, gptsan-japanese, granite, granite_speech, granitemoe, granitemoehybrid, granitemoeshared, granitevision, graphormer, grounding-dino, groupvit, helium, hgnet_v2, hiera, hubert, ibert, idefics, idefics2, idefics3, idefics3_vision, ijepa, imagegpt, informer, instructblip, instructblipvideo, internvl, internvl_vision, jamba, janus, jetmoe, jukebox, kosmos-2, kyutai_speech_to_text, layoutlm, layoutlmv2, layoutlmv3, led, levit, lightglue, lilt, llama, llama4, llama4_text, llava, llava_next, llava_next_video, llava_onevision, longformer, longt5, luke, lxmert, m2m_100, mamba, mamba2, marian, markuplm, mask2former, maskformer, maskformer-swin, mbart, mctct, mega, megatron-bert, mgp-str, mimi, minimax, mistral, mistral3, mixtral, mlcd, mllama, mobilebert, mobilenet_v1, mobilenet_v2, mobilevit, mobilevitv2, modernbert, moonshine, moshi, mpnet, mpt, mra, mt5, musicgen, musicgen_melody, mvp, nat, nemotron, nezha, nllb-moe, nougat, nystromformer, olmo, olmo2, olmoe, omdet-turbo, oneformer, open-llama, openai-gpt, opt, owlv2, owlvit, paligemma, patchtsmixer, patchtst, pegasus, pegasus_x, perceiver, persimmon, phi, phi3, phi4_multimodal, phimoe, pix2struct, pixtral, plbart, poolformer, pop2piano, prompt_depth_anything, prophetnet, pvt, pvt_v2, qdqbert, qwen2, qwen2_5_omni, qwen2_5_vl, qwen2_5_vl_text, qwen2_audio, qwen2_audio_encoder, qwen2_moe, qwen2_vl, qwen2_vl_text, qwen3, qwen3_moe, rag, realm, recurrent_gemma, reformer, regnet, rembert, resnet, retribert, roberta, roberta-prelayernorm, roc_bert, roformer, rt_detr, rt_detr_resnet, rt_detr_v2, rwkv, sam, sam_hq, sam_hq_vision_model, sam_vision_model, seamless_m4t, seamless_m4t_v2, segformer, seggpt, sew, sew-d, shieldgemma2, siglip, siglip2, siglip_vision_model, smollm3, smolvlm, smolvlm_vision, speech-encoder-decoder, speech_to_text, speech_to_text_2, speecht5, splinter, squeezebert, stablelm, starcoder2, superglue, superpoint, swiftformer, swin, swin2sr, swinv2, switch_transformers, t5, t5gemma, table-transformer, tapas, textnet, time_series_transformer, timesfm, timesformer, timm_backbone, timm_wrapper, trajectory_transformer, transfo-xl, trocr, tvlt, tvp, udop, umt5, unispeech, unispeech-sat, univnet, upernet, van, video_llava, videomae, vilt, vipllava, vision-encoder-decoder, vision-text-dual-encoder, visual_bert, vit, vit_hybrid, vit_mae, vit_msn, vitdet, vitmatte, vitpose, vitpose_backbone, vits, vivit, vjepa2, wav2vec2, wav2vec2-bert, wav2vec2-conformer, wavlm, whisper, xclip, xglm, xlm, xlm-prophetnet, xlm-roberta, xlm-roberta-xl, xlnet, xmod, yolos, yoso, zamba, zamba2, zoedepth

## === cell 9
s = torch.nn.Softmax(dim=-1)
fin_output_start = s(fin_output_start)
fin_output_end = s(fin_output_end)
fin_mask = fin_mask.float()

final_outputs = []

for j in range(len(fin_orig_text)):
    orig_text = str(fin_orig_text[j])
    text = " ".join(orig_text.split())
    offsets = fin_offsets[j]  # list of (start_char, end_char)
    mask = fin_mask[j]

    valid_offset_mask = torch.zeros_like(mask)
    n_tok = min(len(offsets), int(mask.numel()))
    for i in range(n_tok):
        off = offsets[i]
        if off is None:
            continue
        if isinstance(off, (list, tuple)) and len(off) == 2 and off[1] > off[0]:
            valid_offset_mask[i] = 1.0

    effective_mask = mask * valid_offset_mask

    mask_start = fin_output_start[j] * effective_mask
    mask_end = fin_output_end[j] * effective_mask

    idx_start = int(torch.argmax(mask_start).item())
    idx_end = int(torch.argmax(mask_end).item())
    if idx_end < idx_start:
        idx_end = idx_start

    idx_start = min(idx_start, n_tok - 1)
    idx_end = min(idx_end, n_tok - 1)

    def _find_next_valid_left(i):
        while i > 0 and (offsets[i] is None or offsets[i][1] <= offsets[i][0]):
            i -= 1
        return i

    def _find_next_valid_right(i):
        while i < n_tok - 1 and (offsets[i] is None or offsets[i][1] <= offsets[i][0]):
            i += 1
        return i

    idx_start = _find_next_valid_right(idx_start)
    idx_end = _find_next_valid_left(idx_end)
    if idx_end < idx_start:
        idx_end = idx_start

    start_char = offsets[idx_start][0]
    end_char = offsets[idx_end][1]

    if end_char <= start_char:
        pred = text
    else:
        pred = text[start_char:end_char].strip()
        if pred == "":
            pred = text

    final_outputs.append(pred)

len(final_outputs), final_outputs[0]



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2493036278.py in <cell line: 0>()
      1 # Core decoding logic preserved; bug fix from prior plan: softmax over tokens (dim=-1), not over batch.
      2 s = torch.nn.Softmax(dim=-1)
----> 3 fin_output_start = s(fin_output_start)
      4 fin_output_end = s(fin_output_end)
      5 fin_mask = fin_mask.float()

NameError: name 'fin_output_start' is not defined

## === cell 10
assert len(final_outputs) == len(
    test_df
), f"Pred length {len(final_outputs)} != test length {len(test_df)}"

test_ids = test_df["textID"].values
sub = pd.DataFrame({"textID": test_ids, "selected_text": final_outputs})
sub.head()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/748946954.py in <cell line: 0>()
----> 1 assert len(final_outputs) == len(
      2     test_df
      3 ), f"Pred length {len(final_outputs)} != test length {len(test_df)}"
      4 
      5 test_ids = test_df["textID"].values

NameError: name 'final_outputs' is not defined

## === cell 11
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.isna().sum())
print(sub.head(3).to_string(index=False))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3116703084.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub.shape)
      3 print(sub.isna().sum())
      4 print(sub.head(3).to_string(index=False))

NameError: name 'sub' is not defined
