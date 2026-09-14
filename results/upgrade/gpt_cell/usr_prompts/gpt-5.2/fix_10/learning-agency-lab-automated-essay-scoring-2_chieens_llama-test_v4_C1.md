# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.12

# 2. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
transformers==4.53.3

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
from time import time
import os
import sys
import subprocess
from pathlib import Path

import torch

torch.set_grad_enabled(False)

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TRANSFORMERS_NO_PROTOBUF", "1")

import transformers  # re-import here so the env vars are in effect for this cell

model_path = Path("/kaggle/input/llama-3/transformers/8b-chat-hf/1")
if model_path.exists():
    model = str(model_path)
    _local_files_only = True
else:
    model = "gpt2"
    _local_files_only = False

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model, local_files_only=_local_files_only, trust_remote_code=True
)
model_obj = transformers.AutoModelForCausalLM.from_pretrained(
    model,
    local_files_only=_local_files_only,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)



## === cell 1
system_message = """
You are an AI assistant designed to score student essay.
The score is a value from 1 to 6.
You must answer only the score.
"""

_HAS_CHAT_TEMPLATE = bool(getattr(pipeline.tokenizer, "chat_template", None))
_TERMINATORS = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]


def _make_prompt(system_message: str, user_message: str) -> str:
    user_message = "Essay: " + user_message + " Score:"
    if _HAS_CHAT_TEMPLATE:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prompt = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"{system_message.strip()}\n\n{user_message}"
    return prompt


@torch.inference_mode()
def query_model_batch(
    system_message, user_messages, temperature=0.7, max_length=5, batch_size=16
):
    out = []
    n = len(user_messages)
    for i in range(0, n, batch_size):
        batch = user_messages[i : i + batch_size]
        start_time = time()

        prompts = [_make_prompt(system_message, um) for um in batch]
        sequences = pipeline(
            prompts,
            do_sample=True,
            top_p=0.9,
            temperature=temperature,
            eos_token_id=_TERMINATORS,
            max_new_tokens=max_length,
            return_full_text=False,
            pad_token_id=pipeline.model.config.eos_token_id,
            batch_size=batch_size,
        )

        end_time = time()
        ttime = f"Total time: {round(end_time-start_time, 2)} sec."

        for um, seq in zip(batch, sequences):
            um_fmt = "Essay: " + um + " Score:"
            answer = (
                seq[0]["generated_text"]
                if isinstance(seq, list)
                else seq["generated_text"]
            )
            out.append(um_fmt + " " + answer + " " + ttime)
    return out


def _parse_score_from_response(response: str) -> int:
    score = response.split("Score: ")[1][0]
    if score == "1":
        return 1
    if score == "2":
        return 2
    if score == "4":
        return 4
    if score == "5":
        return 5
    if score == "6":
        return 6
    return 3




## === cell 2
import pandas as pd



## === cell 3
test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)



## === cell 4
texts = test["full_text"].tolist()
responses = query_model_batch(system_message, texts, max_length=5, batch_size=16)
prediction = [_parse_score_from_response(r) for r in responses]



## --- ERROR in cell 4, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/682053883.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;31m# Core logic preserved: identical prompt construction, same decoding params, same score parsing.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0mtexts[0m [0;34m=[0m [0mtest[0m[0;34m[[0m[0;34m"full_text"[0m[0;34m][0m[0;34m.[0m[0mtolist[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0mresponses[0m [0;34m=[0m [0mquery_model_batch[0m[0;34m([0m[0msystem_message[0m[0;34m,[0m [0mtexts[0m[0;34m,[0m [0mmax_length[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m [0mbatch_size[0m[0;34m=[0m[0;36m16[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0mprediction[0m [0;34m=[0m [0;34m[[0m[0m_parse_score_from_response[0m[0;34m([0m[0mr[0m[0;34m)[0m [0;32mfor[0m [0mr[0m [0;32min[0m [0mresponses[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py[0m in [0;36mdecorate_context[0;34m(*args, **kwargs)[0m
[1;32m    114[0m     [0;32mdef[0m [0mdecorate_context[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    115[0m         [0;32mwith[0m [0mctx_factory[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 116[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    117[0m [0;34m[0m[0m
[1;32m    118[0m     [0;32mreturn[0m [0mdecorate_context[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1163683832.py[0m in [0;36mquery_model_batch[0;34m(system_message, user_messages, temperature, max_length, batch_size)[0m
[1;32m     44[0m [0;34m[0m[0m
[1;32m     45[0m         [0mprompts[0m [0;34m=[0m [0;34m[[0m[0m_make_prompt[0m[0;34m([0m[0msystem_message[0m[0;34m,[0m [0mum[0m[0;34m)[0m [0;32mfor[0m [0mum[0m [0;32min[0m [0mbatch[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 46[0;31m         sequences = pipeline(
[0m[1;32m     47[0m             [0mprompts[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m             [0mdo_sample[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/pipelines/text_generation.py[0m in [0;36m__call__[0;34m(self, text_inputs, **kwargs)[0m
[1;32m    314[0m                     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    315[0m                         [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__call__[0m[0;34m([0m[0mlist[0m[0;34m([0m[0mchats[0m[0;34m)[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 316[0;31m         [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__call__[0m[0;34m([0m[0mtext_inputs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    317[0m [0;34m[0m[0m
[1;32m    318[0m     def preprocess(

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/pipelines/base.py[0m in [0;36m__call__[0;34m(self, inputs, num_workers, batch_size, *args, **kwargs)[0m
[1;32m   1440[0m         [0;32mif[0m [0mis_list[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1441[0m             [0;32mif[0m [0mcan_use_iterator[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1442[0;31m                 final_iterator = self.get_iterator(
[0m[1;32m   1443[0m                     [0minputs[0m[0;34m,[0m [0mnum_workers[0m[0;34m,[0m [0mbatch_size[0m[0;34m,[0m [0mpreprocess_params[0m[0;34m,[0m [0mforward_params[0m[0;34m,[0m [0mpostprocess_params[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1444[0m                 )

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/pipelines/base.py[0m in [0;36mget_iterator[0;34m(self, inputs, num_workers, batch_size, preprocess_params, forward_params, postprocess_params)[0m
[1;32m   1394[0m         [0;31m# TODO hack by collating feature_extractor and image_processor[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1395[0m         [0mfeature_extractor[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfeature_extractor[0m [0;32mif[0m [0mself[0m[0;34m.[0m[0mfeature_extractor[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32melse[0m [0mself[0m[0;34m.[0m[0mimage_processor[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1396[0;31m         [0mcollate_fn[0m [0;34m=[0m [0mno_collate_fn[0m [0;32mif[0m [0mbatch_size[0m [0;34m==[0m [0;36m1[0m [0;32melse[0m [0mpad_collate_fn[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mtokenizer[0m[0;34m,[0m [0mfeature_extractor[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1397[0m         [0mdataloader[0m [0;34m=[0m [0mDataLoader[0m[0;34m([0m[0mdataset[0m[0;34m,[0m [0mnum_workers[0m[0;34m=[0m[0mnum_workers[0m[0;34m,[0m [0mbatch_size[0m[0;34m=[0m[0mbatch_size[0m[0;34m,[0m [0mcollate_fn[0m[0;34m=[0m[0mcollate_fn[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1398[0m         [0mmodel_iterator[0m [0;34m=[0m [0mPipelineIterator[0m[0;34m([0m[0mdataloader[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mforward[0m[0;34m,[0m [0mforward_params[0m[0;34m,[0m [0mloader_batch_size[0m[0;34m=[0m[0mbatch_size[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/transformers/pipelines/base.py[0m in [0;36mpad_collate_fn[0;34m(tokenizer, feature_extractor)[0m
[1;32m    153[0m     [0;32mif[0m [0mtokenizer[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    154[0m         [0;32mif[0m [0mtokenizer[0m[0;34m.[0m[0mpad_token_id[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 155[0;31m             raise ValueError(
[0m[1;32m    156[0m                 [0;34m"Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    157[0m                 [0;34m"`pipe.tokenizer.pad_token_id = model.config.eos_token_id`."[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with `pipe.tokenizer.pad_token_id = model.config.eos_token_id`.

## === cell 6
submission = test[["essay_id"]].copy()
submission["score"] = prediction
