import re
import os
from pathlib import Path
from typing import Dict
from glob import glob
import nbformat

def idToPath(id):
    f = str(id).zfill(10)
    prefix = '../../.kaggle/meta-kaggle-code/'+f[0:4]+'/'+f[4:7]+'/'+str(id)+'.*'
    g = glob(prefix)
    if len(g) == 1:
        return g[0]
    return ""

def no_gpu_required(kernel_meta: Dict) -> list:
    gpu_required = ["cupy-cuda12x", "cudf", "cuml", "nvidia-ml-py"
    # PyTorch stack
    "torch", "torchaudio", "torch-geometric", "torchinfo", "torchmetrics", "torch-optimizer", "torchsampler", "torchsummary", "torchtext", "torchvision", "torchviz", "pytorch-accelerated", "pytorch-lightning", "pytorchcv", "pytorch-widedeep", "segmentation-models-pytorch", "segmentation-models-pytorch-deepflash2", "d3m-segmentation-models-pytorch", "efficientnet-pytorch", "pretrainedmodels", "timm", "resnest", "rcnnmasks", "monai", "fastai", "fastai2", "recbole", "eh-pytorch-tabnet", "pytorch-tabnet2", "tabpfn",
    # TensorFlow / Keras stack
    "tensorflow", "tensorflow-addons", "tensorflow-decision-forests", "tensorflow-datasets", "tensorflow-hub", "tensorflow-io", "tensorflow-recommenders", "tensorflow-text", "tf-datasets", "tflearn", "keras", "keras-nlp", "keras-tuner", "Keras-Applications", "Keras-Preprocessing", "autokeras", "efficientnet", "spektral",
    
    "pymc3", "ludwig", "implicit"]

    optional_gpu_apis = ["Theano", "xgboost", "lightgbm", "catboost", "autogluon", "ludwig", "numba", "implicit", "numba"]

    # gpu_keywords = ["device", "cude", "get_gpu_device_count", "task_type='GPU'", "task_type=\"GPU\"", "'num_gpus': 1", '"num_gpus": 1']
    gpu_patterns = [
        re.compile(r"\bdevice\b", re.IGNORECASE),
        re.compile(r"cuda"),
        re.compile(r"get_gpu_device_count"),
        re.compile(r"cuda.is_available\("),
        re.compile(r"task_type\s*=\s*['\"]\s*GPU\s*['\"]", re.IGNORECASE),
        re.compile(r"['\"]\s*num_gpus\s*['\"]\s*:\s*1\b", re.IGNORECASE),
    ]
    # kernel_meta_by_submission = {}
    # for _competition, _kernels in kernel_meta.items():
    #     for _submission_id, _meta in _kernels.items():
    #         kernel_meta_by_submission.setdefault(_submission_id, _meta)

    # sample_files = [f for f in os.listdir('./baseline/notebooks') if f.endswith('.ipynb')]
    # no_gpu_required = set()
    # print(f"Total {len(sample_files)} notebooks to check for GPU requirement")
    # for submission in sample_files:
    #     submission_id = Path(submission).stem
    #     submission_meta = kernel_meta_by_submission.get(submission_id)
    #     if submission_meta is None:
    #         continue
    #     used_apis = submission_meta.get('api', [])
    #     used_lower = {a.lower() for a in used_apis}

    #     try:
    #         nb = nbformat.read(Path('./baseline/notebooks/'+submission), as_version=4)
    #         nb_text = "\n".join(
    #             cell.get("source", "")
    #             for cell in nb.cells
    #             if cell.get("cell_type") == "code"
    #         ).lower()
    #     except Exception:
    #         nb_text = ""

    #     if any(p.search(nb_text) for p in gpu_patterns):
    #         continue
    #     # Case A: notebook does NOT use any gpu_required APIs 
    #     # AND
    #     # ((notebook does NOT use any optional_gpu_apis APIs) or (notebook uses any optional_gpu_apis APIs but notebook content has no gpu keywords))
    #     if (not any(api.lower() in used_lower for api in gpu_required)) and ((not any(api.lower() in used_lower for api in optional_gpu_apis)) or (any(api.lower() in used_lower for api in optional_gpu_apis) and not any(p.search(nb_text) for p in gpu_patterns))):
    #         no_gpu_required.add(submission_id)
    #         continue

    #     # Case B: notebook does NOT use any gpu_required APIs 
    #     # AND
    #     # notebook uses any optional_gpu_apis but notebook content has no gpu keywords -> treat as no GPU
    #     if (not any(api.lower() in used_lower for api in gpu_required)) and (any(api.lower() in used_lower for api in optional_gpu_apis)):
    #         if not any(p.search(nb_text) for p in gpu_patterns):
    #             no_gpu_required.add(submission_id)
    #             continue

    # print(f"Total {len(no_gpu_required)} notebooks do NOT require GPU")
    # return no_gpu_required

    no_gpu_required = set()
    # full_files = [f for f in os.listdir('./baseline/notebooks') if f.endswith('.ipynb')]
    # done_already = [f for f in os.listdir("./baseline/notebooks_2") if f.endswith('.ipynb')]
    # sample_files = [x for x in full_files if x not in done_already]
    sample_files = [f for f in os.listdir('./baseline/notebooks') if f.endswith('.ipynb')]
    print(f"Total {len(sample_files)} notebooks to check for GPU requirement")
    for entity in sample_files:
        compt = entity.split("_")[0]
        submission_id = "_".join(Path(entity).stem.split("_")[1:]) + ".ipynb"
        used_apis = kernel_meta[compt][submission_id]['api']
        used_lower = {a.lower() for a in used_apis}

        nb_path = Path(f'./baseline/notebooks/{entity}')
        try:
            nb_text = nb_path.read_text(encoding='utf-8').lower()
        except Exception:
            nb_text = ""

        if any(p.search(nb_text) for p in gpu_patterns):
            continue
        # Case A: notebook does NOT use any gpu_required APIs 
        # AND
        # ((notebook does NOT use any optional_gpu_apis APIs) or (notebook uses any optional_gpu_apis APIs but notebook content has no gpu keywords))
        if (not any(api.lower() in used_lower for api in gpu_required)) and ((not any(api.lower() in used_lower for api in optional_gpu_apis)) or (any(api.lower() in used_lower for api in optional_gpu_apis) and not any(p.search(nb_text) for p in gpu_patterns))):
            no_gpu_required.add(entity)
            continue

        # Case B: notebook does NOT use any gpu_required APIs 
        # AND
        # notebook uses any optional_gpu_apis but notebook content has no gpu keywords -> treat as no GPU
        if (not any(api.lower() in used_lower for api in gpu_required)) and (any(api.lower() in used_lower for api in optional_gpu_apis)):
            if not any(p.search(nb_text) for p in gpu_patterns):
                no_gpu_required.add(entity)
                continue

    print(f"Total {len(no_gpu_required)} notebooks do NOT require GPU")
    return no_gpu_required