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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8949546090873821

# 6. Current score

-0.04397

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0763) has done: 'I fix the runtime by making the model weight loading robust to missing Kaggle input files: if the ensemble checkpoints aren’t available, the code fall back to a single pretrained timm model so inference can still run end-to-end. I also fix the ensemble weighting logic so weights are computed only for actually loaded models, preventing the empty `torch.cat()` crash. To preserve your core inference semantics, the prediction still uses softmax probabilities and argmax over 5 classes, just with a safe fallback path. Finally, I ensure the submission CSV is always written with the required columns and correct row alignment to `test.csv`.'
- What this solution (achieved 0.0748) has done: 'Your low score is almost certainly because the fallback path uses an ImageNet-pretrained model with a randomly initialized 5-class head, so predictions are essentially noise. To move the score toward the target with minimal core-logic change, I keep your exact inference pipeline (resize/normalize, softmax, weighted averaging, argmax) but make the fallback choose a timm model that already has pretrained weights for this exact competition (`hf_hub` APTOS fine-tuned weights), which is still “pretrained weights + num_classes=5” and doesn’t change architecture or evaluation semantics. I also add a tiny robustness tweak to handle common checkpoint formats (`state_dict` key) without changing behavior when your original ensemble files exist. This should lift QWK substantially toward your target while keeping runtime and code structure nearly identical.'
- What this solution (achieved 0.02995) has done: 'Your score is far below the target, and the most likely cause is that the current fallback model isn’t actually loading APTOS-finetuned weights (HF hub access is typically unavailable in Kaggle offline), so the 5-class head remains randomly initialized and predictions are near-random. To move the score toward the target with minimal change and without altering your inference semantics (resize/normalize → model → softmax → weighted average → argmax), I make the fallback load a locally-available APTOS-finetuned checkpoint from the Kaggle dataset if present, and only then fall back to the HF-hub attempt. I also make checkpoint loading slightly more robust to common key prefixes (`module.`, `model.`), which can otherwise silently prevent using the good weights. This should substantially increase QWK toward your target while keeping the architecture and prediction pipeline the same.'
- What this solution (achieved 0.25963) has done: 'Your current score indicates the model is almost surely running with an untrained/random 5-class head because the intended ensemble checkpoints aren’t being found/loaded (path typo variants) and the HF hub APTOS weights are offline, so predictions are near-random. To move toward the target with minimal change, I (1) robustly discover and load any available local ensemble checkpoints by searching the `/kaggle/input` directory for the expected `.pth` files, (2) make weight loading tolerant to common head key mismatches (so it can still load the backbone and any matching head weights instead of failing silently into the bad fallback), and (3) keep your exact inference semantics (224 resize/normalize → softmax → weighted average → argmax) and submission formatting unchanged.'
- What this solution (achieved -0.14704) has done: 'Your current score (0.25963) is far below the target, so we should improve it with the smallest change that directly affects predictions while keeping your inference pipeline (resize/normalize → model → softmax → weighted average → argmax) identical. The most likely reason for near-random performance is that your checkpoint discovery uses misspelled filenames (`efficentNet_...`) and therefore fails to load the intended fine-tuned weights, falling back to an ImageNet-pretrained model with a random 5-class head. I fix checkpoint discovery to search for both the misspelled and correctly spelled variants (plus a couple common extensions) and also broaden the search to accept “contains” matches (case-insensitive), so your intended local checkpoints are actually found. This should move QWK substantially upward toward the target without changing model architectures, transforms, ensembling math, or post-processing.'
- What this solution (achieved -0.03428) has done: 'Your very low QWK indicates the intended fine-tuned checkpoints still aren’t being loaded, so the pipeline falls back to an ImageNet model with a randomly initialized 5-class head (near-random predictions). I make the checkpoint discovery broader (recursive, case-insensitive, and not dependent on exact basenames) and also make the model selection infer the architecture from the found filename so we instantiate the correct `timm` model before loading weights. I also fix a key mismatch in your weighting dict: when the fallback model key is a full timm name (`tf_efficientnet_b3_ns`) it won’t match the existing `validation_scores` keys unless we align them, which can currently distort the ensemble. These changes keep your exact inference semantics (resize/normalize → model → softmax → weighted average → argmax) but should move the score sharply upward toward your target by actually using the fine-tuned weights.'
- What this solution (achieved -0.05826) has done: 'Your current score is far below the target, which strongly suggests you’re still not loading any real APTOS-finetuned 5-class checkpoints and are effectively predicting with an ImageNet backbone plus a random 5-class head. To move the score toward the target with minimal changes and identical inference semantics (resize/normalize → model → softmax → weighted average → argmax), I (1) broaden checkpoint discovery to include Kaggle *working* and *data* directories (not just `/kaggle/input`), (2) prioritize loading any found APTOS/DR checkpoints while making key-cleaning tolerant of common prefixes, and (3) fix the weight-key alignment so inferred model names used during loading always match the ensemble-weight dictionary. These changes should materially improve QWK by ensuring you actually use the intended finetuned weights instead of the random-head fallback, without changing the model forward pass, transforms, or post-processing.'
- What this solution (achieved -0.01328) has done: 'Your current QWK (-0.058) strongly suggests the code is still falling back to an ImageNet-pretrained model with a randomly initialized 5-class head, producing near-random labels. The smallest change that should move the score materially toward your target (without changing your inference semantics: resize/normalize → logits → softmax → weighted average → argmax) is to make checkpoint discovery actually find *any* `.pth/.pt/.bin` in your available folders, then try loading it by instantiating a small set of likely timm architectures until one loads with reasonable key coverage. I also make the ensemble weight keys consistent by always using the actual timm model name for `loaded_model_keys`, so weighting never misaligns. If no usable checkpoint exists locally, the fallback behavior remains the same as you have now, ensuring the notebook still completes and writes a valid `submission.csv`.'
- What this solution (achieved -0.04397) has done: 'Your current negative QWK strongly indicates you’re still not loading any APTOS-finetuned 5-class weights and are effectively predicting with an ImageNet backbone plus a random classification head. With minimal change and without altering your inference semantics (resize/normalize → logits → softmax → weighted average → argmax), I add a robust “head-adaptation” loader: when a checkpoint contains a 5-class classifier head under a different key name (common in timm checkpoints), we remap those keys to the instantiated model’s head so the trained classifier actually gets loaded. I also prefer architectures that match your loaded checkpoint filenames (e.g., `tf_efficientnet_b3_ns` for b3-ns checkpoints) so the correct model is instantiated before loading, but keep your ensemble logic intact. These changes should materially increase QWK toward the target while keeping the model forward pass, transforms, and post-processing unchanged and still producing `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm




## === cell 1
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 2
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 3
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"

test_df = pd.read_csv(test_csv_file)
test_dataset = BlindnessDataset(
    test_csv_file, test_root_dir, transform=transform, test=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 2 if torch.cuda.is_available() else 0
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 4
SEARCH_ROOTS = [
    "/kaggle/input",
    "/kaggle/working",
    "/kaggle/data",
]


def _iter_search_roots(search_root: str | None):
    if search_root is not None:
        yield search_root
    else:
        for r in SEARCH_ROOTS:
            if os.path.exists(r):
                yield r


def _find_checkpoint(
    basename: str,
    search_root: str | None = None,
    allow_contains: bool = True,
) -> str | None:
    target = basename.lower()

    for sr in _iter_search_roots(search_root):
        for root, _, files in os.walk(sr):
            for f in files:
                fl = f.lower()
                if fl == target:
                    return os.path.join(root, f)

    if allow_contains:
        for sr in _iter_search_roots(search_root):
            for root, _, files in os.walk(sr):
                for f in files:
                    fl = f.lower()
                    if target in fl:
                        return os.path.join(root, f)

    return None


def _find_any_checkpoint(
    basenames: list[str], search_root: str | None = None
) -> str | None:
    for b in basenames:
        p = _find_checkpoint(b, search_root=search_root, allow_contains=False)
        if p:
            return p
    for b in basenames:
        p = _find_checkpoint(b, search_root=search_root, allow_contains=True)
        if p:
            return p
    return None


def _ckpt_candidates(stem: str) -> list[str]:
    cands = []
    for ext in [".pth", ".pt", ".bin"]:
        cands.append(f"{stem}{ext}")
    return cands


def _find_checkpoints_by_keywords(
    keywords: list[str],
    search_root: str | None = None,
    exts: tuple[str, ...] = (".pth", ".pt", ".bin"),
) -> list[str]:
    kws = [k.lower() for k in keywords]
    out = []
    for sr in _iter_search_roots(search_root):
        for root, _, files in os.walk(sr):
            for f in files:
                fl = f.lower()
                if not fl.endswith(exts):
                    continue
                if any(k in fl for k in kws):
                    out.append(os.path.join(root, f))
    out = sorted(set(out))
    return out


def _infer_model_name_from_path(p: str) -> str | None:
    b = os.path.basename(p).lower()
    if "seresnext101_32x4d" in b or "se_resnext101_32x4d" in b:
        return "seresnext101_32x4d"
    if "seresnext50_32x4d" in b or "se_resnext50_32x4d" in b:
        return "seresnext50_32x4d"
    if "inception_resnet_v2" in b:
        return "inception_resnet_v2"
    if "inception_v4" in b:
        return "inception_v4"
    if "resnet18" in b:
        return "resnet18"
    if "tf_efficientnet_b3_ns" in b:
        return "tf_efficientnet_b3_ns"
    if (
        "b3" in b
        and ("efficientnet" in b or "efficentnet" in b)
        and ("_ns" in b or "tf_" in b)
    ):
        return "tf_efficientnet_b3_ns"
    if "efficientnet" in b or "efficentnet" in b:
        for n in ["b5", "b4", "b3", "b2", "b1", "b0"]:
            if n in b:
                return f"efficientnet_{n}"
    return None


model_paths = {
    "efficientnet_b1": _find_any_checkpoint(
        _ckpt_candidates("efficentNet_b1")
        + _ckpt_candidates("efficientNet_b1")
        + _ckpt_candidates("efficientnet_b1")
    ),
    "efficientnet_b2": _find_any_checkpoint(
        _ckpt_candidates("efficentNet_b2")
        + _ckpt_candidates("efficientNet_b2")
        + _ckpt_candidates("efficientnet_b2")
    ),
    "efficientnet_b3": _find_any_checkpoint(
        _ckpt_candidates("efficentNet_b3")
        + _ckpt_candidates("efficientNet_b3")
        + _ckpt_candidates("efficientnet_b3")
    ),
    "seresnext101_32x4d": _find_any_checkpoint(_ckpt_candidates("seresnext101_32x4d")),
}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
    "tf_efficientnet_b3_ns": "tf_efficientnet_b3_ns",
}


def _extract_state_dict(maybe_ckpt):
    if isinstance(maybe_ckpt, dict):
        if "state_dict" in maybe_ckpt and isinstance(maybe_ckpt["state_dict"], dict):
            return maybe_ckpt["state_dict"]
        if "model" in maybe_ckpt and isinstance(maybe_ckpt["model"], dict):
            return maybe_ckpt["model"]
    return maybe_ckpt


def _clean_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


def _remap_classifier_keys_to_model(state: dict, model: nn.Module) -> dict:
    if not isinstance(state, dict):
        return state

    try:
        model_sd = model.state_dict()
        cls_name = model.get_classifier()
    except Exception:
        return state

    if not isinstance(cls_name, str) or cls_name == "":
        return state

    cls_w = f"{cls_name}.weight"
    cls_b = f"{cls_name}.bias"
    if cls_w not in model_sd:
        return state

    target_shapes = {
        cls_w: tuple(model_sd[cls_w].shape),
        cls_b: tuple(model_sd[cls_b].shape) if cls_b in model_sd else None,
    }

    src_w_keys = [
        "classifier.weight",
        "fc.weight",
        "head.weight",
        "last_linear.weight",
    ]
    src_b_keys = [
        "classifier.bias",
        "fc.bias",
        "head.bias",
        "last_linear.bias",
    ]

    out = dict(state)

    def _find_matching_key(cands, shape):
        for k in cands:
            if k in out and hasattr(out[k], "shape") and tuple(out[k].shape) == shape:
                return k
        return None

    sw = _find_matching_key(src_w_keys, target_shapes[cls_w])
    if sw is not None and cls_w not in out:
        out[cls_w] = out[sw]

    if target_shapes[cls_b] is not None:
        sb = _find_matching_key(src_b_keys, target_shapes[cls_b])
        if sb is not None and cls_b not in out:
            out[cls_b] = out[sb]

    return out


def _try_load_weights(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = _clean_state_dict_keys(_extract_state_dict(ckpt))

    if isinstance(state, dict):
        state = _remap_classifier_keys_to_model(state, model)

    try:
        model.load_state_dict(state, strict=True)
        return model
    except RuntimeError as e:
        missing, unexpected = model.load_state_dict(state, strict=False)
        n_params = len(list(model.state_dict().keys()))
        if len(missing) > 0.6 * n_params:
            raise RuntimeError(
                f"Too many missing keys when loading {path}: {len(missing)}/{n_params}. Original error: {e}"
            )
        return model


def _try_load_with_arch_guesses(
    path: str, arch_candidates: list[str]
) -> tuple[str, nn.Module] | tuple[None, None]:
    last_err = None
    for arch in arch_candidates:
        try:
            m = timm.create_model(arch, pretrained=False, num_classes=5)
            m = _try_load_weights(m, path)
            return arch, m
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        print(f"Exhausted arch guesses for {path}. Last error: {repr(last_err)}")
    return None, None


models_list = []
loaded_model_keys = []

for model_key, path in model_paths.items():
    if not path or (not os.path.exists(path)):
        continue

    inferred = _infer_model_name_from_path(path)
    model_name = inferred if inferred is not None else model_names[model_key]

    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    try:
        model = _try_load_weights(model, path)
        model.to(device)
        model.eval()
        models_list.append(model)
        loaded_model_keys.append(model_name)
        print(f"Loaded checkpoint for {model_name} from: {path}")
    except Exception as e:
        print(f"Failed loading {model_name} from {path}: {repr(e)}")

if len(models_list) == 0:
    keyword_hits = _find_checkpoints_by_keywords(
        keywords=[
            "aptos",
            "blindness",
            "diabetic",
            "retinopathy",
            "efficentnet",
            "efficientnet",
            "seresnext101_32x4d",
            "se_resnext101_32x4d",
            "seresnext50_32x4d",
            "se_resnext50_32x4d",
            "inception_resnet_v2",
            "inception_v4",
            "tf_efficientnet_b3_ns",
        ],
        search_root=None,
    )

    loaded = False
    for p in keyword_hits:
        inferred = _infer_model_name_from_path(p)
        if inferred is None:
            continue
        model = timm.create_model(inferred, pretrained=False, num_classes=5)
        try:
            model = _try_load_weights(model, p)
            model.to(device).eval()
            loaded_model_keys = [inferred]
            models_list = [model]
            loaded = True
            print("Loaded keyword-discovered checkpoint:", p, "as model:", inferred)
            break
        except Exception as e:
            print("Failed loading keyword-discovered checkpoint:", p, "Error:", repr(e))

    if not loaded:
        local_fallback_candidates = []
        for b in (
            _ckpt_candidates("efficentNet_b3")
            + _ckpt_candidates("efficientNet_b3")
            + _ckpt_candidates("efficientnet_b3")
            + _ckpt_candidates("efficentNet_b2")
            + _ckpt_candidates("efficientNet_b2")
            + _ckpt_candidates("efficientnet_b2")
            + _ckpt_candidates("efficentNet_b1")
            + _ckpt_candidates("efficientNet_b1")
            + _ckpt_candidates("efficientnet_b1")
            + _ckpt_candidates("seresnext101_32x4d")
        ):
            p = _find_checkpoint(b, search_root=None)
            if p:
                local_fallback_candidates.append(p)

        for p in local_fallback_candidates:
            inferred = _infer_model_name_from_path(p)
            if inferred is None:
                base = os.path.basename(p).lower()
                if "b1" in base:
                    inferred = "efficientnet_b1"
                elif "b2" in base:
                    inferred = "efficientnet_b2"
                elif "b3" in base:
                    inferred = "efficientnet_b3"
                elif "seresnext101" in base:
                    inferred = "seresnext101_32x4d"
                elif "tf_efficientnet_b3_ns" in base:
                    inferred = "tf_efficientnet_b3_ns"
                else:
                    continue

            model = timm.create_model(inferred, pretrained=False, num_classes=5)
            try:
                model = _try_load_weights(model, p)
                model.to(device).eval()
                loaded_model_keys = [inferred]
                models_list = [model]
                loaded = True
                print("Loaded local fallback checkpoint:", p, "as model:", inferred)
                break
            except Exception as e:
                print("Failed loading local fallback checkpoint:", p, "Error:", repr(e))

    if not loaded:
        any_ckpts = []
        for sr in _iter_search_roots(None):
            for root, _, files in os.walk(sr):
                for f in files:
                    fl = f.lower()
                    if fl.endswith((".pth", ".pt", ".bin")):
                        any_ckpts.append(os.path.join(root, f))
        any_ckpts = sorted(set(any_ckpts))

        preferred = []
        other = []
        for p in any_ckpts:
            bl = os.path.basename(p).lower()
            if any(k in bl for k in ["aptos", "blind", "retina", "dr", "diabetic"]):
                preferred.append(p)
            else:
                other.append(p)

        candidate_paths = (preferred + other)[:40]

        arch_guesses = [
            "tf_efficientnet_b3_ns",
            "efficientnet_b3",
            "efficientnet_b2",
            "efficientnet_b1",
            "efficientnet_b0",
            "seresnext101_32x4d",
            "seresnext50_32x4d",
            "inception_resnet_v2",
            "inception_v4",
            "resnet18",
        ]

        for p in candidate_paths:
            inferred = _infer_model_name_from_path(p)
            if inferred is not None:
                arch_try = [inferred] + [a for a in arch_guesses if a != inferred]
            else:
                arch_try = arch_guesses

            arch, m = _try_load_with_arch_guesses(p, arch_try)
            if arch is not None:
                m.to(device).eval()
                loaded_model_keys = [arch]
                models_list = [m]
                loaded = True
                print("Loaded generic-discovered checkpoint:", p, "as model:", arch)
                break

    if not loaded:
        fallback_key = "tf_efficientnet_b3_ns"
        try:
            model = timm.create_model(
                fallback_key,
                pretrained=True,
                num_classes=5,
                pretrained_cfg_overlay={
                    "hf_hub_id": "timm/tf_efficientnet_b3_ns.aptos2019"
                },
            )
            loaded_model_keys = [fallback_key]
            models_list = [model.to(device).eval()]
            print("Loaded HF-hub APTOS finetuned fallback:", fallback_key)
        except Exception as e:
            print(
                "APTOS finetuned fallback unavailable, using ImageNet fallback. Error:",
                repr(e),
            )
            fallback_key = "efficientnet_b0"
            model = timm.create_model(fallback_key, pretrained=True, num_classes=5)
            loaded_model_keys = [fallback_key]
            models_list = [model.to(device).eval()]

print("Loaded models:", loaded_model_keys)



## === cell 5
validation_scores = {
    "resnet18": 0.887,
    "efficientnet_b0": 0.8922,
    "efficientnet_b1": 0.894,
    "efficientnet_b2": 0.898,
    "efficientnet_b3": 0.9127,
    "efficientnet_b4": 0.893,
    "efficientnet_b5": 0.870,
    "inception_resnet_v2": 0.896,
    "inception_v4": 0.8875,
    "seresnext50_32x4d": 0.8652,
    "seresnext101_32x4d": 0.9083,
    "tf_efficientnet_b3_ns": 0.91,
}

available_scores = {}
for k in loaded_model_keys:
    if k in validation_scores:
        available_scores[k] = validation_scores[k]
    else:
        available_scores[k] = 1.0

total_score = float(sum(available_scores.values()))
weights = {
    k: (v / total_score if total_score > 0 else 1.0 / len(available_scores))
    for k, v in available_scores.items()
}

print("Ensemble weights:", weights)



## === cell 6
all_outputs = []

with torch.no_grad():
    for images in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device, non_blocking=True)

        probs_list = []
        for model_key, model in zip(loaded_model_keys, models_list):
            logits = model(images)
            probs = nn.functional.softmax(logits, dim=1)
            probs_list.append(weights[model_key] * probs)

        weighted_outputs = torch.stack(probs_list, dim=0).sum(dim=0)
        all_outputs.append(weighted_outputs.detach().cpu().numpy())

all_outputs = np.concatenate(all_outputs, axis=0)
final_predictions = np.argmax(all_outputs, axis=1).astype(int)

assert len(final_predictions) == len(test_df), (len(final_predictions), len(test_df))



## === cell 7
submission_df = pd.DataFrame(
    {
        "id_code": test_df["id_code"].values,
        "diagnosis": final_predictions,
    }
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape=", submission_df.shape)
print(submission_df.head())
