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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8360531882744031

# 6. Current score

0.20404

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'Your code didn’t yield a score mainly because it likely didn’t run end-to-end on Kaggle: it imports TensorFlow (unneeded) and references two external model checkpoints that are not in the provided data paths, so the notebook error before writing `submission.csv`. To keep the core inference logic the same (single model, argmax class), I add a safe fallback that uses torchvision’s built-in pretrained weights when the checkpoint file is missing, ensuring a valid submission is always produced. I also ensure `image_id` order matches `sample_submission.csv` (prevents accidental misalignment due to `os.listdir()` ordering), which can materially improve accuracy without changing the model. Finally, I remove the unused TensorFlow/Keras imports to avoid environment/package issues.'
- What this solution (achieved 0.18311) has done: 'I fix the immediate runtime errors so the notebook runs end-to-end and always writes a valid `submission.csv`. First, I correct the pretrained-weight fallback for ViT by loading only non-head weights (the previous `strict=False` still errors on size-mismatched classifier params). Second, I fix the CUDA/CPU dtype/device mismatch by ensuring any fallback model used for weight loading is created on CPU and then moving the final model to `device` after weights are loaded. These changes are minimal, preserve your core inference logic (single model + argmax), and should substantially improve score versus the broken fallback (which was failing before producing a proper submission).'
- What this solution (achieved 0.39163) has done: 'Your current score (0.183) is far below the target (0.836), so we need a real accuracy lift while keeping the same single-model argmax inference core. The biggest issue is that your ViT inference is almost certainly running with the fallback ImageNet weights (since the referenced checkpoints aren’t in the provided data paths), and the custom “invert_square_pad” transform is likely harming accuracy because it destroys natural spatial structure. I keep the exact model + argmax logic, but (1) automatically fall back to EfficientNetV2-L (ImageNet weights) if the cassava ViT checkpoint is missing, and (2) replace the destructive padding transform with a standard reflect-pad-to-square (no quadrant inversion) while keeping the rest of your preprocessing unchanged. These are minimal changes that should move your score substantially toward the target without changing the overall approach.'
- What this solution (achieved 0.09865) has done: 'Your score gap is large (0.39163 vs target 0.83605), so we need a meaningful accuracy lift while keeping your single-model argmax inference core unchanged. The biggest blocker is that you’re almost certainly running an ImageNet-pretrained fallback (not cassava-trained), and ImageNet normalization/resize behavior for EfficientNetV2-L is better matched by using the official torchvision weight transforms. I keep the same model selection + preprocessing structure, but when falling back to ImageNet EfficientNetV2-L I swap the normalization/resize pipeline to `EfficientNet_V2_L_Weights.DEFAULT.transforms()` (still with your reflect-square-pad first), and I also add `torch.inference_mode()` + autocast on CUDA to stabilize outputs and speed up without changing semantics. This is minimal, runs end-to-end, and should move accuracy substantially toward your target when checkpoints are missing.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.09865) is far below the target (0.83605), so we need a real accuracy lift while keeping the same single-model argmax inference. The largest issue is that your fallback path uses an ImageNet-pretrained EfficientNetV2-L with a randomly initialized 5-class head, which makes predictions nearly random for Cassava. The smallest change that keeps your exact inference semantics is to use a torchvision model that already ships with 5-class weights for Cassava (EfficientNet-B4 from `Cassava_Weights`) when your external checkpoints are missing, and to use its official preprocessing transforms. This should move accuracy strongly toward your target without changing the overall approach (single model + preprocessing + argmax + submission alignment).'
- What this solution (achieved 0.0) has done: 'I fix the immediate runtime error caused by `EfficientNet_B4_Weights.CASSAVA` not existing in your installed torchvision, by adding a robust fallback that keeps the same single-model + argmax inference flow. To move accuracy toward the target when your external ViT checkpoint is missing, the fallback use the best available built-in ImageNet weights (EfficientNetV2-L preferred, else EfficientNet-B4) and correctly swap in the official torchvision preprocessing transforms for those weights. I also ensure `val_transforms` is always defined and that model selection results in exactly one model being called per image (to avoid accidental overwrites). These are minimal, execution-blocking fixes plus a legitimate accuracy nudge without changing your overall approach.'
- What this solution (achieved 0.16629) has done: 'Your current 0.0 score is consistent with a submission that is valid CSV but has a nearly-always-wrong label space: when your cassava checkpoint is missing you fall back to an ImageNet model that outputs 1000 classes, yet you still submit `argmax` as if it were in 0–4, which destroys accuracy. I keep the same single-model + transforms + argmax inference core, but make the fallback produce 5-class logits by swapping in a 5-class EfficientNet-V2-L head and loading only compatible ImageNet backbone weights (head left random). I also keep your submission alignment to `sample_submission.csv` unchanged so ordering can’t hurt accuracy. This should move you materially upward from 0.0 toward your target without changing the overall approach.'
- What this solution (achieved 0.10613) has done: 'Your current score is far below the target, and the main reason is that when the cassava-trained checkpoint is missing you’re effectively using an ImageNet backbone with a random 5-class head, which can’t reach the target accuracy. The smallest legitimate change that keeps your exact single-model + transforms + argmax inference semantics is to add a robust “cassava pretrained weights” fallback from torchvision (`resnet50`/`resnext50_32x4d` with `ResNet50_Weights` / `ResNeXt50_32X4D_Weights`) when available; these weights were released for this competition and typically jump accuracy massively. I also keep your submission ordering anchored to `sample_submission.csv` and preserve your existing ViT checkpoint path behavior (no architecture/training loop changes). If cassava weights are not present in the installed torchvision, the code falls back to your current ImageNet EfficientNetV2-L path unchanged.'
- What this solution (achieved 0.11547) has done: 'Your score is far below the target, and the biggest “minimal” lever is to ensure you actually use Cassava-trained weights when available, because ImageNet-backbone + random 5-class head can’t approach 0.83. I keep your single-model + preprocessing + argmax inference flow unchanged, but (1) make the Cassava-pretrained torchvision fallback more robust by checking multiple likely weight enum names/locations (so it works across torchvision versions), and (2) if Cassava weights are found, force the model output to exactly 5 classes by replacing the classifier head (some weight variants ship with different head shapes). I also keep submission ordering anchored to `sample_submission.csv` and preserve your padding+weight-transforms pipeline so evaluation semantics stay the same.'
- What this solution (achieved 0.20404) has done: 'Your current score is far below the target, so we need a real accuracy lift with minimal changes while keeping the same single-model + preprocessing + argmax inference semantics. The biggest correctness bug hurting accuracy is in the `torchvision_cassava_auto` branch: you replace the `fc` head *after* loading Cassava weights, which discards the Cassava-trained classifier and makes predictions near-random. I keep the exact same model selection flow, but ensure we preserve the Cassava 5-class head when Cassava weights are available (and only resize/replace the head if it truly doesn’t match 5 classes). I also remove mixed-precision autocast for inference to avoid small-but-real accuracy regressions on some models; this is a minimal change that typically improves stability/accuracy (at the cost of speed) and still finishes within the time limit.'

# 9. Code solution

## === cell 0
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
import os



## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/3/efficientnetv2_l_480_8450.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/5/vit_h_14_518_8369_base.pth"
)
vit_image_size = 518


def _file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def _try_get_torchvision_cassava_weights():
    """
    Returns (arch, weights_enum) where arch is "resnet50" or "resnext50_32x4d",
    or (None, None) if not available in this torchvision build.
    """
    candidates = []

    candidates.append(("resnet50", "ResNet50_Weights", "CASSAVA"))
    candidates.append(("resnext50_32x4d", "ResNeXt50_32X4D_Weights", "CASSAVA"))

    candidates.append(("resnet50", "ResNet50_Weights", "CASSAVA_V1"))
    candidates.append(("resnext50_32x4d", "ResNeXt50_32X4D_Weights", "CASSAVA_V1"))

    for arch, enum_name, member in candidates:
        enum_obj = getattr(models, enum_name, None)
        if enum_obj is None:
            continue
        w = getattr(enum_obj, member, None)
        if w is not None:
            return arch, w
    return None, None


model_select = "vit"
if model_select == "vit" and (not _file_exists(vit_model_path)):
    print(
        f"ViT checkpoint not found at {vit_model_path}. "
        f"Trying Cassava-pretrained torchvision fallback; otherwise switching to ImageNet fallback."
    )
    model_select = "torchvision_cassava_auto"

if model_select == "vit":
    model_image_size = vit_image_size
elif model_select == "en":
    model_image_size = en_image_size
elif model_select == "imagenet_en_v2_l":
    model_image_size = 480
elif model_select == "imagenet_b4":
    model_image_size = 380
elif model_select == "torchvision_cassava_auto":
    model_image_size = 224
else:
    raise ValueError(f"Unknown model_select={model_select}")




## === cell 2
def square_pad_reflect(img: Image.Image) -> Image.Image:
    width, height = img.size
    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,  # left
        (max_side - height) // 2,  # top
        (max_side - width) - (max_side - width) // 2,  # right
        (max_side - height) - (max_side - height) // 2,  # bottom
    )
    return transforms.functional.pad(img, padding, padding_mode="reflect")




## === cell 3
if model_select == "torchvision_cassava_auto":
    arch, _w = _try_get_torchvision_cassava_weights()
    if _w is None:
        print(
            "No torchvision Cassava weights found. Falling back to ImageNet EfficientNetV2-L."
        )
        model_select = "imagenet_en_v2_l"
    else:
        cassava_arch = arch
        val_transforms = transforms.Compose(
            [transforms.Lambda(square_pad_reflect), _w.transforms()]
        )

if model_select == "imagenet_en_v2_l":
    try:
        _w = models.EfficientNet_V2_L_Weights.DEFAULT
    except Exception:
        _w = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1

    val_transforms = transforms.Compose(
        [
            transforms.Lambda(square_pad_reflect),
            _w.transforms(),
        ]
    )
elif model_select == "imagenet_b4":
    try:
        _w = models.EfficientNet_B4_Weights.DEFAULT
    except Exception:
        _w = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    val_transforms = transforms.Compose(
        [
            transforms.Lambda(square_pad_reflect),
            _w.transforms(),
        ]
    )
elif model_select in ("vit", "en"):
    val_transforms = transforms.Compose(
        [
            transforms.Lambda(square_pad_reflect),
            transforms.ToTensor(),
            transforms.Resize((model_image_size, model_image_size)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
elif model_select == "torchvision_cassava_auto":
    pass
else:
    raise ValueError(f"Unknown model_select={model_select}")




## === cell 4
def _load_state_dict_skip_mismatch(model: torch.nn.Module, state_dict: dict) -> None:
    """
    Why: strict=False does NOT ignore size mismatches for parameters that exist in both
    model and checkpoint. We filter out mismatched keys (e.g., classifier head) so we can
    still reuse the pretrained backbone.
    """
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if v.shape == model_sd[k].shape:
                filtered[k] = v
    model.load_state_dict(filtered, strict=False)


if model_select == "vit":
    vit_model = models.vit_h_14(weights=None, image_size=518)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )

    try:
        ckpt = torch.load(vit_model_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(vit_model_path, map_location="cpu")
    vit_model.load_state_dict(ckpt)
    using = f"loaded checkpoint: {vit_model_path}"

    vit_model.to(device)
    vit_model.eval()
    print(f"ViT ready ({using}).")

elif model_select == "en":
    en_model = models.efficientnet_v2_l(weights=None)
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )

    if _file_exists(en_model_path):
        try:
            ckpt = torch.load(en_model_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(en_model_path, map_location="cpu")
        en_model.load_state_dict(ckpt)
        using = f"loaded checkpoint: {en_model_path}"
    else:
        try:
            en_imagenet = models.efficientnet_v2_l(
                weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1
            )
        except Exception:
            en_imagenet = models.efficientnet_v2_l(
                weights=models.EfficientNet_V2_L_Weights.DEFAULT
            )

        _load_state_dict_skip_mismatch(en_model, en_imagenet.state_dict())
        using = "fallback torchvision pretrained EfficientNetV2-L weights (checkpoint missing; head skipped)"
        del en_imagenet

    en_model.to(device)
    en_model.eval()
    print(f"EfficientNetV2-L ready ({using}).")

elif model_select == "torchvision_cassava_auto":
    arch, _w = _try_get_torchvision_cassava_weights()
    if _w is None:
        raise RuntimeError(
            "model_select=torchvision_cassava_auto but cassava weights disappeared."
        )

    if cassava_arch == "resnet50":
        fallback_model = models.resnet50(weights=_w)
        if getattr(fallback_model, "fc", None) is not None:
            if fallback_model.fc.out_features != num_classes:
                fallback_model.fc = torch.nn.Linear(
                    fallback_model.fc.in_features, num_classes
                )
    elif cassava_arch == "resnext50_32x4d":
        fallback_model = models.resnext50_32x4d(weights=_w)
        if getattr(fallback_model, "fc", None) is not None:
            if fallback_model.fc.out_features != num_classes:
                fallback_model.fc = torch.nn.Linear(
                    fallback_model.fc.in_features, num_classes
                )
    else:
        raise ValueError(f"Unknown cassava_arch={cassava_arch}")

    fallback_model.to(device)
    fallback_model.eval()
    print(f"Fallback model ready (torchvision {cassava_arch} Cassava weights).")

elif model_select == "imagenet_en_v2_l":
    try:
        _w = models.EfficientNet_V2_L_Weights.DEFAULT
    except Exception:
        _w = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1

    fallback_model = models.efficientnet_v2_l(weights=None)
    fallback_model.classifier[1] = torch.nn.Linear(
        fallback_model.classifier[1].in_features, num_classes
    )
    imagenet_model = models.efficientnet_v2_l(weights=_w)
    _load_state_dict_skip_mismatch(fallback_model, imagenet_model.state_dict())
    del imagenet_model

    fallback_model.to(device)
    fallback_model.eval()
    print(
        "Fallback model ready (EfficientNetV2-L ImageNet backbone weights; 5-class head)."
    )

elif model_select == "imagenet_b4":
    try:
        _w = models.EfficientNet_B4_Weights.DEFAULT
    except Exception:
        _w = models.EfficientNet_B4_Weights.IMAGENET1K_V1

    fallback_model = models.efficientnet_b4(weights=None)
    fallback_model.classifier[1] = torch.nn.Linear(
        fallback_model.classifier[1].in_features, num_classes
    )
    imagenet_model = models.efficientnet_b4(weights=_w)
    _load_state_dict_skip_mismatch(fallback_model, imagenet_model.state_dict())
    del imagenet_model

    fallback_model.to(device)
    fallback_model.eval()
    print(
        "Fallback model ready (EfficientNet-B4 ImageNet backbone weights; 5-class head)."
    )

else:
    raise ValueError(f"Unknown model_select={model_select}")



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

predictions = []
image_ids = []

use_amp = False

for image_name in tqdm(test_image_ids, desc="Test"):
    image_path = os.path.join(test_data_directory, image_name)

    image = Image.open(image_path).convert("RGB")
    transformed_image = val_transforms(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if model_select == "vit":
                    out = vit_model(transformed_image)
                elif model_select == "en":
                    out = en_model(transformed_image)
                else:
                    out = fallback_model(transformed_image)
        else:
            if model_select == "vit":
                out = vit_model(transformed_image)
            elif model_select == "en":
                out = en_model(transformed_image)
            else:
                out = fallback_model(transformed_image)

        predicted_class = torch.argmax(out, dim=1)

    predictions.append(int(predicted_class.item()))
    image_ids.append(image_name)



## === cell 6
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

submission_df = submission_df.merge(
    sample_sub[["image_id"]], on="image_id", how="right"
)
submission_df["label"] = submission_df["label"].fillna(0).astype(int)
submission_df.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")
print(submission_df.head())
print(f"Rows: {len(submission_df)} (expected {len(sample_sub)})")
