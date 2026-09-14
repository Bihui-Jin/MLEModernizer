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

3.9

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.9228320167757752

# 6. Current score

0.03939

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.02261) has done: 'The fix adds a safe check for the missing `ThreeStage_Model` class so the inference loop no longer raises a `NameError`. The code now falls back to the standard timm model path when the custom class isn’t defined, allowing predictions to be generated and the submission DataFrame to be populated correctly.'
- What this solution (achieved 0.05346) has done: 'I add a safe placeholder definition for `ThreeStage_Model` so the `isinstance` check does not raise a `NameError`, and I replace that check with the existing `fallback` flag which already tells us whether the custom model was loaded. These minimal changes prevent the runtime error and ensure predictions are generated, producing a non‑empty submission CSV.'
- What this solution (achieved -0.04007) has done: 'I adjust the loading logic so that the provided fine‑tuned weights are applied to the EfficientNet model when the custom ThreeStage_Model is unavailable, and I use the standard ImageNet normalization (which matches the pretrained EfficientNet). These minimal changes keep the overall architecture unchanged while allowing the model to benefit from the fine‑tuned parameters, which should raise the Quadratic Weighted Kappa score toward the target.'
- What this solution (achieved 0.02007) has done: 'I replace the simple threshold‑based `regress2class` conversion with the ordinal probability mapping already defined (`ordinal2class_prob`). This uses the model’s four‑output logits to compute a proper class probability distribution and then picks the most likely class via `argmax`, which better matches the quadratic weighted kappa metric and should raise the score toward the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved -0.07358) has done: 'I load the fine‑tuned checkpoint with `strict=False` so any compatible weights are applied to the EfficientNet backbone, and simplify the inference loop to always use the model’s logits with an arg‑max prediction. This keeps the original architecture but gives the network a better starting point and removes the fragile tuple handling, moving the QWK score toward the target.'
- What this solution (achieved 0.26547) has done: 'I adjust the device selection to use GPU when available and change the prediction logic to use the soft‑max probabilities with an expected‑value rounding (which aligns better with the quadratic weighted kappa). This small, metric‑aware change should move the score upward toward the target without altering the core model architecture.'
- What this solution (achieved 0.26737) has done: 'I keep the core model and data pipeline unchanged, but ensure that any fine‑tuned weights are actually applied (handling the common “state_dict” nesting) and switch the final prediction from an expected‑value rounding to a straightforward argmax on the averaged soft‑max probabilities. This small, metric‑aware tweak should raise the quadratic weighted kappa toward the target while preserving the original architecture and inference flow.'
- What this solution (achieved 0.03939) has done: 'I replace the naïve arg‑max on the raw soft‑max probabilities with a metric‑aware conversion that uses the `ordinal2class_prob` helper when the model outputs four logits (the format expected from the fine‑tuned checkpoint). This aligns the predicted class distribution with the ordinal nature of the task and should raise the quadratic weighted kappa toward the target while keeping the core architecture unchanged.'

# 9. Code solution

## === cell 0
import random, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import timm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ThreeStage_Model(nn.Module):
    """Minimal stub used only when the real ThreeStage_Model is not provided."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError(
            "ThreeStage_Model placeholder does not implement forward."
        )




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=device)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)




## === cell 2
try:
    net = ThreeStage_Model()
    net.load_state_dict(
        torch.load(
            "../input/weights/B4_3stage_7epoch_finetune3.pkl", map_location=device
        )
    )
    net = net.to(device)
    fallback = False
except Exception:
    net = timm.create_model("tf_efficientnet_b4_ns", pretrained=True, num_classes=5)
    weight_path = "../input/weights/B4_3stage_7epoch_finetune3.pkl"
    if os.path.exists(weight_path):
        try:
            state = torch.load(weight_path, map_location=device)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            net.load_state_dict(state, strict=False)
            fallback = False  # fine‑tuned weights (as much as possible) loaded
        except Exception:
            fallback = True  # unable to load fine‑tuned weights
    else:
        fallback = True
    net = net.to(device)

net.eval()




## === cell 3
test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = np.squeeze(test_ids.values)

input_size = 380
transform = transforms.Compose(
    [
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, ids, transform):
        self.ids = ids
        self.transform = transform
        self.base_path = "../input/aptos2019-blindness-detection/test_images/"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = f"{self.base_path}{img_id}.png"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_id


test_dataset = TestDataset(test_ids, transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)




## === cell 4
submission = []
net.eval()
with torch.no_grad():
    for batch_idx, (imgs, ids) in enumerate(test_loader):
        if batch_idx % 2 == 0:
            processed = batch_idx * test_loader.batch_size
            print(f"Processing {processed}/{len(test_ids)}")
        imgs = imgs.to(device)

        logits = net(imgs)

        imgs_flipped = torch.flip(imgs, dims=[3])  # flip width dimension
        logits_flipped = net(imgs_flipped)

        if logits.shape[1] == 4:
            probs = ordinal2class_prob(logits)
            probs_flipped = ordinal2class_prob(logits_flipped)
        else:
            probs = torch.softmax(logits, dim=1)
            probs_flipped = torch.softmax(logits_flipped, dim=1)

        avg_probs = (probs + probs_flipped) / 2.0
        pred_labels = torch.argmax(avg_probs, dim=1).cpu().numpy().astype(int)

        for img_id, label in zip(ids, pred_labels):
            submission.append([img_id, int(label)])

submission = np.array(submission)




## === cell 5
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, rows:", len(df))
