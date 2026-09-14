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
Create a model to automatically segment the stomach and intestines on MRI scans.

## Metric
Mean Dice coefficient and 3D Hausdorff distance. 

The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where $X$ is the predicted set of pixels and $Y$ is the ground truth. The Dice coefficient is defined to be 0 when both $X$ and $Y$ are empty. 

Hausdorff distance is a method for calculating the distance between segmentation objects A and B, by calculating the furthest point on object A from the nearest point on object B. For 3D Hausdorff, we construct 3D volumes by combining each 2D segmentation with slice depth as the Z coordinate and then find the Hausdorff distance between them. (Here the slice depth for all scans is set to 1). The expected / predicted pixel locations are normalized by image size to create a bounded 0-1 score.

The two metrics are combined, with a weight of 0.4 for the Dice metric and 0.6 for the Hausdorff distance.

## Submission Format
Use run-length encoding on the pixel values.  Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The file should contain a header and have the following format:

```
id,class,predicted
1,large_bowel,1 1 5 1
1,small_bowel,1 1
1,stomach,1 1
2,large_bowel,1 5 2 17
etc.
```

## Dataset
Each case is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. The goal is to be able to generalize to both partially and wholly unseen cases.

### Files
- train.csv - IDs and masks for all training objects.
- sample_submission.csv - a sample submission file in the correct format
- train - a folder of case/day folders, each containing slice images for a particular case on a given day.

Note that the image filenames include 4 numbers (ex. 276_276_1.63_1.63.png). These four numbers are slice width / height (integers in pixels) and width/height pixel spacing (floating points in mm). The first two defines the resolution of the slide. The last two record the physical size of each pixel.

Physical pixel thickness in superior-inferior direction is 3mm.

### Columns
- `id` - unique identifier for object
- `class` - the predicted class for the object
- `segmentation` - RLE-encoded pixels for the identified object

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        input/
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        working/
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
```

-> data/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> data/uw-madison-gi-tract-image-segmentation/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/uw-madison-gi-tract-image-segmentation/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/uw-madison-gi-tract-image-segmentation/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> input/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.5540628775178745

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I added a safe fallback for the missing `segmentation_models_pytorch` library by defining a very small UNet‑like model when the import fails, fixed the missing Albumentations alias `A`, imported `Image` from PIL, and ensured all cells are correctly indexed so the script runs from start to finish and writes a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import os, gc, glob, pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


def find_base_path():
    """
    Search common directories for a 'test.csv' file and return its parent folder.
    Falls back to the original relative 'data' folder if none are found.
    """
    candidates = [
        pathlib.Path.cwd() / "data",
        pathlib.Path.cwd() / "input",
        pathlib.Path.cwd() / "working",
        pathlib.Path("/kaggle/input"),
        pathlib.Path("/kaggle/working"),
    ]
    for p in candidates:
        if (p / "test.csv").exists():
            return str(p)
    return str(candidates[0])


BASE_PATH = find_base_path()
CKPT_DIR = os.path.join(BASE_PATH, "ckpts")  # placeholder, may be empty




## === cell 1
class CFG:
    seed = 42
    debug = False
    model_name = "UNET"
    encoder_name = "resnet34"
    encoder_weights = "imagenet"
    train_batch_size = 32
    val_batch_size = 32
    img_size = (224, 224)
    scheduler = "CosineAnnealingLR"
    epochs = 22
    lr = 2e-3
    min_lr = 1e-6
    weight_decay = 1e-6
    T_max = int(30000 / train_batch_size * epochs) + 50
    num_classes = 3
    val_split_percentage = 0.2
    n_accumulate = max(1, 32 // train_batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thr = 0.5  # original 0.4 → 0.5




## === cell 2
def load_dummy_model():
    """
    Returns a minimal model that outputs zeros of the correct shape.
    Kept for compatibility but not used in the final inference.
    """

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            batch, _, h, w = x.shape
            return torch.zeros((batch, CFG.num_classes, h, w), device=x.device)

    return DummyModel()


def get_pretrained_models():
    """
    Load two torchvision pretrained segmentation models (FCN‑ResNet50 and DeepLabV3‑ResNet50),
    trim them to the first `CFG.num_classes` channels, move them to the device and set eval mode.
    Returns a list of models.
    """
    fcn = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    deeplab = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)

    for m in (fcn, deeplab):
        m = m.to(CFG.device)
        m.eval()
    return [fcn, deeplab]


def masks2rles(mask_np, ids, heights, widths):
    """
    Convert a batch of binary masks (B, H, W, C) to RLE strings.
    Returns three lists: RLE strings, corresponding ids, and class names.
    """
    class_names = ["large_bowel", "small_bowel", "stomach"]
    B, H, W, C = mask_np.shape
    rles, id_list, cls_list = [], [], []

    for b in range(B):
        for c in range(C):
            flat = mask_np[b, :, :, c].flatten(order="F")  # column‑major flatten
            runs = []
            i = 0
            while i < flat.size:
                if flat[i]:
                    start = i + 1  # positions are 1‑based
                    length = 1
                    i += 1
                    while i < flat.size and flat[i]:
                        length += 1
                        i += 1
                    runs.append(f"{start} {length}")
                else:
                    i += 1
            rle_str = " ".join(runs)
            rles.append(rle_str)
            id_list.append(ids[b])
            cls_list.append(class_names[c])
    return rles, id_list, cls_list




## === cell 3
class TestDataset(Dataset):
    """
    Very lightweight dataset returning a constant tensor.
    Each row from test.csv provides an id and a class; we ignore the actual
    image files and supply a dummy image of the configured size.
    """

    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.zeros((3, CFG.img_size[0], CFG.img_size[1]), dtype=np.float32)
        if self.transforms:
            img = self.transforms(img)
        img = torch.from_numpy(img)

        h, w = CFG.img_size
        return img, row["id"], h, w


test_transforms = {"test": lambda x: x}




## === cell 4
@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    """
    Perform inference.
    If `model_paths` is empty we fall back to two pretrained torchvision
    segmentation models (FCN‑ResNet50 & DeepLabV3‑ResNet50) and average their
    probability maps. This provides a richer baseline while preserving the
    original inference flow.
    """
    pred_strings = []
    pred_ids = []
    pred_classes = []
    imgs, msks = [], []

    use_pretrained = len(model_paths) == 0
    if use_pretrained:
        pretrained_models = get_pretrained_models()  # list of models

    for idx, (img, ids, heights, widths) in enumerate(
        tqdm(test_loader, total=len(test_loader), desc="Infer ")
    ):
        img = img.to(CFG.device, dtype=torch.float)
        batch_size = img.size(0)

        if use_pretrained:
            ensemble_msk = torch.zeros(
                (batch_size, CFG.num_classes, img.size(2), img.size(3)),
                device=CFG.device,
                dtype=torch.float32,
            )
            for model in pretrained_models:
                out = model(img)["out"]
                out = out[:, : CFG.num_classes, :, :]  # keep first 3 channels
                out = torch.sigmoid(out)  # convert logits to probabilities
                ensemble_msk += out / len(pretrained_models)
        else:
            ensemble_msk = torch.zeros(
                (batch_size, CFG.num_classes, img.size(2), img.size(3)),
                device=CFG.device,
                dtype=torch.float32,
            )
            for path in model_paths:
                model = load_dummy_model()  # placeholder for real checkpoints
                out = model(img)
                out = torch.sigmoid(out)
                ensemble_msk += out / len(model_paths)

        mask_np = (ensemble_msk.permute(0, 2, 3, 1) > thr).to(torch.uint8).cpu().numpy()
        strings, ids_out, classes_out = masks2rles(mask_np, ids, heights, widths)
        pred_strings.extend(strings)
        pred_ids.extend(ids_out)
        pred_classes.extend(classes_out)

        if idx < num_log:
            imgs.append(img.permute(0, 2, 3, 1).cpu().numpy())
            msks.append(mask_np)

        del img, ensemble_msk, mask_np
        gc.collect()
        torch.cuda.empty_cache()

    return pred_strings, pred_ids, pred_classes, imgs, msks




## === cell 5
test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))

test_dataset = TestDataset(test_df, transforms=test_transforms["test"])
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.val_batch_size,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
)

model_paths = []  # empty list triggers the pretrained model ensemble
pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)




## === cell 6
pred_df = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
)

if not CFG.debug:
    sub_template = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
    sub_template = sub_template.drop(columns=["predicted"])
else:
    sub_template = pd.read_csv(os.path.join(BASE_PATH, "train.csv")).head(1000 * 3)
    sub_template = sub_template.drop(columns=["segmentation"])

final_sub = sub_template.merge(pred_df, on=["id", "class"], how="left")
final_sub.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
print(final_sub.head())
