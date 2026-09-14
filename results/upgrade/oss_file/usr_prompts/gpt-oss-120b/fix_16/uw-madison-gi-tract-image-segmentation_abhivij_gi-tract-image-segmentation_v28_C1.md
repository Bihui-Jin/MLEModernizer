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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.8077588400405604

# 6. Current score

0.40766

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00413) has done: 'Implemented fixes to get the pipeline running and produce a valid `submission.csv`.

Key changes:
1. Added proper import for `ToTensorV2` from Albumentations.
2. Guarded the creation of `idclass_to_rle` in `GITractDataset` to handle test data that lacks the `segmentation` column, preventing the AttributeError.
3. Updated the test transformation pipeline to use the correctly imported `ToTensorV2`.'
- What this solution (achieved 0.0) has done: 'Implemented a fix to use the merged dataframe (`data`) that contains image paths when creating the training dataset, preventing the missing `image_path` attribute error. This allows the training loop to run, defines `dataloader_test` correctly, and ensures a valid `submission.csv` is written.'
- What this solution (achieved 0.0) has done: 'We extend the brief training phase from 2 to 5 epochs so the model moves away from random weights and produces non‑empty predictions, which should raise the score toward the target while keeping the overall architecture and pipeline unchanged.'
- What this solution (achieved 0.0) has done: 'We lower the prediction threshold during inference so that the model outputs non‑empty masks, turning a zero score into a non‑zero one and moving the result toward the target. The change is limited to the post‑processing step in the test loop, preserving all other training and data handling logic.'
- What this solution (achieved 0.0) has done: 'I lower the prediction threshold by applying a sigmoid to the logits and using a 0.4 cutoff, which should produce more meaningful masks and improve the Dice‑based score. I also extend training from 5 to 8 epochs to let the model learn a bit more without altering its architecture or loss. These minimal tweaks keep the core pipeline intact while moving the evaluation metric toward the target.'
- What this solution (achieved 0.04779) has done: 'Implemented two minimal adjustments to move the submission score away from zero:  
1. **Use ImageNet‑pretrained encoder weights** for the Unet model (instead of disabling them), giving the network a better starting point while keeping the same architecture.  
2. **Lower the prediction threshold** from 0.4 to 0.2 so the model outputs non‑empty masks, ensuring a non‑zero Dice contribution and bringing the score closer to the target.'
- What this solution (achieved 0.40766) has done: 'I correct the data path so the CSV files are found, which resolves the `FileNotFoundError` and the subsequent `NameError` for the missing `data` variable. I also add a safe fallback that selects the first existing directory among typical Kaggle input locations. No other logic is altered, preserving the original model and training pipeline.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

_possible_paths = [
    "/kaggle/input/uw-madison-gi-tract-image-segmentation",
    "/kaggle/input",
    os.path.join(os.getcwd(), "data"),
]
DIR_PATH = next((p for p in _possible_paths if os.path.isdir(p)), None)
if DIR_PATH is None:
    raise FileNotFoundError(
        "Data directory not found. Checked locations: " + ", ".join(_possible_paths)
    )

CLASS_NAMES = ["large_bowel", "small_bowel", "stomach"]
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_RESIZE = (256, 256)
IMAGE_NORMALIZE_MEAN = (0.5, 0.5, 0.5)
IMAGE_NORMALIZE_SD = (0.5, 0.5, 0.5)

TRAIN_VALID_SPLIT = True  # keep training loop but we’ll limit epochs
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 8
DATA_LOADER_NUM_WORKERS = 0
LOAD_SAVED_MASKS = False  # masks are generated on‑the‑fly


def rle_decode(mask_rle: str, shape):
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def load_image(id_, impath_dict):
    """Load and normalise a single image."""
    img_path = impath_dict[id_]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype("float32")
    mx = np.max(img)
    if mx > 0:
        img /= mx
    return img


def get_mask(id_, id_dicts):
    if LOAD_SAVED_MASKS:
        mask_path = os.path.splitext(id_dicts["impath"][id_])[0] + ".npy"
        return np.load(mask_path)
    else:
        h, w = id_dicts["shape"][id_]
        shape = (h, w, 3)
        mask = np.zeros(shape, dtype=np.uint8)
        for i, class_ in enumerate(CLASS_NAMES):
            rle = id_dicts["rle"].get((id_, class_))
            if rle:
                mask[..., i] = rle_decode(rle, shape[:2])
        return mask




## === cell 1
def get_path_df(train=True):
    if train:
        paths = glob.glob(os.path.join(DIR_PATH, "train/*/*/*/*"))
    else:
        paths = glob.glob(os.path.join(DIR_PATH, "test/*/*/*/*"))
    df = pd.DataFrame(paths, columns=["image_path"])
    df[["case", "day", "slice", "slice_w", "slice_h", "px_w", "px_h"]] = df[
        "image_path"
    ].str.extract(
        r".*/case(\d+)_day(\d+)/scans/slice_(\d+)_(\d+)_(\d+)_(\d+\.\d+)_(\d+\.\d+)\.png"
    )
    return df




## === cell 2
train_df = pd.read_csv(os.path.join(DIR_PATH, "train.csv"))
train_df[["case", "day", "slice"]] = train_df["id"].str.extract(
    r"case(\d+)_day(\d+)_slice_(\d+)"
)
path_df = get_path_df(train=True)
data = train_df.merge(path_df, on=["case", "day", "slice"])

id_to_impath = dict(zip(data.id, data.image_path))
id_to_shape = dict(
    zip(
        data.id,
        zip(data.slice_h.astype(np.uint32), data.slice_w.astype(np.uint32)),
    )
)
idclass_to_rle = {
    (row.id, row["class"]): row.segmentation
    for _, row in data.iterrows()
    if pd.notna(row.segmentation)
}
id_dicts = {"impath": id_to_impath, "shape": id_to_shape, "rle": idclass_to_rle}

test_df = pd.read_csv(os.path.join(DIR_PATH, "test.csv"))
test_df[["case", "day", "slice"]] = test_df["id"].str.extract(
    r"case(\d+)_day(\d+)_slice_(\d+)"
)
test_path_df = get_path_df(train=False)
test_df = test_df.merge(test_path_df, on=["case", "day", "slice"])




## === cell 3
class GITractDataset(Dataset):
    def __init__(self, df, is_test=False, transforms=None):
        self.ids = df.id.unique()
        self.is_test = is_test
        self.transforms = transforms
        self.id_to_impath = dict(zip(df.id, df.image_path))
        self.id_to_shape = dict(
            zip(df.id, zip(df.slice_h.astype(np.uint32), df.slice_w.astype(np.uint32)))
        )
        if "segmentation" in df.columns:
            self.idclass_to_rle = {
                (row.id, row["class"]): row.segmentation
                for _, row in df.iterrows()
                if pd.notna(row.segmentation)
            }
        else:
            self.idclass_to_rle = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img = load_image(id_, self.id_to_impath)  # (H,W) normalized float32
        img = np.repeat(img[..., None], 3, axis=2)  # to (H,W,3)
        if not self.is_test:
            mask = get_mask(
                id_, {"shape": self.id_to_shape, "rle": self.idclass_to_rle}
            )
            if self.transforms:
                aug = self.transforms(image=img, mask=mask)
                img, mask = aug["image"], aug["mask"]
            return img, mask, id_
        else:
            h, w = self.id_to_shape[id_]
            if self.transforms:
                img = self.transforms(image=img)["image"]
            return img, id_, h, w


class SimpleCNN(nn.Module):
    def __init__(self, in_ch=3, out_ch=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


try:
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )
except Exception:
    model = SimpleCNN()
model.to(DEVICE)

weights_path = (
    "/kaggle/input/git-seg/pytorch/256x256/1/GIT-Seg-256x256-efficientnet-b1.pth"
)
if os.path.isfile(weights_path):
    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print("Loaded pretrained weights.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
else:
    print("Pretrained weight file not found – using random init.")

transform_test = A.Compose(
    [
        A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation=cv2.INTER_NEAREST),
        A.Normalize(
            mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
        ),
        ToTensorV2(),
    ]
)

if TRAIN_VALID_SPLIT:
    transform_train = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation=cv2.INTER_NEAREST),
            A.Normalize(
                mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
            ),
            ToTensorV2(),
        ]
    )

    train_dataset = GITractDataset(data, is_test=False, transforms=transform_train)
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    bce_criterion = nn.BCEWithLogitsLoss()

    def dice_loss(logits, targets, eps=1e-6):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.shape[0], probs.shape[1], -1)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)
        intersect = (probs * targets).sum(dim=2)
        denominator = probs.sum(dim=2) + targets.sum(dim=2)
        loss = 1 - (2 * intersect + eps) / (denominator + eps)
        return loss.mean()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(2):  # limited epochs for quick run
        epoch_loss = 0.0
        for imgs, masks, _ in dataloader_train:
            imgs = imgs.to(DEVICE, dtype=torch.float)
            masks = masks.to(DEVICE, dtype=torch.float).permute(0, 3, 1, 2)

            optimizer.zero_grad()
            logits = model(imgs)

            loss_bce = bce_criterion(logits, masks)
            loss_dice = dice_loss(logits, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/2 - loss: {epoch_loss/len(dataloader_train):.4f}")

    test_dataset = GITractDataset(test_df, is_test=True, transforms=transform_test)
    dataloader_test = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=DATA_LOADER_NUM_WORKERS,
        pin_memory=True,
    )




## === cell 4
model.eval()
test_ids, test_class, test_pred_RLE = [], [], []

with torch.no_grad():
    for imgs, ids, heights, widths in dataloader_test:
        imgs = imgs.to(DEVICE, dtype=torch.float)
        logits = model(imgs)  # (B, C, H, W)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.2).int()  # binary mask

        preds = preds.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)

        for mask, id_, h, w in zip(preds, ids, heights, widths):
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                dsize=(w.item(), h.item()),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_bin = (mask_resized > 0).astype(np.uint8)

            rles = [rle_encode(mask_bin[..., c]) for c in range(NUM_CLASSES)]

            test_ids.extend([id_] * NUM_CLASSES)
            test_class.extend(CLASS_NAMES)
            test_pred_RLE.extend(rles)

submission_df = pd.DataFrame(
    {"id": test_ids, "class": test_class, "predicted": test_pred_RLE}
)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission_df.head())
