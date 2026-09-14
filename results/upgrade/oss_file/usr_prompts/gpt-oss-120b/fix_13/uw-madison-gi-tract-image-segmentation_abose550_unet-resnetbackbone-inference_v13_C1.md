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

0.5711150869012114

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix replaces the use of the deprecated `pretrained=True` flag with an explicit construction that loads an ImageNet‑pretrained ResNet‑50 backbone while allowing a custom number of output classes (3). This prevents the `ValueError` about mismatched `num_classes`. With the corrected model builder, inference runs, producing the expected prediction lists, and the final submission CSV is generated without the previous `NameError`. No other logic is altered, preserving the original workflow.'
- What this solution (achieved 0.0) has done: 'I fix the RLE conversion so that masks are used directly (removing the incorrect padding that can produce empty masks) and slightly tighten the inference threshold to 0.5, which better matches the sigmoid output distribution. These minimal adjustments keep the model architecture unchanged while allowing non‑zero predictions, moving the score toward the target.'

# 9. Code solution

## === cell 0
import os
import gc
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CFG:
    seed = 42
    debug = False
    model_name = "UNET"
    encoder_name = "resnet50"
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
    thr = 0.15  # lowered threshold to generate non‑empty masks


def build_model():
    """
    Build a simple FCN model with a ResNet‑50 backbone pretrained on ImageNet.
    The final classifier is adapted to output `CFG.num_classes` channels.
    """
    model = models.segmentation.fcn_resnet50(pretrained=True, progress=False)
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = torch.nn.Conv2d(in_channels, CFG.num_classes, kernel_size=1)
    return model


def load_model(path):
    """
    Load a checkpoint saved with torch.save(model.state_dict()).
    If the file does not exist we fall back to a freshly built model.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = build_model()
    state = torch.load(path, map_location=CFG.device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(CFG.device)
    model.eval()
    return model


def rle_encode(mask):
    """
    Encode a binary mask (2‑D) using Run‑Length Encoding as required by the competition.
    The mask is flattened column‑wise (top‑to‑bottom, then left‑to‑right).
    """
    pixels = mask.ravel(order="F")
    pads = np.array([0])
    pixels = np.concatenate([pads, pixels, pads])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)


def masks2rles(preds, ids, heights, widths):
    """
    Convert a batch of predicted masks to RLE strings.
    preds: numpy array (B, H, W, C) of uint8 (0/1)
    ids, heights, widths: lists of original identifiers and dimensions for each sample.
    Returns three parallel lists: rle strings, ids, class names.
    """
    rles, out_ids, out_classes = [], [], []
    class_names = None
    for b in range(preds.shape[0]):
        for c in range(preds.shape[3]):
            mask = preds[b, :, :, c]
            h, w = heights[b], widths[b]
            if (h, w) != mask.shape:
                mask = Image.fromarray(mask.astype(np.uint8) * 255)
                mask = mask.resize((w, h), resample=Image.NEAREST)
                mask = np.array(mask) // 255
            rle = rle_encode(mask)
            rles.append(rle)
            out_ids.append(ids[b])
            out_classes.append(c)
    return rles, out_ids, out_classes


class TestDataset(Dataset):
    """
    Reads the test CSV and loads the corresponding images.
    Returns (image_tensor, id, original_height, original_width).
    """

    def __init__(self, csv_path, image_root):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transforms.Compose(
            [
                transforms.Resize(CFG.img_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        cls = row["class"]
        img_dir = os.path.join(self.image_root, img_id)
        scan_dir = os.path.join(img_dir, "scans")
        png_files = [f for f in os.listdir(scan_dir) if f.lower().endswith(".png")]
        if not png_files:
            raise FileNotFoundError(f"No PNG found in {scan_dir}")
        img_path = os.path.join(scan_dir, png_files[0])
        img = Image.open(img_path).convert("RGB")
        h, w = img.size[1], img.size[0]  # width, height order in PIL
        img_tensor = self.transform(img)
        return img_tensor, img_id, h, w


@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    """
    Perform inference. If no checkpoint is found, fall back to a freshly built
    pretrained FCN model so that predictions are not all zeros.
    """
    pred_strings, pred_ids, pred_classes = [], [], []
    imgs, msks = [], []

    if model_paths:
        models_list = [load_model(p) for p in model_paths]
    else:
        single_model = build_model()
        single_model.to(CFG.device)
        single_model.eval()
        models_list = [single_model]

    for idx, (img, ids, heights, widths) in enumerate(
        tqdm(test_loader, total=len(test_loader), desc="Infer")
    ):
        img = img.to(CFG.device, dtype=torch.float)
        batch_size = img.size(0)
        agg_msk = torch.zeros(
            (batch_size, CFG.num_classes, img.size(2), img.size(3)),
            device=CFG.device,
            dtype=torch.float32,
        )
        for model in models_list:
            out = model(img)["out"]
            out = torch.sigmoid(out)
            agg_msk += out / len(models_list)

        preds = (agg_msk.permute(0, 2, 3, 1) > thr).to(torch.uint8).cpu().numpy()
        rs, ri, rc = masks2rles(preds, ids, heights, widths)
        pred_strings.extend(rs)
        pred_ids.extend(ri)
        pred_classes.extend(rc)

        if idx < num_log:
            imgs.append(img.permute(0, 2, 3, 1).cpu().numpy())
            msks.append(preds)

        del img, agg_msk, preds, rs, ri, rc
        torch.cuda.empty_cache()
        gc.collect()
    return pred_strings, pred_ids, pred_classes, imgs, msks


def write_submission(
    pred_strings, pred_ids, pred_classes, class_map, out_path="submission.csv"
):
    """
    Save predictions to a CSV file compatible with the competition format.
    `class_map` converts integer class indices to the textual class names.
    """
    class_names = [class_map[c] for c in pred_classes]
    df = pd.DataFrame({"id": pred_ids, "class": class_names, "predicted": pred_strings})
    df = df[["id", "class", "predicted"]]
    df.to_csv(out_path, index=False)
    print(f"Submission written to {out_path} (rows: {len(df)})")




## === cell 1
if __name__ == "__main__":
    base_dir = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
    test_csv_path = os.path.join(base_dir, "test.csv")
    test_image_root = os.path.join(
        base_dir, "test"
    )  # folder containing case sub‑folders

    test_dataset = TestDataset(csv_path=test_csv_path, image_root=test_image_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    unique_classes = pd.read_csv(test_csv_path)["class"].unique()
    class_map = {i: unique_classes[i] for i in range(len(unique_classes))}
    if len(class_map) < CFG.num_classes:
        for i in range(len(class_map), CFG.num_classes):
            class_map[i] = f"class_{i}"
    elif len(class_map) > CFG.num_classes:
        class_map = {i: unique_classes[i] for i in range(CFG.num_classes)}

    model_paths = []  # empty → fallback to pretrained FCN‑ResNet50
    pred_strings, pred_ids, pred_classes, _, _ = infer(
        model_paths=model_paths,
        test_loader=test_loader,
        num_log=1,
        thr=CFG.thr,
    )

    write_submission(
        pred_strings,
        pred_ids,
        pred_classes,
        class_map=class_map,
        out_path="submission.csv",
    )

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1361548338.py in <cell line: 0>()
     38     # ------------------------------------------------------------------
     39     model_paths = []  # empty → fallback to pretrained FCN‑ResNet50
---> 40     pred_strings, pred_ids, pred_classes, _, _ = infer(
     41         model_paths=model_paths,
     42         test_loader=test_loader,

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/1986017482.py in infer(model_paths, test_loader, num_log, thr)
    170         models_list = [single_model]
    171 
--> 172     for idx, (img, ids, heights, widths) in enumerate(
    173         tqdm(test_loader, total=len(test_loader), desc="Infer")
    174     ):

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/1986017482.py in __getitem__(self, idx)
    140         # locate a png file inside the scans sub‑folder
    141         scan_dir = os.path.join(img_dir, "scans")
--> 142         png_files = [f for f in os.listdir(scan_dir) if f.lower().endswith(".png")]
    143         if not png_files:
    144             raise FileNotFoundError(f"No PNG found in {scan_dir}")

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/uw-madison-gi-tract-image-segmentation/test/case123_day20_slice_0001/scans'
