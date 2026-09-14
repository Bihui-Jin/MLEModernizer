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

3.13

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.7516228812032423

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.00213) has done: 'I fixed the model construction to work with the fallback `SimpleUNet` when the full `segmentation_models_pytorch` library is unavailable, and wrapped the optimizer creation so it runs after a successful model instantiation. This resolves the `TypeError` and subsequent `NameError`, allowing the script to run through training (if enabled) and generate a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
print(f"Number of available CPUs: {os.cpu_count()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1140487450.py in <cell line: 0>()
----> 1 print(f"Number of available CPUs: {os.cpu_count()}")
      2 print(f"Number of available GPUs: {torch.cuda.device_count()}")
      3 
      4 

NameError: name 'os' is not defined

## === cell 1
DIR_PATH = "/kaggle/input/uw-madison-gi-tract-image-segmentation/"

pd.set_option("display.max_colwidth", 400)

CMAP1 = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 1]])  # black transparent, red opaque
CMAP2 = ListedColormap([[0, 0, 0, 0], [0, 1, 0, 1]])  # black transparent, green opaque
CMAP3 = ListedColormap([[0, 0, 0, 0], [0, 0, 1, 1]])  # black transparent, blue opaque

RANDOM_SEED = 0

IMAGE_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGE_NORMALIZE_SD = (0.229, 0.224, 0.225)

IMAGE_RESIZE = [288, 288]

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = BATCH_SIZE_TRAIN * 2
BATCH_SIZE_TEST = BATCH_SIZE_TRAIN * 2

DATA_LOADER_NUM_WORKERS = 4

NUM_CLASSES = 3
CLASS_NAMES = ["large_bowel", "small_bowel", "stomach"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 5

MODEL_PARAMS_FILE_NAME = "GIT-Seg-efficientnet-b1-subset.pth"
MODEL_PARAMS_LOAD_FILE_PATH = (
    "/kaggle/input/git-seg/pytorch/default/1/GIT-Seg-efficientnet-b1-subset.pth"
)

TRAIN_VALID_SPLIT = True
TEST_PREDICT = True

SAVE_TRAIN_VALID_MODEL = False
LOAD_MODEL_FOR_TEST_PREDICT = False




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/818997717.py in <cell line: 0>()
      1 DIR_PATH = "/kaggle/input/uw-madison-gi-tract-image-segmentation/"
      2 
----> 3 pd.set_option("display.max_colwidth", 400)
      4 
      5 CMAP1 = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 1]])  # black transparent, red opaque

NameError: name 'pd' is not defined

## === cell 2
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3959886339.py in <cell line: 0>()
----> 1 random.seed(RANDOM_SEED)
      2 np.random.seed(RANDOM_SEED)
      3 torch.manual_seed(RANDOM_SEED)
      4 torch.cuda.manual_seed_all(RANDOM_SEED)
      5 

NameError: name 'random' is not defined

## === cell 3
data = pd.read_csv(DIR_PATH + "train.csv")
data.head()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3598993885.py in <cell line: 0>()
----> 1 data = pd.read_csv(DIR_PATH + "train.csv")
      2 data.head()
      3 
      4 

NameError: name 'pd' is not defined

## === cell 4
data_nonnaseg = data.loc[data.segmentation.notna(), :]
data_nonnaseg.head()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1984853436.py in <cell line: 0>()
----> 1 data_nonnaseg = data.loc[data.segmentation.notna(), :]
      2 data_nonnaseg.head()
      3 
      4 

NameError: name 'data' is not defined

## === cell 5
data[["case", "day", "slice"]] = data["id"].str.extract(
    r"case(\d+)_day(\d+)_slice_(\d+)"
)
data




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/801339358.py in <cell line: 0>()
----> 1 data[["case", "day", "slice"]] = data["id"].str.extract(
      2     r"case(\d+)_day(\d+)_slice_(\d+)"
      3 )
      4 data
      5 

NameError: name 'data' is not defined

## === cell 6
def get_path_df(train=True):
    if train:
        paths = glob(DIR_PATH + "train/*/*/*/*")
    else:
        paths = glob(DIR_PATH + "test/*/*/*/*")
    path_df = pd.DataFrame(paths, columns=["image_path"])
    path_df[["case", "day", "slice", "slice_w", "slice_h", "px_w", "px_h"]] = (
        path_df.image_path.str.extract(
            r".*/case(\d+)_day(\d+)/scans/slice_(\d+)_(\d+)_(\d+)_(\d+\.\d+)_(\d+\.\d+)\.png"
        )
    )
    return path_df


path_df = get_path_df()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2434072929.py in <cell line: 0>()
     13 
     14 
---> 15 path_df = get_path_df()
     16 
     17 

/tmp/ipykernel_56/2434072929.py in get_path_df(train)
      1 def get_path_df(train=True):
      2     if train:
----> 3         paths = glob(DIR_PATH + "train/*/*/*/*")
      4     else:
      5         paths = glob(DIR_PATH + "test/*/*/*/*")

NameError: name 'glob' is not defined

## === cell 7
data.info()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2699117437.py in <cell line: 0>()
----> 1 data.info()
      2 
      3 

NameError: name 'data' is not defined

## === cell 8
path_df.info()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/96925661.py in <cell line: 0>()
----> 1 path_df.info()
      2 
      3 

NameError: name 'path_df' is not defined

## === cell 9
data = data.merge(path_df, on=["case", "day", "slice"])
data




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1089280659.py in <cell line: 0>()
----> 1 data = data.merge(path_df, on=["case", "day", "slice"])
      2 data
      3 
      4 

NameError: name 'data' is not defined

## === cell 10
data.info()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2699117437.py in <cell line: 0>()
----> 1 data.info()
      2 
      3 

NameError: name 'data' is not defined

## === cell 11
data.px_w.unique(), data.px_h.unique()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/157282275.py in <cell line: 0>()
----> 1 data.px_w.unique(), data.px_h.unique()
      2 
      3 

NameError: name 'data' is not defined

## === cell 12
data.case.unique(), data.day.unique(), data.slice.unique(), data.slice_w.unique(), data.slice_h.unique()




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/4221479212.py in <cell line: 0>()
----> 1 data.case.unique(), data.day.unique(), data.slice.unique(), data.slice_w.unique(), data.slice_h.unique()
      2 
      3 

NameError: name 'data' is not defined

## === cell 13
int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
data[int_cols] = data[int_cols].astype(np.uint32)

float_cols = ["px_w", "px_h"]
data[float_cols] = data[float_cols].astype(np.float32)

data.info()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3008184326.py in <cell line: 0>()
      1 int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
----> 2 data[int_cols] = data[int_cols].astype(np.uint32)
      3 
      4 float_cols = ["px_w", "px_h"]
      5 data[float_cols] = data[float_cols].astype(np.float32)

NameError: name 'data' is not defined

## === cell 14
def rle_decode(mask_rle, shape):
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




## === cell 15
def get_mask(id_, data):
    data_subset_id = data.loc[data["id"] == id_]
    if data_subset_id.empty:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    slice_dim = data_subset_id[["slice_h", "slice_w"]].iloc[0]
    shape = (slice_dim.slice_h, slice_dim.slice_w, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(CLASS_NAMES):
        data_subset_class = data_subset_id[data_subset_id["class"] == class_]
        rle = data_subset_class.segmentation.squeeze()
        if not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask




## === cell 16
def load_image(id_, data):
    data_subset = data.loc[data.id == id_]
    if data_subset.empty:
        return np.zeros((256, 256), dtype="float32")
    img_path = data_subset.image_path.iloc[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        h = int(data_subset.slice_h.iloc[0])
        w = int(data_subset.slice_w.iloc[0])
        return np.zeros((h, w), dtype="float32")
    img = img.astype("float32")
    mx = np.max(img)
    if mx > 0:
        img /= mx
    return img




## === cell 17
def display_image(
    id_,
    data,
    pred_mask=None,
    apply_CLAHE=False,
    show_orig_img=True,
    show_true_mask=True,
    show_pred_mask=False,
):
    img = load_image(id_, data)
    img = (img * 255).astype(np.uint8)  # 0-255 range required for CLAHE.
    if apply_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
        img = clahe.apply(img)

    mask = get_mask(id_, data)

    plt.figure(figsize=(9, 3))

    i = 1
    if show_orig_img:
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(img, cmap="bone")
        plt.title(f"{id_} image")
        plt.axis("off")

    if show_true_mask:
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(img, cmap="bone")
        plt.title("Image with true mask")
        plt.imshow(mask[..., 0], cmap=CMAP1)
        plt.imshow(mask[..., 1], cmap=CMAP2)
        plt.imshow(mask[..., 2], cmap=CMAP3)
        handles = [
            Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP3(1.0)),
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.axis("off")
        plt.legend(
            handles,
            labels,
            bbox_to_anchor=(1.0, -0.4),
            loc="lower right",
            borderaxespad=0.0,
        )

    if show_pred_mask and pred_mask is not None:
        plt.subplot(1, 3, i)
        plt.imshow(img, cmap="bone")
        plt.title("Image with predicted mask")
        plt.imshow(pred_mask[..., 0], cmap=CMAP1)
        plt.imshow(pred_mask[..., 1], cmap=CMAP2)
        plt.imshow(pred_mask[..., 2], cmap=CMAP3)
        handles = [
            Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP3(1.0)),
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.axis("off")
        plt.legend(
            handles,
            labels,
            bbox_to_anchor=(1.0, -0.4),
            loc="lower right",
            borderaxespad=0.0,
        )

    plt.tight_layout()
    plt.show()




## === cell 18
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
index_train, index_valid = next(
    sgkf.split(data.id, data.segmentation.isna(), data.case)
)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3590770032.py in <cell line: 0>()
----> 1 sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
      2 index_train, index_valid = next(
      3     sgkf.split(data.id, data.segmentation.isna(), data.case)
      4 )
      5 

NameError: name 'StratifiedGroupKFold' is not defined

## === cell 19
len(index_train), len(index_valid)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1354749846.py in <cell line: 0>()
----> 1 len(index_train), len(index_valid)
      2 
      3 

NameError: name 'index_train' is not defined

## === cell 20
data_train = data.iloc[index_train, :]
data_valid = data.iloc[index_valid, :]




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2782564665.py in <cell line: 0>()
----> 1 data_train = data.iloc[index_train, :]
      2 data_valid = data.iloc[index_valid, :]
      3 
      4 

NameError: name 'data' is not defined

## === cell 21
print(len(data_train.case.unique()), len(data_valid.case.unique()))




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/484774141.py in <cell line: 0>()
----> 1 print(len(data_train.case.unique()), len(data_valid.case.unique()))
      2 
      3 

NameError: name 'data_train' is not defined

## === cell 22
data_train_sub = data_train.loc[data_train.case.isin(data_train.case.unique()[:11]), :]
data_valid_sub = data_valid.loc[data_valid.case.isin(data_valid.case.unique()[:2]), :]

print(
    len(data_train_sub), len(data_valid_sub), len(data_train_sub) / len(data_valid_sub)
)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/232508348.py in <cell line: 0>()
----> 1 data_train_sub = data_train.loc[data_train.case.isin(data_train.case.unique()[:11]), :]
      2 data_valid_sub = data_valid.loc[data_valid.case.isin(data_valid.case.unique()[:2]), :]
      3 
      4 print(
      5     len(data_train_sub), len(data_valid_sub), len(data_train_sub) / len(data_valid_sub)

NameError: name 'data_train' is not defined

## === cell 23
missing_masks_train = data_train_sub.segmentation.isna().sum()
missing_masks_valid = data_valid_sub.segmentation.isna().sum()
print(missing_masks_train, missing_masks_train * 100 / len(data_train_sub))
print(missing_masks_valid, missing_masks_valid * 100 / len(data_valid_sub))




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2966508055.py in <cell line: 0>()
----> 1 missing_masks_train = data_train_sub.segmentation.isna().sum()
      2 missing_masks_valid = data_valid_sub.segmentation.isna().sum()
      3 print(missing_masks_train, missing_masks_train * 100 / len(data_train_sub))
      4 print(missing_masks_valid, missing_masks_valid * 100 / len(data_valid_sub))
      5 

NameError: name 'data_train_sub' is not defined

## === cell 24
na_counts_train = (
    data_train_sub.groupby("class")["segmentation"]
    .apply(lambda s: s.isna().sum())
    .reset_index(name="count")
)
na_counts_train["percent"] = (
    100
    * na_counts_train["count"]
    / data_train_sub.groupby("class")["segmentation"].size().values
)
display(na_counts_train)

na_counts_valid = (
    data_valid_sub.groupby("class")["segmentation"]
    .apply(lambda s: s.isna().sum())
    .reset_index(name="count")
)
na_counts_valid["percent"] = (
    100
    * na_counts_valid["count"]
    / data_valid_sub.groupby("class")["segmentation"].size().values
)
display(na_counts_valid)




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/498101797.py in <cell line: 0>()
      1 na_counts_train = (
----> 2     data_train_sub.groupby("class")["segmentation"]
      3     .apply(lambda s: s.isna().sum())
      4     .reset_index(name="count")
      5 )

NameError: name 'data_train_sub' is not defined

## === cell 25
data_train_sub = data_train_sub.reset_index(drop=True)
data_valid_sub = data_valid_sub.reset_index(drop=True)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1042551053.py in <cell line: 0>()
----> 1 data_train_sub = data_train_sub.reset_index(drop=True)
      2 data_valid_sub = data_valid_sub.reset_index(drop=True)
      3 
      4 

NameError: name 'data_train_sub' is not defined

## === cell 26
class GITractDataset(Dataset):
    def __init__(self, df, is_train=True, transforms=None):
        self.df = df
        self.id_ = df["id"].unique()
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.id_)

    def __getitem__(self, idx):
        id_ = self.id_[idx]
        img = load_image(id_, self.df)
        img = np.tile(img[..., None], [1, 1, 3])
        if self.is_train:
            mask = get_mask(id_, self.df)
            if self.transforms:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            return img, mask, id_
        else:
            if self.transforms:
                augmented = self.transforms(image=img)
                img = augmented["image"]
                data_sub = self.df.loc[self.df.id == id_]
                height = data_sub.slice_h.iloc[0]
                width = data_sub.slice_w.iloc[0]
            return img, id_, height, width




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/626621059.py in <cell line: 0>()
----> 1 class GITractDataset(Dataset):
      2     def __init__(self, df, is_train=True, transforms=None):
      3         self.df = df
      4         self.id_ = df["id"].unique()
      5         self.is_train = is_train

NameError: name 'Dataset' is not defined

## === cell 27
transform_train = A.Compose(
    [
        A.Resize(
            IMAGE_RESIZE[0],
            IMAGE_RESIZE[1],
            interpolation=cv2.INTER_NEAREST,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.Normalize(
            mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
        ),
        A.ToTensorV2(transpose_mask=True),
    ]
)

transform_valid = A.Compose(
    [
        A.Resize(
            IMAGE_RESIZE[0],
            IMAGE_RESIZE[1],
            interpolation=cv2.INTER_NEAREST,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.Normalize(
            mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
        ),
        A.ToTensorV2(transpose_mask=True),
    ]
)




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3195203155.py in <cell line: 0>()
----> 1 transform_train = A.Compose(
      2     [
      3         A.Resize(
      4             IMAGE_RESIZE[0],
      5             IMAGE_RESIZE[1],

NameError: name 'A' is not defined

## === cell 28
dataset_train = GITractDataset(data_train_sub, transforms=transform_train)
dataset_valid = GITractDataset(data_valid_sub, transforms=transform_valid)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    num_workers=DATA_LOADER_NUM_WORKERS,
)
dataloader_valid = DataLoader(
    dataset_valid,
    batch_size=BATCH_SIZE_VALID,
    shuffle=False,
    num_workers=DATA_LOADER_NUM_WORKERS,
)




## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1495149793.py in <cell line: 0>()
----> 1 dataset_train = GITractDataset(data_train_sub, transforms=transform_train)
      2 dataset_valid = GITractDataset(data_valid_sub, transforms=transform_valid)
      3 
      4 dataloader_train = DataLoader(
      5     dataset_train,

NameError: name 'GITractDataset' is not defined

## === cell 29
batch = next(iter(dataloader_train))
imgs, masks, ids = batch
print("Batch shapes:", imgs.shape, masks.shape, len(ids))




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2289966987.py in <cell line: 0>()
----> 1 batch = next(iter(dataloader_train))
      2 imgs, masks, ids = batch
      3 print("Batch shapes:", imgs.shape, masks.shape, len(ids))
      4 
      5 

NameError: name 'dataloader_train' is not defined

## === cell 30
smp_encoder_weights = (
    None if TEST_PREDICT and LOAD_MODEL_FOR_TEST_PREDICT else "imagenet"
)

try:
    model = smp.Unet(
        encoder_name="efficientnet-b1",
        encoder_weights=smp_encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
    )
except TypeError:
    model = smp.Unet(in_channels=3, out_channels=NUM_CLASSES)

model.to(DEVICE)




## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/744883473.py in <cell line: 0>()
      1 smp_encoder_weights = (
----> 2     None if TEST_PREDICT and LOAD_MODEL_FOR_TEST_PREDICT else "imagenet"
      3 )
      4 
      5 try:

NameError: name 'TEST_PREDICT' is not defined

## === cell 31
optimizer = optim.Adam(model.parameters(), lr=1e-3)




## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3875936518.py in <cell line: 0>()
----> 1 optimizer = optim.Adam(model.parameters(), lr=1e-3)
      2 
      3 

NameError: name 'optim' is not defined

## === cell 32
dice_loss = (
    smp.losses.DiceLoss(mode="multilabel")
    if hasattr(smp, "losses")
    else smp.losses.DiceLoss()
)
BCE_loss = (
    smp.losses.SoftBCEWithLogitsLoss()
    if hasattr(smp, "losses")
    else smp.losses.SoftBCEWithLogitsLoss()
)


def loss_fn(y_pred, y_true, loss_wt=0.5):
    return dice_loss(y_pred, y_true) * loss_wt + BCE_loss(y_pred, y_true) * (
        1 - loss_wt
    )




## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/315690287.py in <cell line: 0>()
      1 dice_loss = (
      2     smp.losses.DiceLoss(mode="multilabel")
----> 3     if hasattr(smp, "losses")
      4     else smp.losses.DiceLoss()
      5 )

NameError: name 'smp' is not defined

## === cell 33
class DiceScoreCustom:
    def __init__(self, num_classes, eps=1e-6):
        self.num_classes = num_classes
        self.eps = eps
        self.reset()

    def reset(self):
        self.dice_sum = 0.0
        self.image_count = 0
        self.organ_dice_sum = torch.zeros(self.num_classes)
        self.organ_count = torch.zeros(self.num_classes)

    def update(self, preds, targets):
        I = (targets & preds).sum((2, 3))
        U = (targets | preds).sum((2, 3))
        dice = (2 * I) / (U + I + self.eps)
        non_empty = U > 0
        organ_counts = non_empty.sum(dim=1)
        dice_per_image = dice.sum(dim=1) / organ_counts.clamp(min=1)
        self.dice_sum += dice_per_image.sum().item()
        self.image_count += dice_per_image.numel()
        self.organ_dice_sum += dice.sum(dim=0).detach().cpu()
        self.organ_count += non_empty.sum(dim=0).detach().cpu()

    def compute(self):
        overall = (
            torch.tensor(self.dice_sum / self.image_count)
            if self.image_count > 0
            else torch.tensor(0.0)
        )
        per_organ = torch.where(
            self.organ_count > 0,
            self.organ_dice_sum / self.organ_count,
            torch.tensor(0.0),
        )
        return overall, per_organ




## === cell 34
class HausdorffDistanceCustom:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.h3d_sum = 0.0
        self.image3d_count = 0
        self.organ_h3d_sum = np.zeros(self.num_classes)
        self.organ_count_sum = np.zeros(self.num_classes)

    def _compute_hausdorff_per_organ(self, preds, targets):
        if np.all(preds == targets):
            return 0.0
        (edges_preds, edges_targets) = get_mask_edges(preds, targets)
        surface_distance = get_surface_distance(
            edges_preds, edges_targets, distance_metric="euclidean"
        )
        if surface_distance.shape == (0,):
            return 0.0
        dist = surface_distance.max()
        max_dist = np.sqrt(np.sum((np.array(preds.shape) - 1) ** 2))
        if dist > max_dist:
            return 1.0
        return dist / max_dist

    def update(self, preds, targets):
        U = (targets | preds).sum((1, 2, 3))
        hausdorff = np.array(
            [
                self._compute_hausdorff_per_organ(preds[i, ...], targets[i, ...])
                for i in range(NUM_CLASSES)
            ]
        )
        non_empty = U > 0
        organ_count = non_empty.sum()
        if organ_count != 0:
            hausdorff_per_3dimage = hausdorff.sum() / organ_count
            self.h3d_sum += hausdorff_per_3dimage
            self.image3d_count += 1
        self.organ_h3d_sum += hausdorff
        self.organ_count_sum += non_empty

    def compute(self):
        overall = self.h3d_sum / self.image3d_count if self.image3d_count > 0 else 0.0
        per_organ = self.organ_h3d_sum / self.organ_count_sum
        return overall, per_organ




## === cell 35
dice_score_obj = DiceScoreCustom(num_classes=NUM_CLASSES)
hausdorff_obj = HausdorffDistanceCustom(num_classes=NUM_CLASSES)




## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3153266537.py in <cell line: 0>()
----> 1 dice_score_obj = DiceScoreCustom(num_classes=NUM_CLASSES)
      2 hausdorff_obj = HausdorffDistanceCustom(num_classes=NUM_CLASSES)
      3 
      4 

NameError: name 'NUM_CLASSES' is not defined

## === cell 36
def one_epoch_train(epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for data in loop:
        imgs, masks, ids = data
        imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(
            DEVICE, dtype=torch.float
        )
        optimizer.zero_grad()
        pred_masks = model(imgs)
        loss = loss_fn(pred_masks, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_loss = running_loss / len(dataloader_train)
    return avg_loss




## === cell 37
slices80_casedays = set(
    data_valid[["case", "day", "slice"]]
    .drop_duplicates()
    .value_counts(["case", "day"])
    .loc[lambda s: s == 80]
    .index
)




## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2242064719.py in <cell line: 0>()
      1 slices80_casedays = set(
----> 2     data_valid[["case", "day", "slice"]]
      3     .drop_duplicates()
      4     .value_counts(["case", "day"])
      5     .loc[lambda s: s == 80]

NameError: name 'data_valid' is not defined

## === cell 38
def one_epoch_valid():
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        pred_masks_dict, masks_dict = {}, {}
        for data in dataloader_valid:
            imgs, masks, ids = data
            imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(
                DEVICE, dtype=torch.float
            )
            pred_masks = model(imgs)
            loss = loss_fn(pred_masks, masks)
            running_loss += loss.item()
            pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()
            masks = masks.int()
            dice_score_obj.update(pred_masks, masks)
            for p, m, id_ in zip(pred_masks, masks, ids):
                match = re.match(r"case(\d+)_day(\d+)_slice_(\d+)", id_)
                if match:
                    caseid, dayid, sliceid = map(int, match.groups())
                casedayid = (caseid, dayid)
                pred_masks_dict.setdefault(casedayid, []).append((sliceid, p))
                masks_dict.setdefault(casedayid, []).append((sliceid, m))
                if (len(pred_masks_dict[casedayid]) == 144) or (
                    casedayid in slices80_casedays
                    and len(pred_masks_dict[casedayid]) == 80
                ):
                    pred_masks_sorted = [
                        p.cpu().numpy()
                        for sid, p in sorted(
                            pred_masks_dict[casedayid], key=lambda x: x[0]
                        )
                    ]
                    masks_sorted = [
                        m.cpu().numpy()
                        for sid, m in sorted(masks_dict[casedayid], key=lambda x: x[0])
                    ]
                    pred_masks_volume = np.stack(pred_masks_sorted, axis=1)
                    masks_volume = np.stack(masks_sorted, axis=1)
                    hausdorff_obj.update(pred_masks_volume, masks_volume)
                    del pred_masks_dict[casedayid], masks_dict[casedayid]
        avg_loss = running_loss / len(dataloader_valid)
        epoch_dice_score = dice_score_obj.compute()
        dice_score_obj.reset()
        epoch_hausdorff = hausdorff_obj.compute()
        hausdorff_obj.reset()
    return avg_loss, epoch_dice_score, epoch_hausdorff




## === cell 39
if TRAIN_VALID_SPLIT:
    for epoch in range(EPOCHS):
        loss_train = one_epoch_train(epoch)
        loss_valid, dice_score, hausdorff = one_epoch_valid()
        dice_overall, dice_per_organ = dice_score
        hausdorff_overall, hausdorff_per_organ = hausdorff
        combined_metric = 0.4 * dice_overall + 0.6 * (1 - hausdorff_overall)
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {loss_train:.3f} | Valid Loss: {loss_valid:.3f} | "
            f"Combined metric: {combined_metric:.3f} | "
            f"Dice: {dice_overall:.3f} (LB {dice_per_organ[0]:.3f}, SB {dice_per_organ[1]:.3f}, S {dice_per_organ[2]:.3f}) | "
            f"Hausdorff: {hausdorff_overall:.3f} (LB {hausdorff_per_organ[0]:.3f}, SB {hausdorff_per_organ[1]:.3f}, S {hausdorff_per_organ[2]:.3f})"
        )




## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3156834231.py in <cell line: 0>()
----> 1 if TRAIN_VALID_SPLIT:
      2     for epoch in range(EPOCHS):
      3         loss_train = one_epoch_train(epoch)
      4         loss_valid, dice_score, hausdorff = one_epoch_valid()
      5         dice_overall, dice_per_organ = dice_score

NameError: name 'TRAIN_VALID_SPLIT' is not defined

## === cell 40
if TRAIN_VALID_SPLIT and SAVE_TRAIN_VALID_MODEL:
    torch.save(model.state_dict(), MODEL_PARAMS_FILE_NAME)




## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1258924253.py in <cell line: 0>()
----> 1 if TRAIN_VALID_SPLIT and SAVE_TRAIN_VALID_MODEL:
      2     torch.save(model.state_dict(), MODEL_PARAMS_FILE_NAME)
      3 
      4 

NameError: name 'TRAIN_VALID_SPLIT' is not defined

## === cell 41
if TEST_PREDICT:
    if LOAD_MODEL_FOR_TEST_PREDICT and os.path.exists(MODEL_PARAMS_LOAD_FILE_PATH):
        model.load_state_dict(
            torch.load(MODEL_PARAMS_LOAD_FILE_PATH, map_location=DEVICE)
        )
    model.eval()
    data_test = pd.read_csv(DIR_PATH + "sample_submission.csv")
    test_set_hidden = not bool(len(data_test))
    if test_set_hidden:
        data_test = data_valid_sub
    else:
        data_test[["case", "day", "slice"]] = data_test["id"].str.extract(
            r"case(\d+)_day(\d+)_slice_(\d+)"
        )
        path_df = get_path_df(train=False)
        data_test = data_test.merge(path_df, on=["case", "day", "slice"])
        int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
        data_test[int_cols] = data_test[int_cols].astype(np.uint32)
        float_cols = ["px_w", "px_h"]
        data_test[float_cols] = data_test[float_cols].astype(np.float32)

    transform_test = A.Compose(
        [
            A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation=cv2.INTER_NEAREST),
            A.Normalize(
                mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
            ),
            A.ToTensorV2(transpose_mask=False),
        ]
    )
    dataset_test = GITractDataset(data_test, is_train=False, transforms=transform_test)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=DATA_LOADER_NUM_WORKERS,
    )




## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2439606402.py in <cell line: 0>()
----> 1 if TEST_PREDICT:
      2     # Load a checkpoint only if explicitly requested and the file exists
      3     if LOAD_MODEL_FOR_TEST_PREDICT and os.path.exists(MODEL_PARAMS_LOAD_FILE_PATH):
      4         model.load_state_dict(
      5             torch.load(MODEL_PARAMS_LOAD_FILE_PATH, map_location=DEVICE)

NameError: name 'TEST_PREDICT' is not defined

## === cell 42
if TEST_PREDICT:
    test_ids, test_class, test_pred_RLE = (
        [],
        [],
        [],
    )  # data to be written to submission file

    with torch.no_grad():
        for imgs, ids, heights, widths in dataloader_test:
            imgs = imgs.to(DEVICE, dtype=torch.float)
            pred_masks = model(imgs)
            pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()
            pred_masks = pred_masks.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]

            for mask, id_, h, w in zip(pred_masks, ids, heights, widths):
                mask_orig_size = cv2.resize(
                    mask,
                    dsize=(int(w.item()), int(h.item())),
                    interpolation=cv2.INTER_NEAREST,
                )
                rles = [
                    rle_encode(mask_orig_size[..., chid]) for chid in range(NUM_CLASSES)
                ]
                test_ids.extend([id_] * NUM_CLASSES)
                test_class.extend(CLASS_NAMES)
                test_pred_RLE.extend(rles)

    submission_df = pd.DataFrame(
        {"id": test_ids, "class": test_class, "predicted": test_pred_RLE}
    )
    submission_df.to_csv("submission.csv", index=False)
    print(submission_df.head())

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2267126919.py in <cell line: 0>()
----> 1 if TEST_PREDICT:
      2     test_ids, test_class, test_pred_RLE = (
      3         [],
      4         [],
      5         [],

NameError: name 'TEST_PREDICT' is not defined
