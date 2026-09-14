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
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

dill==0.4.0
fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.6023259240548947

# 6. Current score

0.00209

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00209) has done: 'The notebook fails because it references pretrained `.pkl` models in `../input/fgvc8hotel/` that do not exist in your dataset paths; that leaves `probs=None` and cascades into later errors. I fix this by (1) automatically locating any available exported fastai learners (`.pkl`) under the provided `../input/` tree and using them if present, and (2) adding a safe fallback that still produces a valid MAP@5 submission (using the top-5 most frequent `hotel_id` from `train.csv`) when no models are found. I also fix path handling for test images so the test dataloader receives proper file paths, and ensure the submission is written as `submission.csv` with the required columns.'
- What this solution (achieved 0.00209) has done: 'Your current score is low because the notebook almost certainly falls back to the “top-5 most frequent hotels” baseline (meaning no usable exported learner is found/loaded), and even when a learner loads, `test_dl` is built from string paths in a way that often won’t be recognized as valid images by the learner’s pipeline. I make two minimal, score-relevant fixes: (1) reliably locate and load the specific exported fastai learner(s) if present (prefer `export.pkl`), and (2) construct the test dataloader from actual image `Path` objects (not a renamed string column) so transforms work and predictions aren’t garbage. I also correctly ensemble multiple learners by averaging probabilities (instead of summing without normalization) and ensure vocab-to-hotel_id mapping is robust (handles list/CategoryMap). These changes keep the same core approach (load exported learners → TTA → top-5) but should move MAP@5 substantially upward toward your target if any valid exports exist; if not, it still writes a valid submission.'
- What this solution (achieved 0.00209) has done: 'I keep your fastai inference approach (load exported learners → build test_dl → TTA → top-5) but make two minimal, score-relevant fixes that commonly cause near-random MAP@5: (1) only search for exported learners that are likely compatible (and avoid accidentally loading unrelated `.pkl` files from other Kaggle datasets), and (2) construct the test dataloader from a list of `PILImage` objects (not bare Paths) to reliably trigger the same item transforms as training. I also normalize the ensemble probabilities safely and ensure the predicted hotel IDs are always space-delimited strings matching submission order. These changes should move you off the “fallback/top5” behavior and toward your target if an actual export exists, while still producing a valid `submission.csv` in all cases.'
- What this solution (achieved 0.00209) has done: 'Your very low score strongly suggests you’re still in the fallback path (no valid exported learner found/loaded), so the smallest meaningful move toward the target is to train a quick in-notebook fastai classifier and use it for inference when no compatible `.pkl` is available. I keep your existing inference core (fastai learner → test_dl → TTA → top-5) and only add a minimal, time-bounded training fallback that uses the competition’s train_images + train.csv to build a standard `ImageDataLoaders` pipeline. I also make the test image loading more robust by feeding Paths (not PIL objects) into `test_dl`, which is the most reliable way to trigger the same item transforms. This should move MAP@5 substantially upward toward your target while still producing a valid `submission.csv` within Kaggle constraints.'
- What this solution (achieved 0.00209) has done: 'The timeout is driven by two expensive behaviors: scanning the whole `/kaggle/input` tree for many `.pkl` candidates and (when no export is used) training a fallback model + running `tta(n=5)` which is far too slow for the full dataset under 600s. The optimized script keeps identical inference semantics (load exported learner(s) → `test_dl` → `tta` → top-5) but makes export discovery O(1) by checking only a few exact expected filenames under the competition folder and avoids repeated/slow filesystem checks. It also makes prediction faster (without changing results) by using `torch.inference_mode()` and by reusing a single `test_dl` per learner rather than rebuilding from scratch in ways that trigger extra pipeline work. The fallback training block is kept logically identical but is guarded so it only runs when absolutely necessary (no exports found), and its path existence filtering is vectorized to avoid Python overhead.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import pandas as pd
import numpy as np
import fastai
from fastai.vision.all import *
import dill

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
fastai.__version__, torch.__version__



## === cell 2
DATA_ROOT = Path("../input/hotel-id-2021-fgvc8")
TEST_IMG_DIR = DATA_ROOT / "test_images"
TRAIN_IMG_DIR = DATA_ROOT / "train_images"
TRAIN_CSV = DATA_ROOT / "train.csv"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"

assert SAMPLE_SUB.exists(), f"Missing sample submission at: {SAMPLE_SUB}"
assert TRAIN_CSV.exists(), f"Missing train.csv at: {TRAIN_CSV}"
assert TEST_IMG_DIR.exists(), f"Missing test_images dir at: {TEST_IMG_DIR}"
assert TRAIN_IMG_DIR.exists(), f"Missing train_images dir at: {TRAIN_IMG_DIR}"

submission = pd.read_csv(SAMPLE_SUB)

test_paths = [TEST_IMG_DIR / img for img in submission["image"].tolist()]
missing_ct = int(np.sum([not os.path.exists(str(p)) for p in test_paths]))
missing_ct



## === cell 3
INPUT_ROOT = Path("../input")


def find_exported_learners(root: Path, max_candidates: int = 25) -> list:
    if not root.exists():
        return []

    comp_root = root / "hotel-id-2021-fgvc8"

    likely = []
    if comp_root.exists():
        likely += [
            comp_root / "export.pkl",
            comp_root / "models" / "export.pkl",
            comp_root / "export_0.pkl",
            comp_root / "export_1.pkl",
            comp_root / "export_resnet34.pkl",
            comp_root / "export_resnet50.pkl",
        ]

        for sub in ("", "hotel-id-2021-fgvc8"):
            base = comp_root / sub if sub else comp_root
            models_dir = base / "models"
            if models_dir.exists():
                likely += [
                    models_dir / "export.pkl",
                    models_dir / "export_0.pkl",
                    models_dir / "export_1.pkl",
                ]

    existing = [p for p in likely if p.is_file() and p.stat().st_size > 1024]
    existing = sorted(set(existing), key=lambda p: str(p))

    if len(existing) > 0:
        return existing[:max_candidates] if max_candidates is not None else existing

    candidates = []
    if comp_root.exists():
        candidates += list(comp_root.rglob("export.pkl"))
        candidates += list(comp_root.rglob("export_*.pkl"))
        candidates += list(comp_root.rglob("export*.pkl"))

    pkl_files = [p for p in candidates if p.is_file() and p.stat().st_size > 1024]
    pkl_files = sorted(set(pkl_files), key=lambda p: str(p))
    if max_candidates is not None and len(pkl_files) > max_candidates:
        pkl_files = pkl_files[:max_candidates]
    return pkl_files


models = find_exported_learners(INPUT_ROOT)
models[:10], len(models)




## === cell 4
def _vocab_to_list(v):
    """Normalize vocab to a plain list for correct index->hotel_id mapping."""
    if v is None:
        return None
    if hasattr(v, "items"):
        try:
            return list(v.items)
        except Exception:
            pass
    try:
        return list(v)
    except Exception:
        return None


def _try_load_and_predict_with_exports(models, test_paths):
    probs_sum = None
    n_models_used = 0
    learn_last = None

    if len(models) == 0:
        return None, None, 0

    if not all(os.path.exists(str(p)) for p in test_paths):
        return None, None, 0

    test_items = list(test_paths)

    for model_path in models:
        try:
            learn_tmp = load_learner(
                fname=Path(model_path), cpu=False, pickle_module=dill
            )

            test_dl = learn_tmp.dls.test_dl(test_items, num_workers=0)

            with torch.inference_mode():
                probs_temp, _ = learn_tmp.tta(dl=test_dl, n=5)

            probs_sum = probs_temp if probs_sum is None else (probs_sum + probs_temp)
            n_models_used += 1
            learn_last = learn_tmp
        except Exception as e:
            print(f"Skipping model {model_path} due to error: {type(e).__name__}: {e}")
            continue

    if probs_sum is None or learn_last is None or n_models_used == 0:
        return None, None, 0
    return probs_sum / float(n_models_used), learn_last, n_models_used


probs, learn, n_models_used = _try_load_and_predict_with_exports(models, test_paths)
(n_models_used, probs is not None)




## === cell 5
def _train_fallback_learner_and_predict(train_csv_path, train_img_dir, test_paths):
    df = pd.read_csv(train_csv_path)

    df["hotel_id"] = df["hotel_id"].astype(str)
    df["chain"] = df["chain"].astype(str)

    rel_paths = df["chain"].astype(str).str.cat(df["image"].astype(str), sep=os.sep)
    img_paths = (train_img_dir / rel_paths).astype(str).to_numpy()

    exists = np.fromiter(
        (os.path.exists(p) for p in img_paths), dtype=np.bool_, count=img_paths.shape[0]
    )
    df = df.loc[exists].reset_index(drop=True)

    if len(df) < 1000:
        return None, None

    max_train = 30000
    if len(df) > max_train:
        df = df.sample(n=max_train, random_state=42).reset_index(drop=True)

    dls = ImageDataLoaders.from_df(
        df,
        fn_col=(
            "image_path" if "image_path" in df.columns else None
        ),  # kept for safety below
        label_col="hotel_id",
        valid_pct=0.1,
        seed=42,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(size=224, min_scale=0.75),
        bs=64,
        num_workers=0,
    )

    if dls.items is None or len(dls.items) == 0:
        df = df.copy()
        df["image_path"] = img_paths[exists]
        dls = ImageDataLoaders.from_df(
            df,
            fn_col="image_path",
            label_col="hotel_id",
            valid_pct=0.1,
            seed=42,
            item_tfms=Resize(224),
            batch_tfms=aug_transforms(size=224, min_scale=0.75),
            bs=64,
            num_workers=0,
        )

    learn_fb = vision_learner(dls, resnet34, metrics=[])
    learn_fb.fine_tune(1)

    test_dl = learn_fb.dls.test_dl(list(test_paths), num_workers=0)
    with torch.inference_mode():
        probs_fb, _ = learn_fb.tta(dl=test_dl, n=5)

    return probs_fb, learn_fb


if probs is None or learn is None or n_models_used == 0:
    print(
        "No usable exported learner found; training a minimal fallback model for better MAP@5."
    )
    probs_fb, learn_fb = _train_fallback_learner_and_predict(
        TRAIN_CSV, TRAIN_IMG_DIR, test_paths
    )
    if probs_fb is not None and learn_fb is not None:
        probs, learn = probs_fb, learn_fb

(probs is not None, learn is not None)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/311258098.py in <cell line: 0>()
     67         "No usable exported learner found; training a minimal fallback model for better MAP@5."
     68     )
---> 69     probs_fb, learn_fb = _train_fallback_learner_and_predict(
     70         TRAIN_CSV, TRAIN_IMG_DIR, test_paths
     71     )

/tmp/ipykernel_55/311258098.py in _train_fallback_learner_and_predict(train_csv_path, train_img_dir, test_paths)
     23         df = df.sample(n=max_train, random_state=42).reset_index(drop=True)
     24 
---> 25     dls = ImageDataLoaders.from_df(
     26         df,
     27         fn_col=(

/usr/local/lib/python3.11/dist-packages/fastai/vision/data.py in from_df(cls, df, path, valid_pct, seed, fn_col, folder, suff, label_col, label_delim, y_block, valid_col, item_tfms, batch_tfms, img_cls, **kwargs)
    177                            item_tfms=item_tfms,
    178                            batch_tfms=batch_tfms)
--> 179         return cls.from_dblock(dblock, df, path=path, **kwargs)
    180 
    181     @classmethod

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in from_dblock(cls, dblock, source, path, bs, val_bs, shuffle, device, **kwargs)
    278         **kwargs
    279     ):
--> 280         return dblock.dataloaders(source, path=path, bs=bs, val_bs=val_bs, shuffle=shuffle, device=device, **kwargs)
    281 
    282     _docs=dict(__getitem__="Retrieve `DataLoader` at `i` (`0` is training, `1` is validation)",

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in dataloaders(self, source, path, verbose, **kwargs)
    157         dsets = self.datasets(source, verbose=verbose)
    158         kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
--> 159         return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)
    160 
    161     _docs = dict(new="Create a new `DataBlock` with other `item_tfms` and `batch_tfms`",

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in dataloaders(self, bs, shuffle_train, shuffle, val_shuffle, n, path, dl_type, dl_kwargs, device, drop_last, val_bs, **kwargs)
    331         dl = dl_type(self.subset(0), **merge(kwargs,def_kwargs, dl_kwargs[0]))
    332         def_kwargs = {'bs':bs if val_bs is None else val_bs,'shuffle':val_shuffle,'n':None,'drop_last':False}
--> 333         dls = [dl] + [dl.new(self.subset(i), **merge(kwargs,def_kwargs,val_kwargs,dl_kwargs[i]))
    334                       for i in range(1, self.n_subsets)]
    335         return self._dbunch_type(*dls, path=path, device=device)

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in <listcomp>(.0)
    331         dl = dl_type(self.subset(0), **merge(kwargs,def_kwargs, dl_kwargs[0]))
    332         def_kwargs = {'bs':bs if val_bs is None else val_bs,'shuffle':val_shuffle,'n':None,'drop_last':False}
--> 333         dls = [dl] + [dl.new(self.subset(i), **merge(kwargs,def_kwargs,val_kwargs,dl_kwargs[i]))
    334                       for i in range(1, self.n_subsets)]
    335         return self._dbunch_type(*dls, path=path, device=device)

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in new(self, dataset, cls, **kwargs)
    102         if not hasattr(self, '_n_inp') or not hasattr(self, '_types'):
    103             try:
--> 104                 self._one_pass()
    105                 res._n_inp,res._types = self._n_inp,self._types
    106             except Exception as e:

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in _one_pass(self)
     85         b = self.do_batch([self.do_item(None)])
     86         if self.device is not None: b = to_device(b, self.device)
---> 87         its = self.after_batch(b)
     88         self._n_inp = 1 if not isinstance(its, (list,tuple)) or len(its)==1 else len(its)-1
     89         self._types = explode_types(its)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in __call__(self, o)
    246         self.fs = self.fs.sorted(key='order')
    247 
--> 248     def __call__(self, o): return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
    249     def __repr__(self): return f"Pipeline: {' -> '.join([f.name for f in self.fs if f.name != 'noop'])}"
    250     def __getitem__(self,i): return self.fs[i]

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in compose_tfms(x, tfms, is_enc, reverse, **kwargs)
    195     for f in tfms:
    196         if not is_enc: f = f.decode
--> 197         x = f(x, **kwargs)
    198     return x
    199 

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in __call__(self, b, split_idx, **kwargs)
     48         **kwargs
     49     ):
---> 50         self.before_call(b, split_idx=split_idx)
     51         return super().__call__(b, split_idx=split_idx, **kwargs) if self.do else b
     52 

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in before_call(self, b, split_idx)
    479         while isinstance(b, tuple): b = b[0]
    480         self.split_idx = split_idx
--> 481         self.do,self.mat = True,self._get_affine_mat(b)
    482         for t in self.coord_fs: t.before_call(b)
    483 

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in _get_affine_mat(self, x)
    490 
    491     def _get_affine_mat(self, x):
--> 492         aff_m = _init_mat(x)
    493         if self.split_idx: return _prepare_mat(x, aff_m)
    494         ms = [f(x) for f in self.aff_fs]

/usr/local/lib/python3.11/dist-packages/fastai/vision/augment.py in _init_mat(x)
    348 # %% ../../nbs/09_vision.augment.ipynb 77
    349 def _init_mat(x):
--> 350     mat = torch.eye(3, device=x.device).float()
    351     return mat.unsqueeze(0).expand(x.size(0), 3, 3).contiguous()
    352 

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __getattr__(self, k)
    551         if self._component_attr_filter(k):
    552             attr = getattr(self,self._default,None)
--> 553             if attr is not None: return getattr(attr,k)
    554         raise AttributeError(k)
    555     def __dir__(self): return custom_dir(self,self._dir())

AttributeError: 'list' object has no attribute 'device'

## === cell 6
if probs is not None and learn is not None:
    preds_idx = probs.topk(5)[1].detach().cpu().numpy()

    vocab_raw = getattr(learn.dls, "vocab", None)
    vocab_list = _vocab_to_list(vocab_raw)

    if vocab_list is None or len(vocab_list) == 0:
        preds = [" ".join(map(str, row)) for row in preds_idx]
    else:
        preds = [" ".join([str(vocab_list[i]) for i in row]) for row in preds_idx]
else:
    train_df = pd.read_csv(TRAIN_CSV)
    top5 = train_df["hotel_id"].value_counts().head(5).index.tolist()
    top5_str = " ".join(map(str, top5))
    preds = [top5_str] * len(submission)

preds[:3], len(preds)



## === cell 7
submission["hotel_id"] = preds
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)

submission.head(), str(submission_path.resolve())
