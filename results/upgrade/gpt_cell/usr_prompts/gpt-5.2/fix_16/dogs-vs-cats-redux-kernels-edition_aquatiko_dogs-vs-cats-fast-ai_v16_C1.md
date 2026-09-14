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

3.7

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 4. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
BASE = "../input/"
COMP_SUB = "dogs-vs-cats-redux-kernels-edition"
if os.path.isdir(os.path.join(BASE, COMP_SUB)):
    PATH = os.path.join(BASE, COMP_SUB) + "/"
else:
    PATH = BASE

TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz = 224

print("Using PATH:", PATH)
print("PATH contents:", os.listdir(PATH)[:20])



## === cell 2
train_cat_dir = os.path.join(PATH, "train", "cat")
train_dog_dir = os.path.join(PATH, "train", "dog")

if os.path.isdir(train_cat_dir) and os.path.isdir(train_dog_dir):
    cat_files = sorted(
        [
            os.path.join("train", "cat", f)
            for f in os.listdir(train_cat_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
    )
    dog_files = sorted(
        [
            os.path.join("train", "dog", f)
            for f in os.listdir(train_dog_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
    )
    fnames = np.array(cat_files + dog_files)
    labels = np.array([0] * len(cat_files) + [1] * len(dog_files))
else:
    train_dir = os.path.join(PATH, "train")
    fnames = np.array(
        [
            f"train/{f}"
            for f in sorted(os.listdir(train_dir))
            if os.path.isfile(os.path.join(train_dir, f))
            and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
    )
    labels = np.array([(0 if "cat" in fname else 1) for fname in fnames])

print("Num train images:", len(fnames), "Num labels:", len(labels))



## === cell 3
print(fnames[-2], labels[-2])



## === cell 4
try:
    from fastai.imports import *
    from fastai.transforms import *
    from fastai.conv_learner import *
    from fastai.model import *
    from fastai.dataset import *
    from fastai.sgdr import *
    from fastai.plots import *
except ModuleNotFoundError:
    try:
        from torchvision.models import (
            resnet18,
            resnet34,
            resnet50,
            resnet101,
            resnet152,
        )
    except Exception as e:
        raise ModuleNotFoundError(
            "Required fastai v0.7 modules are unavailable, and torchvision could not be imported "
            "to provide resnet architectures needed by later cells."
        ) from e



## === cell 5
arch = resnet101



## === cell 6
if (
    "ImageClassifierData" not in globals()
    or "tfms_from_model" not in globals()
    or "ConvLearner" not in globals()
):
    import os
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    try:
        import cv2  # optional; if unavailable we'll keep PIL

        _HAS_CV2 = True
    except Exception:
        cv2 = None
        _HAS_CV2 = False

    from PIL import Image

    try:
        Image.MAX_IMAGE_PIXELS = None
        try:
            from PIL import ImageFile

            ImageFile.LOAD_TRUNCATED_IMAGES = True
        except Exception:
            pass
    except Exception:
        pass

    try:
        import torchvision.transforms as T
    except Exception as e:
        raise ModuleNotFoundError(
            "torchvision is required to provide image transforms in the fastai-compat shim."
        ) from e

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def tfms_from_model(arch, sz):
        return {
            "train": T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
            "valid": T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        }

    _NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    _NORM_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

    if _HAS_CV2:
        try:
            cv2.setNumThreads(max(1, (os.cpu_count() or 2) // 2))
        except Exception:
            pass

    _CACHE_DIR = "/tmp/dvc_memmap_cache_sz{}".format(int(sz))
    os.makedirs(_CACHE_DIR, exist_ok=True)

    def _safe_name(tag: str) -> str:
        return "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in tag])

    def _collect_image_relpaths_under(root_dir, base_path):
        _ends = (".jpg", ".jpeg", ".png", ".bmp")
        out = []
        stack = [root_dir]
        while stack:
            d = stack.pop()
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir():
                        stack.append(entry.path)
                    elif entry.is_file():
                        n = entry.name
                        if n.lower().endswith(_ends):
                            out.append(os.path.relpath(entry.path, base_path))
        out.sort()
        return out

    def _read_rgb_resized_tensor_nocache(path, sz, transform):
        if _HAS_CV2:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_LINEAR)
                t = (
                    torch.from_numpy(img)
                    .permute(2, 0, 1)
                    .contiguous()
                    .float()
                    .div_(255.0)
                )
                t.sub_(_NORM_MEAN).div_(_NORM_STD)
                return t
        pil = Image.open(path).convert("RGB").resize((sz, sz), resample=Image.BILINEAR)
        return transform(pil)

    def _build_or_load_memmap(root, relpaths, transform, sz, tag):
        import multiprocessing as mp

        tag = _safe_name(tag)
        data_path = os.path.join(_CACHE_DIR, f"{tag}.mmap")
        meta_path = os.path.join(_CACHE_DIR, f"{tag}.npz")

        n = len(relpaths)
        shape = (n, 3, int(sz), int(sz))
        dtype = np.float32

        if os.path.exists(data_path) and os.path.exists(meta_path):
            try:
                meta = np.load(meta_path, allow_pickle=False)
                if (
                    int(meta["n"]) == n
                    and tuple(meta["shape"]) == shape
                    and str(meta["dtype"]) == str(np.dtype(dtype))
                    and np.array_equal(
                        meta["relpaths"], np.array(relpaths, dtype=object)
                    )
                ):
                    mm = np.memmap(data_path, mode="r", dtype=dtype, shape=shape)
                    return mm, relpaths
            except Exception:
                pass

        mm = np.memmap(data_path, mode="w+", dtype=dtype, shape=shape)

        _mp_ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")

        cpu_cnt = os.cpu_count() or 2
        n_workers = min(8, max(2, cpu_cnt // 2))

        chunksize = 32

        global _MM_ROOT, _MM_SZ, _MM_HAS_CV2
        _MM_ROOT = root
        _MM_SZ = int(sz)
        _MM_HAS_CV2 = _HAS_CV2

        def _worker_init():
            import random as _r
            import numpy as _np
            import torch as _torch

            _r.seed(SEED)
            _np.random.seed(SEED)
            _torch.manual_seed(SEED)

        def _decode_one(args):
            i, rp = args
            p = os.path.join(_MM_ROOT, rp)
            t = _read_rgb_resized_tensor_nocache(p, _MM_SZ, transform)
            return i, t.numpy()

        it = ((i, rp) for i, rp in enumerate(relpaths))
        if n_workers <= 1:
            for i, arr in map(_decode_one, it):
                mm[i, :, :, :] = arr
        else:
            with _mp_ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
                for i, arr in pool.imap_unordered(_decode_one, it, chunksize=chunksize):
                    mm[i, :, :, :] = arr

        mm.flush()
        np.savez_compressed(
            meta_path,
            n=np.int64(n),
            shape=np.array(shape, dtype=np.int64),
            dtype=np.dtype(dtype).str,
            relpaths=np.array(relpaths, dtype=object),
        )
        mm = np.memmap(data_path, mode="r", dtype=dtype, shape=shape)
        return mm, relpaths

    class _MemmapImageDataset(Dataset):
        def __init__(self, memmap_arr, relpaths, y=None):
            self.mm = memmap_arr
            self.relpaths = list(relpaths)
            self.y = None if y is None else np.array(y, dtype=np.int64)

        def __len__(self):
            return self.mm.shape[0]

        def __getitem__(self, idx):
            x = torch.from_numpy(self.mm[idx])  # float32 CHW already normalized
            if self.y is None:
                return x
            return x, int(self.y[idx])

    class ImageClassifierData:
        def __init__(self, path, trn_dl, val_dl, classes, test_dl=None):
            self.path = path
            self.trn_dl = trn_dl
            self.val_dl = val_dl
            self.classes = classes
            self.test_dl = test_dl

        @classmethod
        def from_names_and_array(cls, path, fnames, y, classes, test_name, tfms, bs=32):
            n = len(fnames)
            idx = np.arange(n)
            rng = np.random.RandomState(42)
            rng.shuffle(idx)
            val_sz = max(1, int(0.2 * n))
            val_idx = idx[:val_sz]
            trn_idx = idx[val_sz:]

            trn_fnames = np.array(fnames)[trn_idx].tolist()
            trn_y = np.array(y)[trn_idx]
            val_fnames = np.array(fnames)[val_idx].tolist()
            val_y = np.array(y)[val_idx]

            trn_mm, trn_rel = _build_or_load_memmap(
                path, trn_fnames, tfms["train"], sz=sz, tag="train"
            )
            val_mm, val_rel = _build_or_load_memmap(
                path, val_fnames, tfms["valid"], sz=sz, tag="valid"
            )

            trn_ds = _MemmapImageDataset(trn_mm, trn_rel, y=trn_y)
            val_ds = _MemmapImageDataset(val_mm, val_rel, y=val_y)

            device_is_cuda = torch.cuda.is_available()
            cpu_cnt = os.cpu_count() or 2

            num_workers = min(8, max(2, cpu_cnt // 2))
            pin_memory = bool(device_is_cuda)

            def _wif(worker_id):
                seed = SEED + worker_id
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

            dl_kwargs = dict(
                batch_size=bs,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=_wif if num_workers > 0 else None,
            )

            _sig = None
            try:
                import inspect

                _sig = inspect.signature(DataLoader.__init__)
            except Exception:
                _sig = None

            if _sig is not None:
                params = _sig.parameters
                if "persistent_workers" in params and num_workers > 0:
                    dl_kwargs["persistent_workers"] = True
                if "prefetch_factor" in params and num_workers > 0:
                    dl_kwargs["prefetch_factor"] = 4

            trn_dl = DataLoader(trn_ds, shuffle=True, **dl_kwargs)
            val_dl = DataLoader(val_ds, shuffle=False, **dl_kwargs)

            test_dl = None
            test_dir = os.path.join(path, test_name)
            if os.path.isdir(test_dir):
                test_files = _collect_image_relpaths_under(test_dir, path)
                if len(test_files) > 0:
                    test_mm, test_rel = _build_or_load_memmap(
                        path, test_files, tfms["valid"], sz=sz, tag="test"
                    )
                    test_ds = _MemmapImageDataset(test_mm, test_rel, y=None)
                    test_dl = DataLoader(test_ds, shuffle=False, **dl_kwargs)
                    test_dl._cached_fnames = test_rel

            return cls(
                path=path,
                trn_dl=trn_dl,
                val_dl=val_dl,
                classes=classes,
                test_dl=test_dl,
            )

    class ConvLearner:
        def __init__(self, model, data, device):
            self.model = model
            self.data = data
            self.device = device
            self.model.to(self.device)
            self.crit = nn.CrossEntropyLoss()

        @classmethod
        def pretrained(
            cls, arch, data, precompute=True, tmp_name=None, models_name=None
        ):
            model = arch(pretrained=True)
            if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, len(data.classes))
            else:
                if hasattr(model, "classifier") and isinstance(
                    model.classifier, nn.Linear
                ):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, len(data.classes))
                else:
                    raise RuntimeError(
                        "Unsupported architecture for minimal ConvLearner shim."
                    )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                model = model.to(memory_format=torch.channels_last)
            return cls(model=model, data=data, device=device)

        def fit(self, lr, epochs):
            self.model.train()
            opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            model = self.model
            crit = self.crit
            device = self.device
            use_cuda = device.type == "cuda"

            for _ in range(int(epochs)):
                for xb, yb in self.data.trn_dl:
                    if use_cuda:
                        xb = xb.to(device, non_blocking=True).to(
                            memory_format=torch.channels_last
                        )
                        yb = yb.to(device, non_blocking=True)
                    else:
                        xb = xb.to(device)
                        yb = yb.to(device)

                    opt.zero_grad(set_to_none=True)
                    out = model(xb)
                    loss = crit(out, yb)
                    loss.backward()
                    opt.step()
            return None


data = ImageClassifierData.from_names_and_array(
    path=PATH,
    fnames=fnames,
    y=labels,
    classes=["cat", "dog"],  # explicit mapping: 0=cat, 1=dog
    test_name="test",
    tfms=tfms_from_model(arch, sz),
    bs=32,
)
learn = ConvLearner.pretrained(
    arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH
)



## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2239430312.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    366[0m [0;34m[0m[0m
[1;32m    367[0m [0;34m[0m[0m
[0;32m--> 368[0;31m data = ImageClassifierData.from_names_and_array(
[0m[1;32m    369[0m     [0mpath[0m[0;34m=[0m[0mPATH[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    370[0m     [0mfnames[0m[0;34m=[0m[0mfnames[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2239430312.py[0m in [0;36mfrom_names_and_array[0;34m(cls, path, fnames, y, classes, test_name, tfms, bs)[0m
[1;32m    239[0m             [0mval_y[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0my[0m[0;34m)[0m[0;34m[[0m[0mval_idx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    240[0m [0;34m[0m[0m
[0;32m--> 241[0;31m             trn_mm, trn_rel = _build_or_load_memmap(
[0m[1;32m    242[0m                 [0mpath[0m[0;34m,[0m [0mtrn_fnames[0m[0;34m,[0m [0mtfms[0m[0;34m[[0m[0;34m"train"[0m[0;34m][0m[0;34m,[0m [0msz[0m[0;34m=[0m[0msz[0m[0;34m,[0m [0mtag[0m[0;34m=[0m[0;34m"train"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    243[0m             )

[0;32m/tmp/ipykernel_11/2239430312.py[0m in [0;36m_build_or_load_memmap[0;34m(root, relpaths, transform, sz, tag)[0m
[1;32m    187[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    188[0m             [0;32mwith[0m [0m_mp_ctx[0m[0;34m.[0m[0mPool[0m[0;34m([0m[0mprocesses[0m[0;34m=[0m[0mn_workers[0m[0;34m,[0m [0minitializer[0m[0;34m=[0m[0m_worker_init[0m[0;34m)[0m [0;32mas[0m [0mpool[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 189[0;31m                 [0;32mfor[0m [0mi[0m[0;34m,[0m [0marr[0m [0;32min[0m [0mpool[0m[0;34m.[0m[0mimap_unordered[0m[0;34m([0m[0m_decode_one[0m[0;34m,[0m [0mit[0m[0;34m,[0m [0mchunksize[0m[0;34m=[0m[0mchunksize[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    190[0m                     [0mmm[0m[0;34m[[0m[0mi[0m[0;34m,[0m [0;34m:[0m[0;34m,[0m [0;34m:[0m[0;34m,[0m [0;34m:[0m[0;34m][0m [0;34m=[0m [0marr[0m[0;34m[0m[0;34m[0m[0m
[1;32m    191[0m [0;34m[0m[0m

[0;32m/usr/lib/python3.11/multiprocessing/pool.py[0m in [0;36m<genexpr>[0;34m(.0)[0m
[1;32m    449[0m                     [0mresult[0m[0;34m.[0m[0m_set_length[0m[0;34m[0m[0;34m[0m[0m
[1;32m    450[0m                 ))
[0;32m--> 451[0;31m             [0;32mreturn[0m [0;34m([0m[0mitem[0m [0;32mfor[0m [0mchunk[0m [0;32min[0m [0mresult[0m [0;32mfor[0m [0mitem[0m [0;32min[0m [0mchunk[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    452[0m [0;34m[0m[0m
[1;32m    453[0m     def apply_async(self, func, args=(), kwds={}, callback=None,

[0;32m/usr/lib/python3.11/multiprocessing/pool.py[0m in [0;36mnext[0;34m(self, timeout)[0m
[1;32m    871[0m         [0;32mif[0m [0msuccess[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    872[0m             [0;32mreturn[0m [0mvalue[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 873[0;31m         [0;32mraise[0m [0mvalue[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    874[0m [0;34m[0m[0m
[1;32m    875[0m     [0m__next__[0m [0;34m=[0m [0mnext[0m                    [0;31m# XXX[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/multiprocessing/pool.py[0m in [0;36m_handle_tasks[0;34m(taskqueue, put, outqueue, pool, cache)[0m
[1;32m    538[0m                         [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m
[1;32m    539[0m                     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 540[0;31m                         [0mput[0m[0;34m([0m[0mtask[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    541[0m                     [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    542[0m                         [0mjob[0m[0;34m,[0m [0midx[0m [0;34m=[0m [0mtask[0m[0;34m[[0m[0;34m:[0m[0;36m2[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/multiprocessing/connection.py[0m in [0;36msend[0;34m(self, obj)[0m
[1;32m    204[0m         [0mself[0m[0;34m.[0m[0m_check_closed[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    205[0m         [0mself[0m[0;34m.[0m[0m_check_writable[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 206[0;31m         [0mself[0m[0;34m.[0m[0m_send_bytes[0m[0;34m([0m[0m_ForkingPickler[0m[0;34m.[0m[0mdumps[0m[0;34m([0m[0mobj[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    207[0m [0;34m[0m[0m
[1;32m    208[0m     [0;32mdef[0m [0mrecv_bytes[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mmaxlength[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/multiprocessing/reduction.py[0m in [0;36mdumps[0;34m(cls, obj, protocol)[0m
[1;32m     49[0m     [0;32mdef[0m [0mdumps[0m[0;34m([0m[0mcls[0m[0;34m,[0m [0mobj[0m[0;34m,[0m [0mprotocol[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     50[0m         [0mbuf[0m [0;34m=[0m [0mio[0m[0;34m.[0m[0mBytesIO[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 51[0;31m         [0mcls[0m[0;34m([0m[0mbuf[0m[0;34m,[0m [0mprotocol[0m[0;34m)[0m[0;34m.[0m[0mdump[0m[0;34m([0m[0mobj[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     52[0m         [0;32mreturn[0m [0mbuf[0m[0;34m.[0m[0mgetbuffer[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     53[0m [0;34m[0m[0m

[0;31mAttributeError[0m: Can't pickle local object '_build_or_load_memmap.<locals>._decode_one'

## === cell 7
learn.fit(lr=0.01, epochs=1)
