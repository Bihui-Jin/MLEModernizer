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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

19.137224051093728

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, warnings, pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")



## === cell 1
epochs = 2  # training epochs
batch_size = 32
num_workers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_INPUT = pathlib.Path("/kaggle/input/petfinder-pawpularity-score")
if not BASE_INPUT.exists():
    BASE_INPUT = pathlib.Path("../input/petfinder-pawpularity-score")

train_pic_path = BASE_INPUT / "train"
test_pic_path = BASE_INPUT / "test"
train_csv_path = BASE_INPUT / "train.csv"
test_csv_path = BASE_INPUT / "test.csv"
model_path = pathlib.Path("../input/notebookd97ba6b497/pic_net_model.pt")




## === cell 2
class FineTuneResnet50(nn.Module):
    def __init__(self, num_class=1):
        super(FineTuneResnet50, self).__init__()
        resnet50_net = models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(resnet50_net.children())[:-1])
        self.classifier = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x




## === cell 3
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))




## === cell 4
class PicSet(Dataset):
    def __init__(self, root_folder, train=False):
        self.train = train
        self.root = pathlib.Path(root_folder)
        self.img_files = list(self.root.iterdir())  # list of image paths
        self.transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        csv_path = self.root.with_suffix(".csv")
        self.csv_data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_id = self.csv_data.loc[idx, "Id"]
        img_path = self.root / f"{img_id}.jpg"
        pil_img = Image.open(img_path).convert("RGB")
        pic = self.transforms(pil_img)

        if self.train:
            label = torch.tensor(
                self.csv_data.loc[idx, "Pawpularity"], dtype=torch.float32
            )
            return {"pic": pic, "label": label}
        else:
            return {"pic": pic}




## === cell 5
def load_pic(train_path, test_path):
    full_train = PicSet(train_path, train=True)
    n = len(full_train)
    train_sz = int(n * 0.9)
    val_sz = n - train_sz
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [train_sz, val_sz]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_dataset = PicSet(test_path, train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader




## === cell 6
if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_pic(train_pic_path, test_pic_path)

    test_df = pd.read_csv(test_csv_path)

    pic_net = FineTuneResnet50(num_class=1).to(device)

    if model_path.is_file():
        try:
            pic_net.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded pretrained weights.")
        except Exception as e:
            print(f"Failed to load pretrained model ({e}); training from scratch.")
    else:
        print("Pretrained model file not found; training from scratch.")

    criterion = RMSELoss()
    optimizer = optim.Adam(pic_net.parameters(), lr=1e-4)

    for epoch in range(epochs):
        pic_net.train()
        train_losses = []
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as t:
            for batch in t:
                imgs = batch["pic"].to(device)
                labels = batch["label"].view(-1, 1).to(device)

                preds = pic_net(imgs)
                loss = criterion(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                t.set_postfix(loss=loss.item())

        pic_net.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["pic"].to(device)
                labels = batch["label"].view(-1, 1).to(device)
                preds = pic_net(imgs)
                loss = criterion(preds, labels)
                val_losses.append(loss.item())

        print(
            f"Epoch {epoch+1}: train RMSE={np.mean(train_losses):.4f}, "
            f"val RMSE={np.mean(val_losses):.4f}"
        )

    torch.save(pic_net.state_dict(), "pic_net_model.pt")

    pic_net.eval()
    test_scores = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["pic"].to(device)
            preds = pic_net(imgs)  # shape (B,1)
            preds = preds.squeeze(1).cpu().numpy()  # (B,)
            test_scores.extend(preds.astype(float).tolist())

    submission = pd.DataFrame({"Id": test_df["Id"].values, "Pawpularity": test_scores})
    submission.to_csv("submission.csv", index=False)
    print("Submission file saved as submission.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    412             try:
--> 413                 return self._range.index(new_key)
    414             except ValueError as err:

ValueError: 8920 is not in range

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3080382009.py in <cell line: 0>()
     27         train_losses = []
     28         with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as t:
---> 29             for batch in t:
     30                 imgs = batch["pic"].to(device)
     31                 labels = batch["label"].view(-1, 1).to(device)

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
     48         if self.auto_collation:
     49             if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
---> 50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
     52                 data = [self.dataset[idx] for idx in possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataset.py in __getitems__(self, indices)
    418             return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
    419         else:
--> 420             return [self.dataset[self.indices[idx]] for idx in indices]
    421 
    422     def __len__(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataset.py in <listcomp>(.0)
    418             return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
    419         else:
--> 420             return [self.dataset[self.indices[idx]] for idx in indices]
    421 
    422     def __len__(self):

/tmp/ipykernel_55/3619150059.py in __getitem__(self, idx)
     21 
     22     def __getitem__(self, idx):
---> 23         img_id = self.csv_data.loc[idx, "Id"]
     24         img_path = self.root / f"{img_id}.jpg"
     25         pil_img = Image.open(img_path).convert("RGB")

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1181             key = tuple(com.apply_if_callable(x, self.obj) for x in key)
   1182             if self._is_scalar_access(key):
-> 1183                 return self.obj._get_value(*key, takeable=self._takeable)
   1184             return self._getitem_tuple(key)
   1185         else:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _get_value(self, index, col, takeable)
   4219             #  results if our categories are integers that dont match our codes
   4220             # IntervalIndex: IntervalTree has no get_loc
-> 4221             row = self.index.get_loc(index)
   4222             return series._values[row]
   4223 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    413                 return self._range.index(new_key)
    414             except ValueError as err:
--> 415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
    417             raise KeyError(key)

KeyError: 8920
