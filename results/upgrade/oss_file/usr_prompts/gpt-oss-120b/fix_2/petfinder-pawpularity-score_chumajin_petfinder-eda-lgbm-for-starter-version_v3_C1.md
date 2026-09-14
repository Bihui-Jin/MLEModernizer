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
lightgbm==4.6.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

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

20.47022

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2




## === cell 1
sample = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")
sample




## === cell 2
test = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
test




## === cell 3
train = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
train




## === cell 4
train_path = "../input/petfinder-pawpularity-score/train"




## === cell 5
path = os.path.join(train_path, train["Id"].iloc[0] + ".jpg")




## === cell 6
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)




## === cell 7
train.iloc[0]




## === cell 8
explain_dict = {
    "Id": "画像ファイル名",
    "Subject Focus": "ペットは整頓された背景に対して際立っており、近すぎたり遠すぎたりすることはありません。",
    "Eyes": "両方の目が正面または正面近くを向いており、少なくとも1つの目/瞳孔が適切にクリアされています。",
    "Face": "正面または正面近くを向いた、きちんとクリアな顔。",
    "Near": "写真のかなりの部分を占める1匹のペット（写真の幅または高さの約50％以上）。",
    "Action": "アクション（ジャンプなど）の途中でペットを飼う。",
    "Accessory": "付属の物理的またはデジタルの付属品/支柱（おもちゃ、デジタルステッカーなど）、首輪と鎖を除く。",
    "Group": "写真に写っているペットが1匹以上。",
    "Collage": "デジタルレタッチされた写真（つまり、デジタルフォトフレーム、複数の写真の組み合わせ）。",
    "Human": "写真の中の人間。",
    "Occlusion": "ペットの一部をブロックする特定の望ましくないオブジェクト（つまり、人間、ケージ、または柵）。すべてのブロッキングオブジェクトがオクルージョンと見なされるわけではないことに注意してください。",
    "Info": "カスタム追加されたテキストまたはラベル（つまり、ペットの名前、説明）。",
    "Blur": "特にペットの目や顔の焦点がはっきりしていないか、ノイズが多い。ぼかしエントリの場合、「目」列は常に0に設定されます。",
    "Pawpularity": "ペットの人気度",
}




## === cell 9
train_jap = train.copy()




## === cell 10
train_jap.columns = train.columns.map(explain_dict)




## === cell 11
train_jap.head(3)




## === cell 12
tmpdf = train_jap[train_jap.index == 0].T




## === cell 13
tmpdf




## === cell 14
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)




## === cell 15
tmpdf[tmpdf[0] == True].index




## === cell 16
tmpdf[tmpdf[0] == False].index




## === cell 17
tmp2 = tmpdf[tmpdf[0] == False].index.to_list()




## === cell 18
path = os.path.join(train_path, train["Id"].iloc[0] + ".jpg")

img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

tmpdf = train_jap[train_jap.index == 0].T

print("--------ペットの人気度------------")

print(tmpdf[0].iloc[-1])

print("--------1の項目------------")

for a in tmpdf[tmpdf[0] == True].index:
    print("・ " + a)


print("")

print("--------0の項目------------")

for a in tmpdf[tmpdf[0] == False].index:
    print("・ " + a)




## === cell 19
def showimg(id):

    plt.figure()
    path = os.path.join(train_path, train["Id"].iloc[id] + ".jpg")

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

    plt.show()

    tmpdf = train_jap[train_jap.index == id].T

    print("--------ペットの人気度------------")

    print(tmpdf[id].iloc[-1])

    print("--------1の項目------------")

    for a in tmpdf[tmpdf[id] == True].index:

        if a == "Pawpularity":
            continue

        print("・ " + a)

    print("")

    print("--------0の項目------------")

    for a in tmpdf[tmpdf[id] == False].index:

        if a == "Pawpularity":
            continue
        print("・ " + a)




## === cell 20
showimg(1)




## === cell 21
showimg(2)




## === cell 22
train[train["Pawpularity"] == 100]




## === cell 23
showimg(19)




## === cell 24
showimg(50)




## === cell 25
train[train["Pawpularity"] == 1]




## === cell 26
showimg(2442)




## === cell 27
showimg(3232)




## === cell 28
showimg(4235)




## === cell 29
tmpdf3 = train.groupby("Pawpularity").head(1).sort_values("Pawpularity").reset_index()
tmpdf3




## === cell 30
tmpdf4 = tmpdf3.iloc[::10, :]
tmpdf4




## === cell 31
for a in tmpdf4["index"]:
    showimg(a)
    print("")
    print("#################################################")




## === cell 32
train




## === cell 33
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold




## === cell 34
folds = train.copy()
Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds["Pawpularity"])):
    folds.loc[val_index, "fold"] = int(n)
folds["fold"] = folds["fold"].astype(int)
print(folds.groupby(["fold", "Pawpularity"]).size())




## === cell 35
import lightgbm as lgb




## === cell 36
import random


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


SEED = 42
fix_seed(SEED)




## === cell 37
features = train.columns.to_list()[1:-1]
features




## === cell 38
target = "Pawpularity"




## === cell 39
lgbm_params = {
    "objective": "rmse",
    "seed": 42,
    "metric": "rmse",
    "learning_rate": 0.1,
    "max_bin": 800,
    "num_leaves": 80,
    "verbose": -1,
}




## === cell 40
from sklearn.metrics import mean_squared_error


def rmsescore(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))




## === cell 41
test




## === cell 42
scores = []
allpreds = []

allvaliddf = pd.DataFrame()

for fold in range(5):

    fix_seed(SEED)  # reproducibility

    p_train = folds[folds["fold"] != fold].reset_index(drop=True)
    p_val = folds[folds["fold"] == fold].reset_index(drop=True)

    lgb_train = lgb.Dataset(p_train[features], p_train[target])
    lgb_eval = lgb.Dataset(p_val[features], p_val[target])

    model = lgb.train(
        lgbm_params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=1000,
        early_stopping_rounds=100,
        verbose=-1,
    )

    import pickle

    model_name = f"LGBMmodel{fold}.bin"
    pickle.dump(model, open(model_name, "wb"))
    model = pickle.load(open(model_name, "rb"))

    oof_pred = model.predict(p_val[features], num_iteration=model.best_iteration)
    scores.append(rmsescore(p_val[target], oof_pred))

    preds = model.predict(test[features])
    allpreds.append(preds)

    p_val["preds"] = oof_pred
    allvaliddf = pd.concat([allvaliddf, p_val], ignore_index=True)




## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3175979039.py in <cell line: 0>()
     15 
     16     # removed unsupported verbose_eval argument
---> 17     model = lgb.train(
     18         lgbm_params,
     19         lgb_train,

TypeError: train() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 43
scores




## === cell 44
np.mean(scores)




## === cell 45
rmsescore(allvaliddf[target], allvaliddf["preds"])




## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3824793213.py in <cell line: 0>()
----> 1 rmsescore(allvaliddf[target], allvaliddf["preds"])
      2 
      3 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'Pawpularity'

## === cell 46
allpreds = np.mean(allpreds, axis=0)




## === cell 47
allpreds = np.clip(allpreds, 1, 100)




## === cell 48
sample["Pawpularity"] = allpreds
sample.to_csv("submission.csv", index=False)




## === cell 49
sample

## --- ERROR in outputing the csv:
Invalid submission: Pawpularity in submission should be between 1 and 100
