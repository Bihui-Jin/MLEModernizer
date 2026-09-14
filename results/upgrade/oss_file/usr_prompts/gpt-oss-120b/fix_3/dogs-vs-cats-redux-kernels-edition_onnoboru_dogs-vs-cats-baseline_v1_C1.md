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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

# 3. Installed packages

albumentations==2.0.8
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
tqdm==4.67.1
transformers==4.53.3

# 4. Data file paths

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

# 5. Target score

0.05676

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
class CFG:
    debug_one_epoch = True
    debug_one_fold = False
    only_infer = False
    num_workers = 8
    batch_size = 64
    num_epochs = 10
    lr = 1e-3
    early_stopping_round = 5
    random_seed = 42
    n_splits = 5
    model_name = "resnet18"  # timm で使うモデル名
    pretrained_path = None
    train_dir = None  # 学習データセットのパス
    test_dir = None  # テストデータセットのパス
    optimizer = torch.optim.AdamW
    criterion = nn.BCEWithLogitsLoss()
    scheduler = transformers.get_linear_schedule_with_warmup
    input_imgsize = 224
    data_dir = "../input/dogs-vs-cats-redux-kernels-edition/"
    kaggle_working_dir = "/kaggle/working/"


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(CFG.random_seed)

if CFG.debug_one_epoch:
    CFG.num_epochs = 1

print("KAGGLE_URL_BASE" in set(os.environ.keys()))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3197046015.py in <cell line: 0>()
----> 1 class CFG:
      2     debug_one_epoch = True
      3     debug_one_fold = False
      4     only_infer = False
      5     num_workers = 8

/tmp/ipykernel_55/3197046015.py in CFG()
     14     train_dir = None  # 学習データセットのパス
     15     test_dir = None  # テストデータセットのパス
---> 16     optimizer = torch.optim.AdamW
     17     criterion = nn.BCEWithLogitsLoss()
     18     scheduler = transformers.get_linear_schedule_with_warmup

NameError: name 'torch' is not defined

## === cell 1
submission = pd.read_csv(os.path.join(CFG.data_dir, "sample_submission.csv"))
submission.head(3)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1468467363.py in <cell line: 0>()
----> 1 submission = pd.read_csv(os.path.join(CFG.data_dir, "sample_submission.csv"))
      2 submission.head(3)
      3 
      4 

NameError: name 'pd' is not defined

## === cell 2
if "KAGGLE_URL_BASE" in set(os.environ.keys()):
    kaggle_train_dir = os.path.join(CFG.kaggle_working_dir, "train")
    if not os.path.exists(kaggle_train_dir):
        shutil.unpack_archive(
            os.path.join(CFG.data_dir, "train.zip"), CFG.kaggle_working_dir
        )

    kaggle_test_dir = os.path.join(CFG.kaggle_working_dir, "test")
    if not os.path.exists(kaggle_test_dir):
        shutil.unpack_archive(
            os.path.join(CFG.data_dir, "test.zip"), CFG.kaggle_working_dir
        )

    possible_nested = os.path.join(
        CFG.kaggle_working_dir, "dogs-vs-cats-redux-kernels-edition"
    )
    if os.path.isdir(possible_nested):
        CFG.data_dir = possible_nested
    else:
        CFG.data_dir = CFG.kaggle_working_dir

CFG.train_dir = os.path.join(CFG.data_dir, "train")
CFG.test_dir = os.path.join(CFG.data_dir, "test")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2693552495.py in <cell line: 0>()
----> 1 if "KAGGLE_URL_BASE" in set(os.environ.keys()):
      2     kaggle_train_dir = os.path.join(CFG.kaggle_working_dir, "train")
      3     if not os.path.exists(kaggle_train_dir):
      4         shutil.unpack_archive(
      5             os.path.join(CFG.data_dir, "train.zip"), CFG.kaggle_working_dir

NameError: name 'os' is not defined

## === cell 3
train_list = glob.glob(
    os.path.join(CFG.data_dir, "train", "**", "*.jpg"), recursive=True
)
test_list = glob.glob(os.path.join(CFG.data_dir, "test", "**", "*.jpg"), recursive=True)

print(f"train data : {len(train_list)}")
print(f"test data : {len(test_list)}")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4170900603.py in <cell line: 0>()
----> 1 train_list = glob.glob(
      2     os.path.join(CFG.data_dir, "train", "**", "*.jpg"), recursive=True
      3 )
      4 test_list = glob.glob(os.path.join(CFG.data_dir, "test", "**", "*.jpg"), recursive=True)
      5 

NameError: name 'glob' is not defined

## === cell 4
print("the number of dog : ", len([i for i in train_list if "dog" in i]))
print("the number of cat : ", len([i for i in train_list if "cat" in i]))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3854243652.py in <cell line: 0>()
----> 1 print("the number of dog : ", len([i for i in train_list if "dog" in i]))
      2 print("the number of cat : ", len([i for i in train_list if "cat" in i]))
      3 
      4 

NameError: name 'train_list' is not defined

## === cell 5
random_img = random.choice(train_list)
img = Image.open(random_img).convert("RGB")
print(random_img)
print(img.size)
plt.imshow(img)
plt.axis("off")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3749094390.py in <cell line: 0>()
----> 1 random_img = random.choice(train_list)
      2 img = Image.open(random_img).convert("RGB")
      3 print(random_img)
      4 print(img.size)
      5 plt.imshow(img)

NameError: name 'random' is not defined

## === cell 6
img_array = np.array(img)
print(img_array.shape)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/893379348.py in <cell line: 0>()
----> 1 img_array = np.array(img)
      2 print(img_array.shape)
      3 
      4 

NameError: name 'np' is not defined

## === cell 7
transform_tmp = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
    ]
)
img_transformed_tmp = transform_tmp(
    image=img_array
)  # albumentations expects a numpy array
print(img_transformed_tmp.keys())
img_transformed_tmp = Image.fromarray(img_transformed_tmp["image"])
print(img_transformed_tmp.size)
plt.imshow(img_transformed_tmp)
plt.axis("off")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1757233566.py in <cell line: 0>()
----> 1 transform_tmp = A.Compose(
      2     [
      3         A.Resize(CFG.input_imgsize, CFG.input_imgsize),
      4     ]
      5 )

NameError: name 'A' is not defined

## === cell 8
transform_tmp = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.HorizontalFlip(p=1.0),
    ]
)
img_transformed_tmp = transform_tmp(image=img_array)
img_transformed_tmp = Image.fromarray(img_transformed_tmp["image"])
plt.imshow(img_transformed_tmp)
plt.axis("off")




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4017816247.py in <cell line: 0>()
----> 1 transform_tmp = A.Compose(
      2     [
      3         A.Resize(CFG.input_imgsize, CFG.input_imgsize),
      4         A.HorizontalFlip(p=1.0),
      5     ]

NameError: name 'A' is not defined

## === cell 9
transform_tmp = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.Normalize(),
    ]
)
img_transformed_tmp = transform_tmp(image=img_array)["image"]
plt.imshow(img_transformed_tmp)
plt.axis("off")




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3745175157.py in <cell line: 0>()
----> 1 transform_tmp = A.Compose(
      2     [
      3         A.Resize(CFG.input_imgsize, CFG.input_imgsize),
      4         A.Normalize(),
      5     ]

NameError: name 'A' is not defined

## === cell 10
transform_tmp = A.Compose(
    [A.Resize(CFG.input_imgsize, CFG.input_imgsize), ToTensorV2()]
)
img_transformed_tmp = transform_tmp(image=img_array)
print(img_transformed_tmp.keys())
print(type(img_transformed_tmp["image"]))
print(img_transformed_tmp["image"].shape)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2266706859.py in <cell line: 0>()
----> 1 transform_tmp = A.Compose(
      2     [A.Resize(CFG.input_imgsize, CFG.input_imgsize), ToTensorV2()]
      3 )
      4 img_transformed_tmp = transform_tmp(image=img_array)
      5 print(img_transformed_tmp.keys())

NameError: name 'A' is not defined

## === cell 11
train_df = pd.DataFrame(train_list, columns=["path"])
train_df["class"] = train_df["path"].apply(lambda x: x.split("/")[-1].split(".")[0])
train_df["class"] = train_df["class"].map({"dog": 1, "cat": 0})
test_df = pd.DataFrame(test_list, columns=["path"])
test_df["class"] = -1
test_df["id"] = test_df["path"].apply(lambda x: int(x.split("/")[-1].split(".")[0]))
test_df = test_df.sort_values("id").reset_index(drop=True)

train_df.head(3)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3284699319.py in <cell line: 0>()
----> 1 train_df = pd.DataFrame(train_list, columns=["path"])
      2 train_df["class"] = train_df["path"].apply(lambda x: x.split("/")[-1].split(".")[0])
      3 train_df["class"] = train_df["class"].map({"dog": 1, "cat": 0})
      4 test_df = pd.DataFrame(test_list, columns=["path"])
      5 test_df["class"] = -1

NameError: name 'pd' is not defined

## === cell 12
test_df.head(3)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3023781903.py in <cell line: 0>()
----> 1 test_df.head(3)
      2 
      3 

NameError: name 'test_df' is not defined

## === cell 13
train_transform = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.HorizontalFlip(p=0.5),  # 50% の確率で水平反転
        A.Normalize(),
        ToTensorV2(),
    ]
)
test_transform = A.Compose(
    [A.Resize(CFG.input_imgsize, CFG.input_imgsize), A.Normalize(), ToTensorV2()]
)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1372965739.py in <cell line: 0>()
----> 1 train_transform = A.Compose(
      2     [
      3         A.Resize(CFG.input_imgsize, CFG.input_imgsize),
      4         A.HorizontalFlip(p=0.5),  # 50% の確率で水平反転
      5         A.Normalize(),

NameError: name 'A' is not defined

## === cell 14
class DogsCatsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(image=np.array(img))["image"]
        label = torch.tensor(self.df.iloc[idx, 1], dtype=torch.float32)
        return img, label




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2527303384.py in <cell line: 0>()
----> 1 class DogsCatsDataset(Dataset):
      2     def __init__(self, df, transform=None):
      3         self.df = df
      4         self.transform = transform
      5 

NameError: name 'Dataset' is not defined

## === cell 15
def train_one_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    losses = []
    for img, label in tqdm(dataloader, leave=False):
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output.squeeze(-1), label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return np.mean(losses)




## === cell 16
def eval_one_epoch(model, dataloader, criterion):
    model.eval()
    losses = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for img, label in tqdm(dataloader, leave=False):
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            loss = criterion(output.squeeze(-1), label)
            losses.append(loss.item())
            all_labels.extend(label.cpu().numpy())
            pred = torch.sigmoid(output).cpu().numpy()
            all_outputs.extend(pred)

    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)

    return {
        "bce_loss": np.mean(losses),
        "log_loss": log_loss(all_labels, all_outputs),
        "labels": all_labels,
        "outputs": all_outputs,
    }




## === cell 17
def infer(model, dataloader, test=False):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for img, label in tqdm(dataloader, leave=False):
            if test:
                assert label[0] == -1
            img = img.to(device)
            output = model(img)
            all_outputs.extend(torch.sigmoid(output).cpu().numpy())

    return np.array(all_outputs)




## === cell 18
def run_train_cv(train, test):
    kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed)
    oof = np.zeros((len(train), 1))
    predictions = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
        print(f"====================fold : {fold}====================")
        train_df_fold = train.iloc[train_idx].reset_index(drop=True)
        valid_df_fold = train.iloc[valid_idx].reset_index(drop=True)

        train_dataset = DogsCatsDataset(train_df_fold, transform=train_transform)
        valid_dataset = DogsCatsDataset(valid_df_fold, transform=test_transform)
        test_dataset = DogsCatsDataset(test, transform=test_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
        )

        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        model.to(device)

        optimizer = CFG.optimizer(model.parameters(), lr=CFG.lr)
        training_steps = len(train_loader) * CFG.num_epochs
        warmup_steps = int(training_steps * 0.1)
        scheduler = CFG.scheduler(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
        )

        best_loss = np.inf
        early_stopping_round = 0

        for epoch in range(CFG.num_epochs):
            start_time = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, CFG.criterion
            )
            valid_result = eval_one_epoch(model, valid_loader, CFG.criterion)
            print(
                f"epoch {epoch} - train loss {train_loss:.4f} - valid bce {valid_result['bce_loss']:.4f} - valid logloss {valid_result['log_loss']:.4f}"
            )

            if valid_result["bce_loss"] < best_loss:
                best_loss = valid_result["bce_loss"]
                early_stopping_round = 0
                torch.save(model.state_dict(), f"{CFG.model_name}_fold{fold}.pth")
            else:
                early_stopping_round += 1
                if early_stopping_round > CFG.early_stopping_round:
                    break

            print(f"epoch time : {time.time() - start_time:.2f}s")

        model.load_state_dict(torch.load(f"{CFG.model_name}_fold{fold}.pth"))
        oof[valid_idx] = infer(model, valid_loader).reshape(-1, 1)

        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

        if CFG.debug_one_fold:
            break

    for fold in range(CFG.n_splits):
        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        model.load_state_dict(torch.load(f"{CFG.model_name}_fold{fold}.pth"))
        model.to(device)
        predictions.append(infer(model, test_loader, test=True))
        if CFG.debug_one_fold:
            break

    predictions = np.mean(predictions, axis=0)

    return {"oof": oof, "predictions": predictions}




## === cell 19
def main():
    if CFG.only_infer:
        test_dataset = DogsCatsDataset(test_df, transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
        )
        model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        model.to(device)
        predictions = infer(model, test_loader, test=True)
        submission["label"] = predictions
        submission.to_csv("submission.csv", index=False)
    else:
        result = run_train_cv(train_df, test_df)
        oof_preds = result["oof"]
        predictions = result["predictions"]
        submission["label"] = predictions
        submission.to_csv("submission.csv", index=False)
        train_df["oof_preds"] = oof_preds
        train_df.to_csv("oof_preds.csv", index=False)
        if not CFG.debug_one_fold:
            print(f"oof log loss : {log_loss(train_df['class'], oof_preds)}")


if __name__ == "__main__":
    main()

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1228398684.py in <cell line: 0>()
     26 
     27 if __name__ == "__main__":
---> 28     main()

/tmp/ipykernel_55/1228398684.py in main()
      1 def main():
----> 2     if CFG.only_infer:
      3         test_dataset = DogsCatsDataset(test_df, transform=test_transform)
      4         test_loader = DataLoader(
      5             test_dataset,

NameError: name 'CFG' is not defined
