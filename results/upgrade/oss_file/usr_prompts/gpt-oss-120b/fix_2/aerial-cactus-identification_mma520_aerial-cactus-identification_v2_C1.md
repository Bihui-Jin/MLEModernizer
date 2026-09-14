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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.995

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will adjust the data‑folder paths so the extracted archives are correctly located, add a small helper that picks the right folder whether the zip creates a top‑level `train`/`test` directory or nests them under `aerial-cactus-identification`. This resolves the FileNotFoundError, restores the dataset variables (`all_dl`, `test_data`, etc.) and therefore removes the subsequent NameError failures. No core modeling code is changed, so the approach and metrics stay intact while the script now runs end‑to‑end and writes a proper `submission.csv`.

```


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/3432597170.py", line 1
    I will adjust the data‑folder paths so the extracted archives are correctly located, add a small helper that picks the right folder whether the zip creates a top‑level `train`/`test` directory or nests them under `aerial-cactus-identification`. This resolves the FileNotFoundError, restores the dataset variables (`all_dl`, `test_data`, etc.) and therefore removes the subsequent NameError failures. No core modeling code is changed, so the approach and metrics stay intact while the script now runs end‑to‑end and writes a proper `submission.csv`.
                          ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 1
DATA_DIR        = '../input/aerial-cactus-identification/'
TRAIN_ZIP_DIR   = DATA_DIR + 'train.zip'
TEST_ZIP_DIR    = DATA_DIR + 'test.zip'
SAMPLE_SUBMIS   = DATA_DIR + 'sample_submission.csv'
ANNOTATIONS_DIR = DATA_DIR + 'train.csv'

TRAIN_DIR_DEFAULT = './train'
TEST_DIR_DEFAULT  = './test'

DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
N_LABELS        = 2
N_EPOCHS        = 10
BATCH_SIZE      = 64
LEARNING_RATE   = 0.001
MOMENTUM        = 0.9
LABELS_MAP      = {0: 'No Cactus', 1: 'Cactus'}

def init_weights(layer):
    if type(layer) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

def display_data(data, n=10, classes=None):
    fig, ax = plt.subplots(1, n, figsize=(15,3))
    indices = np.random.randint(0, len(data), size=n)
    for i, j in enumerate(indices):
        ax[i].imshow(np.transpose(data[j][0], (1, 2, 0)))
        ax[i].axis('off')
        if classes:
            ax[i].set_title(classes[data[j][1]])

def train_epoch(model, 
                dataloader, 
                lr=LEARNING_RATE, 
                optimizer=None, 
                loss_fn=nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    total_loss, accuracy, count = 0, 0, 0
    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss
        predicted = torch.max(out, 1)[1]
        accuracy += (predicted == y).sum()
        count += len(y)
    return total_loss.item() / count, accuracy.item() / count

def validate(model, 
             dataloader, 
             loss_fn=nn.NLLLoss()):
    model.eval()
    total_loss, accuracy, count = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            total_loss += loss_fn(out, y)
            predicted = torch.max(out, 1)[1]
            accuracy += (predicted == y).sum()
            count += len(y)
    return total_loss.item() / count, accuracy.item() / count 
    
def train(model, 
          train_loader, 
          valid_loader=None, 
          optimizer=None, 
          lr=LEARNING_RATE, 
          epochs=N_EPOCHS, 
          loss_fn=nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(model.parameters(),lr=lr)
    history = {'train_loss': [], 'train_accuracy': []}
    if valid_loader is not None:
        history['validation_loss'] = []
        history['validation_accuracy'] = []
    for epoch in range(epochs):
        tl, ta = train_epoch(model, 
                             train_loader, 
                             lr=lr, 
                             optimizer=optimizer, 
                             loss_fn=loss_fn)
        history['train_loss'].append(tl)
        history['train_accuracy'].append(ta)
        if valid_loader is not None:
            vl, va = validate(model, valid_loader, loss_fn=loss_fn)
            print(f"Epoch {epoch:2}, Train Acc = {ta:.3f}, Val Acc = {va:.3f}, Train Loss = {tl:.3f}, Val Loss={vl:.3f}")
            history['validation_loss'].append(vl)
            history['validation_accuracy'].append(va)
        else:
            print(f"Epoch {epoch:2}, Train Acc = {ta:.3f}, Train Loss = {tl:.3f}")
    return history

def plot_history(history, validation=False):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(history['train_accuracy'], label='Training')
    if validation:
        plt.plot(history['validation_accuracy'], label='Validation')
    plt.legend()
    plt.subplot(122)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(history['train_loss'], label='Training')
    if validation:
        plt.plot(history['validation_loss'], label='Validation')
    plt.legend()

def submission(dataset, model):
    model.eval()
    result = []
    with torch.no_grad():
        for datapoint in dataset:
            X = datapoint[0][None, ...].to(DEVICE)
            out = model(X)
            prob = float(torch.exp(out)[0][1])
            result.append([datapoint[1], prob])
    df = pd.DataFrame(result, columns = ['id', 'has_cactus'])
    df = df.set_index('id')
    df = df.sort_values('id')
    df.to_csv('./submission.csv')
    return df




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1456975858.py in <cell line: 0>()
      9 TEST_DIR_DEFAULT  = './test'
     10 
---> 11 DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
     12 N_LABELS        = 2
     13 N_EPOCHS        = 10

NameError: name 'torch' is not defined

## === cell 2
if not os.path.exists(TRAIN_DIR_DEFAULT):
    with zipfile.ZipFile(TRAIN_ZIP_DIR, 'r') as zip_ref:
        zip_ref.extractall('./')
if not os.path.exists(TEST_DIR_DEFAULT):
    with zipfile.ZipFile(TEST_ZIP_DIR, 'r') as zip_ref:
        zip_ref.extractall('./')

possible_train_paths = [
    TRAIN_DIR_DEFAULT,
    os.path.join('.', 'aerial-cactus-identification', 'train')
]
possible_test_paths = [
    TEST_DIR_DEFAULT,
    os.path.join('.', 'aerial-cactus-identification', 'test')
]

TRAIN_DIR = next((p for p in possible_train_paths if os.path.isdir(p)), None)
TEST_DIR  = next((p for p in possible_test_paths  if os.path.isdir(p)), None)

if TRAIN_DIR is None or TEST_DIR is None:
    raise FileNotFoundError("Could not locate train or test directories after extraction.")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1352397375.py in <cell line: 0>()
----> 1 if not os.path.exists(TRAIN_DIR_DEFAULT):
      2     with zipfile.ZipFile(TRAIN_ZIP_DIR, 'r') as zip_ref:
      3         zip_ref.extractall('./')
      4 if not os.path.exists(TEST_DIR_DEFAULT):
      5     with zipfile.ZipFile(TEST_ZIP_DIR, 'r') as zip_ref:

NameError: name 'os' is not defined

## === cell 3
class ACIDataset(Dataset):
    def __init__(self, 
                 img_dir,
                 annotations_file=None, 
                 transform=None, 
                 target_transform=None):
        self.img_dir = img_dir
        self.is_labeled = False
        if annotations_file is not None:
            self.is_labeled = True
            self.img_labels = pd.read_csv(annotations_file)
        else:
            self.img_labels = pd.DataFrame(os.listdir(img_dir))
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 
                                self.img_labels.iloc[idx, 0])
        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        if self.is_labeled:
            label = self.img_labels.iloc[idx, 1]
            if self.target_transform:
                label = self.target_transform(label)
            sample = [image, label]
        else:
            sample = [image, self.img_labels.iloc[idx, 0]]
        return sample

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) 

data = ACIDataset(
    img_dir=TRAIN_DIR,
    annotations_file=ANNOTATIONS_DIR,
    transform=transform,
    target_transform=None)

test_data = ACIDataset(
    img_dir=TEST_DIR,
    transform=transform)

train_data, val_data = random_split(data, [len(data) * 8 // 10, 
                                           len(data) * 2 // 10])

train_dl = DataLoader(train_data, batch_size=BATCH_SIZE)
valid_dl = DataLoader(val_data, batch_size=BATCH_SIZE)
all_dl   = DataLoader(data, batch_size=BATCH_SIZE)
test_dl  = DataLoader(test_data, batch_size=BATCH_SIZE)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3842594714.py in <cell line: 0>()
----> 1 class ACIDataset(Dataset):
      2     def __init__(self, 
      3                  img_dir,
      4                  annotations_file=None,
      5                  transform=None,

NameError: name 'Dataset' is not defined

## === cell 4
display_data(data, n=12, classes=LABELS_MAP)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1113172803.py in <cell line: 0>()
----> 1 display_data(data, n=12, classes=LABELS_MAP)
      2 

NameError: name 'display_data' is not defined

## === cell 5
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=10,
            kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(
            kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=10,
            out_channels=20,
            kernel_size=(3, 3))
        self.fc = nn.Linear(
            in_features = 20*6*6,
            out_features = 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20*6*6)
        x = F.log_softmax(self.fc(x), dim=1)
        return x



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2049512165.py in <cell line: 0>()
----> 1 class Net(nn.Module):
      2     def __init__(self):
      3         super(Net, self).__init__()
      4         self.conv1 = nn.Conv2d(
      5             in_channels=3,

NameError: name 'nn' is not defined

## === cell 6
%%capture
model_ = Net().to(DEVICE)
model_.apply(init_weights)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1212191778.py in <cell line: 0>()
----> 1 model_ = Net().to(DEVICE)
      2 model_.apply(init_weights)
      3 

NameError: name 'Net' is not defined

## === cell 7
optimizer = torch.optim.Adam(model_.parameters(), 
                             lr=LEARNING_RATE)
history = train(model_, 
                all_dl, 
                optimizer=optimizer, 
                lr=LEARNING_RATE, 
                epochs=N_EPOCHS, 
                loss_fn=nn.NLLLoss())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2793355743.py in <cell line: 0>()
----> 1 optimizer = torch.optim.Adam(model_.parameters(), 
      2                              lr=LEARNING_RATE)
      3 history = train(model_, 
      4                 all_dl,
      5                 optimizer=optimizer,

NameError: name 'torch' is not defined

## === cell 8
plot_history(history, validation=False)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1141050978.py in <cell line: 0>()
----> 1 plot_history(history, validation=False)
      2 

NameError: name 'plot_history' is not defined

## === cell 9
submission(test_data, model_)
```

## --- ERROR in cell 9, traceback:
  File "/tmp/ipykernel_55/1001601673.py", line 2
    ```
    ^
SyntaxError: invalid syntax
