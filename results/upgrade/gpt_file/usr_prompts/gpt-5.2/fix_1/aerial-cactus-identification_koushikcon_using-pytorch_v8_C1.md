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

3.7

# 3. Installed packages

No external packages required in the script and installed.

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

0.9516666666666668

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import cv2
from glob import glob
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import TensorDataset, DataLoader,Dataset
from sklearn.model_selection import train_test_split


import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print('No GPU found. Please use a GPU to train your neural network.')

## === cell 1
label_frame = pd.read_csv('../input/train.csv')


## === cell 2
class ImageLabelDataset(Dataset):
    def __init__(self, df_data, prediction):
        super().__init__()
        self.df = df_data.values
        self.prediction = prediction.values

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        tensorimage = self.preprocess_image(self.df[index])
        label = self.prediction[index]
        return tensorimage, label
    
    def preprocess_image(self, img_path):
        data_transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(224), 
                                             transforms.CenterCrop(224), 
                                             transforms.RandomRotation(30), 
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                                            ])
        image = cv2.imread("../input/train/train/{}".format(img_path))
        image = data_transform(image)
        return image

## === cell 3
def preprocess_image(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    data_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(224), 
                                         transforms.CenterCrop(224), 
                                         transforms.RandomRotation(30), 
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                                        ])
    image = cv2.imread("../input/train/train/{}".format(img_path))
    image = data_transform(image)
    
    return image # predicted class index

## === cell 4
print(label_frame.dtypes)

## === cell 6
training_set = label_frame["id"]
prediction_set = label_frame.has_cactus
batch_size = 1

X_train, X_val, Y_train, Y_val = train_test_split(training_set, prediction_set, test_size=0.1, random_state=42)
print(len(X_train))
print(len(Y_train))

print(len(X_val))
print(len(Y_val))


train_set = ImageLabelDataset(df_data=X_train, prediction=Y_train)
val_set = ImageLabelDataset(df_data=X_val, prediction=Y_val)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

## === cell 7
batch_no = len(X_train) // batch_size  #batches
n_output=1

sequence_length = 6  # of words in a sequence 892110
num_epochs = 20
learning_rate = 0.002
output_size = 1
embedding_dim = 128
hidden_dim = 256
n_layers = 2

show_every_n_batches = 1

## === cell 8
def train_rnn(model, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    train_loss = 0
    valid_loss = 0
    batch_losses = []
    val_batch_losses = []
    valid_loss_min = np.Inf
    
    model.train()
    
    previousLoss = np.Inf
    minLoss = np.Inf

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            
        model.eval()
        for batch_idx, (data, target) in enumerate(val_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
        
        valid_loss = valid_loss/len(val_loader.dataset)
        train_loss = train_loss/len(train_loader.dataset)
        
        if epoch_i%show_every_n_batches == 0:
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch_i, train_loss, valid_loss))
        
            if valid_loss < valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                with open('../save/trained_rnn_new', 'wb') as pickle_file:
                    torch.save(model.state_dict(), '../save/trained_rnn_new')
                valid_loss_min = valid_loss
                train_loss = 0
                valid_loss = 0
  
    return model

## === cell 9
model_transfer = models.vgg16(pretrained=True)

## === cell 11
for param in model_transfer.features.parameters():
    param.requires_grad = False
    
custom_model = nn.Sequential(nn.Linear(25088, 1024), 
                  nn.ReLU(),
                  nn.Dropout(p=0.5), 
                  nn.Linear(1024, 512), 
                  nn.ReLU(),
                  nn.Dropout(p=0.5), 
                  nn.Linear(512, 2),
                 )

model_transfer.classifier = custom_model

if use_cuda:
    model_transfer = model_transfer.cuda()


criterion_scratch = nn.CrossEntropyLoss()

optimizer_scratch = optim.SGD(model_transfer.parameters(), lr=learning_rate)



## === cell 12
model_transfer.load_state_dict(torch.load('../save/trained_rnn_new'))
test_files = np.array(glob("../input/test/test/*"))
print(test_files[:3])

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1922734963.py in <cell line: 0>()
----> 1 model_transfer.load_state_dict(torch.load('../save/trained_rnn_new'))
      2 test_files = np.array(glob("../input/test/test/*"))
      3 print(test_files[:3])

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '../save/trained_rnn_new'

## === cell 13
def predict(file):
    test_transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(224), 
                                             transforms.CenterCrop(224), 
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
                                            ])
    image = cv2.imread(file)
    image = test_transform(image)
    image = image.unsqueeze(0)
    if use_cuda:
        image = image.cuda()
    
    output = model_transfer(image)
    values, indices = output.max(1)
    return os.path.basename(file), indices.item()

## === cell 14
model_transfer.eval()
submission = pd.DataFrame(columns=['id', 'has_cactus'])

for file in test_files:
    name,value = predict(file)
    submission = submission.append({'id': name, 'has_cactus': value}, ignore_index=True)

print(submission.head())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3524609839.py in <cell line: 0>()
      2 submission = pd.DataFrame(columns=['id', 'has_cactus'])
      3 
----> 4 for file in test_files:
      5     name,value = predict(file)
      6     submission = submission.append({'id': name, 'has_cactus': value}, ignore_index=True)

NameError: name 'test_files' is not defined

## === cell 16
final_submission_frame = pd.read_csv('../input/mydatasetforaerialcactus/my_submission_final.csv')
final_submission_frame.to_csv('outputsubmission.csv', index=False)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3297378879.py in <cell line: 0>()
----> 1 final_submission_frame = pd.read_csv('../input/mydatasetforaerialcactus/my_submission_final.csv')
      2 final_submission_frame.to_csv('outputsubmission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/mydatasetforaerialcactus/my_submission_final.csv'

## === cell 17
from IPython.display import HTML
import base64

def create_download_link(df, title = "Download CSV file", filename="submission_1.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(submission)
