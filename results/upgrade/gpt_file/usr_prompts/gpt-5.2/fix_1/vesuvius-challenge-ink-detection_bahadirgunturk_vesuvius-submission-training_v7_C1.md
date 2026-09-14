# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Detect the presence of ink from 3d x-ray scans of detached fragments of ancient papyrus scrolls.

## Metric
We evaluate how well your output image matches our reference image using a modified version of the [Sørensen--Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), where instead of using the F1 score, we are using the F0.5 score. The F0.5 score is given by:

$$
\frac{\left(1+\beta^2\right) p r}{\beta^2 p+r} \text { where } p=\frac{t p}{t p+f p}, r=\frac{t p}{t p+f n}, \beta=0.5
$$

The F0.5 score weights precision higher than recall, which improves the ability to form coherent characters out of detected ink areas.

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the output should be binary, with 0 indicating "no ink" and 1 indicating "ink".

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

Your output should be a single file, **submission.csv**, with this run-length encoded information. This should have a header with two columns, `Id` and `Predicted`, and with one row for every directory under **test/**. For example:

```
Id,Predicted
a,1 1 5 1 etc.
b,10 20 etc.
```

For a real-world example of what these files look like, see `inklabels_rce.csv` in the data directories, which have been generated with [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0).


## Data
- **[train/test]/[fragment_id]/surface_volume/[image_id].tif** slices from the 3d x-ray [surface volume](https://scrollprize.org/tutorial1#3-surface-volumes). Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us `width * height * 65` number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
- **[train/test]/[fragment_id]/mask.png** --- a binary mask of which pixels contain data.
- **train/[fragment_id]/inklabels.png** --- a binary mask of the ink vs no-ink labels.
- **train/[fragment_id]/inklabels_rle.csv** --- a run-length-encoded version of the labels, generated using [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0). This is the same format as you should make your submission in.
- **train/[fragment_id]/ir.png** --- the infrared photo on which the binary mask is based.
- **sample_submission.csv**, an example of a submission file in the correct format. You need to output the following file in the home directory: **submission.csv**.

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        input/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        working/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
```

-> data/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> input/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.1964498661486244

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0


! pip install --no-index --no-deps /kaggle/input/segmentationpackage/wheelhouse/pretrainedmodels-0.7.4-py3-none-any.whl
! pip install --no-index --no-deps /kaggle/input/segmentationpackage/wheelhouse/efficientnet_pytorch-0.7.1-py3-none-any.whl
! pip install --no-index --no-deps /kaggle/input/segmentationpackage/wheelhouse/timm-0.6.12-py3-none-any.whl
! pip install --no-index --no-deps /kaggle/input/segmentationpackage/wheelhouse/segmentation_models_pytorch-0.3.2-py3-none-any.whl

## === cell 1
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


import torchvision.transforms as T2     

import segmentation_models_pytorch as smp

import glob

import numpy as np
import PIL.Image as Image
from matplotlib import pyplot as plt
import pandas as pd




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3986064476.py in <cell line: 0>()
      8 #import torchvision.transforms.v2 as T2  # if data augmentation is needed
      9 
---> 10 import segmentation_models_pytorch as smp
     11 
     12 import glob

ModuleNotFoundError: No module named 'segmentation_models_pytorch'

## === cell 2
class config:
    TILE_DEPTH = 5
    TILE_HEIGHT = 224
    TILE_WIDTH = 224
    TILE_STRIDE = 112
    TILE_PADDING = 28

    Z_MID = 32

    INKLABEL_REGION_THRESHOLD = 0.10

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TRAIN = False 

    BASE_PATH = "/kaggle/input/" # for submission

    DEBUG = False


## === cell 3
class CustomModel1(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.encoder = smp.Unet(
            encoder_name="resnext50_32x4d", 
            encoder_weights="imagenet",
            in_channels=config.TILE_DEPTH,
            classes=1
        )

    def forward(self,X):
        y = self.encoder(X)
        return y

## === cell 5
def prepare_data(config,folder_list):


    fragment_dict = np.empty(len(folder_list),object)

    z_start = config.Z_MID - (config.TILE_DEPTH // 2)
    z_end = config.Z_MID + (config.TILE_DEPTH // 2) + 1

    for fragment_id, fragment_path in enumerate(folder_list): 
        surface_volume_paths = sorted( glob.glob(fragment_path+"surface_volume/*.tif") )

        surface_volume_paths = surface_volume_paths[z_start:z_end]

        image = np.array(Image.open(surface_volume_paths[0]),  dtype=np.float32)/65535.0
        height,width = image.shape 

        image_stack = np.zeros([config.TILE_DEPTH,height,width],dtype=np.float32)
        
        for i, filename in enumerate(surface_volume_paths):
            image = np.array(Image.open(filename),  dtype=np.float32)/65535.0
            image_stack[i,:,:] = image 
        
        inklabels = np.expand_dims(np.array(Image.open( fragment_path + "inklabels.png" ).convert("1"),dtype=np.float32), axis=0)

        image_stack = np.concatenate((image_stack,inklabels),axis=0)

        print(image_stack.shape)

        fragment_dict[fragment_id] = image_stack
    
    return fragment_dict

## === cell 6
class SubVolumeDataset(Dataset):
    def __init__(self,config,fragment_dict,transform=None):

        self.fragment_dict = fragment_dict


        self.TILE_HEIGHT_WITHOUT_PADDING = config.TILE_HEIGHT
        self.TILE_WIDTH_WITHOUT_PADDING = config.TILE_WIDTH

        self.TILE_DEPTH = config.TILE_DEPTH
        self.TILE_HEIGHT = config.TILE_HEIGHT + config.TILE_PADDING
        self.TILE_WIDTH = config.TILE_WIDTH + config.TILE_PADDING
        self.TILE_STRIDE = config.TILE_STRIDE
        
        self.Z_MID = config.Z_MID

        self.transform = transform

        self.valid_fragments_pixels = []

        for fragment_id in range(len(self.fragment_dict)): 

            inklabels = self.fragment_dict[fragment_id][self.TILE_DEPTH,:,:]
            height, width = inklabels.shape

            for x in range(0,width-self.TILE_WIDTH,self.TILE_STRIDE):
                for y in range(0,height-self.TILE_HEIGHT,self.TILE_STRIDE):
                    tile_inklabels = inklabels[y:y+self.TILE_HEIGHT,x:x+self.TILE_WIDTH] 

                    if np.sum(tile_inklabels) > config.INKLABEL_REGION_THRESHOLD * (self.TILE_HEIGHT * self.TILE_WIDTH):
                        self.valid_fragments_pixels.append([fragment_id,x,y])
                        
    def __len__(self):
        return len(self.valid_fragments_pixels)
    

    def segmentation_transform(self,image_stack):
        transform = T2.Compose([
            T2.CenterCrop((self.TILE_HEIGHT_WITHOUT_PADDING,self.TILE_WIDTH_WITHOUT_PADDING))
        ])
        transformed_image_stack = transform(image_stack)
        return transformed_image_stack


    def __getitem__(self,index):
        
        fragment_id = self.valid_fragments_pixels[index][0]

        tile_x = self.valid_fragments_pixels[index][1]
        tile_y = self.valid_fragments_pixels[index][2]

        tile_image_stack = self.fragment_dict[fragment_id][:,tile_y:tile_y+self.TILE_HEIGHT,tile_x:tile_x+self.TILE_WIDTH] 

        if self.transform is None:

            tile_image_stack = torch.from_numpy(tile_image_stack)
            
            tile_image_stack = self.segmentation_transform(tile_image_stack)

        else:
            tile_image_stack = self.transform(tile_image_stack)
        
        tile_subvolume = tile_image_stack[0:self.TILE_DEPTH,:,:]
        tile_inklabels = tile_image_stack[self.TILE_DEPTH,:,:].unsqueeze(dim=0) # A channel is taken, that dim is dropped. Add it back.
    
        return tile_subvolume, tile_inklabels

## === cell 7
class DiceBCEwithLogitsLoss(nn.Module):
    def __init__(self,alpha=0.5,epsilon=0.1):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self,y_pred,y):

        y_pred = nn.functional.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y = y.view(-1)

        intersection = (y_pred * y).sum()
        dice_score = (2*intersection) / (y_pred.sum() + y.sum() + self.epsilon)
        dice_loss = 1 - dice_score

        BCE_loss = nn.functional.binary_cross_entropy(y_pred,y,reduction='mean')

        DiceBCE = self.alpha * dice_loss + (1 - self.alpha) * BCE_loss

        return DiceBCE 

## === cell 8
def train_model(model, optimizer, loss_function, train_loader):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    total_loss = 0
    dataset_size = len(train_loader.dataset)

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X)

        loss = loss_function(y_pred,y)  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss_function(y_pred, y).item()

    print(f"Training Loss:{total_loss/dataset_size:0.6f}")


## === cell 9
def prepare_test_data(config,test_folder):
    
    z_start = config.Z_MID - (config.TILE_DEPTH // 2)
    z_end = config.Z_MID + (config.TILE_DEPTH // 2) + 1

    fragment_path = test_folder
    surface_volume_paths = sorted( glob.glob(fragment_path+"surface_volume/*.tif") )
    surface_volume_paths = surface_volume_paths[z_start:z_end]

    image = np.array(Image.open(surface_volume_paths[0]),  dtype=np.float32)/65535.0
    height,width = image.shape 
    image_stack = np.zeros([config.TILE_DEPTH,height,width],dtype=np.float32)

    for i, filename in enumerate(surface_volume_paths):
        image = np.array(Image.open(filename),  dtype=np.float32)/65535.0
        image_stack[i,:,:] = image     

    return image_stack

## === cell 10
def test_model(config,model,image_stack):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    depth, height, width = image_stack.shape

    pred_image = np.zeros((height,width),dtype=np.float32)

    with torch.no_grad():
        for x in range(0,width-config.TILE_WIDTH,config.TILE_WIDTH):
            for y in range(0,height-config.TILE_HEIGHT,config.TILE_HEIGHT):

                X_tile = image_stack[:,y:y+config.TILE_HEIGHT,x:x+config.TILE_WIDTH] 
                X_tile = torch.from_numpy(X_tile).unsqueeze(0)
                X_tile = X_tile.to(device)

                y_tile = model(X_tile)
                y_tile = y_tile.view(-1,config.TILE_HEIGHT,config.TILE_WIDTH)

                pred_image[y:y+config.TILE_HEIGHT,x:x+config.TILE_WIDTH] = y_tile.squeeze().cpu().numpy()
        

    return pred_image

## === cell 12
if config.TRAIN:
    train_folder_list = [config.BASE_PATH + "vesuvius-challenge/train/1/", 
                        config.BASE_PATH + "vesuvius-challenge/train/2/", 
                        config.BASE_PATH + "vesuvius-challenge/train/3/"]

    train_data_dict = prepare_data(config,train_folder_list)


## === cell 13

if config.DEBUG:

    fragment_id = 0
    print(train_data_dict[fragment_id].shape)

    print(len(train_data_dict))


    fig, ax = plt.subplots(1,5)
    for i in range(5):
        ax[i].imshow(train_data_dict[fragment_id][i])

    X = train_data_dict[fragment_id][2]
    plt.figure()
    plt.imshow(X)

    y = train_data_dict[fragment_id][config.TILE_DEPTH]
    plt.figure()
    plt.imshow(y)


## === cell 14
if config.DEBUG:

    sample_id = 0
    slice_id = 0

    tile_image_stack, tile_inklabels = train_dataset[sample_id]

    image = tile_image_stack[slice_id,:,:].view(config.TILE_HEIGHT,config.TILE_WIDTH).numpy()
    target = tile_inklabels[0,:,:].view(config.TILE_HEIGHT,config.TILE_WIDTH).numpy()

    plt.imshow(image)
    plt.figure()
    plt.imshow(target)

    print(tile_image_stack.shape)

    BATCH_SIZE = 1
    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE)
    X, y = next(iter(train_dataloader))
    image = X[0,slice_id,:,:].view(config.TILE_HEIGHT,config.TILE_WIDTH).numpy()
    target = y[0,0,:,:].view(config.TILE_HEIGHT,config.TILE_WIDTH).numpy()

    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(target)

    print(X.shape)

## === cell 15
if config.TRAIN:
    model1 = CustomModel1(config)   
    


## === cell 16
if config.DEBUG:


    from torchinfo import summary
    summary(model1,(1,config.TILE_DEPTH,config.TILE_HEIGHT,config.TILE_WIDTH))

## === cell 17
if config.DEBUG:

    BATCH_SIZE = 1

    train_dataset = SubVolumeDataset(config,train_data_dict)
    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE)

    model1 = CustomModel1(config)   
    model1.load_state_dict(torch.load("model1_augmented_300epochs.pth"))

    X, y = next(iter(train_dataloader))

    X = X.to(config.DEVICE)
    y = y.to(config.DEVICE)
    model1.to(config.DEVICE)

    y_pred = model1(X)

    slice_id = 0
    image = X[0,slice_id,:,:].view(config.TILE_HEIGHT,config.TILE_WIDTH).cpu().numpy()
    target = y[0,0,:,:].view(config.TILE_HEIGHT,config.TILE_WIDTH).cpu().numpy()
    pred = y_pred[0,0,:,:].view(config.TILE_HEIGHT,config.TILE_WIDTH).detach().cpu().numpy()  # detach because tensor has autograd connection

    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(target)
    plt.figure()
    plt.imshow(pred)


## === cell 18
if config.DEBUG:

    loss_function = torch.nn.BCEWithLogitsLoss()
    loss = loss_function(y_pred,y) 
    print(loss)


    loss_function = DiceBCEwithLogitsLoss()
    loss = loss_function(y_pred,y) 
    print(loss)

## === cell 19

if config.TRAIN == True:


  loss_function = DiceBCEwithLogitsLoss()

  optimizer = torch.optim.AdamW(model1.parameters(),lr=0.001)

  BATCH_SIZE = 32
  
  train_dataset = SubVolumeDataset(config,train_data_dict)
  train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE)

  num_epochs = 100

  for epoch in range(num_epochs):
    print('Epoch:',epoch)
    train_model(model1, optimizer, loss_function, train_dataloader)


else: 

  model1 = torch.load(config.BASE_PATH + "pretrained/model1.pth")


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3808584736.py in <cell line: 0>()
     34 
     35   # Load the entire model
---> 36   model1 = torch.load(config.BASE_PATH + "pretrained/model1.pth")

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/pretrained/model1.pth'

## === cell 20
if config.DEBUG:


    test_folder = config.BASE_PATH + "vesuvius-challenge/train/1/"
    test_image_stack = prepare_test_data(config,test_folder)


    test_result = test_model(config,model1,test_image_stack)



    mask_file = test_folder+"mask.png"
    mask = np.array(Image.open(mask_file),  dtype=np.float32)


    ground_truth = np.array(Image.open( test_folder + "inklabels.png" ).convert("1"), dtype=np.float32)

    test_result = mask * test_result


    plt.imshow(test_result)

    threshold = 0.0
    test_result2 = (test_result > threshold).astype(int) 

    plt.figure()
    plt.imshow(test_result2)
    plt.title("Prediction")

    plt.figure()
    plt.imshow(ground_truth)
    plt.title("Ground truth")

## === cell 22
def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)



base_path = config.BASE_PATH + "vesuvius-challenge/test/"
fragment_ids = ["a","b"]

results = []
for fragment_id in fragment_ids:

    test_folder = base_path + fragment_id + "/"

    mask_file = test_folder + "mask.png"
    mask = np.array(Image.open(mask_file),  dtype=np.float32)


    test_image_stack = prepare_test_data(config,test_folder)
    test_result = test_model(config,model1,test_image_stack)

    test_result = mask * test_result

    threshold = 0.0
    test_result = (test_result > threshold).astype(int) 

    plt.figure()
    plt.imshow(test_result)


    inklabels_rle = rle(test_result)

    results.append((fragment_id, inklabels_rle))


sub = pd.DataFrame(results, columns=['Id', 'Predicted'])
sub.to_csv("/kaggle/working/submission.csv", index=False)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/838613402.py in <cell line: 0>()
     24 
     25     mask_file = test_folder + "mask.png"
---> 26     mask = np.array(Image.open(mask_file),  dtype=np.float32)
     27 
     28 

NameError: name 'np' is not defined
