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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.8

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
sklearn-pandas==2.2.0
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 5. Target score

0.03227

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%reload_ext autoreload
%autoreload 2
%matplotlib inline


## === cell 1
import pathlib
import fastai
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn
from subprocess import check_output


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/582469950.py in <cell line: 0>()
      3 from fastai import *
      4 from fastai.vision import *
----> 5 from fastai.callbacks import *
      6 from fastai.utils.mem import *
      7 

ModuleNotFoundError: No module named 'fastai.callbacks'

## === cell 2
input_path = Path('/kaggle/input/denoising-dirty-documents')
items = list(input_path.glob("*.zip"))
print([x for x in items])


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1850065450.py in <cell line: 0>()
----> 1 input_path = Path('/kaggle/input/denoising-dirty-documents')
      2 items = list(input_path.glob("*.zip"))
      3 print([x for x in items])

NameError: name 'Path' is not defined

## === cell 3
import zipfile

for item in items:
    print(item)
    with zipfile.ZipFile(str(item), "r") as z:
        z.extractall(".")


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2249817689.py in <cell line: 0>()
      1 import zipfile
      2 
----> 3 for item in items:
      4     print(item)
      5     with zipfile.ZipFile(str(item), "r") as z:

NameError: name 'items' is not defined

## === cell 4
bs, size = 4, 128
arch = models.resnet34
path_train = Path("train")
path_train_cleaned = Path("train_cleaned")
path_test = Path("test")
path_submission = Path("submission")


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/955868706.py in <cell line: 0>()
      1 bs, size = 4, 128
----> 2 arch = models.resnet34
      3 path_train = Path("train")
      4 path_train_cleaned = Path("train_cleaned")
      5 path_test = Path("test")

NameError: name 'models' is not defined

## === cell 5
src = ImageImageList.from_folder(path_train).split_by_rand_pct(0.2, seed=42)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1217177935.py in <cell line: 0>()
----> 1 src = ImageImageList.from_folder(path_train).split_by_rand_pct(0.2, seed=42)

NameError: name 'ImageImageList' is not defined

## === cell 6
def get_data(src, bs, size):
    data = (
        src.label_from_func(lambda x: path_train_cleaned / x.name)
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
           .databunch(bs=bs)       
           .normalize(imagenet_stats, do_y=True)
    )
    data.c = 3
    return data


## === cell 7
data = get_data(src, bs, size)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3939034675.py in <cell line: 0>()
----> 1 data = get_data(src, bs, size)

NameError: name 'src' is not defined

## === cell 8
data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(5, 5), title="Some image")


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2876590753.py in <cell line: 0>()
      1 # Show some validation examples
----> 2 data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(5, 5), title="Some image")

NameError: name 'data' is not defined

## === cell 9
t = data.valid_ds[0][1].data
t = torch.stack([t,t])


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4098144961.py in <cell line: 0>()
----> 1 t = data.valid_ds[0][1].data
      2 t = torch.stack([t,t])

NameError: name 'data' is not defined

## === cell 10
def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)


## === cell 11
base_loss = F.l1_loss


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2201258191.py in <cell line: 0>()
----> 1 base_loss = F.l1_loss

NameError: name 'F' is not defined

## === cell 12
vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4251567762.py in <cell line: 0>()
----> 1 vgg_m = vgg16_bn(True).features.cuda().eval()
      2 requires_grad(vgg_m, False)

NameError: name 'vgg16_bn' is not defined

## === cell 13
blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1679394010.py in <cell line: 0>()
      2 # This is right before the grid size changes in the VGG model, which we are using
      3 # for feature generation.
----> 4 blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
      5 blocks, [vgg_m[i] for i in blocks]

NameError: name 'children' is not defined

## === cell 14
class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        """ m_feat is the pretrained model """
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4259781460.py in <cell line: 0>()
----> 1 class FeatureLoss(nn.Module):
      2     def __init__(self, m_feat, layer_ids, layer_wgts):
      3         """ m_feat is the pretrained model """
      4         super().__init__()
      5         self.m_feat = m_feat

NameError: name 'nn' is not defined

## === cell 15
feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1529403934.py in <cell line: 0>()
----> 1 feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])

NameError: name 'FeatureLoss' is not defined

## === cell 16
wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
gc.collect();


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3861058962.py in <cell line: 0>()
      1 wd = 1e-3
----> 2 learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
      3                      blur=True, norm_type=NormType.Weight)
      4 gc.collect();

NameError: name 'unet_learner' is not defined

## === cell 17
learn.model_dir = Path('models').absolute()


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3173352520.py in <cell line: 0>()
----> 1 learn.model_dir = Path('models').absolute()

NameError: name 'Path' is not defined

## === cell 18
learn.lr_find()
learn.recorder.plot()


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4001406990.py in <cell line: 0>()
----> 1 learn.lr_find()
      2 learn.recorder.plot()

NameError: name 'learn' is not defined

## === cell 19
print(f"Validation set size: {len(data.valid_ds.items)}")


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2943907846.py in <cell line: 0>()
----> 1 print(f"Validation set size: {len(data.valid_ds.items)}")

NameError: name 'data' is not defined

## === cell 20
lr = 1e-3


## === cell 21
def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(10, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=5)


## === cell 22
do_fit('1a', slice(lr*10))


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3469956024.py in <cell line: 0>()
----> 1 do_fit('1a', slice(lr*10))

/tmp/ipykernel_11/1035204811.py in do_fit(save_name, lrs, pct_start)
      1 def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
----> 2     learn.fit_one_cycle(10, lrs, pct_start=pct_start)
      3     learn.save(save_name)
      4     learn.show_results(rows=1, imgsize=5)

NameError: name 'learn' is not defined

## === cell 23
learn.unfreeze()


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1773196037.py in <cell line: 0>()
----> 1 learn.unfreeze()

NameError: name 'learn' is not defined

## === cell 24
do_fit('1b', slice(1e-5, lr))


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2607391871.py in <cell line: 0>()
----> 1 do_fit('1b', slice(1e-5, lr))

/tmp/ipykernel_11/1035204811.py in do_fit(save_name, lrs, pct_start)
      1 def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
----> 2     learn.fit_one_cycle(10, lrs, pct_start=pct_start)
      3     learn.save(save_name)
      4     learn.show_results(rows=1, imgsize=5)

NameError: name 'learn' is not defined

## === cell 25
data = get_data(src, 12, size*2)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2703133602.py in <cell line: 0>()
      1 # Increase resolution of the images.
----> 2 data = get_data(src, 12, size*2)

NameError: name 'src' is not defined

## === cell 26
learn.data = data
learn.freeze()
gc.collect()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3848777200.py in <cell line: 0>()
----> 1 learn.data = data
      2 learn.freeze()
      3 gc.collect()

NameError: name 'data' is not defined

## === cell 27
learn.load('1b');


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2839821256.py in <cell line: 0>()
----> 1 learn.load('1b');

NameError: name 'learn' is not defined

## === cell 28
do_fit('2a')


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1735490575.py in <cell line: 0>()
----> 1 do_fit('2a')

/tmp/ipykernel_11/1035204811.py in do_fit(save_name, lrs, pct_start)
      1 def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
----> 2     learn.fit_one_cycle(10, lrs, pct_start=pct_start)
      3     learn.save(save_name)
      4     learn.show_results(rows=1, imgsize=5)

NameError: name 'learn' is not defined

## === cell 29
learn.unfreeze()


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1773196037.py in <cell line: 0>()
----> 1 learn.unfreeze()

NameError: name 'learn' is not defined

## === cell 30
do_fit('2b', slice(1e-6,1e-4), pct_start=0.3)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1471242765.py in <cell line: 0>()
----> 1 do_fit('2b', slice(1e-6,1e-4), pct_start=0.3)
      2 
      3 # save entire configuration
      4 #learn.export(file = model_path)

/tmp/ipykernel_11/1035204811.py in do_fit(save_name, lrs, pct_start)
      1 def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
----> 2     learn.fit_one_cycle(10, lrs, pct_start=pct_start)
      3     learn.save(save_name)
      4     learn.show_results(rows=1, imgsize=5)

NameError: name 'learn' is not defined

## === cell 31
fn = data.valid_ds.x.items[10]; fn


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3573881783.py in <cell line: 0>()
----> 1 fn = data.valid_ds.x.items[10]; fn

NameError: name 'data' is not defined

## === cell 32
img = open_image(fn); img.shape


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2455244274.py in <cell line: 0>()
----> 1 img = open_image(fn); img.shape

NameError: name 'open_image' is not defined

## === cell 33
p,img_pred,b = learn.predict(img)
show_image(img, figsize=(8,5), interpolation='nearest');


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3715577578.py in <cell line: 0>()
----> 1 p,img_pred,b = learn.predict(img)
      2 show_image(img, figsize=(8,5), interpolation='nearest');

NameError: name 'learn' is not defined

## === cell 34
Image(img_pred).show(figsize=(8,5))


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3262287777.py in <cell line: 0>()
----> 1 Image(img_pred).show(figsize=(8,5))

NameError: name 'Image' is not defined

## === cell 35
learn.data.single_ds.tfmargs['size'] = None


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3393970290.py in <cell line: 0>()
      1 # Turn off resizing transformations for inference time.
      2 # https://forums.fast.ai/t/segmentation-mask-prediction-on-different-input-image-sizes/44389
----> 3 learn.data.single_ds.tfmargs['size'] = None

NameError: name 'learn' is not defined

## === cell 36
test_images = ImageImageList.from_folder(path_test)
print(test_images)


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1575681004.py in <cell line: 0>()
----> 1 test_images = ImageImageList.from_folder(path_test)
      2 print(test_images)

NameError: name 'ImageImageList' is not defined

## === cell 37
img = test_images[0]
img.show()
img.shape


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1961519078.py in <cell line: 0>()
----> 1 img = test_images[0]
      2 img.show()
      3 img.shape

NameError: name 'test_images' is not defined

## === cell 38
p, img_pred, b = learn.predict(img)


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2612105952.py in <cell line: 0>()
----> 1 p, img_pred, b = learn.predict(img)

NameError: name 'learn' is not defined

## === cell 39
def rgb2gray(_img):
    """ Convert from 3 channels to 1 channel """
    from skimage.color import rgb2gray as _rgb2gray

    _img_pred_np = _img.permute(1, 2, 0).numpy()
    _img_pred_2d = Tensor(_rgb2gray(_img_pred_np))
    _img_pred = _img_pred_2d.unsqueeze(0)
    return _img_pred
  
Image(rgb2gray(img_pred)).show(figsize=(8,5))


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2118983621.py in <cell line: 0>()
     10     return _img_pred
     11 
---> 12 Image(rgb2gray(img_pred)).show(figsize=(8,5))

NameError: name 'Image' is not defined

## === cell 40
def write_image(fname, _img_tensor):
    _img_tensor = (_img_tensor * 255).to(dtype=torch.uint8)
    imwrite(path_submission/fname, _img_tensor.squeeze().numpy())


## === cell 41
import csv
from imageio import imread, imwrite

path_submission.mkdir(exist_ok=True)

with Path('submission.csv').open('w', encoding='utf-8', newline='') as outf:
    writer = csv.writer(outf)
    writer.writerow(('id', 'value'))
    for i, fname in enumerate(path_test.glob("*.png")):
        img = open_image(fname)
        img_id = int(fname.name[:-4])
        print('Processing: {} '.format(img_id))
        p, img_pred, b = learn.predict(img)
        img_2d = rgb2gray(img_pred).clamp(0, 1)
        write_image(fname.name, img_2d)
        for r in range(img_2d.shape[1]):
            for c in range(img_2d.shape[2]):
                id = str(img_id)+'_'+str(r + 1)+'_'+str(c + 1)
                val = img_2d[0, r, c].item()
                writer.writerow((id, val))


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2338450312.py in <cell line: 0>()
      2 from imageio import imread, imwrite
      3 
----> 4 path_submission.mkdir(exist_ok=True)
      5 
      6 with Path('submission.csv').open('w', encoding='utf-8', newline='') as outf:

NameError: name 'path_submission' is not defined
