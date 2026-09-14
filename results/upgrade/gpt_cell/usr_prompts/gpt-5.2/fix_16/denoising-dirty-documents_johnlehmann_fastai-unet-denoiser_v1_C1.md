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

3.8

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
%reload_ext autoreload
%autoreload 2
%matplotlib inline


## === cell 1
import pathlib
from pathlib import Path

import fastai

from fastai.vision.all import *
from fastai.callback.all import *

try:
    from fastai.utils.mem import *  # type: ignore
except ModuleNotFoundError:
    pass

from torchvision.models import vgg16_bn
from subprocess import check_output


## === cell 2
input_path = Path('/kaggle/input/denoising-dirty-documents')
items = list(input_path.glob("*.zip"))
print([x for x in items])


## === cell 3
import zipfile

for item in items:
    print(item)
    with zipfile.ZipFile(str(item), "r") as z:
        z.extractall(".")


## === cell 4
bs, size = 4, 128
arch = models.resnet34
path_train = Path("train")
path_train_cleaned = Path("train_cleaned")
path_test = Path("test")
path_submission = Path("submission")


## === cell 5
src = DataBlock(
    blocks=(ImageBlock, ImageBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=lambda x: path_train_cleaned / x.name,
)

dls = src.dataloaders(path_train, bs=bs, item_tfms=Resize(size))


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
def get_data(src, bs, size):
    block = DataBlock(
        blocks=(ImageBlock, ImageBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=lambda x: path_train_cleaned / x.name,
        item_tfms=Resize(size),
        batch_tfms=[
            *aug_transforms(max_zoom=2.0),
            Normalize.from_stats(*imagenet_stats, do_y=True),
        ],
    )
    data = block.dataloaders(path_train, bs=bs)
    data.c = 3
    return data


## === cell 8
_orig_from_stats = Normalize.from_stats


def _from_stats_compat(*args, **kwargs):
    kwargs.pop("do_y", None)
    return _orig_from_stats(*args, **kwargs)


Normalize.from_stats = _from_stats_compat

try:
    data
except NameError:
    data = get_data(src, bs, size)

try:
    data.show_batch(rows=2, figsize=(5, 5), max_n=4)
except Exception:
    pass


## === cell 9
y = data.valid_ds[0][1]
t = TensorImage(np.array(y)).permute(2, 0, 1).float() / 255.0
t = torch.stack([t, t])


## === cell 10
def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)


## === cell 11
base_loss = F.l1_loss


## === cell 12
vgg_m = vgg16_bn(True).features.cuda().eval()

for p in vgg_m.parameters():
    p.requires_grad_(False)


## === cell 13
blocks = [i - 1 for i, o in enumerate(list(vgg_m)) if isinstance(o, nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]


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


## === cell 15
feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])


## === cell 16
wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
gc.collect();


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3861058962.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mwd[0m [0;34m=[0m [0;36m1e-3[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
[0m[1;32m      3[0m                      blur=True, norm_type=NormType.Weight)
[1;32m      4[0m [0mgc[0m[0;34m.[0m[0mcollect[0m[0;34m([0m[0;34m)[0m[0;34m;[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py[0m in [0;36munet_learner[0;34m(dls, arch, normalize, n_out, pretrained, weights, config, loss_func, opt_func, lr, splitter, cbs, metrics, path, model_dir, wd, wd_bn_bias, train_bn, moms, **kwargs)[0m
[1;32m    281[0m     [0mimg_size[0m [0;34m=[0m [0mdls[0m[0;34m.[0m[0mone_batch[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;34m-[0m[0;36m2[0m[0;34m:[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    282[0m     [0;32massert[0m [0mimg_size[0m[0;34m,[0m [0;34m"image size could not be inferred from data"[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 283[0;31m     [0mmodel[0m [0;34m=[0m [0mcreate_unet_model[0m[0;34m([0m[0march[0m[0;34m,[0m [0mn_out[0m[0;34m,[0m [0mimg_size[0m[0;34m,[0m [0mpretrained[0m[0;34m=[0m[0mpretrained[0m[0;34m,[0m [0mweights[0m[0;34m=[0m[0mweights[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    284[0m [0;34m[0m[0m
[1;32m    285[0m     [0msplitter[0m [0;34m=[0m [0mifnone[0m[0;34m([0m[0msplitter[0m[0;34m,[0m [0mmeta[0m[0;34m[[0m[0;34m'split'[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py[0m in [0;36mcreate_unet_model[0;34m(arch, n_out, img_size, pretrained, weights, cut, n_in, **kwargs)[0m
[1;32m    258[0m         [0mmodel[0m [0;34m=[0m [0march[0m[0;34m([0m[0mpretrained[0m[0;34m=[0m[0mpretrained[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    259[0m     [0mbody[0m [0;34m=[0m [0mcreate_body[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mn_in[0m[0;34m,[0m [0mpretrained[0m[0;34m,[0m [0mifnone[0m[0;34m([0m[0mcut[0m[0;34m,[0m [0mmeta[0m[0;34m[[0m[0;34m'cut'[0m[0;34m][0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 260[0;31m     [0mmodel[0m [0;34m=[0m [0mmodels[0m[0;34m.[0m[0munet[0m[0;34m.[0m[0mDynamicUnet[0m[0;34m([0m[0mbody[0m[0;34m,[0m [0mn_out[0m[0;34m,[0m [0mimg_size[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    261[0m     [0;32mreturn[0m [0mmodel[0m[0;34m[0m[0;34m[0m[0m
[1;32m    262[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastcore/meta.py[0m in [0;36m__call__[0;34m(cls, *args, **kwargs)[0m
[1;32m     40[0m         [0;32mif[0m [0mtype[0m[0;34m([0m[0mres[0m[0;34m)[0m[0;34m==[0m[0mcls[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     41[0m             [0;32mif[0m [0mhasattr[0m[0;34m([0m[0mres[0m[0;34m,[0m[0;34m'__pre_init__'[0m[0;34m)[0m[0;34m:[0m [0mres[0m[0;34m.[0m[0m__pre_init__[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 42[0;31m             [0mres[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     43[0m             [0;32mif[0m [0mhasattr[0m[0;34m([0m[0mres[0m[0;34m,[0m[0;34m'__post_init__'[0m[0;34m)[0m[0;34m:[0m [0mres[0m[0;34m.[0m[0m__post_init__[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     44[0m         [0;32mreturn[0m [0mres[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/vision/models/unet.py[0m in [0;36m__init__[0;34m(self, encoder, n_out, img_size, blur, blur_final, self_attention, y_range, last_cross, bottle, act_cls, init, norm_type, **kwargs)[0m
[1;32m     66[0m [0;34m[0m[0m
[1;32m     67[0m         [0mni[0m [0;34m=[0m [0msizes[0m[0;34m[[0m[0;34m-[0m[0;36m1[0m[0;34m][0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 68[0;31m         middle_conv = nn.Sequential(ConvLayer(ni, ni*2, act_cls=act_cls, norm_type=norm_type, **kwargs),
[0m[1;32m     69[0m                                     ConvLayer(ni*2, ni, act_cls=act_cls, norm_type=norm_type, **kwargs)).eval()
[1;32m     70[0m         [0mx[0m [0;34m=[0m [0mmiddle_conv[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/layers.py[0m in [0;36m__init__[0;34m(self, ni, nf, ks, stride, padding, bias, ndim, norm_type, bn_1st, act_cls, transpose, init, xtra, bias_std, **kwargs)[0m
[1;32m    249[0m         [0;32mif[0m [0mbias[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m [0mbias[0m [0;34m=[0m [0;32mnot[0m [0;34m([0m[0mbn[0m [0;32mor[0m [0minn[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    250[0m         [0mconv_func[0m [0;34m=[0m [0m_conv_func[0m[0;34m([0m[0mndim[0m[0;34m,[0m [0mtranspose[0m[0;34m=[0m[0mtranspose[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 251[0;31m         [0mconv[0m [0;34m=[0m [0mconv_func[0m[0;34m([0m[0mni[0m[0;34m,[0m [0mnf[0m[0;34m,[0m [0mkernel_size[0m[0;34m=[0m[0mks[0m[0;34m,[0m [0mbias[0m[0;34m=[0m[0mbias[0m[0;34m,[0m [0mstride[0m[0;34m=[0m[0mstride[0m[0;34m,[0m [0mpadding[0m[0;34m=[0m[0mpadding[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    252[0m         [0mact[0m [0;34m=[0m [0;32mNone[0m [0;32mif[0m [0mact_cls[0m [0;32mis[0m [0;32mNone[0m [0;32melse[0m [0mact_cls[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    253[0m         [0minit_linear[0m[0;34m([0m[0mconv[0m[0;34m,[0m [0mact[0m[0;34m,[0m [0minit[0m[0;34m=[0m[0minit[0m[0;34m,[0m [0mbias_std[0m[0;34m=[0m[0mbias_std[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Conv2d.__init__() got an unexpected keyword argument 'callback_fns'

## === cell 17
learn.model_dir = Path('models').absolute()
