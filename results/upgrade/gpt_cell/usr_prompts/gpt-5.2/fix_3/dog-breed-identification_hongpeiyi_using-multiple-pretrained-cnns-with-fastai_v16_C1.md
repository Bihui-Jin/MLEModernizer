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

3.9

# 2. Installed packages

fastai==2.8.5
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torchvision==0.21.0+cu124

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
from fastai.vision.all import *

labels = pd.read_csv("../input/dog-breed-identification/labels.csv")
labels


## === cell 1
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_ids, valid_ids = next(split.split(labels, labels["breed"]))
labels["is_valid"] = [i in valid_ids for i in range(len(labels))]

labels["id"] = labels["id"].apply(lambda x: x + ".jpg")


## === cell 2
path = "../input/dog-breed-identification/train"

dls = ImageDataLoaders.from_df(labels, path,
                               item_tfms=Resize(460, method="squeeze"),
                               batch_tfms=[*aug_transforms(size=300),
                                           Normalize.from_stats(*imagenet_stats)],
                               bs=32, valid_col="is_valid")
dls.show_batch()


## === cell 3
from torchvision.models import inception_v3, mobilenet_v2, Inception_V3_Weights

inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
inception = nn.Sequential(*list(inception.children())[:-2], nn.Flatten()).eval()


## === cell 4
resnet = nn.Sequential(*list(resnet50(pretrained=True).children())[:-1], 
                          nn.Flatten()).eval()


## === cell 5
mobile = nn.Sequential(*list(mobilenet_v2(pretrained=True).children())[:-1],
                       nn.AdaptiveAvgPool2d((1,1)),
                       nn.Flatten()).eval()


## === cell 6
class NeuralNet(Module):
    def __init__(self, extractors, hidden_size, vocab_size, device):
        
        self.extractors = extractors
        for conv in self.extractors:
            conv.to(device)
                  
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, vocab_size)
        )
        
    def forward(self, x):
        
        features = torch.cat([conv(x) for conv in self.extractors], dim=1)
        
        return self.classifier(features)


## === cell 7
extractors = [inception, resnet, mobile]
hidden_size = 2048 + 2048 + 1280
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNet(extractors, hidden_size, len(dls.vocab), device)


## === cell 8
weights = [labels.shape[0] / (120 * labels["breed"].value_counts()[breed]) for breed in dls.vocab]
weights = tensor(weights, device=device)


## === cell 9

learn = Learner(dls, model, metrics=accuracy, path=".").to_fp16()


def _patched_forward(self, x):
    feats = []

    for conv in self.extractors:
        if (
            isinstance(conv, nn.Sequential)
            and len(conv) > 0
            and isinstance(conv[0], torchvision.models.inception.Inception3)
        ):
            inc = conv[0]
            y = inc.Conv2d_1a_3x3(x)
            y = inc.Conv2d_2a_3x3(y)
            y = inc.Conv2d_2b_3x3(y)
            y = inc.maxpool1(y)
            y = inc.Conv2d_3b_1x1(y)
            y = inc.Conv2d_4a_3x3(y)
            y = inc.maxpool2(y)
            y = inc.Mixed_5b(y)
            y = inc.Mixed_5c(y)
            y = inc.Mixed_5d(y)
            y = inc.Mixed_6a(y)
            y = inc.Mixed_6b(y)
            y = inc.Mixed_6c(y)
            y = inc.Mixed_6d(y)
            y = inc.Mixed_6e(y)
            y = inc.Mixed_7a(y)
            y = inc.Mixed_7b(y)
            y = inc.Mixed_7c(y)
            y = torch.nn.functional.adaptive_avg_pool2d(y, (1, 1))
            y = torch.flatten(y, 1)
            feats.append(y)
        else:
            feats.append(conv(x))

    features = torch.cat(feats, dim=1)
    return self.classifier(features)


model.forward = _patched_forward.__get__(model, type(model))

learn.lr_find()


## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2150715308.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     49[0m [0mmodel[0m[0;34m.[0m[0mforward[0m [0;34m=[0m [0m_patched_forward[0m[0;34m.[0m[0m__get__[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mtype[0m[0;34m([0m[0mmodel[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     50[0m [0;34m[0m[0m
[0;32m---> 51[0;31m [0mlearn[0m[0;34m.[0m[0mlr_find[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/fastai/callback/schedule.py[0m in [0;36mlr_find[0;34m(self, start_lr, end_lr, num_it, stop_div, show_plot, suggest_funcs)[0m
[1;32m    294[0m     [0mn_epoch[0m [0;34m=[0m [0mnum_it[0m[0;34m//[0m[0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdls[0m[0;34m.[0m[0mtrain[0m[0;34m)[0m [0;34m+[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    295[0m     [0mcb[0m[0;34m=[0m[0mLRFinder[0m[0;34m([0m[0mstart_lr[0m[0;34m=[0m[0mstart_lr[0m[0;34m,[0m [0mend_lr[0m[0;34m=[0m[0mend_lr[0m[0;34m,[0m [0mnum_it[0m[0;34m=[0m[0mnum_it[0m[0;34m,[0m [0mstop_div[0m[0;34m=[0m[0mstop_div[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 296[0;31m     [0;32mwith[0m [0mself[0m[0;34m.[0m[0mno_logging[0m[0;34m([0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mn_epoch[0m[0;34m,[0m [0mcbs[0m[0;34m=[0m[0mcb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    297[0m     [0;32mif[0m [0msuggest_funcs[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    298[0m         [0mlrs[0m[0;34m,[0m [0mlosses[0m [0;34m=[0m [0mtensor[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mrecorder[0m[0;34m.[0m[0mlrs[0m[0;34m[[0m[0mnum_it[0m[0;34m//[0m[0;36m10[0m[0;34m:[0m[0;34m-[0m[0;36m5[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0mtensor[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mrecorder[0m[0;34m.[0m[0mlosses[0m[0;34m[[0m[0mnum_it[0m[0;34m//[0m[0;36m10[0m[0;34m:[0m[0;34m-[0m[0;36m5[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mfit[0;34m(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)[0m
[1;32m    270[0m             [0mself[0m[0;34m.[0m[0mopt[0m[0;34m.[0m[0mset_hypers[0m[0;34m([0m[0mlr[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mlr[0m [0;32mif[0m [0mlr[0m [0;32mis[0m [0;32mNone[0m [0;32melse[0m [0mlr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    271[0m             [0mself[0m[0;34m.[0m[0mn_epoch[0m [0;34m=[0m [0mn_epoch[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 272[0;31m             [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_do_fit[0m[0;34m,[0m [0;34m'fit'[0m[0;34m,[0m [0mCancelFitException[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_end_cleanup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    273[0m [0;34m[0m[0m
[1;32m    274[0m     [0;32mdef[0m [0m_end_cleanup[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mdl[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0mxb[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0myb[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0mpred[0m[0;34m,[0m[0mself[0m[0;34m.[0m[0mloss[0m [0;34m=[0m [0;32mNone[0m[0;34m,[0m[0;34m([0m[0;32mNone[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m[0;34m([0m[0;32mNone[0m[0;34m,[0m[0;34m)[0m[0;34m,[0m[0;32mNone[0m[0;34m,[0m[0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_do_fit[0;34m(self)[0m
[1;32m    259[0m         [0;32mfor[0m [0mepoch[0m [0;32min[0m [0mrange[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mn_epoch[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    260[0m             [0mself[0m[0;34m.[0m[0mepoch[0m[0;34m=[0m[0mepoch[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 261[0;31m             [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_do_epoch[0m[0;34m,[0m [0;34m'epoch'[0m[0;34m,[0m [0mCancelEpochException[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    262[0m [0;34m[0m[0m
[1;32m    263[0m     [0;32mdef[0m [0mfit[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mn_epoch[0m[0;34m,[0m [0mlr[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mwd[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mcbs[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mreset_opt[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mstart_epoch[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_do_epoch[0;34m(self)[0m
[1;32m    253[0m [0;34m[0m[0m
[1;32m    254[0m     [0;32mdef[0m [0m_do_epoch[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 255[0;31m         [0mself[0m[0;34m.[0m[0m_do_epoch_train[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    256[0m         [0mself[0m[0;34m.[0m[0m_do_epoch_validate[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    257[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_do_epoch_train[0;34m(self)[0m
[1;32m    245[0m     [0;32mdef[0m [0m_do_epoch_train[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    246[0m         [0mself[0m[0;34m.[0m[0mdl[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdls[0m[0;34m.[0m[0mtrain[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 247[0;31m         [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mall_batches[0m[0;34m,[0m [0;34m'train'[0m[0;34m,[0m [0mCancelTrainException[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    248[0m [0;34m[0m[0m
[1;32m    249[0m     [0;32mdef[0m [0m_do_epoch_validate[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mds_idx[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m [0mdl[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mall_batches[0;34m(self)[0m
[1;32m    211[0m     [0;32mdef[0m [0mall_batches[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    212[0m         [0mself[0m[0;34m.[0m[0mn_iter[0m [0;34m=[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 213[0;31m         [0;32mfor[0m [0mo[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdl[0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mone_batch[0m[0;34m([0m[0;34m*[0m[0mo[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    214[0m [0;34m[0m[0m
[1;32m    215[0m     [0;32mdef[0m [0m_backward[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mloss_grad[0m[0;34m.[0m[0mbackward[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36mone_batch[0;34m(self, i, b)[0m
[1;32m    241[0m         [0mb[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_set_device[0m[0;34m([0m[0mb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    242[0m         [0mself[0m[0;34m.[0m[0m_split[0m[0;34m([0m[0mb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 243[0;31m         [0mself[0m[0;34m.[0m[0m_with_events[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_do_one_batch[0m[0;34m,[0m [0;34m'batch'[0m[0;34m,[0m [0mCancelBatchException[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    244[0m [0;34m[0m[0m
[1;32m    245[0m     [0;32mdef[0m [0m_do_epoch_train[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_with_events[0;34m(self, f, event_type, ex, final)[0m
[1;32m    205[0m [0;34m[0m[0m
[1;32m    206[0m     [0;32mdef[0m [0m_with_events[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mf[0m[0;34m,[0m [0mevent_type[0m[0;34m,[0m [0mex[0m[0;34m,[0m [0mfinal[0m[0;34m=[0m[0mnoop[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 207[0;31m         [0;32mtry[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'before_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mf[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    208[0m         [0;32mexcept[0m [0mex[0m[0;34m:[0m [0mself[0m[0;34m([0m[0;34mf'after_cancel_{event_type}'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m         [0mself[0m[0;34m([0m[0;34mf'after_{event_type}'[0m[0;34m)[0m[0;34m;[0m  [0mfinal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fastai/learner.py[0m in [0;36m_do_one_batch[0;34m(self)[0m
[1;32m    222[0m [0;34m[0m[0m
[1;32m    223[0m     [0;32mdef[0m [0m_do_one_batch[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 224[0;31m         [0mself[0m[0;34m.[0m[0mpred[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mmodel[0m[0;34m([0m[0;34m*[0m[0mself[0m[0;34m.[0m[0mxb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    225[0m         [0mself[0m[0;34m([0m[0;34m'after_pred'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    226[0m         [0;32mif[0m [0mlen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0myb[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2150715308.py[0m in [0;36m_patched_forward[0;34m(self, x)[0m
[1;32m     15[0m             [0misinstance[0m[0;34m([0m[0mconv[0m[0;34m,[0m [0mnn[0m[0;34m.[0m[0mSequential[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m             [0;32mand[0m [0mlen[0m[0;34m([0m[0mconv[0m[0;34m)[0m [0;34m>[0m [0;36m0[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 17[0;31m             [0;32mand[0m [0misinstance[0m[0;34m([0m[0mconv[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m,[0m [0mtorchvision[0m[0;34m.[0m[0mmodels[0m[0;34m.[0m[0minception[0m[0;34m.[0m[0mInception3[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m         ):
[1;32m     19[0m             [0minc[0m [0;34m=[0m [0mconv[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: name 'torchvision' is not defined

## === cell 10
learn.fit_one_cycle(3, 1e-3)
