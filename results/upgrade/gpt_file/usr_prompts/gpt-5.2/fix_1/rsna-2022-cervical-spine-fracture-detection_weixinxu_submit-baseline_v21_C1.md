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
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Target score

0.6699109061407365

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import argparse
import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm
import csv

import torch
from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.interpolation import zoom


## === cell 1
import copy

class Dict(dict):

    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        object.__setattr__(__self, '__frozen', False)
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in kwargs.items():
            __self[key] = __self._hook(val)

    def __setattr__(self, name, value):
        if hasattr(self.__class__, name):
            raise AttributeError("'Dict' object attribute "
                                 "'{0}' is read-only".format(name))
        else:
            self[name] = value

    def __setitem__(self, name, value):
        isFrozen = (hasattr(self, '__frozen') and
                    object.__getattribute__(self, '__frozen'))
        if isFrozen and name not in super(Dict, self).keys():
                raise KeyError(name)
        super(Dict, self).__setitem__(name, value)
        try:
            p = object.__getattribute__(self, '__parent')
            key = object.__getattribute__(self, '__key')
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, '__parent')
            object.__delattr__(self, '__key')

    def __add__(self, other):
        if not self.keys():
            return other
        else:
            self_type = type(self).__name__
            other_type = type(other).__name__
            msg = "unsupported operand type(s) for +: '{}' and '{}'"
            raise TypeError(msg.format(self_type, other_type))

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __missing__(self, name):
        if object.__getattribute__(self, '__frozen'):
            raise KeyError(name)
        return self.__class__(__parent=self, __key=name)

    def __delattr__(self, name):
        del self[name]

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, type(self)) else
                    item for item in value)
            else:
                base[key] = value
        return base

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def update(self, *args, **kwargs):
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if ((k not in self) or
                (not isinstance(self[k], dict)) or
                (not isinstance(v, dict))):
                self[k] = v
            else:
                self[k].update(v)

    def __getnewargs__(self):
        return tuple(self.items())

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def __or__(self, other):
        if not isinstance(other, (Dict, dict)):
            return NotImplemented
        new = Dict(self)
        new.update(other)
        return new

    def __ror__(self, other):
        if not isinstance(other, (Dict, dict)):
            return NotImplemented
        new = Dict(other)
        new.update(self)
        return new

    def __ior__(self, other):
        self.update(other)
        return self

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    def freeze(self, shouldFreeze=True):
        object.__setattr__(self, '__frozen', shouldFreeze)
        for key, val in self.items():
            if isinstance(val, Dict):
                val.freeze(shouldFreeze)

    def unfreeze(self):
        self.freeze(False)

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

## === cell 2
import ast
import os.path as osp
import platform
import shutil
import sys
import tempfile
import copy

from argparse import Action, ArgumentParser
from collections import abc
from importlib import import_module
from yapf.yapflib.yapf_api import FormatCode

if platform.system() == 'Windows':
    import regex as re
else:
    import re


BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text']
class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, 'r') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    def _substitute_predefined_vars(filename, temp_config_name):
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(filename, 'r') as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w') as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _file2dict(filename, use_predefined_variables=True):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
            raise IOError('Only py/yml/yaml/json type are supported now!')

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            if platform.system() == 'Windows':
                temp_config_file.close()
            temp_config_name = osp.basename(temp_config_file.name)
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename,
                                                   temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)

            if filename.endswith('.py'):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                }
                del sys.modules[temp_module_name]
            elif filename.endswith(('.yml', '.yaml', '.json')):
                import mmcv
                cfg_dict = mmcv.load(temp_config_file.name)
            temp_config_file.close()

        cfg_text = filename + '\n'
        with open(filename, 'r') as f:
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError('Duplicate key is not allowed among bases')
                base_cfg_dict.update(c)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b):
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b and not v.pop(DELETE_KEY, False):
                if not isinstance(b[k], dict):
                    raise TypeError(
                        f'{k}={v} in child config cannot inherit from base '
                        f'because {k} is a dict in the child config but is of '
                        f'type {type(b[k])} in base config. You may set '
                        f'`{DELETE_KEY}=True` to ignore the base config')
                b[k] = Config._merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    @staticmethod
    def fromfile(filename, use_predefined_variables=True):
        cfg_dict, cfg_text = Config._file2dict(filename,
                                               use_predefined_variables)
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from config file automatically (experimental)"""
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    @property
    def pretty_text(self):

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        yapf_style = dict(
            based_on_style='pep8',
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True)
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def __repr__(self):
        return f'Config (path: {self.filename}): {self._cfg_dict.__repr__()}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._filename, self._text)

    def __setstate__(self, state):
        _cfg_dict, _filename, _text = state
        super(Config, self).__setattr__('_cfg_dict', _cfg_dict)
        super(Config, self).__setattr__('_filename', _filename)
        super(Config, self).__setattr__('_text', _text)

    def dump(self, file=None):
        cfg_dict = super(Config, self).__getattribute__('_cfg_dict').to_dict()
        if self.filename.endswith('.py'):
            if file is None:
                return self.pretty_text
            else:
                with open(file, 'w') as f:
                    f.write(self.pretty_text)
        else:
            import mmcv
            if file is None:
                file_format = self.filename.split('.')[-1]
                return mmcv.dump(cfg_dict, file_format=file_format)
            else:
                mmcv.dump(cfg_dict, file)

    def merge_from_dict(self, options):
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

        Args:
            options (dict): dict of configs to merge from.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
        super(Config, self).__setattr__(
            '_cfg_dict', Config._merge_a_into_b(option_cfg_dict, cfg_dict))

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1181507813.py in <cell line: 0>()
     11 from collections import abc
     12 from importlib import import_module
---> 13 from yapf.yapflib.yapf_api import FormatCode
     14 
     15 if platform.system() == 'Windows':

ModuleNotFoundError: No module named 'yapf'

## === cell 3
def dcm2nii(dcms_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    spacing = image2.GetSpacing()  # x, y, z
    return image_array, spacing

## === cell 4
class SegConfig:

    def __init__(self,  network_f):
        self.network_f = network_f
        if self.network_f is not None:

            if isinstance(self.network_f, str):
                self.network_cfg = Config.fromfile(self.network_f)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py') as temp_config_file:
                        with open(temp_config_file.name, 'wb') as f:
                            f.write(self.network_f.read())

                        self.network_cfg = Config.fromfile(temp_config_file.name)

    def __repr__(self) -> str:
        return str(self.__dict__)

## === cell 5
class SegModel:

    def __init__(self, model_f, network_f):
        self.model_f = model_f
        self.network_f = network_f

## === cell 6
class SegPredictor:

    def __init__(self, gpu: int, model: SegModel):
        self.gpu = gpu
        self.model = model
        self.config = SegConfig(self.model.network_f)
        self.load_model()

    def load_model(self):
        self.net = self._load_model(self.model.model_f, self.config.network_cfg, half=False)

    def _load_model(self, model_f, network_f, half=False) -> None:
        if isinstance(model_f, str):
            if model_f.endswith(".pth"):
                net = self.load_model_pth(model_f, network_f, half)
            else:
                net = self.load_model_jit(model_f, half)
        else:
            model_f.seek(0)
            headers = model_f.peek(2)
            if headers[0] == 0x80 and headers[1] == 0x02:
                net = self.load_model_pth(model_f, network_f, half)
            else:
                net = self.load_model_jit(model_f, half)
        return net

    def load_model_pth(self, model_f, network_cfg, half) -> None:
        config = network_cfg
        print(config.model["backbone"])
        if config.model["backbone"] == "ResUnet":
            backbone = ResUnet(config.in_ch, channels=config.model["channels"])
        else:
            raise TypeError("<<<<<<<<<<<wrong backbone>>>>>>>>>>>>")

        if config.model["head_type"] == "SegSoftHead" or config.model["head_type"] == "SegSoftHead3D":
            head = SegSoftHead(config.model["channels"], classes=8)
        elif config.model["head_type"]== "SegSigHead":
            head = SegSigHead(config.model["channels"], classes=1)
        else:
            raise TypeError("<<<<<<<<<<<wrong head>>>>>>>>>>>>")
            
        net = SegNetwork(backbone, head)
        checkpoint = torch.load(model_f, map_location=f"cpu")
        if list(checkpoint["state_dict"].keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        net.half()
        net.cuda()
        net = net.forward_test
        return net

    def forward(self, seg, spacing):

        config = self.config.network_cfg
        patch_size = np.array(config.patch_size)
        ori_shape = np.array(seg.shape)
        seg_np = zoom(seg, np.array(np.array(patch_size) / ori_shape), order=0)

        data = torch.from_numpy(seg_np).float()[None, None]
        data = data.detach()
        
        with autocast():
            data = data.cuda().detach()
            pred_seg = self.net(data)
            del data
            if pred_seg.size()[1] > 1:
                pred_seg = F.softmax(pred_seg, dim=1)
                pred_seg = torch.argmax(pred_seg, dim=1, keepdim=True)
                pred_seg[pred_seg == 8] = 0
            else:
                pred_seg = torch.sigmoid(pred_seg)
                pred_seg[pred_seg >= config.threshold] = 1
                pred_seg[pred_seg < config.threshold] = 0
     
        heatmap = pred_seg.cpu().detach().numpy()[0, 0].astype(np.uint8)
        heatmap = zoom(heatmap, np.array(ori_shape / np.array(patch_size)), order=0)

        heatmap *= (seg > 0)
        heatmap = heatmap.astype(np.uint8)

        return heatmap

class SegPredictor_:

    def __init__(self, gpu: int, model: SegModel):
        self.gpu = gpu
        self.model = model
        self.config = SegConfig(self.model.network_f)
        self.load_model()

    def load_model(self):
        self.net = self._load_model(self.model.model_f, self.config.network_cfg, half=False)

    def _load_model(self, model_f, network_f, half=False) -> None:
        if isinstance(model_f, str):
            net = self.load_model_pth(model_f, network_f, half)
        else:
            model_f.seek(0)
            headers = model_f.peek(2)
            net = self.load_model_pth(model_f, network_f, half)
        return net
    
    def load_model_pth(self, model_f, network_cfg, half) -> None:
        config = network_cfg
        backbone = ResUnet(config.model["in_ch"], channels=config.model["channels"])

        if config.model["head_type"] == "SegSoftHead" or config.model["head_type"] == "SegSoftHead3D":
            head = SegSoftHead(in_channels=config.model["channels"], classes=config.model["classes"])
        elif config.model["head_type"] == "SegSigHead" or config.model["head_type"] == "SegSigHead3D":
            head = SegSigHead(in_channels=config.model["channels"], classes=1)
        else:
            raise TypeError("<<<<<<<<<<<wrong head>>>>>>>>>>>>")
            
        net = SegNetwork(backbone, head)
        checkpoint = torch.load(model_f, map_location=f"cpu")
        if list(checkpoint["state_dict"].keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        net.half()
        net.cuda()
        net = net.forward_test
        return net

    def _get_input(self, vol, spacing_zyx):

        config = self.config.network_cfg

        def _window_array(vol, win_level, win_width):
            win = [
                win_level - win_width / 2,
                win_level + win_width / 2,
            ]
            vol = torch.clamp(vol, win[0], win[1])
            vol -= win[0]
            vol /= win_width
            return vol

        vol = torch.from_numpy(vol).float()[None, None]
        vol = [_window_array(vol, wl, wd) for wl, wd in zip(config.win_level, config.win_width)]
        vol = torch.cat(vol, dim=1)
        vol_shape = np.array(vol.shape[2:], dtype=np.float32)
        patch_size = np.array(config.patch_size)
        if np.any(vol_shape != patch_size):
            vol = torch.nn.functional.interpolate(
                vol.float(), size=tuple(patch_size), mode="trilinear", align_corners=False
            )
        vol = vol.detach()
        return vol

    def forward(self, vol, spacing, sup_mask=None):

        config = self.config.network_cfg
        patch_size = np.array(config.patch_size)
        ori_shape = np.array(vol.shape)
        spacing_zyx = np.array(spacing)
        data = self._get_input(vol, spacing_zyx)


        with autocast():
            data = data.cuda().detach()
            pred_seg = self.net(data)
            del data
            if pred_seg.size()[1] > 1:
                pred_seg = F.softmax(pred_seg, dim=1)
                pred_seg = torch.argmax(pred_seg, dim=1, keepdim=True)
            else:
                pred_seg = torch.sigmoid(pred_seg)
                pred_seg[pred_seg >= config.threshold] = 1
                pred_seg[pred_seg < config.threshold] = 0
     
        heatmap = pred_seg.cpu().detach().numpy()[0, 0].astype(np.uint8)
        heatmap = zoom(heatmap, np.array(ori_shape / np.array(patch_size)), order=0)

        heatmap = heatmap.astype(np.uint8)

        return heatmap

## === cell 7
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

## === cell 8
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        del identity
        del x
        torch.cuda.empty_cache()
        
        out = self.relu(out)

        return out

## === cell 9
def make_res_layer(inplanes, planes, blocks, stride=1):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)

## === cell 10
class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

## === cell 11
def _ASPPConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    asppconv = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )
    return asppconv

class ASPP(nn.Module):
    """
    ASPP module in `DeepLabV3, see also in <https://arxiv.org/abs/1706.05587>` 
    """
    def __init__(self, in_channels, out_channels, output_stride=16):
        super(ASPP, self).__init__()

        if output_stride == 16:
            astrous_rates = [0, 4, 8, 12]
        elif output_stride == 8:
            astrous_rates = [0, 2, 4, 8]
        else:
            raise Warning('Output stride must be 8 or 16!')

        self.conv1 = _ASPPConv(in_channels, out_channels, 1, 1)
        self.conv2 = _ASPPConv(in_channels, out_channels, 3, 1, padding=astrous_rates[1], dilation=astrous_rates[1])
        self.conv3 = _ASPPConv(in_channels, out_channels, 3, 1, padding=astrous_rates[2], dilation=astrous_rates[2])
        self.conv4 = _ASPPConv(in_channels, out_channels, 3, 1, padding=astrous_rates[3], dilation=astrous_rates[3])

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, input):
        input1 = self.conv1(input)
        input2 = self.conv2(input)
        input3 = self.conv3(input)
        input4 = self.conv4(input)
        
        input5 = F.interpolate(self.pool(input), size=input4.size()[2:], mode='trilinear', align_corners=False)
        output = torch.cat((input1, input2, input3, input4, input5), dim=1)

        del input
        del input1
        del input2
        del input3
        del input4
        del input5
        torch.cuda.empty_cache()

        output = self.bottleneck(output)
        return output

## === cell 12
class ResUnet(nn.Module):

    def __init__(self, in_ch, channels=16, blocks=3, use_aspp=False, is_aux=False):
        super(ResUnet, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.aspp = ASPP(channels * 16, channels * 16)
        self.is_aux = is_aux
        self.use_aspp = use_aspp

    def forward(self, input):
        c1 = self.in_conv(input)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        if self.use_aspp:
            c4_ap = self.aspp(c4) 
        else:
            c4_ap = c4

        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        if self.is_aux:
            return [up_8, c7, c6, c5]
        else:
            return up_8

## === cell 13
class SegSigHead(nn.Module):

    def __init__(self, in_channels, classes=1):
        super(SegSigHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, 1)
        self.bce_loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, inputs):
        features = self.conv(inputs)
        return features

    def forward_test(self, inputs):
        return self.forward(inputs)

class SegSoftHead(nn.Module):

    def __init__(self, in_channels, classes=14):
        super(SegSoftHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, classes, 1)
        self.multi_loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self._classes = classes

    def forward(self, inputs):
        seg_predict = self.conv(inputs)
        return seg_predict

    def forward_test(self, inputs):
        return self.forward(inputs)
    
class SegSoftHead_cls_vert(nn.Module):

    def __init__(self, in_channels, classes=5):
        super(SegSoftHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, classes, 1)
        self.bce_loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)
        self.multi_loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self._classes = classes

    def forward(self, inputs):
        inputs = F.interpolate(inputs, scale_factor=1.0, mode='trilinear')
        seg_predict = self.conv(inputs)

        del inputs
        torch.cuda.empty_cache()

        return seg_predict

    def forward_test(self, inputs):
        return self.forward(inputs)

## === cell 14
class SegNetwork(nn.Module):

    def __init__(self, backbone,
                head, apply_sync_batchnorm=False,
                train_cfg=None,
                test_cfg=None
                ):
        super(SegNetwork, self).__init__()
        self.backbone = backbone
        self.head = head
        self._show_count = 0
        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward(self, vol, seg):
        vol = vol.float()
        seg = seg.float()
        features = self.backbone(vol)
        head_outs = self.head(features)
        del features
        del vol
        del seg
        torch.cuda.empty_cache()
        return head_outs

    @torch.jit.export
    def forward_test(self, img):
        features = self.backbone(img)
        del img 
        torch.cuda.empty_cache()
        seg_predict = self.head.forward_test(features)
        del features
        torch.cuda.empty_cache()
        return seg_predict

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)

## === cell 15
class SpineModel:

    def __init__(self, model_f, network_f):
        self.model_f = model_f
        self.network_f = network_f

class SpineConfig:

    def __init__(self,  network_f):
        self.network_f = network_f
        if self.network_f is not None:
            if isinstance(self.network_f, str):
                self.network_cfg = Config.fromfile(self.network_f)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py') as temp_config_file:
                        with open(temp_config_file.name, 'wb') as f:
                            f.write(self.network_f.read())

                        self.network_cfg = Config.fromfile(temp_config_file.name)

    def __repr__(self) -> str:
        return str(self.__dict__)

class ClsPredictor2D:

    def __init__(self, gpu: int, model: SpineModel):
        self.gpu = gpu
        self.model = model
        self.config = SpineConfig(self.model.network_f)
        self.load_model()
    def load_model(self):
        self.net = self._load_model(self.model.model_f, self.config.network_cfg, half=False)

    def _load_model(self, model_f, network_f, half=False) -> None:
        net = self.load_model_pth(model_f, network_f, half)
        return net

    def load_model_pth(self, model_f, network_cfg, half) -> None:
        config = network_cfg
        backbone = ResNet2D_50()
        head = ClsHead_Res(in_channels=config.model["in_channels"], num_classes=config.model["num_classes"])
        net = ClsNetwork_Res(backbone, head)
        checkpoint = torch.load(model_f, map_location=f"cpu")
        if list(checkpoint["state_dict"].keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        net.half()
        net.cuda()
        net = net.forward_test
        return net
    
    def _get_input(self, img, spacing_zyx):

        config = self.config.network_cfg


        def _window_array(img, win_level, win_width):
            win = [
                win_level - win_width / 2,
                win_level + win_width / 2,
            ]
            img = torch.clamp(img, win[0], win[1])
            img -= win[0]
            img /= win_width
            return img

        patch_size = np.array(config.patch_size)
        all_size = np.array(list(img.shape[:2]) + list(patch_size))
        if np.any(all_size != img.shape):
            img = zoom(img, np.array(np.array(all_size) / img.shape), order=1)
        img = torch.from_numpy(img).float()
        img = [_window_array(img, wl, wd) for wl, wd in zip(config.win_level, config.win_width)]
        img = torch.cat(img, dim=1)
        img = img.detach()
        return img

    def forward(self, img, spacing):
        spacing_zyx = np.array(spacing)
        data = self._get_input(img, spacing_zyx)

        with autocast():
            data = data.cuda().detach()
            pred = self.net(data)
            del data
            if pred.size()[1] > 1:
                pred = F.softmax(pred, dim=1)
            else:
                pred = torch.sigmoid(pred)
        pred = pred.cpu().detach().numpy()
        return pred


class ClsPredictor2D_local(ClsPredictor2D):

    def load_model_pth(self, model_f, network_cfg, half) -> None:
        config = network_cfg
        backbone = ResNet2D_50()
        head = ClsHead_Res(in_channels=config.model["in_channels"], num_classes=config.model["num_classes"])
        net = ClsNetwork_Res(backbone, head)
        checkpoint = torch.load(model_f, map_location=f"cpu")
        if list(checkpoint["state_dict"].keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        net.half()
        net.cuda()
        net = net.forward_test
        return net

## === cell 16

class ClsHead_Res(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(ClsHead_Res, self).__init__()
        self.fc = nn.Linear(in_channels, num_classes)
        self.v = nn.NLLLoss()
        self.classes = num_classes
    def forward(self, inputs):
        cls_out = torch.flatten(inputs, 1)        
        cls_out = self.fc(cls_out)
        return cls_out

    def forward_test(self, inputs):
        return self.forward(inputs)

## === cell 17
class ClsNetwork_Res(nn.Module):

    def __init__(self, backbone,
                head, apply_sync_batchnorm=False,
                train=True,
                train_cfg=None,
                test_cfg=None
                ):
        super(ClsNetwork_Res, self).__init__()
        self.backbone = backbone
        self.head = head
        self._show_count = 0
        self._train = train
        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward(self, img, label):
        with torch.no_grad():
            img = img.float()
            label = label.float()
        features = self.backbone(img)
        head_outs = self.head(features)
        return head_outs

    @torch.jit.export
    def forward_test(self, img):
        features = self.backbone(img)
        label_predict = self.head(features)
        return label_predict

    def single_test(self, img, label):
        with torch.no_grad():
            img = img.float()
            features = self.backbone(img)
        predict = self.head(features)
        predict = F.softmax(predict, dim=1)
        predict = torch.argmax(predict, dim=1)
        acc_value = []
        label = torch.squeeze(label)
        labels = torch.argmax(label, dim=1)
        for i in range(labels.shape[0]):
            if (predict[i]==labels[i]):
                acc_value.append([1])
            else:
                acc_value.append([0])
        
        return acc_value

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)

## === cell 18
import torch
import torch.nn as nn
 
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']
def conv3x3_2d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
 
def conv1x1_2d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)   

class BasicBlock2D(nn.Module):
    expansion = 1
    __constants__ = ['downsample']
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # Batch Normalization
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock2D only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock2D")
        self.conv1 = conv3x3_2d(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True) # Relu激活函数
        self.conv2 = conv3x3_2d(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out
    
class Bottleneck2D(nn.Module):
    expansion = 4
    __constants__ = ['downsample']
 
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1_2d(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_2d(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1_2d(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out
    
class ResNet(nn.Module):
     
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
 
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck2D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock2D):
                    nn.init.constant_(m.bn2.weight, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        ''' layer1的时候不执行下面这个if，layer2往后，由于self.inplanes不等于
        planes*block.expansion，所以需要经过下面的if段，作用是将通道数由self.planes变为
        planes*block.expansion'''
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_2d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
 
        return nn.Sequential(*layers)
 
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
 
        return x
 
    def forward(self, x):
        return self._forward_impl(x)


class ResNet2D_18(nn.Module):

    def __init__(self, **kwargs):
        super(ResNet2D_18, self).__init__()
        self.feature_extractor = ResNet(BasicBlock2D, [2, 2, 2, 2], **kwargs)

    def forward(self, input):
        return self.feature_extractor(input)

class ResNet2D_34(nn.Module):

    def __init__(self, **kwargs):
        super(ResNet2D_34, self).__init__()
        self.feature_extractor = ResNet(BasicBlock2D, [3, 4, 6, 3], **kwargs)

    def forward(self, input):
        return self.feature_extractor(input)

class ResNet2D_50(nn.Module):

    def __init__(self, **kwargs):
        super(ResNet2D_50, self).__init__()
        self.feature_extractor = ResNet(Bottleneck2D, [3, 4, 6, 3], **kwargs)

    def forward(self, input):
        return self.feature_extractor(input)

class ResNet2D_101(nn.Module):

    def __init__(self, **kwargs):
        super(ResNet2D_101, self).__init__()
        self.feature_extractor = ResNet(Bottleneck2D, [3, 4, 23, 3], **kwargs)

    def forward(self, input):
        return self.feature_extractor(input)

## === cell 19
def inference(predictor: SegPredictor, hu_volume, spacing):
    pred_array = predictor.forward(hu_volume, spacing)
    return pred_array

def crop_patch(vol, z_num=10):
    z, x, y = vol.shape
    if z%z_num:
        z_stride = int(z/z_num) + 1
    else:
        z_stride = int(z/z_num)
    vol_crop = []
    z_max_list = []
    z_min_list = []

    for z_item in range(z_num):
        z_min = z_item*z_stride
        z_max = (z_item+1)*z_stride
        if z_max > z:
            z_max = z
            z_min = z-z_stride

        vol_item = vol[z_min:z_max]
        vol_crop.append(vol_item)
        z_max_list.append(z_max)
        z_min_list.append(z_min)
    return vol_crop, z_max_list, z_min_list

def find_valid_region(mask, values, low_margin=[0,0,0], up_margin=[0,0,0]):
    for v in values:
        mask[mask == v] = 100
    nonzero_points = np.argwhere((mask > 20))
    if len(nonzero_points) == 0:
        return None, None
    else:
        v_min = np.min(nonzero_points, axis=0)
        v_max = np.max(nonzero_points, axis=0)
        assert len(v_min) == len(low_margin), f'the length of margin is not equal the mask dims {len(v_min)}!'
        for idx in range(len(v_min)):
            v_min[idx] = max(0, v_min[idx] - low_margin[idx])
            v_max[idx] = min(mask.shape[idx], v_max[idx] + up_margin[idx])
        return v_min, v_max

## === cell 20
try:
    import cc3d
except:
    !cp ../input/mirror-cc3d/connected_components_3d-3.10.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl ./
    !pip install ./connected_components_3d-3.10.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    import cc3d

def max_connected_region(mask, values):
    out = np.zeros(mask.shape, dtype=np.uint8)
    for v in values:
        temp = mask.copy()
        temp[temp != v] = 0
        if v in list(np.unique(temp)):
            labeled, N = cc3d.connected_components((temp > 0), return_N=True)
            area = np.sum(labeled == 1); target = 1
            if N >= 2:
                for idx in range(2, N+1):
                    if np.sum(labeled == idx) > area:
                        target = idx
                        area = np.sum(labeled == idx)
            assert (target > 0)
            out[labeled == target] = v
    return out

def main(input_path, output_path, gpu):
    
    model_seg_global = '../input/test-model/seg_model/coarse_epoch_60.pth'
    network_seg_global = '../input/test-model/seg_model/train_config_coarse.py'
    
    model_seg_fine = '../input/model-0919/fine_eight_epoch_90.pth'
    network_seg_fine = '../input/model-0919/train_config_fine_eight.py'
    
    
    
    model_seg_crop = '../input/test-model/seg_model/crop_epoch_30.pth'
    network_seg_crop = '../input/test-model/seg_model/train_config_crop.py'
    
    model_cls_vert = '../input/test-model/cls_epoch_120_16.pth'
    network_cls_vert = '../input/test-model/train_config_cls_16.py'
    
    model_cls_frac = '../input/model-0919/frac_2D_epoch_43.pth'
    network_cls_frac = '../input/model-0919/train_config_res_frac.py'
    
    model_segmask_global = SegModel(
        model_f=model_seg_global,
        network_f=network_seg_global,
    )
    model_segmask_fine = SegModel(
        model_f=model_seg_fine,
        network_f=network_seg_fine,
    )
    model_segmask_crop = SegModel(
        model_f=model_seg_crop,
        network_f=network_seg_crop,
    )
    model_seg_cls = SegModel(
        model_f=model_cls_vert,
        network_f=network_cls_vert,
    )
    model_frac_cls = SpineModel(
        model_f = model_cls_frac,
        network_f = network_cls_frac
    )
    
    predictor_segmask_global = SegPredictor_(
        gpu = gpu,
        model = model_segmask_global,
    )
    predictor_segmask_fine = SegPredictor_(
        gpu = gpu,
        model = model_segmask_fine,
    )
    predictor_segmask_crop = SegPredictor_(
        gpu = gpu,
        model = model_segmask_crop,
    )
    predictor_seg_cls = SegPredictor(
        gpu = gpu,
        model = model_seg_cls,
    )
    predictor_frac_cls = ClsPredictor2D_local(
        gpu = gpu,
        model = model_frac_cls,
    )


    os.makedirs(output_path, exist_ok=True)
    file_result = open('submission.csv', 'w')
    writer = csv.writer(file_result)
    row_head = ['StudyInstanceUID','row_id', 'fractured']
    writer.writerow(row_head)
    
    pids = sorted(os.listdir(input_path))
    
    predictions = []
    batch_size_2D = 256
    batch_size_3D = 2
    
    for pid in tqdm(pids):
        sub_res = [pid]
        pid_path = os.path.join(input_path, pid)
        if not os.path.isdir(pid_path):
            continue
        dcms = sorted(os.listdir(pid_path))
        if not dcms[0].endswith('.dcm'):
            continue
        print('data load ......') 
        hu_volume, spacing = dcm2nii(pid_path)
        print(pid, hu_volume.shape)
        src_shape = hu_volume.shape
        spacing = spacing[::-1]

        global_seg = inference(predictor_segmask_global, hu_volume, spacing)
        global_seg_shape = global_seg.shape
        heatmap = global_seg.copy()
        
        pmin, pmax = find_valid_region(heatmap.copy(), [1])
        vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] 
        pred_case = inference(predictor_segmask_fine, vol_case, spacing) 
        heatmap[heatmap > 0] = 0
        heatmap[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = pred_case
        
        

        
        vert_labels = sorted(list(np.unique(heatmap)))
        vert_labels.remove(0)
        frac_probs = []
        patch_size = (224, 224)
        slices2D = []
        for vert_num in vert_labels:
            pmin, pmax = find_valid_region(heatmap.copy(), [vert_num])
            vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] 
            target_shape = np.array(list(vol_case.shape[:1]) + list(patch_size))
            vol_case = zoom(vol_case, np.array(target_shape / vol_case.shape), order=1)
            slices2D.append([vol_case, [vert_num] * (pmax[0]-pmin[0])])
        vol_case = [s[0] for s in slices2D]
        C_ID_case = [s[1] for s in slices2D]
        vol_case = np.concatenate(vol_case, axis=0)
        C_ID_case = np.concatenate(C_ID_case, axis=0)
        assert vol_case.shape[0] == C_ID_case.shape[0], "2D slices num and C indexes dismatch!"
        idx = 0
        while idx < vol_case.shape[0]:
            slices2D = vol_case[idx:idx+batch_size_2D]
            slices2D = slices2D[:, None, :, :]
            
            pred_case = inference(predictor_frac_cls, slices2D, spacing)[:, 1]
            frac_probs.extend(list(pred_case))
            idx += batch_size_2D
        frac_probs = np.array(frac_probs)
        C_frac_prob = []
        assert frac_probs.shape == C_ID_case.shape, "2D slices num and C indexes dismatch!"
        for vert_num in vert_labels:
            frac_vert_prob = sorted(frac_probs[C_ID_case == vert_num])
               
            frac_vert = max(1e-7, min(np.mean(frac_vert_prob), 0.9995))
            predictions.append([pid, pid+'_C'+str(vert_num), frac_vert])
            C_frac_prob.append(frac_vert)
            print('fracture prob of vert: ', vert_num, frac_vert)
        patient_overall = patient_overall = max(min(0.99, 1 - np.prod(1 - np.array(C_frac_prob))), 1e-7)
        predictions.append([pid, pid+'_patient_overall', patient_overall])
        
        

    data_res = pd.DataFrame(
        predictions, columns=["StudyInstanceUID", "row_id", "fractured"]
    )
    data_res.to_csv('predictions.csv', encoding='utf8', index=None)
    data_res[['row_id', 'fractured']].to_csv('submission_.csv', encoding='utf8', index=None)
    print(data_res[['row_id', 'fractured']])
    !mv 'submission_.csv' 'submission.csv'

        
        

        
        


if __name__ == '__main__':
    input_path = '../input/rsna-2022-cervical-spine-fracture-detection/test_images'
    output_path = './'
    main(
        input_path=input_path,
        output_path=output_path,
        gpu=0,
    )

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1818804820.py in <cell line: 0>()
      1 try:
----> 2     import cc3d
      3 except:

ModuleNotFoundError: No module named 'cc3d'

During handling of the above exception, another exception occurred:

ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1818804820.py in <cell line: 0>()
      4     get_ipython().system('cp ../input/mirror-cc3d/connected_components_3d-3.10.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl ./')
      5     get_ipython().system('pip install ./connected_components_3d-3.10.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')
----> 6     import cc3d
      7 
      8 def max_connected_region(mask, values):

ModuleNotFoundError: No module named 'cc3d'
