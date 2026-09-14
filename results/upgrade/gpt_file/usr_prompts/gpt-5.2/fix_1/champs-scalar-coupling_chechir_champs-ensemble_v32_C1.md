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
Predict the `scalar_coupling_constant` between atom pairs in molecules, given the two atom types (e.g., C and H), the coupling type (e.g., `2JHC`), and any features you are able to create from the molecule structure (`xyz`) files.

## Metric
Log of the Mean Absolute Error, calculated for each scalar coupling type, and then averaged across types.

## Submission Format
```
id,scalar_coupling_constant
2324604,0.0
2324605,0.0
2324606,0.0
etc.
```

## Dataset
The training and test splits are by *molecule*, so that no molecule in the training data is found in the test data.

- **train.csv** - the training set, where the first column (`molecule_name`) is the name of the molecule where the coupling constant originates (the corresponding XYZ file is located at ./structures/.xyz), the second (`atom_index_0`) and third column (`atom_index_1`) is the atom indices of the atom-pair creating the coupling and the fourth column (`scalar_coupling_constant`) is the scalar coupling constant that we want to be able to predict
- **test.csv** - the test set; same info as train, without the target variable
- **sample_submission.csv** - a sample submission file in the correct format
- **structures.zip** - folder containing molecular structure (xyz) files, where the first line is the number of atoms in the molecule, followed by a blank line, and then a line for every atom, where the first column contains the atomic element (H for hydrogen, C for carbon etc.) and the remaining columns contain the X, Y and Z cartesian coordinates (a standard format for chemists and molecular visualization programs)
- **structures.csv** - this file contains the **same** information as the individual xyz structure files, but in a single file
- **dipole_moments.csv** - contains the molecular electric dipole moments. These are three dimensional vectors that indicate the charge distribution in the molecule. The first column (`molecule_name`) are the names of the molecule, the second to fourth column are the `X`, `Y` and `Z` components respectively of the dipole moment.
- **magnetic_shielding_tensors.csv** - contains the magnetic shielding tensors for all atoms in the molecules. The first column (`molecule_name`) contains the molecule name, the second column (`atom_index`) contains the index of the atom in the molecule, the third to eleventh columns contain the `XX`, `YX`, `ZX`, `XY`, `YY`, `ZY`, `XZ`, `YZ` and `ZZ` elements of the tensor/matrix respectively.
- **mulliken_charges.csv** - contains the mulliken charges for all atoms in the molecules. The first column (`molecule_name`) contains the name of the molecule, the second column (`atom_index`) contains the index of the atom in the molecule, the third column (`mulliken_charge`) contains the mulliken charge of the atom.
- **potential_energy.csv** - contains the potential energy of the molecules. The first column (`molecule_name`) contains the name of the molecule, the second column (`potential_energy`) contains the potential energy of the molecule.
- **scalar_coupling_contributions.csv** - The scalar coupling constants in `train.csv` (or corresponding files) are a sum of four terms. `scalar_coupling_contributions.csv` contain all these terms. The first column (`molecule_name`) are the name of the molecule, the second (`atom_index_0`) and third column (`atom_index_1`) are the atom indices of the atom-pair, the fourth column indicates the type of coupling, the fifth column (`fc`) is the Fermi Contact contribution, the sixth column (`sd`) is the Spin-dipolar contribution, the seventh column (`pso`) is the Paramagnetic spin-orbit contribution and the eighth column (`dso`) is the Diamagnetic spin-orbit contribution.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (111 lines)
            dipole_moments.csv (76511 lines)
            dipole_moments.csv.zip (892.4 kB)
            magnetic_shielding_tensors.csv (1379965 lines)
            magnetic_shielding_tensors.csv.zip (47.9 MB)
            mulliken_charges.csv (1379965 lines)
            mulliken_charges.csv.zip (9.5 MB)
            potential_energy.csv (76511 lines)
            potential_energy.csv.zip (641.9 kB)
            sample_submission.csv (467814 lines)
            sample_submission.csv.zip (846.9 kB)
            scalar_coupling_contributions.csv (4191264 lines)
            scalar_coupling_contributions.csv.zip (90.0 MB)
            structures.csv (1379965 lines)
            structures.csv.zip (33.0 MB)
            structures.zip (44.3 MB)
            test.csv (467814 lines)
            test.csv.zip (2.6 MB)
            train.csv (4191264 lines)
            train.csv.zip (43.6 MB)
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
            structures/
                dsgdb9nsd_000001.xyz (212 Bytes)
                dsgdb9nsd_000002.xyz (171 Bytes)
                ... and 76508 other files
        input/
            description.md (111 lines)
            dipole_moments.csv (76511 lines)
            dipole_moments.csv.zip (892.4 kB)
            magnetic_shielding_tensors.csv (1379965 lines)
            magnetic_shielding_tensors.csv.zip (47.9 MB)
            mulliken_charges.csv (1379965 lines)
            mulliken_charges.csv.zip (9.5 MB)
            potential_energy.csv (76511 lines)
            potential_energy.csv.zip (641.9 kB)
            sample_submission.csv (467814 lines)
            sample_submission.csv.zip (846.9 kB)
            scalar_coupling_contributions.csv (4191264 lines)
            scalar_coupling_contributions.csv.zip (90.0 MB)
            structures.csv (1379965 lines)
            structures.csv.zip (33.0 MB)
            structures.zip (44.3 MB)
            test.csv (467814 lines)
            test.csv.zip (2.6 MB)
            train.csv (4191264 lines)
            train.csv.zip (43.6 MB)
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
            structures/
                dsgdb9nsd_000001.xyz (212 Bytes)
                dsgdb9nsd_000002.xyz (171 Bytes)
                ... and 76508 other files
        working/
            champs-scalar-coupling/
                description.md (111 lines)
                dipole_moments.csv (76511 lines)
                ... and 18 other files
                champs-scalar-coupling/
                structures/
                    dsgdb9nsd_000001.xyz (212 Bytes)
                    dsgdb9nsd_000002.xyz (171 Bytes)
                    ... and 76508 other files
```

-> data/champs-scalar-coupling/dipole_moments.csv has 76510 rows and 4 columns.
The columns are: molecule_name, X, Y, Z

-> data/champs-scalar-coupling/magnetic_shielding_tensors.csv has 1379964 rows and 11 columns.
The columns are: molecule_name, atom_index, XX, YX, ZX, XY, YY, ZY, XZ, YZ, ZZ

-> data/champs-scalar-coupling/mulliken_charges.csv has 1379964 rows and 3 columns.
The columns are: molecule_name, atom_index, mulliken_charge

-> data/champs-scalar-coupling/potential_energy.csv has 76510 rows and 2 columns.
The columns are: molecule_name, potential_energy

-> data/champs-scalar-coupling/sample_submission.csv has 467813 rows and 2 columns.
The columns are: id, scalar_coupling_constant

-> data/champs-scalar-coupling/scalar_coupling_contributions.csv has 4191263 rows and 8 columns.
The columns are: molecule_name, atom_index_0, atom_index_1, type, fc, sd, pso, dso

-> data/champs-scalar-coupling/structures.csv has 1379964 rows and 6 columns.
The columns are: molecule_name, atom_index, atom, x, y, z

-> data/champs-scalar-coupling/test.csv has 467813 rows and 5 columns.
The columns are: id, molecule_name, atom_index_0, atom_index_1, type

-> data/champs-scalar-coupling/train.csv has 4191263 rows and 6 columns.
The columns are: id, molecule_name, atom_index_0, atom_index_1, type, scalar_coupling_constant

-> (stopped after 10 files for performance)

# 5. Target score

-2.4226869287246378

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!ls ../input/1-mpn


## === cell 1
!ls ../input/
!ls ../input/nnet-b-seed-10
!ls ../input/nnet-b-seed-11
!ls ../input/nnet-b-seed-12

## === cell 2
!ls ../input/champ-preds

## === cell 3
import pandas as pd

def get_median_from_files(files):
    print(len(files))
    outs = [pd.read_csv(f, index_col=0) for f in files]
    concat_sub = pd.concat(outs, axis=1, sort=True)
    champ_median = concat_sub.median(axis=1).values
    return champ_median


test = pd.read_csv(f"../input/champs-scalar-coupling/test.csv")
TARGET = 'scalar_coupling_constant'

test['conservative']  =  pd.read_csv('../input/champs-ensemble-conservative/ensemble_sub_conservative.csv')[TARGET]
test['n1'] =  get_median_from_files(
    [
        '../input/champ-preds/gnn_median_2302.csv',
        '../input/champ-preds/gnn_median_2301.csv',
        '../input/champ-preds/gnn_median_adjusted_1JHC_2296.csv', 
    ]
)
gnn_2312 = pd.read_csv('../input/champ-preds/gnn_median_65_68_2312.csv')[TARGET]
lastgnn = pd.read_csv('../input/champ-preds/gnn_median_69_73.csv')[TARGET]  # 2318
test['n1'] = lastgnn * 0.5 + gnn_2312 * 0.3 + test['n1'] *0.2

test['n2'] =  pd.read_csv('../input/champ-preds/gnn0_median_2068.csv')[TARGET]
test['lgb_a'] = get_median_from_files(
    [
        '../input/champ-preds/submission_type_2100.csv',
        '../input/champ-preds/submission_type_2085.csv', 
    ]
)
test['lgb_m'] = get_median_from_files(
    [
        '../input/champ-preds/lgb_type_full_f286_10.csv', # -2.027
        '../input/champ-preds/lgb_type_full_f262_10.csv', # -2.016
    ]
)
test['nnet'] = get_median_from_files(
    [
        '../input/nnpvals-seed-20/nnet_sub_s11.csv',
        '../input/champ-preds/nnet_sub.csv',  # version 6
        '../input/nn-seed-10/nnet_sub.csv',
        '../input/nn-seed-11/nnet_sub.csv',
        '../input/nnet-c-seed-10/lgb_type_cv-1.7126_mae0.23572_fd5_10.csv',
        '../input/nnet-c-seed-11/lgb_type_cv-1.70994_mae0.23497_fd5_11.csv',
        '../input/nnet-c-seed-12/lgb_type_cv-1.71029_mae0.23523_fd5_12.csv',
        '../input/nnet-b-seed-10/lgb_type_cv-1.72296_mae0.24019_bags-1_f120_fd5_10.csv',
        '../input/nnet-b-seed-11/lgb_type_cv-1.70647_mae0.2376_bags-1_f120_fd5_11.csv',
        '../input/nnet-b-seed-12/lgb_type_cv-1.69833_mae0.24106_bags-1_f120_fd5_12.csv',
        '../input/nnet-try-seed-11/lgb_type_cv-1.64944_mae0.23965_bags-1_f120_fd5_11.csv',
    ]    
)
test['nnet_cont'] = get_median_from_files(
    [
        '../input/nncont-seed-23-p/nnetCont_sub_only_predict.csv', 
        '../input/nncont-seed-22/nnetCont_sub-1.8213.csv', # cv
    ]   
)
nn_contof =  pd.read_csv('../input/nncont-seed-26/nnetCont_sub_-2.6677.csv')[TARGET]
test['nnet_cont'] = nn_contof * 0.6 + test['nnet_cont'] * 0.4
test['final_mpnn']  =  pd.read_csv('../input/champ-preds/final_mpnn.csv')[TARGET]
test['mpnn']  =  pd.read_csv('../input/champ-preds/mpnn_5_fold_pseudo_1449.csv')[TARGET]

test['lb'] = pd.read_csv('../input/chemistry-of-best-models-1-895/stack_median.csv')[TARGET]  # -1.895 or a bit more
test.head(10)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1275091109.py in <cell line: 0>()
     12 TARGET = 'scalar_coupling_constant'
     13 
---> 14 test['conservative']  =  pd.read_csv('../input/champs-ensemble-conservative/ensemble_sub_conservative.csv')[TARGET]
     15 test['n1'] =  get_median_from_files(
     16     [

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/champs-ensemble-conservative/ensemble_sub_conservative.csv'

## === cell 4
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline  

test['nnet_ens'] = test['nnet_cont'] * 0.6 + test['nnet'] * 0.4
test['lgb_ens'] = test['lgb_a'] * 0.8 + test['lgb_m'] * 0.2

test['final_preds'] =(
    test['n1']*0.5 +
    test['n2']*0.1 +
    test['lgb_ens']*0.15 +
    test['nnet_ens']*0.1 +
    test['lb']*0.1 +
    test['final_mpnn'] * 0.03 + test['mpnn'] *0.02
)
test.head(20)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
cols = [col for col in test.columns if col not in ['id', 'molecule_name', 'atom_index_0', 'atom_index_1']]
fig = sns.heatmap(test[cols].corr(), cmap='viridis', square=True)
test[cols].corr()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'nnet_cont'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2958980451.py in <cell line: 0>()
      4 get_ipython().run_line_magic('matplotlib', 'inline')
      5 
----> 6 test['nnet_ens'] = test['nnet_cont'] * 0.6 + test['nnet'] * 0.4
      7 test['lgb_ens'] = test['lgb_a'] * 0.8 + test['lgb_m'] * 0.2
      8 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'nnet_cont'

## === cell 5
submission = pd.DataFrame()
submission['id'] = test.id
submission['scalar_coupling_constant'] = test['final_preds'] * 0.6 + test['conservative'] * 0.4
submission.to_csv('ensemble_sub.csv', index=False)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'final_preds'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/912949375.py in <cell line: 0>()
      3 submission['id'] = test.id
      4 # submission['scalar_coupling_constant'] = test['final_preds']
----> 5 submission['scalar_coupling_constant'] = test['final_preds'] * 0.6 + test['conservative'] * 0.4
      6 submission.to_csv('ensemble_sub.csv', index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'final_preds'

## === cell 6
submission.head(20)
