# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the correct ordering of the cells in a given notebook whose markdown cells have been shuffled.

## Metric
Kendall tau correlation between predicted cell orders and ground truth cell orders accumulated across the entire collection of test set notebooks.

Let $S$ be the number of swaps of adjacent entries needed to sort the predicted cell order into the ground truth cell order. In the worst case, a predicted order for a notebook with $n$ cells will need $\frac{1}{2} n(n-1)$ swaps to sort.

We sum the number of swaps from your predicted cell order across the entire collection of test set notebooks, and similarly with the worst-case number of swaps. We then compute the Kendall tau correlation as:

$K=1-4 \frac{\sum_i S_i}{\sum_i n_i\left(n_i-1\right)}$

## Submission Format
For each `id` in the test set (representing a notebook), you must predict `cell_order`, the correct ordering of its cells in terms of the cell ids. The file should contain a header and have the following format:

```
id,cell_order
0009d135ece78d,ddfd239c c6cd22db 1372ae9b ...
0010483c12ba9b,54c7cab3 fe66203e 7844d5f8 ...
0010a919d60e4f,aafc3d23 80e077ec b190ebb4 ...
0028856e09c5b7,012c9d02 d22526d1 3ae7ece3 ...
etc.
```

## Dataset 
- **train/** - A folder comprising about 140,000 JSON files with the filenames corresponding to the `id` field in the `csv` files. Each file contains the code and markdown cells of a notebook. **The code cells are in their original (correct) order. The markdown cells have been shuffled** and placed after the code cells.
- **train_orders.csv** - Gives the correct order of the cells for each notebook in the `train/` folder.
    - `id` - The notebook in file `{id}.json`.
    - `cell_order` - A space delimited list of the correct cell ordering given in terms of the order in `{id}.json`.
- **train_ancestors.csv** - A user may "fork" (that is, copy) the notebook of another user to create their own version. This file contains the forking history of notebooks in the training set. **Note: There is no corresponding file for the test set.**
    - `ancestor_id` - Identifies sets of notebooks that have a common origin or "ancestor". As no notebook in the test set has an ancestor in the training set, you may find this field to be of use as a grouping factor when constructing validation splits.
    - `parent_id` - Indicates that some version of the notebook `id` was forked from some version of the notebook `parent_id`. The notebook `parent_id` may or may not be present in the training data. (The parent may be missing because someone had forked a private notebook of their own, for instance.)
- **test/** - Notebooks from the test set. 
- **sample_submission.csv** - A sample submission file in the correct format.

# 2. Python version

3.11

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (20001 lines)
            sample_submission.csv.zip (4.8 MB)
            test.zip (104.5 MB)
            train.zip (603.7 MB)
            train_ancestors.csv (119257 lines)
            train_ancestors.csv.zip (1.8 MB)
            train_orders.csv (119257 lines)
            train_orders.csv.zip (29.4 MB)
            AI4Code/
                description.md (122 lines)
                sample_submission.csv (20001 lines)
                ... and 7 other files
                AI4Code/
                test/
                    00015c83e2717b.json (1 lines)
                    0001bdd4021779.json (1 lines)
                    ... and 19998 other files
                    test/
                train/
                    00001756c60be8.json (1 lines)
                    0001daf4c2c76d.json (1 lines)
                    ... and 119254 other files
                    train/
            test/
                00015c83e2717b.json (1 lines)
                0001bdd4021779.json (1 lines)
                ... and 19998 other files
                test/
            train/
                00001756c60be8.json (1 lines)
                0001daf4c2c76d.json (1 lines)
                ... and 119254 other files
                train/
        input/
            description.md (122 lines)
            sample_submission.csv (20001 lines)
            sample_submission.csv.zip (4.8 MB)
            test.zip (104.5 MB)
            train.zip (603.7 MB)
            train_ancestors.csv (119257 lines)
            train_ancestors.csv.zip (1.8 MB)
            train_orders.csv (119257 lines)
            train_orders.csv.zip (29.4 MB)
            AI4Code/
                description.md (122 lines)
                sample_submission.csv (20001 lines)
                ... and 7 other files
                AI4Code/
                test/
                    00015c83e2717b.json (1 lines)
                    0001bdd4021779.json (1 lines)
                    ... and 19998 other files
                    test/
                train/
                    00001756c60be8.json (1 lines)
                    0001daf4c2c76d.json (1 lines)
                    ... and 119254 other files
                    train/
            test/
                00015c83e2717b.json (1 lines)
                0001bdd4021779.json (1 lines)
                ... and 19998 other files
                test/
                    00015c83e2717b.json (1 lines)
                    0001bdd4021779.json (1 lines)
                    ... and 19998 other files
                    test/
            train/
                00001756c60be8.json (1 lines)
                0001daf4c2c76d.json (1 lines)
                ... and 119254 other files
                train/
                    00001756c60be8.json (1 lines)
                    0001daf4c2c76d.json (1 lines)
                    ... and 119254 other files
                    train/
        working/
            AI4Code/
                description.md (122 lines)
                sample_submission.csv (20001 lines)
                ... and 7 other files
                AI4Code/
                test/
                    00015c83e2717b.json (1 lines)
                    0001bdd4021779.json (1 lines)
                    ... and 19998 other files
                    test/
                train/
                    00001756c60be8.json (1 lines)
                    0001daf4c2c76d.json (1 lines)
                    ... and 119254 other files
                    train/
```

-> data/AI4Code/sample_submission.csv has 20000 rows and 2 columns.
The columns are: id, cell_order

-> data/AI4Code/test/00015c83e2717b.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "cell_type": {
      "type": "object",
      "properties": {
        "c417225b": {
          "type": "string"
        },
        "51e3cd89": {
          "type": "string"
        },
        "2600b4eb": {
          "type": "string"
        },
        "75b65993": {
          "type": "string"
        },
        "cf195f8b": {
          "type": "string"
        },
        "25699d02": {
          "type": "string"
        },
        "de148b56": {
          "type": "string"
        },
        "9901472c": {
          "type": "string"
        },
        "10377ef8": {
          "type": "string"
        },
        "1f462e2f": {
          "type": "string"
        },
        "fceeb3e6": {
          "type": "string"
        },
        "2af2a41a": {
          "type": "string"
        },
        "91e68f13": {
          "type": "string"
        },
        "9216a113": {
          "type": "string"
        },
        "63753f10": {
          "type": "string"
        },
        "d5aee1e4": {
          "type": "string"
        },
        "dc8a39a5": {
          "type": "string"
        },
        "3d0a28c2": {
          "type": "string"
        },
        "0eea9701": {
          "type": "string"
        },
        "a6f8f9f1": {
          "type": "string"
        },
        "7223cfc2": {
          "type": "string"
        },
        "df6b3ccb": {
          "type": "string"
        },
        "a9b266d2": {
          "type": "string"
        },
        "3e17f424": {
          "type": "string"
        },
        "42f0c365": {
          "type": "string"
        },
        "cc8d23d8": {
          "type": "string"
        },
        "ad42abc1": {
          "type": "string"
        },
        "7894c4e8": {
          "type": "string"
        },
        "cc1add42": {
          "type": "string"
        },
        "16b0d436": {
          "type": "string"
        },
        "a3e791de": {
          "type": "string"
        },
        "02ef0932": {
          "type": "string"
        },
        "03441163": {
          "type": "string"
        },
        "d429a743": {
          "type": "string"
        },
        "4d5ebd46": {
          "type": "string"
        },
        "3adcd0f8": {
          "type": "string"
        },
        "d8f4dfe0": {
          "type": "string"
        },
        "d47d3338": {
          "type": "string"
        },
        "c5b89474": {
          "type": "string"
        },
        "b5ef409a": {
          "type": "string"
        },
        "7bb6803b": {
          "type": "string"
        },
        "36b95373": {
          "type": "string"
        },
        "31d43a34": {
          "type": "string"
        },
        "cba54be3": {
          "type": "string"
        },
        "c33ab270": {
          "type": "string"
        },
        "23cb507d": {
          "type": "string"
        },
        "017f1c1b": {
          "type": "string"
        },
        "e458b502": {
          "type": "string"
        },
        "10fc0035": {
          "type": "string"
        },
        "61f2723c": {
          "type": "string"
        },
        "06b8472a": {
          "type": "string"
        },
        "704a44aa": {
          "type": "string"
        },
        "c7bb2674": {
          "type": "string"
        },
        "da1357af": {
          "type": "string"
        },
        "7a416873": {
          "type": "string"
        },
        "1554d3bc": {
          "type": "string"
        },
        "512a821e": {
          "type": "string"
        },
        "bf2740de": {
          "type": "string"
        },
        "fa1ee016": {
          "type": "string"
        },
        "ce96953f": {
          "type": "string"
        },
        "96c19678": {
          "type": "string"
        },
        "275d2fa7": {
          "type": "string"
        },
        "4de05ac0": {
          "type": "string"
        },
        "9aadfa3f": {
          "type": "string"
        },
        "60840c05": {
          "type": "string"
        },
        "a829d740": {
          "type": "string"
        },
        "14feb55f": {
          "type": "string"
        },
        "b8f3850a": {
          "type": "string"
        },
        "ccbe6713": {
          "type": "string"
        },
        "1ecd4e35": {
          "type": "string"
        },
        "1e21e4e5": {
          "type": "string"
        },
        "3db2d50e": {
          "type": "string"
        },
        "f2c750d3": {
          "type": "string"
        },
        "cd61f6d1": {
          "type": "string"
        },
        "72b3201a": {
          "type": "string"
        },
        "924c9f0f": {
          "type": "string"
        },
        "da8b817a": {
          "type": "string"
        },
        "8542d6fc": {
          "type": "string"
        },
        "fbe3e811": {
          "type": "string"
        },
        "657a8804": {
          "type": "string"
        },
        "a8fcc3e3": {
          "type": "string"
        },
        "183226e4": {
          "type": "string"
        },
        "be6c9079": {
          "type": "string"
        },
        "41beeead": {
          "type": "string"
        },
        "eab2b130": {
          "type": "string"
        },
        "b5e286ea": {
          "type": "string"
        },
        "2e94bd7a": {
          "type": "string"
        },
        "a166703b": {
          "type": "string"
        },
        "ceba8ae0": {
          "type": "string"
        },
        "f2915b9f": {
          "type": "string"
        },
        "3e99dee9": {
          "type": "string"
        },
        "da4f7550": {
          "type": "string"
        },
        "42749e24": {
          "type": "string"
        }
      },
      "required": [
        "017f1c1b",
        "02ef0932",
        "03441163",
        "06b8472a",
        "0eea9701",
        "10377ef8",
        "10fc0035",
        "14feb55f",
        "1554d3bc",
        "16b0d436",
        "183226e4",
        "1e21e4e5",
        "1ecd4e35",
        "1f462e2f",
        "23cb507d",
        "25699d02",
        "2600b4eb",
        "275d2fa7",
        "2af2a41a",
        "2e94bd7a",
        "31d43a34",
        "36b95373",
        "3adcd0f8",
        "3d0a28c2",
        "3db2d50e",
        "3e17f424",
        "3e99dee9",
        "41beeead",
        "42749e24",
        "42f0c365",
        "4d5ebd46",
        "4de05ac0",
        "512a821e",
        "51e3cd89",
        "60840c05",
        "61f2723c",
        "63753f10",
        "657a8804",
        "704a44aa",
        "7223cfc2",
        "72b3201a",
        "75b65993",
        "7894c4e8",
        "7a416873",
        "7bb6803b",
        "8542d6fc",
        "91e68f13",
        "9216a113",
        "924c9f0f",
        "96c19678",
        "9901472c",
        "9aadfa3f",
        "a166703b",
        "a3e791de",
        "a6f8f9f1",
        "a829d740",
        "a8fcc3e3",
        "a9b266d2",
        "ad42abc1",
        "b5e286ea",
        "b5ef409a",
        "b8f3850a",
        "be6c9079",
        "bf2740de",
        "c33ab270",
        "c417225b",
        "c5b89474",
        "c7bb2674",
        "cba54be3",
        "cc1add42",
        "cc8d23d8",
        "ccbe6713",
        "cd61f6d1",
        "ce96953f",
        "ceba8ae0",
        "cf195f8b",
        "d429a743",
        "d47d3338",
        "d5aee1e4",
        "d8f4dfe0",
        "da1357af",
        "da4f7550",
        "da8b817a",
        "dc8a39a5",
        "de148b56",
        "df6b3ccb",
        "e458b502",
        "eab2b130",
        "f2915b9f",
        "f2c750d3",
        "fa1ee016",
        "fbe3e811",
        "fceeb3e6"
      ]
    },
    "source": {
      "type": "object",
      "properties": {
        "c417225b": {
          "type": "string"
        },
        "51e3cd89": {
          "type": "string"
        },
        "2600b4eb": {
          "type": "string"
        },
        "75b65993": {
          "type": "string"
        },
        "cf195f8b": {
          "type": "string"
        },
        "25699d02": {
          "type": "string"
        },
        "de148b56": {
          "type": "string"
        },
        "9901472c": {
          "type": "string"
        },
        "10377ef8": {
          "type": "string"
        },
        "1f462e2f": {
          "type": "string"
        },
        "fceeb3e6": {
          "type": "string"
        },
        "2af2a41a": {
          "type": "string"
        },
        "91e68f13": {
          "type": "string"
        },
        "9216a113": {
          "type": "string"
        },
        "63753f10": {
          "type": "string"
        },
        "d5aee1e4": {
          "type": "string"
        },
        "dc8a39a5": {
          "type": "string"
        },
        "3d0a28c2": {
          "type": "string"
        },
        "0eea9701": {
          "type": "string"
        },
        "a6f8f9f1": {
          "type": "string"
        },
        "7223cfc2": {
          "type": "string"
        },
        "df6b3ccb": {
          "type": "string"
        },
        "a9b266d2": {
          "type": "string"
        },
        "3e17f424": {
          "type": "string"
        },
        "42f0c365": {
          "type": "string"
        },
        "cc8d23d8": {
          "type": "string"
        },
        "ad42abc1": {
          "type": "string"
        },
        "7894c4e8": {
          "type": "string"
        },
        "cc1add42": {
          "type": "string"
        },
        "16b0d436": {
          "type": "string"
        },
        "a3e791de": {
          "type": "string"
        },
        "02ef0932": {
          "type": "string"
        },
        "03441163": {
          "type": "string"
        },
        "d429a743": {
          "type": "string"
        },
        "4d5ebd46": {
          "type": "string"
        },
        "3adcd0f8": {
          "type": "string"
        },
        "d8f4dfe0": {
          "type": "string"
        },
        "d47d3338": {
          "type": "string"
        },
        "c5b89474": {
          "type": "string"
        },
        "b5ef409a": {
          "type": "string"
        },
        "7bb6803b": {
          "type": "string"
        },
        "36b95373": {
          "type": "string"
        },
        "31d43a34": {
          "type": "string"
        },
        "cba54be3": {
          "type": "string"
        },
        "c33ab270": {
          "type": "string"
        },
        "23cb507d": {
          "type": "string"
        },
        "017f1c1b": {
          "type": "string"
        },
        "e458b502": {
          "type": "string"
        },
        "10fc0035": {
          "type": "string"
        },
        "61f2723c": {
          "type": "string"
        },
        "06b8472a": {
          "type": "string"
        },
        "704a44aa": {
          "type": "string"
        },
        "c7bb2674": {
          "type": "string"
        },
        "da1357af": {
          "type": "string"
        },
        "7a416873": {
          "type": "string"
        },
        "1554d3bc": {
          "type": "string"
        },
        "512a821e": {
          "type": "string"
        },
        "bf2740de": {
          "type": "string"
        },
        "fa1ee016": {
          "type": "string"
        },
        "ce96953f": {
          "type": "string"
        },
        "96c19678": {
          "type": "string"
        },
        "275d2fa7": {
          "type": "string"
        },
        "4de05ac0": {
          "type": "string"
        },
        "9aadfa3f": {
          "type": "string"
        },
        "60840c05": {
          "type": "string"
        },
        "a829d740": {
          "type": "string"
        },
        "14feb55f": {
          "type": "string"
        },
        "b8f3850a": {
          "type": "string"
        },
        "ccbe6713": {
          "type": "string"
        },
        "1ecd4e35": {
          "type": "string"
        },
        "1e21e4e5": {
          "type": "string"
        },
        "3db2d50e": {
          "type": "string"
        },
        "f2c750d3": {
          "type": "string"
        },
        "cd61f6d1": {
          "type": "string"
        },
        "72b3201a": {
          "type": "string"
        },
        "924c9f0f": {
          "type": "string"
        },
        "da8b817a": {
          "type": "string"
        },
        "8542d6fc": {
          "type": "string"
        },
        "fbe3e811": {
          "type": "string"
        },
        "657a8804": {
          "type": "string"
        },
        "a8fcc3e3": {
          "type": "string"
        },
        "183226e4": {
          "type": "string"
        },
        "be6c9079": {
          "type": "string"
        },
        "41beeead": {
          "type": "string"
        },
        "eab2b130": {
          "type": "string"
        },
        "b5e286ea": {
          "type": "string"
        },
        "2e94bd7a": {
          "type": "string"
        },
        "a166703b": {
          "type": "string"
        },
        "ceba8ae0": {
          "type": "string"
        },
        "f2915b9f": {
          "type": "string"
        },
        "3e99dee9": {
          "type": "string"
        },
        "da4f7550": {
          "type": "string"
        },
        "42749e24": {
          "type": "string"
        }
      },
      "required": [
        "017f1c1b",
        "02ef0932",
        "03441163",
        "06b8472a",
        "0eea9701",
        "10377ef8",
        "10fc0035",
        "14feb55f",
        "1554d3bc",
        "16b0d436",
        "183226e4",
        "1e21e4e5",
        "1ecd4e35",
        "1f462e2f",
        "23cb507d",
        "25699d02",
        "2600b4eb",
        "275d2fa7",
        "2af2a41a",
        "2e94bd7a",
        "31d43a34",
        "36b95373",
        "3adcd0f8",
        "3d0a28c2",
        "3db2d50e",
        "3e17f424",
        "3e99dee9",
        "41beeead",
        "42749e24",
        "42f0c365",
        "4d5ebd46",
        "4de05ac0",
        "512a821e",
        "51e3cd89",
        "60840c05",
        "61f2723c",
        "63753f10",
        "657a8804",
        "704a44aa",
        "7223cfc2",
        "72b3201a",
        "75b65993",
        "7894c4e8",
        "7a416873",
        "7bb6803b",
        "8542d6fc",
        "91e68f13",
        "9216a113",
        "924c9f0f",
        "96c19678",
        "9901472c",
        "9aadfa3f",
        "a166703b",
        "a3e791de",
        "a6f8f9f1",
        "a829d740",
        "a8fcc3e3",
        "a9b266d2",
        "ad42abc1",
        "b5e286ea",
        "b5ef409a",
        "b8f3850a",
        "be6c9079",
        "bf2740de",
        "c33ab270",
        "c417225b",
        "c5b89474",
        "c7bb2674",
        "cba54be3",
        "cc1add42",
        "cc8d23d8",
        "ccbe6713",
        "cd61f6d1",
        "ce96953f",
        "ceba8ae0",
        "cf195f8b",
        "d429a743",
        "d47d3338",
        "d5aee1e4",
        "d8f4dfe0",
        "da1357af",
        "da4f7550",
        "da8b817a",
        "dc8a39a5",
        "de148b56",
        "df6b3ccb",
        "e458b502",
        "eab2b130",
        "f2915b9f",
        "f2c750d3",
        "fa1ee016",
        "fbe3e811",
        "fceeb3e6"
      ]
    }
  },
  "required": [
    "cell_type",
    "source"
  ]
}

-> data/AI4Code/test/0001bdd4021779.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "cell_type": {
      "type": "object",
      "properties": {
        "3fdc37be": {
          "type": "string"
        },
        "073782ca": {
          "type": "string"
        },
        "8ea7263c": {
          "type": "string"
        },
        "80543cd8": {
          "type": "string"
        },
        "38310c80": {
          "type": "string"
        },
        "073e27e5": {
          "type": "string"
        },
        "015d52a4": {
          "type": "string"
        },
        "ad7679ef": {
          "type": "string"
        },
        "07c52510": {
          "type": "string"
        },
        "0a1a7a39": {
          "type": "string"
        },
        "0bcd3fef": {
          "type": "string"
        },
        "7fde4f04": {
          "type": "string"
        },
        "58bf360b": {
          "type": "string"
        }
      },
      "required": [
        "015d52a4",
        "073782ca",
        "073e27e5",
        "07c52510",
        "0a1a7a39",
        "0bcd3fef",
        "38310c80",
        "3fdc37be",
        "58bf360b",
        "7fde4f04",
        "80543cd8",
        "8ea7263c",
        "ad7679ef"
      ]
    },
    "source": {
      "type": "object",
      "properties": {
        "3fdc37be": {
          "type": "string"
        },
        "073782ca": {
          "type": "string"
        },
        "8ea7263c": {
          "type": "string"
        },
        "80543cd8": {
          "type": "string"
        },
        "38310c80": {
          "type": "string"
        },
        "073e27e5": {
          "type": "string"
        },
        "015d52a4": {
          "type": "string"
        },
        "ad7679ef": {
          "type": "string"
        },
        "07c52510": {
          "type": "string"
        },
        "0a1a7a39": {
          "type": "string"
        },
        "0bcd3fef": {
          "type": "string"
        },
        "7fde4f04": {
          "type": "string"
        },
        "58bf360b": {
          "type": "string"
        }
      },
      "required": [
        "015d52a4",
        "073782ca",
        "073e27e5",
        "07c52510",
        "0a1a7a39",
        "0bcd3fef",
        "38310c80",
        "3fdc37be",
        "58bf360b",
        "7fde4f04",
        "80543cd8",
        "8ea7263c",
        "ad7679ef"
      ]
    }
  },
  "required": [
    "cell_type",
    "source"
  ]
}

-> data/AI4Code/test/000757b90aaca0.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "cell_type": {
      "type": "object",
      "properties": {
        "8f84d7a9": {
          "type": "string"
        },
        "eb6ca769": {
          "type": "string"
        },
        "bc595bc2": {
          "type": "string"
        },
        "93cceeef": {
          "type": "string"
        },
        "3cb3d383": {
          "type": "string"
        },
        "6e3a3d90": {
          "type": "string"
        },
        "abc159f0": {
          "type": "string"
        },
        "b20690ef": {
          "type": "string"
        },
        "20f10a90": {
          "type": "string"
        },
        "e301d5a4": {
          "type": "string"
        },
        "7905811c": {
          "type": "string"
        },
        "1fa4803c": {
          "type": "string"
        },
        "dcb9b899": {
          "type": "string"
        },
        "ed7ca83b": {
          "type": "string"
        },
        "afc25d5c": {
          "type": "string"
        },
        "0781d626": {
          "type": "string"
        },
        "e895145f": {
          "type": "string"
        },
        "4f3af9d2": {
          "type": "string"
        },
        "c9131ba9": {
          "type": "string"
        },
        "afc62c5a": {
          "type": "string"
        },
        "5b5af988": {
          "type": "string"
        },
        "18cb2ee7": {
          "type": "string"
        },
        "a32974f3": {
          "type": "string"
        },
        "4e2b2854": {
          "type": "string"
        },
        "a2e1ed42": {
          "type": "string"
        },
        "454e8858": {
          "type": "string"
        },
        "1b8c0237": {
          "type": "string"
        },
        "744648dd": {
          "type": "string"
        },
        "0564962b": {
          "type": "string"
        },
        "336fdc76": {
          "type": "string"
        },
        "22686c49": {
          "type": "string"
        },
        "1c301fa2": {
          "type": "string"
        },
        "53c3a2be": {
          "type": "string"
        },
        "6253a226": {
          "type": "string"
        },
        "335d6d82": {
          "type": "string"
        },
        "e0f60ece": {
          "type": "string"
        },
        "72821d9a": {
          "type": "string"
        },
        "771dbec1": {
          "type": "string"
        },
        "eeab7090": {
          "type": "string"
        },
        "6243c12c": {
          "type": "string"
        },
        "d3db4f3e": {
          "type": "string"
        },
        "cecacc55": {
          "type": "string"
        }
      },
      "required": [
        "0564962b",
        "0781d626",
        "18cb2ee7",
        "1b8c0237",
        "1c301fa2",
        "1fa4803c",
        "20f10a90",
        "22686c49",
        "335d6d82",
        "336fdc76",
        "3cb3d383",
        "454e8858",
        "4e2b2854",
        "4f3af9d2",
        "53c3a2be",
        "5b5af988",
        "6243c12c",
        "6253a226",
        "6e3a3d90",
        "72821d9a",
        "744648dd",
        "771dbec1",
        "7905811c",
        "8f84d7a9",
        "93cceeef",
        "a2e1ed42",
        "a32974f3",
        "abc159f0",
        "afc25d5c",
        "afc62c5a",
        "b20690ef",
        "bc595bc2",
        "c9131ba9",
        "cecacc55",
        "d3db4f3e",
        "dcb9b899",
        "e0f60ece",
        "e301d5a4",
        "e895145f",
        "eb6ca769",
        "ed7ca83b",
        "eeab7090"
      ]
    },
    "source": {
      "type": "object",
      "properties": {
        "8f84d7a9": {
          "type": "string"
        },
        "eb6ca769": {
          "type": "string"
        },
        "bc595bc2": {
          "type": "string"
        },
        "93cceeef": {
          "type": "string"
        },
        "3cb3d383": {
          "type": "string"
        },
        "6e3a3d90": {
          "type": "string"
        },
        "abc159f0": {
          "type": "string"
        },
        "b20690ef": {
          "type": "string"
        },
        "20f10a90": {
          "type": "string"
        },
        "e301d5a4": {
          "type": "string"
        },
        "7905811c": {
          "type": "string"
        },
        "1fa4803c": {
          "type": "string"
        },
        "dcb9b899": {
          "type": "string"
        },
        "ed7ca83b": {
          "type": "string"
        },
        "afc25d5c": {
          "type": "string"
        },
        "0781d626": {
          "type": "string"
        },
        "e895145f": {
          "type": "string"
        },
        "4f3af9d2": {
          "type": "string"
        },
        "c9131ba9": {
          "type": "string"
        },
        "afc62c5a": {
          "type": "string"
        },
        "5b5af988": {
          "type": "string"
        },
        "18cb2ee7": {
          "type": "string"
        },
        "a32974f3": {
          "type": "string"
        },
        "4e2b2854": {
          "type": "string"
        },
        "a2e1ed42": {
          "type": "string"
        },
        "454e8858": {
          "type": "string"
        },
        "1b8c0237": {
          "type": "string"
        },
        "744648dd": {
          "type": "string"
        },
        "0564962b": {
          "type": "string"
        },
        "336fdc76": {
          "type": "string"
        },
        "22686c49": {
          "type": "string"
        },
        "1c301fa2": {
          "type": "string"
        },
        "53c3a2be": {
          "type": "string"
        },
        "6253a226": {
          "type": "string"
        },
        "335d6d82": {
          "type": "string"
        },
        "e0f60ece": {
          "type": "string"
        },
        "72821d9a": {
          "type": "string"
        },
        "771dbec1": {
          "type": "string"
        },
        "eeab7090": {
          "type": "string"
        },
        "6243c12c": {
          "type": "string"
        },
        "d3db4f3e": {
          "type": "string"
        },
        "cecacc55": {
          "type": "string"
        }
      },
      "required": [
        "0564962b",
        "0781d626",
        "18cb2ee7",
        "1b8c0237",
        "1c301fa2",
        "1fa4803c",
        "20f10a90",
        "22686c49",
        "335d6d82",
        "336fdc76",
        "3cb3d383",
        "454e8858",
        "4e2b2854",
        "4f3af9d2",
        "53c3a2be",
        "5b5af988",
        "6243c12c",
        "6253a226",
        "6e3a3d90",
        "72821d9a",
        "744648dd",
        "771dbec1",
        "7905811c",
        "8f84d7a9",
        "93cceeef",
        "a2e1ed42",
        "a32974f3",
        "abc159f0",
        "afc25d5c",
        "afc62c5a",
        "b20690ef",
        "bc595bc2",
        "c9131ba9",
        "cecacc55",
        "d3db4f3e",
        "dcb9b899",
        "e0f60ece",
        "e301d5a4",
        "e895145f",
        "eb6ca769",
        "ed7ca83b",
        "eeab7090"
      ]
    }
  },
  "required": [
    "cell_type",
    "source"
  ]
}

-> data/AI4Code/test/000a2f5243e1ca.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "cell_type": {
      "type": "object",
      "properties": {
        "1d968d84": {
          "type": "string"
        },
        "5774aca9": {
          "type": "string"
        },
        "2ddb979d": {
          "type": "string"
        },
        "dc18005d": {
          "type": "string"
        },
        "3c7d2db1": {
          "type": "string"
        },
        "2b434221": {
          "type": "string"
        },
        "573fbd25": {
          "type": "string"
        },
        "18e0b577": {
          "type": "string"
        },
        "b0e8de50": {
          "type": "string"
        },
        "adf7c730": {
          "type": "string"
        },
        "272894fc": {
          "type": "string"
        },
        "61c25d88": {
          "type": "string"
        },
        "cd9212a6": {
          "type": "string"
        },
        "ecbc1cd5": {
          "type": "string"
        },
        "8259cb78": {
          "type": "string"
        },
        "3b716e1d": {
          "type": "string"
        },
        "75dc5015": {
          "type": "string"
        },
        "62c1199d": {
          "type": "string"
        },
        "7b965d59": {
          "type": "string"
        },
        "5f97f35f": {
          "type": "string"
        },
        "144b402a": {
          "type": "string"
        },
        "19d316e8": {
          "type": "string"
        },
        "f6709ac2": {
          "type": "string"
        },
        "848507bb": {
          "type": "string"
        },
        "08eb5eee": {
          "type": "string"
        },
        "6cb1cfb9": {
          "type": "string"
        },
        "0a304ea5": {
          "type": "string"
        },
        "e58e01a6": {
          "type": "string"
        },
        "f5542fa5": {
          "type": "string"
        },
        "40b4f7e8": {
          "type": "string"
        },
        "8769b4a6": {
          "type": "string"
        },
        "a9cff15b": {
          "type": "string"
        },
        "55c9c2af": {
          "type": "string"
        },
        "71486249": {
          "type": "string"
        },
        "779c929f": {
          "type": "string"
        },
        "d7e8668f": {
          "type": "string"
        },
        "13309042": {
          "type": "string"
        },
        "c4de71b5": {
          "type": "string"
        },
        "31080d42": {
          "type": "string"
        },
        "51a44f3a": {
          "type": "string"
        },
        "4883f94d": {
          "type": "string"
        },
        "b46ca469": {
          "type": "string"
        },
        "39ceb8e0": {
          "type": "string"
        },
        "ea468337": {
          "type": "string"
        },
        "f7a66491": {
          "type": "string"
        }
      },
      "required": [
        "08eb5eee",
        "0a304ea5",
        "13309042",
        "144b402a",
        "18e0b577",
        "19d316e8",
        "1d968d84",
        "272894fc",
        "2b434221",
        "2ddb979d",
        "31080d42",
        "39ceb8e0",
        "3b716e1d",
        "3c7d2db1",
        "40b4f7e8",
        "4883f94d",
        "51a44f3a",
        "55c9c2af",
        "573fbd25",
        "5774aca9",
        "5f97f35f",
        "61c25d88",
        "62c1199d",
        "6cb1cfb9",
        "71486249",
        "75dc5015",
        "779c929f",
        "7b965d59",
        "8259cb78",
        "848507bb",
        "8769b4a6",
        "a9cff15b",
        "adf7c730",
        "b0e8de50",
        "b46ca469",
        "c4de71b5",
        "cd9212a6",
        "d7e8668f",
        "dc18005d",
        "e58e01a6",
        "ea468337",
        "ecbc1cd5",
        "f5542fa5",
        "f6709ac2",
        "f7a66491"
      ]
    },
    "source": {
      "type": "object",
      "properties": {
        "1d968d84": {
          "type": "string"
        },
        "5774aca9": {
          "type": "string"
        },
        "2ddb979d": {
          "type": "string"
        },
        "dc18005d": {
          "type": "string"
        },
        "3c7d2db1": {
          "type": "string"
        },
        "2b434221": {
          "type": "string"
        },
        "573fbd25": {
          "type": "string"
        },
        "18e0b577": {
          "type": "string"
        },
        "b0e8de50": {
          "type": "string"
        },
        "adf7c730": {
          "type": "string"
        },
        "272894fc": {
          "type": "string"
        },
        "61c25d88": {
          "type": "string"
        },
        "cd9212a6": {
          "type": "string"
        },
        "ecbc1cd5": {
          "type": "string"
        },
        "8259cb78": {
          "type": "string"
        },
        "3b716e1d": {
          "type": "string"
        },
        "75dc5015": {
          "type": "string"
        },
        "62c1199d": {
          "type": "string"
        },
        "7b965d59": {
          "type": "string"
        },
        "5f97f35f": {
          "type": "string"
        },
        "144b402a": {
          "type": "string"
        },
        "19d316e8": {
          "type": "string"
        },
        "f6709ac2": {
          "type": "string"
        },
        "848507bb": {
          "type": "string"
        },
        "08eb5eee": {
          "type": "string"
        },
        "6cb1cfb9": {
          "type": "string"
        },
        "0a304ea5": {
          "type": "string"
        },
        "e58e01a6": {
          "type": "string"
        },
        "f5542fa5": {
          "type": "string"
        },
        "40b4f7e8": {
          "type": "string"
        },
        "8769b4a6": {
          "type": "string"
        },
        "a9cff15b": {
          "type": "string"
        },
        "55c9c2af": {
          "type": "string"
        },
        "71486249": {
          "type": "string"
        },
        "779c929f": {
          "type": "string"
        },
        "d7e8668f": {
          "type": "string"
        },
        "13309042": {
          "type": "string"
        },
        "c4de71b5": {
          "type": "string"
        },
        "31080d42": {
          "type": "string"
        },
        "51a44f3a": {
          "type": "string"
        },
        "4883f94d": {
          "type": "string"
        },
        "b46ca469": {
          "type": "string"
        },
        "39ceb8e0": {
          "type": "string"
        },
        "ea468337": {
          "type": "string"
        },
        "f7a66491": {
          "type": "string"
        }
      },
      "required": [
        "08eb5eee",
        "0a304ea5",
        "13309042",
        "144b402a",
        "18e0b577",
        "19d316e8",
        "1d968d84",
        "272894fc",
        "2b434221",
        "2ddb979d",
        "31080d42",
        "39ceb8e0",
        "3b716e1d",
        "3c7d2db1",
        "40b4f7e8",
        "4883f94d",
        "51a44f3a",
        "55c9c2af",
        "573fbd25",
        "5774aca9",
        "5f97f35f",
        "61c25d88",
        "62c1199d",
        "6cb1cfb9",
        "71486249",
        "75dc5015",
        "779c929f",
        "7b965d59",
        "8259cb78",
        "848507bb",
        "8769b4a6",
        "a9cff15b",
        "adf7c730",
        "b0e8de50",
        "b46ca469",
        "c4de71b5",
        "cd9212a6",
        "d7e8668f",
        "dc18005d",
        "e58e01a6",
        "ea468337",
        "ecbc1cd5",
        "f5542fa5",
        "f6709ac2",
        "f7a66491"
      ]
    }
  },
  "required": [
    "cell_type",
    "source"
  ]
}

-> data/AI4Code/test/000c1e0e45bb25.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "cell_type": {
      "type": "object",
      "properties": {
        "3255fff2": {
          "type": "string"
        },
        "d6c42868": {
          "type": "string"
        },
        "dd65465b": {
          "type": "string"
        },
        "07c5250a": {
          "type": "string"
        },
        "31aafb08": {
          "type": "string"
        },
        "82bfd4d2": {
          "type": "string"
        },
        "763c6ec3": {
          "type": "string"
        },
        "11ee491c": {
          "type": "string"
        },
        "0a148514": {
          "type": "string"
        },
        "417a3d15": {
          "type": "string"
        },
        "ec509f3c": {
          "type": "string"
        },
        "ba95a1da": {
          "type": "string"
        },
        "58052fe0": {
          "type": "string"
        },
        "c318c15e": {
          "type": "string"
        },
        "b986e993": {
          "type": "string"
        },
        "484f545c": {
          "type": "string"
        },
        "df6c360d": {
          "type": "string"
        },
        "e798ce33": {
          "type": "string"
        },
        "3cb1ce25": {
          "type": "string"
        },
        "ccb33ced": {
          "type": "string"
        },
        "5b50afbe": {
          "type": "string"
        },
        "c307c967": {
          "type": "string"
        },
        "bbbf7553": {
          "type": "string"
        },
        "a2473d70": {
          "type": "string"
        },
        "9d7a122d": {
          "type": "string"
        },
        "e00daaae": {
          "type": "string"
        },
        "64859a58": {
          "type": "string"
        },
        "e84c6098": {
          "type": "string"
        },
        "09185ff0": {
          "type": "string"
        },
        "b14d592a": {
          "type": "string"
        },
        "f8b0aba8": {
          "type": "string"
        },
        "fdb67c74": {
          "type": "string"
        },
        "e2d14d2d": {
          "type": "string"
        },
        "e7513bb8": {
          "type": "string"
        },
        "28c8ae3c": {
          "type": "string"
        },
        "77e86b84": {
          "type": "string"
        },
        "2672b144": {
          "type": "string"
        },
        "d0e0a582": {
          "type": "string"
        },
        "22155376": {
          "type": "string"
        },
        "760c3153": {
          "type": "string"
        },
        "a5b2432a": {
          "type": "string"
        },
        "48cd2325": {
          "type": "string"
        },
        "656b0496": {
          "type": "string"
        },
        "18e4dbe1": {
          "type": "string"
        },
        "c0effaeb": {
          "type": "string"
        },
        "571513f9": {
          "type": "string"
        },
        "0545edbb": {
          "type": "string"
        },
        "7844bd4f": {
          "type": "string"
        },
        "490a04d5": {
          "type": "string"
        },
        "859a2079": {
          "type": "string"
        },
        "0214ecb9": {
          "type": "string"
        },
        "00f5f344": {
          "type": "string"
        },
        "f17a1150": {
          "type": "string"
        },
        "e5b1a66f": {
          "type": "string"
        },
        "73134d5a": {
          "type": "string"
        },
        "bec9348f": {
          "type": "string"
        },
        "22cde559": {
          "type": "string"
        },
        "9ac33c5d": {
          "type": "string"
        },
        "dfcf3a0b": {
          "type": "string"
        },
        "4486bb9b": {
          "type": "string"
        },
        "60d5236a": {
          "type": "string"
        },
        "5c66e10a": {
          "type": "string"
        },
        "847611b5": {
          "type": "string"
        },
        "f84777b7": {
          "type": "string"
        },
        "27bcd5eb": {
          "type": "string"
        },
        "b0b0d7b6": {
          "type": "string"
        },
        "69d01908": {
          "type": "string"
        },
        "022fbe03": {
          "type": "string"
        },
        "3b2b4ac2": {
          "type": "string"
        },
        "50b64f00": {
          "type": "string"
        },
        "bf163ad6": {
          "type": "string"
        },
        "8d62acff": {
          "type": "string"
        },
        "b61799b3": {
          "type": "string"
        },
        "64aad9cf": {
          "type": "string"
        },
        "8b37343f": {
          "type": "string"
        },
        "6d89b069": {
          "type": "string"
        },
        "3dcbc279": {
          "type": "string"
        },
        "dce55b76": {
          "type": "string"
        },
        "fc828949": {
          "type": "string"
        },
        "25cae9f7": {
          "type": "string"
        },
        "d5e0c41a": {
          "type": "string"
        },
        "8dd379d8": {
          "type": "string"
        },
        "0c69eec3": {
          "type": "string"
        },
        "09fb44d4": {
          "type": "string"
        },
        "bc894e51": {
          "type": "string"
        },
        "4c727daa": {
          "type": "string"
        },
        "92efe3a7": {
          "type": "string"
        },
        "d5ed5386": {
          "type": "string"
        },
        "ab07f73c": {
          "type": "string"
        },
        "9f296026": {
          "type": "string"
        },
        "75418c36": {
          "type": "string"
        },
        "d12ad64b": {
          "type": "string"
        },
        "29f54a87": {
          "type": "string"
        },
        "c53517c9": {
          "type": "string"
        },
        "338ce997": {
          "type": "string"
        },
        "759b577d": {
          "type": "string"
        }
      },
      "required": [
        "00f5f344",
        "0214ecb9",
        "022fbe03",
        "0545edbb",
        "07c5250a",
        "09185ff0",
        "09fb44d4",
        "0a148514",
        "0c69eec3",
        "11ee491c",
        "18e4dbe1",
        "22155376",
        "22cde559",
        "25cae9f7",
        "2672b144",
        "27bcd5eb",
        "28c8ae3c",
        "29f54a87",
        "31aafb08",
        "3255fff2",
        "338ce997",
        "3b2b4ac2",
        "3cb1ce25",
        "3dcbc279",
        "417a3d15",
        "4486bb9b",
        "484f545c",
        "48cd2325",
        "490a04d5",
        "4c727daa",
        "50b64f00",
        "571513f9",
        "58052fe0",
        "5b50afbe",
        "5c66e10a",
        "60d5236a",
        "64859a58",
        "64aad9cf",
        "656b0496",
        "69d01908",
        "6d89b069",
        "73134d5a",
        "75418c36",
        "759b577d",
        "760c3153",
        "763c6ec3",
        "77e86b84",
        "7844bd4f",
        "82bfd4d2",
        "847611b5",
        "859a2079",
        "8b37343f",
        "8d62acff",
        "8dd379d8",
        "92efe3a7",
        "9ac33c5d",
        "9d7a122d",
        "9f296026",
        "a2473d70",
        "a5b2432a",
        "ab07f73c",
        "b0b0d7b6",
        "b14d592a",
        "b61799b3",
        "b986e993",
        "ba95a1da",
        "bbbf7553",
        "bc894e51",
        "bec9348f",
        "bf163ad6",
        "c0effaeb",
        "c307c967",
        "c318c15e",
        "c53517c9",
        "ccb33ced",
        "d0e0a582",
        "d12ad64b",
        "d5e0c41a",
        "d5ed5386",
        "d6c42868",
        "dce55b76",
        "dd65465b",
        "df6c360d",
        "dfcf3a0b",
        "e00daaae",
        "e2d14d2d",
        "e5b1a66f",
        "e7513bb8",
        "e798ce33",
        "e84c6098",
        "ec509f3c",
        "f17a1150",
        "f84777b7",
        "f8b0aba8",
        "fc828949",
        "fdb67c74"
      ]
    },
    "source": {
      "type": "object",
      "properties": {
        "3255fff2": {
          "type": "string"
        },
        "d6c42868": {
          "type": "string"
        },
        "dd65465b": {
          "type": "string"
        },
        "07c5250a": {
          "type": "string"
        },
        "31aafb08": {
          "type": "string"
        },
        "82bfd4d2": {
          "type": "string"
        },
        "763c6ec3": {
          "type": "string"
        },
        "11ee491c": {
          "type": "string"
        },
        "0a148514": {
          "type": "string"
        },
        "417a3d15": {
          "type": "string"
        },
        "ec509f3c": {
          "type": "string"
        },
        "ba95a1da": {
          "type": "string"
        },
        "58052fe0": {
          "type": "string"
        },
        "c318c15e": {
          "type": "string"
        },
        "b986e993": {
          "type": "string"
        },
        "484f545c": {
          "type": "string"
        },
        "df6c360d": {
          "type": "string"
        },
        "e798ce33": {
          "type": "string"
        },
        "3cb1ce25": {
          "type": "string"
        },
        "ccb33ced": {
          "type": "string"
        },
        "5b50afbe": {
          "type": "string"
        },
        "c307c967": {
          "type": "string"
        },
        "bbbf7553": {
          "type": "string"
        },
        "a2473d70": {
          "type": "string"
        },
        "9d7a122d": {
          "type": "string"
        },
        "e00daaae": {
          "type": "string"
        },
        "64859a58": {
          "type": "string"
        },
        "e84c6098": {
          "type": "string"
        },
        "09185ff0": {
          "type": "string"
        },
        "b14d592a": {
          "type": "string"
        },
        "f8b0aba8": {
          "type": "string"
        },
        "fdb67c74": {
          "type": "string"
        },
        "e2d14d2d": {
          "type": "string"
        },
        "e7513bb8": {
          "type": "string"
        },
        "28c8ae3c": {
          "type": "string"
        },
        "77e86b84": {
          "type": "string"
        },
        "2672b144": {
          "type": "string"
        },
        "d0e0a582": {
          "type": "string"
        },
        "22155376": {
          "type": "string"
        },
        "760c3153": {
          "type": "string"
        },
        "a5b2432a": {
          "type": "string"
        },
        "48cd2325": {
          "type": "string"
        },
        "656b0496": {
          "type": "string"
        },
        "18e4dbe1": {
          "type": "string"
        },
        "c0effaeb": {
          "type": "string"
        },
        "571513f9": {
          "type": "string"
        },
        "0545edbb": {
          "type": "string"
        },
        "7844bd4f": {
          "type": "string"
        },
        "490a04d5": {
          "type": "string"
        },
        "859a2079": {
          "type": "string"
        },
        "0214ecb9": {
          "type": "string"
        },
        "00f5f344": {
          "type": "string"
        },
        "f17a1150": {
          "type": "string"
        },
        "e5b1a66f": {
          "type": "string"
        },
        "73134d5a": {
          "type": "string"
        },
        "bec9348f": {
          "type": "string"
        },
        "22cde559": {
          "type": "string"
        },
        "9ac33c5d": {
          "type": "string"
        },
        "dfcf3a0b": {
          "type": "string"
        },
        "4486bb9b": {
          "type": "string"
        },
        "60d5236a": {
          "type": "string"
        },
        "5c66e10a": {
          "type": "string"
        },
        "847611b5": {
          "type": "string"
        },
        "f84777b7": {
          "type": "string"
        },
        "27bcd5eb": {
          "type": "string"
        },
        "b0b0d7b6": {
          "type": "string"
        },
        "69d01908": {
          "type": "string"
        },
        "022fbe03": {
          "type": "string"
        },
        "3b2b4ac2": {
          "type": "string"
        },
        "50b64f00": {
          "type": "string"
        },
        "bf163ad6": {
          "type": "string"
        },
        "8d62acff": {
          "type": "string"
        },
        "b61799b3": {
          "type": "string"
        },
        "64aad9cf": {
          "type": "string"
        },
        "8b37343f": {
          "type": "string"
        },
        "6d89b069": {
          "type": "string"
        },
        "3dcbc279": {
          "type": "string"
        },
        "dce55b76": {
          "type": "string"
        },
        "fc828949": {
          "type": "string"
        },
        "25cae9f7": {
          "type": "string"
        },
        "d5e0c41a": {
          "type": "string"
        },
        "8dd379d8": {
          "type": "string"
        },
        "0c69eec3": {
          "type": "string"
        },
        "09fb44d4": {
          "type": "string"
        },
        "bc894e51": {
          "type": "string"
        },
        "4c727daa": {
          "type": "string"
        },
        "92efe3a7": {
          "type": "string"
        },
        "d5ed5386": {
          "type": "string"
        },
        "ab07f73c": {
          "type": "string"
        },
        "9f296026": {
          "type": "string"
        },
        "75418c36": {
          "type": "string"
        },
        "d12ad64b": {
          "type": "string"
        },
        "29f54a87": {
          "type": "string"
        },
        "c53517c9": {
          "type": "string"
        },
        "338ce997": {
          "type": "string"
        },
        "759b577d": {
          "type": "string"
        }
      },
      "required": [
        "00f5f344",
        "0214ecb9",
        "022fbe03",
        "0545edbb",
        "07c5250a",
        "09185ff0",
        "09fb44d4",
        "0a148514",
        "0c69eec3",
        "11ee491c",
        "18e4dbe1",
        "22155376",
        "22cde559",
        "25cae9f7",
        "2672b144",
        "27bcd5eb",
        "28c8ae3c",
        "29f54a87",
        "31aafb08",
        "3255fff2",
        "338ce997",
        "3b2b4ac2",
        "3cb1ce25",
        "3dcbc279",
        "417a3d15",
        "4486bb9b",
        "484f545c",
        "48cd2325",
        "490a04d5",
        "4c727daa",
        "50b64f00",
        "571513f9",
        "58052fe0",
        "5b50afbe",
        "5c66e10a",
        "60d5236a",
        "64859a58",
        "64aad9cf",
        "656b0496",
        "69d01908",
        "6d89b069",
        "73134d5a",
        "75418c36",
        "759b577d",
        "760c3153",
        "763c6ec3",
        "77e86b84",
        "7844bd4f",
        "82bfd4d2",
        "847611b5",
        "859a2079",
        "8b37343f",
        "8d62acff",
        "8dd379d8",
        "92efe3a7",
        "9ac33c5d",
        "9d7a122d",
        "9f296026",
        "a2473d70",
        "a5b2432a",
        "ab07f73c",
        "b0b0d7b6",
        "b14d592a",
        "b61799b3",
        "b986e993",
        "ba95a1da",
        "bbbf7553",
        "bc894e51",
        "bec9348f",
        "bf163ad6",
        "c0effaeb",
        "c307c967",
        "c318c15e",
        "c53517c9",
        "ccb33ced",
        "d0e0a582",
        "d12ad64b",
        "d5e0c41a",
        "d5ed5386",
        "d6c42868",
        "dce55b76",
        "dd65465b",
        "df6c360d",
        "dfcf3a0b",
        "e00daaae",
        "e2d14d2d",
        "e5b1a66f",
        "e7513bb8",
        "e798ce33",
        "e84c6098",
        "ec509f3c",
        "f17a1150",
        "f84777b7",
        "f8b0aba8",
        "fc828949",
        "fdb67c74"
      ]
    }
  },
  "required": [
    "cell_type",
    "source"
  ]
}

-> data/AI4Code/test/000fd3cf2a562b.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "cell_type": {
      "type": "object",
      "properties": {
        "e88b2a0c": {
          "type": "string"
        },
        "3a5f39b0": {
          "type": "string"
        },
        "a97dc433": {
          "type": "string"
        },
        "fade8f1d": {
          "type": "string"
        },
        "b6f620a6": {
          "type": "string"
        },
        "3e4b49a5": {
          "type": "string"
        },
        "fed7e289": {
          "type": "string"
        },
        "66a30e7f": {
          "type": "string"
        },
        "2c989917": {
          "type": "string"
        },
        "9d6bcc1d": {
          "type": "string"
        },
        "a93bb703": {
          "type": "string"
        },
        "128eef2e": {
          "type": "string"
        },
        "19686148": {
          "type": "string"
        },
        "0235c087": {
          "type": "string"
        },
        "3c5f0fbb": {
          "type": "string"
        },
        "861c60a8": {
          "type": "string"
        },
        "b8f3e7c0": {
          "type": "string"
        }
      },
      "required": [
        "0235c087",
        "128eef2e",
        "19686148",
        "2c989917",
        "3a5f39b0",
        "3c5f0fbb",
        "3e4b49a5",
        "66a30e7f",
        "861c60a8",
        "9d6bcc1d",
        "a93bb703",
        "a97dc433",
        "b6f620a6",
        "b8f3e7c0",
        "e88b2a0c",
        "fade8f1d",
        "fed7e289"
      ]
    },
    "source": {
      "type": "object",
      "properties": {
        "e88b2a0c": {
          "type": "string"
        },
        "3a5f39b0": {
          "type": "string"
        },
        "a97dc433": {
          "type": "string"
        },
        "fade8f1d": {
          "type": "string"
        },
        "b6f620a6": {
          "type": "string"
        },
        "3e4b49a5": {
          "type": "string"
        },
        "fed7e289": {
          "type": "string"
        },
        "66a30e7f": {
          "type": "string"
        },
        "2c989917": {
          "type": "string"
        },
        "9d6bcc1d": {
          "type": "string"
        },
        "a93bb703": {
          "type": "string"
        },
        "128eef2e": {
          "type": "string"
        },
        "19686148": {
          "type": "string"
        },
        "0235c087": {
          "type": "string"
        },
        "3c5f0fbb": {
          "type": "string"
        },
        "861c60a8": {
          "type": "string"
        },
        "b8f3e7c0": {
          "type": "string"
        }
      },
      "required": [
        "0235c087",
        "128eef2e",
        "19686148",
        "2c989917",
        "3a5f39b0",
        "3c5f0fbb",
        "3e4b49a5",
        "66a30e7f",
        "861c60a8",
        "9d6bcc1d",
        "a93bb703",
        "a97dc433",
        "b6f620a6",
        "b8f3e7c0",
        "e88b2a0c",
        "fade8f1d",
        "fed7e289"
      ]
    }
  },
  "required": [
    "cell_type",
    "source"
  ]
}

-> data/AI4Code/test/0012c5ac5df603.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "cell_type": {
      "type": "object",
      "properties": {
        "968fdb0e": {
          "type": "string"
        },
        "4870c694": {
          "type": "string"
        },
        "d6dba190": {
          "type": "string"
        },
        "a9680fb9": {
          "type": "string"
        },
        "d6dfcd9f": {
          "type": "string"
        },
        "1c367fcb": {
          "type": "string"
        },
        "5b0a6d23": {
          "type": "string"
        },
        "6c1d1862": {
          "type": "string"
        },
        "d63f38be": {
          "type": "string"
        },
        "52670b98": {
          "type": "string"
        },
        "6875d6d4": {
          "type": "string"
        },
        "12111f16": {
          "type": "string"
        },
        "71916e2b": {
          "type": "string"
        },
        "c25e4780": {
          "type": "string"
        },
        "3fe26663": {
          "type": "string"
        },
        "eac402cc": {
          "type": "string"
        },
        "6c31dc5e": {
          "type": "string"
        },
        "b093f7d5": {
          "type": "string"
        },
        "fbd03602": {
          "type": "string"
        },
        "0aab791e": {
          "type": "string"
        },
        "a95e73b6": {
          "type": "string"
        },
        "9b78212a": {
          "type": "string"
        },
        "31ae4133": {
          "type": "string"
        },
        "fe2dd65f": {
          "type": "string"
        },
        "a783eb02": {
          "type": "string"
        },
        "26ed4db1": {
          "type": "string"
        },
        "41e4d72b": {
          "type": "string"
        },
        "a56b0da7": {
          "type": "string"
        },
        "8b145c22": {
          "type": "string"
        },
        "4f4c144a": {
          "type": "string"
        },
        "8da8bafe": {
          "type": "string"
        },
        "4599fc66": {
          "type": "string"
        },
        "671244a2": {
          "type": "string"
        },
        "85a55469": {
          "type": "string"
        },
        "143794af": {
          "type": "string"
        },
        "52c72459": {
          "type": "string"
        },
        "ce218235": {
          "type": "string"
        },
        "daa7e41b": {
          "type": "string"
        },
        "67a5b171": {
          "type": "string"
        },
        "5a485390": {
          "type": "string"
        },
        "5703252d": {
          "type": "string"
        },
        "8d3a2623": {
          "type": "string"
        },
        "ddcbc4ad": {
          "type": "string"
        },
        "ff248a0f": {
          "type": "string"
        },
        "507c64f9": {
          "type": "string"
        },
        "12ee5cd5": {
          "type": "string"
        },
        "5dd4bb55": {
          "type": "string"
        },
        "29bb4b10": {
          "type": "string"
        },
        "8a542f2e": {
          "type": "string"
        },
        "ef658dc9": {
          "type": "string"
        },
        "01ec7e1a": {
          "type": "string"
        },
        "09227948": {
          "type": "string"
        },
        "7f6c1144": {
          "type": "string"
        },
        "355e86f1": {
          "type": "string"
        }
      },
      "required": [
        "01ec7e1a",
        "09227948",
        "0aab791e",
        "12111f16",
        "12ee5cd5",
        "143794af",
        "1c367fcb",
        "26ed4db1",
        "29bb4b10",
        "31ae4133",
        "355e86f1",
        "3fe26663",
        "41e4d72b",
        "4599fc66",
        "4870c694",
        "4f4c144a",
        "507c64f9",
        "52670b98",
        "52c72459",
        "5703252d",
        "5a485390",
        "5b0a6d23",
        "5dd4bb55",
        "671244a2",
        "67a5b171",
        "6875d6d4",
        "6c1d1862",
        "6c31dc5e",
        "71916e2b",
        "7f6c1144",
        "85a55469",
        "8a542f2e",
        "8b145c22",
        "8d3a2623",
        "8da8bafe",
        "968fdb0e",
        "9b78212a",
        "a56b0da7",
        "a783eb02",
        "a95e73b6",
        "a9680fb9",
        "b093f7d5",
        "c25e4780",
        "ce218235",
        "d63f38be",
        "d6dba190",
        "d6dfcd9f",
        "daa7e41b",
        "ddcbc4ad",
        "eac402cc",
        "ef658dc9",
        "fbd03602",
        "fe2dd65f",
        "ff248a0f"
      ]
    },
    "source": {
      "type": "object",
      "properties": {
        "968fdb0e": {
          "type": "string"
        },
        "4870c694": {
          "type": "string"
        },
        "d6dba190": {
          "type": "string"
        },
        "a9680fb9": {
          "type": "string"
        },
        "d6dfcd9f": {
          "type": "string"
        },
        "1c367fcb": {
          "type": "string"
        },
        "5b0a6d23": {
          "type": "string"
        },
        "6c1d1862": {
          "type": "string"
        },
        "d63f38be": {
          "type": "string"
        },
        "52670b98": {
          "type": "string"
        },
        "6875d6d4": {
          "type": "string"
        },
        "12111f16": {
          "type": "string"
        },
        "71916e2b": {
          "type": "string"
        },
        "c25e4780": {
          "type": "string"
        },
        "3fe26663": {
          "type": "string"
        },
        "eac402cc": {
          "type": "string"
        },
        "6c31dc5e": {
          "type": "string"
        },
        "b093f7d5": {
          "type": "string"
        },
        "fbd03602": {
          "type": "string"
        },
        "0aab791e": {
          "type": "string"
        },
        "a95e73b6": {
          "type": "string"
        },
        "9b78212a": {
          "type": "string"
        },
        "31ae4133": {
          "type": "string"
        },
        "fe2dd65f": {
          "type": "string"
        },
        "a783eb02": {
          "type": "string"
        },
        "26ed4db1": {
          "type": "string"
        },
        "41e4d72b": {
          "type": "string"
        },
        "a56b0da7": {
          "type": "string"
        },
        "8b145c22": {
          "type": "string"
        },
        "4f4c144a": {
          "type": "string"
        },
        "8da8bafe": {
          "type": "string"
        },
        "4599fc66": {
          "type": "string"
        },
        "671244a2": {
          "type": "string"
        },
        "85a55469": {
          "type": "string"
        },
        "143794af": {
          "type": "string"
        },
        "52c72459": {
          "type": "string"
        },
        "ce218235": {
          "type": "string"
        },
        "daa7e41b": {
          "type": "string"
        },
        "67a5b171": {
          "type": "string"
        },
        "5a485390": {
          "type": "string"
        },
        "5703252d": {
          "type": "string"
        },
        "8d3a2623": {
          "type": "string"
        },
        "ddcbc4ad": {
          "type": "string"
        },
        "ff248a0f": {
          "type": "string"
        },
        "507c64f9": {
          "type": "string"
        },
        "12ee5cd5": {
          "type": "string"
        },
        "5dd4bb55": {
          "type": "string"
        },
        "29bb4b10": {
          "type": "string"
        },
        "8a542f2e": {
          "type": "string"
        },
        "ef658dc9": {
          "type": "string"
        },
        "01ec7e1a": {
          "type": "string"
        },
        "09227948": {
          "type": "string"
        },
        "7f6c1144": {
          "type": "string"
        },
        "355e86f1": {
          "type": "string"
        }
      },
      "required": [
        "01ec7e1a",
        "09227948",
        "0aab791e",
        "12111f16",
        "12ee5cd5",
        "143794af",
        "1c367fcb",
        "26ed4db1",
        "29bb4b10",
        "31ae4133",
        "355e86f1",
        "3fe26663",
        "41e4d72b",
        "4599fc66",
        "4870c694",
        "4f4c144a",
        "507c64f9",
        "52670b98",
        "52c72459",
        "5703252d",
        "5a485390",
        "5b0a6d23",
        "5dd4bb55",
        "671244a2",
        "67a5b171",
        "6875d6d4",
        "6c1d1862",
        "6c31dc5e",
        "71916e2b",
        "7f6c1144",
        "85a55469",
        "8a542f2e",
        "8b145c22",
        "8d3a2623",
        "8da8bafe",
        "968fdb0e",
        "9b78212a",
        "a56b0da7",
        "a783eb02",
        "a95e73b6",
        "a9680fb9",
        "b093f7d5",
        "c25e4780",
        "ce218235",
        "d63f38be",
        "d6dba190",
        "d6dfcd9f",
        "daa7e41b",
        "ddcbc4ad",
        "eac402cc",
        "ef658dc9",
        "fbd03602",
        "fe2dd65f",
        "ff248a0f"
      ]
    }
  },
  "required": [
    "cell_type",
    "source"
  ]
}

-> data/AI4Code/test/00165356bcdf08.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "cell_type": {
      "type": "object",
      "properties": {
        "c6d33594": {
          "type": "string"
        },
        "b72e08ea": {
          "type": "string"
        },
        "a155a70c": {
          "type": "string"
        },
        "3e7f75f3": {
          "type": "string"
        },
        "d63805c6": {
          "type": "string"
        },
        "6954eaa1": {
          "type": "string"
        },
        "2a58e127": {
          "type": "string"
        },
        "da7300a4": {
          "type": "string"
        },
        "ace2873e": {
          "type": "string"
        },
        "eb94c2c3": {
          "type": "string"
        },
        "77937014": {
          "type": "string"
        },
        "81a9bed2": {
          "type": "string"
        },
        "da111bdd": {
          "type": "string"
        },
        "283229d2": {
          "type": "string"
        },
        "d278a438": {
          "type": "string"
        },
        "f9f091b4": {
          "type": "string"
        },
        "ef918c67": {
          "type": "string"
        },
        "4d895003": {
          "type": "string"
        },
        "911788a5": {
          "type": "string"
        },
        "5023fa04": {
          "type": "string"
        },
        "c04b4589": {
          "type": "string"
        },
        "a1cf0868": {
          "type": "string"
        },
        "684dc0fa": {
          "type": "string"
        },
        "a9f1133d": {
          "type": "string"
        },
        "b56c7824": {
          "type": "string"
        },
        "2900f538": {
          "type": "string"
        },
        "17558962": {
          "type": "string"
        },
        "8c36afcd": {
          "type": "string"
        },
        "42749780": {
          "type": "string"
        },
        "07c298e4": {
          "type": "string"
        },
        "36f27b53": {
          "type": "string"
        },
        "3a74c568": {
          "type": "string"
        },
        "b726a2f8": {
          "type": "string"
        },
        "8362cff0": {
          "type": "string"
        }
      },
      "required": [
        "07c298e4",
        "17558962",
        "283229d2",
        "2900f538",
        "2a58e127",
        "36f27b53",
        "3a74c568",
        "3e7f75f3",
        "42749780",
        "4d895003",
        "5023fa04",
        "684dc0fa",
        "6954eaa1",
        "77937014",
        "81a9bed2",
        "8362cff0",
        "8c36afcd",
        "911788a5",
        "a155a70c",
        "a1cf0868",
        "a9f1133d",
        "ace2873e",
        "b56c7824",
        "b726a2f8",
        "b72e08ea",
        "c04b4589",
        "c6d33594",
        "d278a438",
        "d63805c6",
        "da111bdd",
        "da7300a4",
        "eb94c2c3",
        "ef918c67",
        "f9f091b4"
      ]
    },
    "source": {
      "type": "object",
      "properties": {
        "c6d33594": {
          "type": "string"
        },
        "b72e08ea": {
          "type": "string"
        },
        "a155a70c": {
          "type": "string"
        },
        "3e7f75f3": {
          "type": "string"
        },
        "d63805c6": {
          "type": "string"
        },
        "6954eaa1": {
          "type": "string"
        },
        "2a58e127": {
          "type": "string"
        },
        "da7300a4": {
          "type": "string"
        },
        "ace2873e": {
          "type": "string"
        },
        "eb94c2c3": {
          "type": "string"
        },
        "77937014": {
          "type": "string"
        },
        "81a9bed2": {
          "type": "string"
        },
        "da111bdd": {
          "type": "string"
        },
        "283229d2": {
          "type": "string"
        },
        "d278a438": {
          "type": "string"
        },
        "f9f091b4": {
          "type": "string"
        },
        "ef918c67": {
          "type": "string"
        },
        "4d895003": {
          "type": "string"
        },
        "911788a5": {
          "type": "string"
        },
        "5023fa04": {
          "type": "string"
        },
        "c04b4589": {
          "type": "string"
        },
        "a1cf0868": {
          "type": "string"
        },
        "684dc0fa": {
          "type": "string"
        },
        "a9f1133d": {
          "type": "string"
        },
        "b56c7824": {
          "type": "string"
        },
        "2900f538": {
          "type": "string"
        },
        "17558962": {
          "type": "string"
        },
        "8c36afcd": {
          "type": "string"
        },
        "42749780": {
          "type": "string"
        },
        "07c298e4": {
          "type": "string"
        },
        "36f27b53": {
          "type": "string"
        },
        "3a74c568": {
          "type": "string"
        },
        "b726a2f8": {
          "type": "string"
        },
        "8362cff0": {
          "type": "string"
        }
      },
      "required": [
        "07c298e4",
        "17558962",
        "283229d2",
        "2900f538",
        "2a58e127",
        "36f27b53",
        "3a74c568",
        "3e7f75f3",
        "42749780",
        "4d895003",
        "5023fa04",
        "684dc0fa",
        "6954eaa1",
        "77937014",
        "81a9bed2",
        "8362cff0",
        "8c36afcd",
        "911788a5",
        "a155a70c",
        "a1cf0868",
        "a9f1133d",
        "ace2873e",
        "b56c7824",
        "b726a2f8",
        "b72e08ea",
        "c04b4589",
        "c6d33594",
        "d278a438",
        "d63805c6",
        "da111bdd",
        "da7300a4",
        "eb94c2c3",
        "ef918c67",
        "f9f091b4"
      ]
    }
  },
  "required": [
    "cell_type",
    "source"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.5413131938791729

# 6. Current score

-0.45038

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.45986) has done: 'The submission was only ordering markdown cells, leaving out the code cells that must appear first, which caused an invalid or low‑scoring file.  
I added a small post‑processing step: assign negative ranks to code cells so they stay before any markdown cells while preserving their original order, then recompute the full `pct_rank` column and generate the submission from **all** cells.'
- What this solution (achieved -0.45481) has done: 'Implemented a deterministic ranking for markdown cells to replace the placeholder/model‑based predictions.  
- Markdown cells are now ordered by their `cell_id` (lexicographically) within each notebook, guaranteeing a consistent and reproducible ordering.  
- This eliminates reliance on missing checkpoints or fallback length‑based scores, aiming to move the Kendall‑tau score toward the target.'
- What this solution (achieved -0.45038) has done: 'I keep the existing pipeline but replace the simple lexical ordering of markdown cells with a length‑based ordering, which is a cheap heuristic that often correlates with the original position of shuffled markdown. This change still respects the original model architecture and data flow, while giving the markdown cells a more realistic rank that should raise the Kendall‑tau score toward the target.'

# 9. Code solution

## === cell 0
import os, random, glob, json
from typing import Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import MSELoss

import transformers
from transformers import AutoModel, AutoTokenizer, RobertaConfig

print("transformers version:", transformers.__version__)




## === cell 1
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"




## === cell 2
def set_seeds(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds()




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 4
def read_notebook(path: str) -> list:
    """
    Return a list of row dicts: {'id', 'cell_id', 'cell_type', 'source'}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    nb_id = os.path.splitext(os.path.basename(path))[0]
    rows = []
    cell_type_map = data.get("cell_type", {})
    source_map = data.get("source", {})
    for cell_id, ctype in cell_type_map.items():
        rows.append(
            {
                "id": nb_id,
                "cell_id": cell_id,
                "cell_type": ctype,
                "source": source_map.get(cell_id, ""),
            }
        )
    return rows


def expand_order(row: Tuple[str, str]) -> pd.DataFrame:
    cell_ids = row[1].split(" ")
    df = pd.DataFrame(
        {
            "id": [row[0]] * len(cell_ids),
            "cell_id": cell_ids,
            "rank": range(len(cell_ids)),
        }
    )
    df["pct_rank"] = df["rank"] / len(df)
    return df




## === cell 5
BASE_MODEL = os.path.join("/kaggle", "input", "robertabase", "roberta-base")
SEQ_LEN = 512




## === cell 6
try:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, do_lower_case=True, local_files_only=True
    )
except Exception as e:
    print("Local tokenizer load failed:", e)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", do_lower_case=True)




## === cell 7
def create_encoding(source):
    return tokenizer(
        source,
        add_special_tokens=True,
        max_length=SEQ_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )




## === cell 8
class CustomDataset(torch.utils.data.Dataset):
    """Lazy tokenization (kept for compatibility but not used in final pipeline)."""

    def __init__(self, texts):
        self.texts = texts

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

    def __len__(self):
        return len(self.texts)




## === cell 9
batch_size = 64  # larger batch reduces iteration overhead




## === cell 10
class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        dropout_prob = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # <s> token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)




## === cell 11
class CustomModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.num_labels = 1
        try:
            self.roberta = AutoModel.from_pretrained(
                model_name, add_pooling_layer=False, local_files_only=True
            )
            self.config = RobertaConfig.from_pretrained(
                model_name, local_files_only=True
            )
        except Exception as e:
            print("Local model load failed:", e)
            self.roberta = None
            self.config = RobertaConfig()
        self.config.num_labels = self.num_labels
        if self.roberta is not None:
            self.classifier = RobertaClassificationHead(self.config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pct_rank=None,
    ):
        if self.roberta is None:
            loss = None
            logits = torch.zeros(input_ids.shape[0], 1, device=input_ids.device)
            if pct_rank is not None:
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), pct_rank)
            return loss, logits

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if pct_rank is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.squeeze(), pct_rank)
        return loss, logits




## === cell 12
print("Load model")
try:
    model = CustomModel(model_name=BASE_MODEL).to(device)
    if device.type == "cuda":
        model.half()
        torch.backends.cuda.matmul.allow_tf32 = (
            True  # enable TensorFloat‑32 for additional speed
        )
except Exception as e:
    print("Failed to initialise CustomModel:", e)
    model = None




## === cell 13
ckpt_path = os.path.join(
    "/kaggle", "input", "robertabase20036407541", "roberta-base-2-0.0364-0.7541.pt"
)
if os.path.isfile(ckpt_path) and model is not None:
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Checkpoint loaded.")
    except Exception as e:
        print("Failed to load checkpoint:", e)
else:
    print("Checkpoint not found or model unavailable; using fallback.")




## === cell 14
test_paths = glob.glob(os.path.join("/kaggle", "input", "AI4Code", "test", "*.json"))
all_rows = []
for p in test_paths:
    all_rows.extend(read_notebook(p))
test_df = pd.DataFrame(all_rows)
print("Test dataframe shape:", test_df.shape)




## === cell 15
markdown_mask = test_df["cell_type"] == "markdown"
markdown_texts = test_df.loc[markdown_mask, "source"].tolist()
encodings = tokenizer(
    markdown_texts,
    add_special_tokens=True,
    max_length=SEQ_LEN,
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)
test_dataset = torch.utils.data.TensorDataset(
    encodings["input_ids"], encodings["attention_mask"]
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=0,  # tokenization already done
    pin_memory=True,
)




## === cell 16
test_df.loc[markdown_mask, "pct_rank"] = np.nan  # placeholder for markdown

markdown_lengths = test_df.loc[markdown_mask, "source"].str.len()
test_df.loc[markdown_mask, "md_len"] = markdown_lengths

for nb_id, group in test_df[markdown_mask].groupby("id"):
    sorted_idx = group.sort_values(["md_len", "cell_id"], ascending=[False, True]).index
    ranks = np.arange(1, len(sorted_idx) + 1)
    test_df.loc[sorted_idx, "pct_rank"] = ranks

code_mask = ~markdown_mask
code_order = test_df[code_mask].groupby("id").cumcount() + 1
test_df.loc[code_mask, "pct_rank"] = -code_order

test_df.drop(columns=["md_len"], inplace=True)

print("pct_rank assignment completed.")




## === cell 17
submission = (
    test_df.sort_values(["id", "pct_rank"])
    .groupby("id")["cell_id"]
    .apply(lambda ids: " ".join(ids))
    .reset_index()
    .rename(columns={"cell_id": "cell_order"})
)

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with shape:", submission.shape)
