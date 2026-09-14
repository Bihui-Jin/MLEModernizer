# Automated Modernization of Machine Learning Engineering Notebooks for Reproducibility

This is replication package of paper "Automated Modernization of Machine Learning Engineering Notebooks for Reproducibility" (https://doi.org/10.1145/3832166) accepted by ISSTA '26.

Bihui Jin, Kaiyuan Wang, and Pengyu Nie. 2026. Automated Modernization of Machine Learning Engineering
Notebooks for Reproducibility. Proc. ACM Softw. Eng. 3, ISSTA, Article ISSTA075 (October 2026), 22 pages.
https://doi.org/10.1145/383216

## Citation
Please cite using the following BibTeX entry:
```
@article{jin2026automatedReproducibility,
  author    = {Jin, Bihui and Wang, Kaiyuan and Nie, Pengyu},
  title     = {Automated Modernization of Machine Learning Engineering Notebooks for Reproducibility},
  journal   = {Proc. ACM Softw. Eng.},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume    = {3},
  numpages = {22},
  number    = {ISSTA},
  articleno = {ISSTA075},
  year      = {2026},
  issue_date = {Oct 2026},
  month     = {10},
  url = {https://doi.org/10.1145/383216},
  doi       = {10.1145/383216}
}
```

## Replication package layout

The replication package is split across two repositories so that source code is easy to browse while large and high-volume data remain on storage designed for datasets:

- [GitHub](https://github.com/Bihui-Jin/MLEModernizer) contains source code, data-processing programs, and compact result artifacts. Every GitHub file is smaller than 50 MB.
- [Hugging Face](https://huggingface.co/datasets/BihuiJ/MLEModernizer) contains the Meta Kaggle archives, collected/processed notebooks, all execution outputs, and any other included artifact that is at least 50 MB.

Paths are identical in the two repositories and do not overlap, apart from
repository administration files and the three packed directories described
below. Reconstruct the complete package by cloning GitHub and overlaying the
Hugging Face dataset:

```bash
git clone https://github.com/Bihui-Jin/MLEModernizer.git
cd MLEModernizer
hf download BihuiJ/MLEModernizer \
  --repo-type dataset \
  --local-dir . \
  --exclude README.md .gitattributes

# Restore directories packed to satisfy Hugging Face's 10,000-entry limit.
tar -xzf baseline/notebooks.tar.gz \
  -C baseline
tar -xzf results/baseline/script_out_allINone.tar.gz \
  -C results/baseline
tar -xzf results/downgrade/baseline/script_out_allINone.tar.gz \
  -C results/downgrade/baseline
```

This download is very large. To retrieve selected content, add one or more `--include` patterns, for example:

```bash
# Raw Meta Kaggle archives only
hf download BihuiJ/MLEModernizer \
  --repo-type dataset \
  --local-dir . \
  --include ".kaggle/**"

# Baseline execution outputs only
hf download BihuiJ/MLEModernizer \
  --repo-type dataset \
  --local-dir . \
  --include "results/baseline/**"
```

The raw archives are restored to:

- `.kaggle/meta-kaggle.zip`
- `.kaggle/meta-kaggle-code/meta-kaggle-code.zip`

Execution notebooks, CSV submissions, and run metadata retain their original
relative paths below `results/` and `verification/`. The only packed paths are
`baseline/notebooks.tar.gz`,
`results/baseline/script_out_allINone.tar.gz`, and
`results/downgrade/baseline/script_out_allINone.tar.gz`; each archive contains
the correspondingly named directory. The Hugging Face tag
`legacy-full-repo-2026-07-24` preserves the earlier monolithic dataset
snapshot; use `main` for the current split package.

## Environment

**Python:** 3.11.11
```
cd ~/MLEModernizer && uv init --python 3.11.11
uv add -r requirements-mle_env.txt
uv sync
```


## Procedure

### Raw Data Collection

The datadump is provided by [Meta Kaggle](https://www.kaggle.com/datasets/kaggle/meta-kaggle) and [Meta Kaggle Code](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code?sort=published).

**Meta Kaggle:**
```
#!/bin/bash
curl -L -o ~/Downloads/meta-kaggle.zip\
  https://www.kaggle.com/api/v1/datasets/download/kaggle/meta-kaggle
```

**Meta Kaggle Code:**
```
#!/bin/bash
curl -L -o ~/Downloads/meta-kaggle-code.zip\
  https://www.kaggle.com/api/v1/datasets/download/kaggle/meta-kaggle-code
```

#### Tables
##### KernelVersions.csv
Each row is one published version of a kernel: a concrete snapshot you can run or submit from. It has Id (filename in Meta Kaggle Code), ScriptId, ScriptLanguageId, RunningTimeInMilliseconds, VersionNumber, CreationDate, EvaluationDate, etc. [ScriptLanguageId](https://www.kaggle.com/datasets/kaggle/meta-kaggle?select=KernelLanguages.csv) tells you the language family (e.g. Python notebook vs R vs other).

##### Kernels.csv
Each row is one kernel (notebook) entity, not a specific version. Id is the kernel id—the same meaning as KernelVersions.ScriptId. Medal records competition medal tier for that kernel (1: gold, 2: silver, 3: bronze). AuthorUserId is the Kaggle user id of the author; it matches Users.Id (same numeric user key). CurrentUrlSlug is a part of the submission url (refer to the next subsection).

##### Submissions.csv
Each row is one leaderboard submission from a team. TeamId points to Teams.Id: which team filed it. SourceKernelVersionId points to KernelVersions.Id: which kernel version was used as the submission source. Since a KernelVersions.Id may exit multi records in Submissions.csv, SubmissionDate and ScoreDate are timestamps used for date matching and cutoff filters. PublicScoreFullPrecision and PrivateScoreFullPrecision hold scores; the scripts use private (and sometimes public) scores to require a scored submission and non-null / non-zero private logic where applied.

##### KernelVersionDatasetSources.csv
Each row links a kernel version to an external dataset version used in that run. KernelVersionId is the same id space as Submissions.SourceKernelVersionId / KernelVersions.Id. SourceDatasetVersionId identifies the attached dataset version (used when listing dataset sources per version and when building the set of “dataset-bound” version ids to exclude).

##### Teams.csv
Each row is one team registered for a competition. Id identifies the team pointing to TeamId in Submissions.csv. CompetitionId points to Competitions.Id: which competition that team belongs to.

##### Competitions.csv
Each row is one competition. Id is its stable numeric identifier. Slug is the URL-style name (used to resolve a competition from a slug). TotalSubmissions is the count Kaggle reports for the competition.

##### Users.csv
Each row is one Kaggle user. Id is exactly what Kernels.AuthorUserId references: the author of a kernel is that user row. UserName is the URL-style name for the submission url (refer to the next subsection).

#### Submission URL

A submission URL is composed as https://www.kaggle.com/code/{UserName}/{CurrentUrlSlug}?scriptVersionId={SourceKernelVersionId}, where UserName, CurrentUrlSlug, and SourceKernelVersionId are from Users.csv and Kernels.csv * 2 respectively.

#### Table Connections
- Competition → teams: Competitions.Id = Teams.CompetitionId.
- Team → submissions: Teams.Id = Submissions.TeamId.
- Submission → kernel version: Submissions.SourceKernelVersionId = KernelVersions.Id.
- Kernel version → kernel: KernelVersions.ScriptId = Kernels.Id.
- Kernel → user (author): Kernels.AuthorUserId = Users.Id.
- Kernel version → external dataset used: KernelVersionDatasetSources.KernelVersionId = KernelVersions.Id (and thus the same id as SourceKernelVersionId on submissions when that link exists).

#### File Naming
Notebooks are named and saved based on their SourceKernelVersionId processed by following:
```
f = str(id).zfill(10)
meta-kaggle-code/'+f[0:4]+'/'+f[4:7]+'/'+str(id)+'.ipynb'
```
To help readability, we rename them by `{Slug}_{UserName}_{CurrentUrlSlug}_v{VersionNumber}_{}_C1.ipynb`

### Baseline (replication of collected scripts)

#### Raw Data Processing

1.`create_kernel.py` retrieves metadata (runtime, submission date, APIs, private score, and external dataset) for all executable scripts

- Use pigar (still updated) to dynamically list dependencies with correct PyPI names
- output: `kernel.json`

2.`baseline/create_fullDataset.py` filters out targeted HTML and converts the code in HTML to Python code in notebooks (`.ipynb`)

- output: `baseline/notebooks/{competition}_{submissionID}_{status}.ipynb`

`<span style="color: red;">`**Make sure to run `create_kernel.py` and `API Preparation 1 2 3` before going through the below sections**

#### Execution

Please refer to the [section here](#file_execution)

### API Backporting

Backporting dependency versions (Rule-based: looking for old version APIs in pypi):

1. For each dependency, query PyPI for the version current at (or just before) the notebook’s original submit date.
2. Create one docker image for the submissions under the same user&competition (choose the oldest version if conflicts).
3. Pin environment to those historical versions and execute the notebook.
4. If failures arise, apply the same cell-level repair loop, but favor backward-compatible edits that align with older APIs.

#### API Preparation

1. `apiDowngrade/python_versions.py` crawls the Python version w.r.t. its release date from the [Python official documentation](https://www.python.org/doc/versions/)

- output: `apiDowngrade/python_versions.json`

1. `apiDowngrade/python_versions_update.py` updates each submission in `kernel.json` with the relevant (max) Python versions

- feature: discern the code syntax (py2 or py3)
- output: updated `apiDowngrade/kernel_w_pyVersion.json`



Python Version Statistics

**Total collected entries:** 51,743  

**Total scripts updated:** 12,106  

**Scripts with no match:** 0  

**Total unique versions:** 59  


## Python Version Distribution


| Python Version | Count | Percentage |
| -------------- | ----- | ---------- |
| 3.14.0         | 617   | 5.10%      |
| 3.13.7         | 31    | 0.26%      |
| 3.13.5         | 73    | 0.60%      |
| 3.13.4         | 2     | 0.02%      |
| 3.13.3         | 43    | 0.36%      |
| 3.13.2         | 156   | 1.29%      |
| 3.13.1         | 51    | 0.42%      |
| 3.13.0         | 244   | 2.02%      |
| 3.12.6         | 79    | 0.65%      |
| 3.12.5         | 61    | 0.50%      |
| 3.12.4         | 124   | 1.02%      |
| 3.12.3         | 410   | 3.39%      |
| 3.12.2         | 406   | 3.35%      |
| 3.12.1         | 163   | 1.35%      |
| 3.12.0         | 84    | 0.69%      |
| 3.11.5         | 54    | 0.45%      |
| 3.11.4         | 414   | 3.42%      |
| 3.11.3         | 111   | 0.92%      |
| 3.11.2         | 67    | 0.55%      |
| 3.11.1         | 137   | 1.13%      |
| 3.11.0         | 88    | 0.73%      |
| 3.10.8         | 52    | 0.43%      |
| 3.10.7         | 9     | 0.07%      |
| 3.10.6         | 39    | 0.32%      |
| 3.10.5         | 119   | 0.98%      |
| 3.10.4         | 353   | 2.92%      |
| 3.10.3         | 28    | 0.23%      |
| 3.10.2         | 107   | 0.88%      |
| 3.10.1         | 326   | 2.69%      |
| 3.10.0         | 836   | 6.91%      |
| 3.9.7          | 488   | 4.03%      |
| 3.9.6          | 317   | 2.62%      |
| 3.9.5          | 309   | 2.55%      |
| 3.9.4          | 137   | 1.13%      |
| 3.9.3          | 4     | 0.03%      |
| 3.9.2          | 177   | 1.46%      |
| 3.9.1          | 934   | 7.72%      |
| 3.9.0          | 326   | 2.69%      |
| 3.8.6          | 175   | 1.45%      |
| 3.8.5          | 955   | 7.89%      |
| 3.8.4          | 103   | 0.85%      |
| 3.8.3          | 347   | 2.87%      |
| 3.8.2          | 193   | 1.59%      |
| 3.8.1          | 91    | 0.75%      |
| 3.8.0          | 179   | 1.48%      |
| 3.7.4          | 313   | 2.59%      |
| 3.7.3          | 622   | 5.14%      |
| 3.7.2          | 123   | 1.02%      |
| 3.7.1          | 59    | 0.49%      |
| 3.7.0          | 474   | 3.92%      |
| 3.6.5          | 84    | 0.69%      |
| 3.6.4          | 117   | 0.97%      |
| 3.6.3          | 82    | 0.68%      |
| 3.6.1          | 2     | 0.02%      |
| 3.6.0          | 70    | 0.58%      |
| 3.5.2          | 91    | 0.75%      |
| 2.7.18         | 48    | 0.40%      |
| 2.7.16         | 1     | 0.01%      |
| 2.7.15         | 1     | 0.01%      |




3.`apiDowngrade/create_apiVersions.py` creates the API match list based on the script submission date and the API required Python version

- feature: pick latest_version - 0.1 (e.g., 3.9 - 0.1 = 3.8) for Python > 3.5, as most scripts in Python 2 are submitted much after end of life of Python 2.6. This should work well in practice because most packages will be backward-compatible with older Python versions for a while.
- outputs:

-`apiDowngrade/submission_noAPI.json`: No external APIs used in the script, note Python version

-`apiDowngrade/pypi_versions.json`: API metadata to save time in API retrieval (yanked versions removed)

-`apiDowngrade/apiDowngradeList/{competition_fileName}.txt`: the list of API names and their matched stable API versions (in the format of api==version) for each submission

-`apiDowngrade/apiDowngradeFamilyList/{submission}.txt`: groups submissions by family (competition + basename without _vN) and keeps the smallest package version across that family.

```
 (record competition name, submission name, and API name)
```

-`apiDowngrade/apiMatch_error.txt`: APIs w/ errors while calling pypi (N/A)

-`apiDowngrade/apiMatch_notFound.txt`: not found APIs on pypi (N/A)

-`apiDowngrade/apiMatch_oldest_Python_compatible.txt`: fallback to the earliest ever version that is Python-compatible (ignores submission date)

-`apiDowngrade/apiMatch_oldest_version_compatible.txt`: fallback to the earliest ever version, regardless of Python compatibility

1. Manually check the python version on **four** submissions from `apiDowngrade/apiMatch_oldest_version_compatible.txt` based on the submission date and required python versions of APIs

- "osic-pulmonary-fibrosis-progression_pradyut23_pulmonary-fibrosis-eda": "3.8"
- "alaska2-image-steganalysis_dzz1th_kernel3de146856b": "3.9"
- "spooky-author-identification_ttetls_a-gentle-mathematical-approach-to-spooky-en-fr": "3.6"
- "statoil-iceberg-classifier-challenge_brassmonkey381_viewing-leak-and-machine-images": "3.6"
- "rsna-2022-cervical-spine-fracture-detection_lsl000ud_rsna2022-7th-place-inference": "3.10"
- "petfinder-pawpularity-score_guillaumes_petfinder-ensemble-xgboost-tabnet": "3.10"
- "cassava-leaf-disease-classification_surayuthpintawong_inference-cassava": "3.5"
- "cassava-leaf-disease-classification_saurabh2mishra_cassava-leaf-disease-inference-label-smoothing": "3.8"
- "aerial-cactus-identification_visali_cactus-classification-using-fastai": "3.9"
- "AI4Code_valentinaliferov_ai4code-submit": "3.11"
- "AI4Code_vaaliferov_ai4code-submit": "3.11"
- **execute** `apiDowngrade/adjust_apiVersions.py` to adjust the API match list
- outputs:

-`apiDowngrade/apiMatch_oldest_Python_compatible2.txt`: final fallbacks that are Python-compatible (ignores submission date)

-`apiDowngrade/apiMatch_oldest_version_compatible2.txt`: final fallbacks regardless of Python compatibility

1. `baseline/check_missingAPIs.py` inspects any APIs not found in the kaggle image

- prerequisite: `apiDowngrade/pypi_versions.json` from `apiDowngrade/create_apiVersions.py` and `apiDowngrade/adjust_apiVersions.py`
- output:  `baseline/requirements.txt`
- details: Found 906 packages in Docker image`</br>`
     416 packages in cache`</br>`
     Found 284 packages in cache but not in Docker image`</br>`
     Filter out these packages for valid reasons; they are not supported by Kaggle

1. `baseline/search_nltkCorpora.py` searches all nltk packages used in the scripts

- output: `baseline/nltkCorpora.txt`
- details: 492 scripts use nltk (8 unique corpora)

#### Execution

Please refer to the [section here](#file_execution)

### API Upgrade

#### Prerequisite

Execute the `.ipynb` file inside Docker to collect replicated scores for Baseline and Backporting.

#### Preparation

1. `upgrade/get_data_preview.py` lists the file paths as a tree in the Docker container/env

- output: `upgrade/data_previews/{competition}.txt` and `upgrade/data_preview.json`

1. `upgrade/group_api.py` groups the environments with the same requirement and python version plus the empty env (env with a specific python version but no API used)

- output: `upgrade/groups_by_exact_content.json`

1. `upgrade/get_api_preview.py` lists installed APIs for each env in `upgrade/groups_by_exact_content.json`

- output: `upgrade/pipList/{filename}.txt` for backporting and `upgrade/pip_list.txt` for baseline

1. `upgrade/select_candidates.py` selects non-replicable scripts used for LLM upgrade

- LLM Keys: LLM keys are used to call the LLM by setting `openai`, `anthropic`, and `gemini` in profile.json
- Note: 
  1. you need to set the key by your own, or set export OPENAI_API_KEY="\<keyhere>"
  2. sys_tokens > 13_485, where 13_485 (token_size) is computed by the 95th percentile of the resulting token counts (found in `dprocess.ipynb`).
- output: `/upgrade/baseline_candidates.json` (baseline) or `/upgrade/backporting_candidates.json` (backporting)

#### Specification

Go over candidates, upgrade each script one by one with the LLM → send to execution if upgrade is done and there is a free GPU.

- Model: GPT-5.2-2025-12-11

1. File-level:

- i. Convert notebooks (`.ipynb`) to python (`.py`) files via nbconvert.
- ii. Execute the `.py` file inside Docker.
- iii. On failure, prompt the LLM w/ the exception to update code to current APIs without altering algorithmic structure (data flow, feature engineering steps, etc.).

1. Cell-level:

- i. Execute the notebook w/o caring the errors (use `--ExecutePreprocessor.allow_errors=True`).
- ii. If errors occur, localize the first failing cell.
- iii. Prompt the LLM with all cells up to (and including) the failing one plus the next cell and the exception of the failing cell (no output of other cells); request API-compatible edits with minimal structural change.
- iv. Re-execute; repeat until the notebook runs cleanly or time budget cap is reached.

`<a id="file_execution"></a>`

### Execution (in Docker)

#### Setup

-`docker/Dockerfile.base`+`docker/install_envs.sh`: build the base image

- prerequisite:
- (1)`baseline/nltk_corpora.txt`
- (2)`apiDowngrade/kernel_w_pyVersion.json`
- (3) files under `apiDowngrade/apiDowngradeList/*.txt`
- (4) docker image `gcr.io/kaggle-gpu-images/python`

#### Baseline

-`baseline/run_baseline_w_time_parallel.py`: execute notebooks for baseline in parallel, capture wall-clock runtime, logs, and exit status, and create the targeted virtual environment for each script in the container with pinned toolchains

- feature: saves output back to the kaggle script
- output:

-`script_out/{competition}/{version}/*.ipynb`: the folder that contains all notebooks w/ sub dirs {competition} and {version} to help visualization

-`script_out/{competition}/result.json`

-`script_out_allINone`: the folder that contains all notebooks w/o sub dirs

-`csv_output/*.csv`: output CSVs

-`executable_files_w_timer_gpu_{gpu_id}.json`: saved results of each thread

-`executable_files_w_timer_parrallel.json`: integrated results from `executable_files_w_timer_gpu_{gpu_id}.json`

#### Upgrade

-`upgrade/run_upgrade_w_timer_parrallel.py` is similar to baseline but uses the first {n} CPUs to update the candidate code at the file-level/cell-level with LLMs, while the rest of the CPUs execute updated notebooks with both processes running in a lock (concurrently)

- inclusion:
- "plan" = upgrade plan
- "code" = upgraded code
- "fixed_path" = where the upgrade notebooks saved
- "request_time" = response time in second
- "in_tokens" = # input tokens
- "out_tokens" = # output tokens
- "cached_tokens" = # cached tokens
- "model" = request model name
- "full_message" = full response message

#### 
`verification/entrypoint.py` adopt one-sample T-test based on repeated runs (up to 10 executions), classifying a modernization as reproducible when the target score and the reproduced score are from the same distribution with statistical significance (p >= 0.05).

## Results

### Grading

`csv_grader.py` concurrently grades each csv generated by `baseline/run_baseline_w_time_parallel.py` by setting --workers {n}

- prerequisite:
- (1) [MLE-bench](https://github.com/openai/mle-bench)
- (2) csv files and parameters "e.g., --save-dir ../MLEModernizer/results/baseline"
- output: `{args.save_dir}/csv_score.json`
- Note: In upgrade the grading process is included, so the user does not need to run this script

### Selecting Non-Reproducible Notebooks

`verification/entrypoint.py` runs result notebooks repeatedly (up to 10 executions) with one-sample T-test, classifying a modernization as successful when the target score and reproduced scores are from the same distribution with statistical significance (p >= 0.05). Script starts with run_2, as run_1 is the initial run.

- output: `baseline_non_reproducible.json`, `file_non_reproducible.json`, `cell_non_reproducible.json`.


## Data Analysis

Execution outputs are stored in the Hugging Face portion of the replication package. After downloading only `script_out` content, `restore_script_out.py` can restore corresponding files under `csv_output`. Variables `ROOT` (path to the type of result) and `N_WORKERS` (total CPUs used by this task) must be configured before execution.

`dprocess.ipynb` (containing all numbers/figures in the paper) processes results from baseline, backporting, and upgrade runs and produces the following:

- **Raw data & file info**: Collects metadata from fetched competitions (total files, R vs Python scripts, date range) and builds a JSON of file info.
- **Notebook statistics**: Competition/submission distribution; script token counts and long-tailed distribution (histogram, CCDF); replicated vs Kaggle score comparison with per-competition normalization.
- **Classification**: Assigns each notebook to one of five categories—Error-free Reproducible, Error Reproducible, Error-free Non-reproducible, Error Non-reproducible, Failed—for Baseline and Backporting.
- **Sankey flow diagrams**: Transitions between categories (Baseline → Backporting, Baseline → File-level upgrade, Baseline → Cell-level upgrade) with counts and percentages.
- **Average fixes (cell vs file)**: Distribution of number of LLM fixes in cell-level vs file-level upgrade; violin plots; Venn diagram of successfully modernized (replicable) notebooks.
- **Fix count vs file metrics**: Relationship between number of fixes and file size, lines of code, and number of errors (for error and error-free subgroups).
- **Crash/error type analysis**: Top exception types; stacked bars comparing baseline vs upgrade and Error Reproducible vs Error Non-reproducible.
- **Cell- vs file-level performance**: Replication success comparison and edit similarity vs #fix (line chart with band and violin by group: Overall, Timeout, Error, Error-free).

`error_causes.py` analyzes crash outputs from executed notebooks (baseline, backporting, or upgrade) and produces:

- **Crash extraction & normalization**: Extracts error outputs (exception name, value, traceback) from notebook cells; normalizes messages (paths, URLs, numbers) for comparison.
- **Crash type classification**: Maps each crash to a type (e.g. module not found, OOM, io error, tensor shape mismatch, feature name mismatch, key/index/attribute/type/value error, invalid argument, other).
- **Root cause inference**: Assigns a root cause per crash (environment setting, insufficient resource, NB specific, implementation error, data confusion, API misuse, ML model confusion, library cause) using crash type, packages from traceback, and context (notebook execution order, prior errors).
- **ML vs Python**: Flags crashes as ML-related when traceback or message involves ML packages (e.g. numpy, sklearn, tensorflow, torch).
- **Clustering**: Clusters crashes by Jaccard similarity on tokenized normalized error text; outputs top clusters with size, representative messages, and top crash types/root causes per cluster.
- **Outputs**: JSON report and CSV of crash rows; counts by exception type, crash type, root cause, and ML vs Python; co-occurrence of crash type × root cause. Run with `--run-type baseline|downgrade|upgrade`; for upgrade, optionally `--upgrade-type file|cell` and `--last-fix` (analyze last fix only, split by Error Reproducible vs Error Non-reproducible). Writes to `crash_report/`.

# Offline Grading
https://github.com/openai/mle-bench. Git clone https://github.com/openai/mle-bench.git (Recommanded) or unzip mle-bench-grader.tar.gz and then follow the instruction to install, e.g., pip install -e . and mlebench prepare --all, etc.

The per-fix CSV outputs for each file-level and cell-level modernization (i.e., `./results/upgrade/{gpt_cell, gpt_file, oss_file}/csv_output/fix_{fix_number}` for fix_number ∈ {1,…,16}) are included in this replication package.

We have also recorded the score for each fix---consistent with MLE-Bench requirement---in `./results/upgrade/{gpt_cell, gpt_file, oss_file}/executable_files_w_timer_parrallel.json`, and we have included all reproducible CSVs at `./results/upgrade/{gpt_cell, gpt_file, oss_file}/csv_output/{notebookName}.csv`, where the notebookName is made of `{competitionName}_{userName}_{submissionName}_{versionNumber}_C1.ipynb`.

If the per-fix CSV outputs are required, we would recommand replicating the corresponding notebook(s) to generate the CSV again.