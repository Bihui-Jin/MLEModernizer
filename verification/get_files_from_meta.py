import json
from pathlib import Path
from glob import glob

compare = {'jigsaw-toxic-comment-classification-challenge': False,
            'google-quest-challenge': False,
            'detecting-insults-in-social-commentary': False,
            'tabular-playground-series-may-2022': False,
            'denoising-dirty-documents': True,
            'aerial-cactus-identification': False,
            'tweet-sentiment-extraction': False,
            'cassava-leaf-disease-classification': False,
            'aptos2019-blindness-detection': False,
            'random-acts-of-pizza': False,
            'new-york-city-taxi-fare-prediction': True,
            'nomad2018-predict-transparent-conductors': True,
            'spooky-author-identification': True,
            'mlsp-2013-birds': False,
            'plant-pathology-2020-fgvc7': False,
            'champs-scalar-coupling': True,
            'uw-madison-gi-tract-image-segmentation': False,
            'histopathologic-cancer-detection': False,
            'bms-molecular-translation': True,
            'predict-volcanic-eruptions-ingv-oe': True,
            'h-and-m-personalized-fashion-recommendations': False,
            'smartphone-decimeter-2022': True,
            'hubmap-kidney-segmentation': False,
            'whale-categorization-playground': False,
            'text-normalization-challenge-russian-language': False,
            'nfl-player-contact-detection': False,
            'hms-harmful-brain-activity-classification': True,
            'tensorflow2-question-answering': False,
            'osic-pulmonary-fibrosis-progression': False,
            'plant-pathology-2021-fgvc8': False,
            'alaska2-image-steganalysis': False,
            'hotel-id-2021-fgvc8': False,
            'multi-modal-gesture-recognition': True,
            'herbarium-2020-fgvc7': False,
            'vesuvius-challenge-ink-detection': False,
            '3d-object-detection-for-autonomous-vehicles': False,
            'tabular-playground-series-dec-2021': False,
            'inaturalist-2019-fgvc6': True,
            'iwildcam-2020-fgvc7': False,
            'seti-breakthrough-listen': False,
            'icecube-neutrinos-in-deep-ice': True,
            'herbarium-2022-fgvc9': False,
            'herbarium-2021-fgvc8': False,
            'vinbigdata-chest-xray-abnormalities-detection': False,
            'rsna-breast-cancer-detection': False,
            'us-patent-phrase-to-phrase-matching': False,
            'chaii-hindi-and-tamil-question-answering': False,
            'leaf-classification': True,
            'statoil-iceberg-classifier-challenge': True,
            'tgs-salt-identification-challenge': False,
            'dog-breed-identification': True,
            'lmsys-chatbot-arena': True,
            'learning-agency-lab-automated-essay-scoring-2': False,
            'ventilator-pressure-prediction': True,
            'dogs-vs-cats-redux-kernels-edition': True,
            'facebook-recruiting-iii-keyword-extraction': False,
            'jigsaw-unintended-bias-in-toxicity-classification': False,
            'ranzcr-clip-catheter-line-classification': False,
            'text-normalization-challenge-english-language': False,
            'billion-word-imputation': True,
            'freesound-audio-tagging-2019': False,
            'the-icml-2013-whale-challenge-right-whale-redux': False,
            'petfinder-pawpularity-score': True,
            'kuzushiji-recognition': False,
            'iwildcam-2019-fgvc6': False,
            'imet-2020-fgvc7': False,
            'siim-isic-melanoma-classification': False,
            'rsna-miccai-brain-tumor-radiogenomic-classification': False,
            'siim-covid19-detection': False,
            'rsna-2022-cervical-spine-fracture-detection': True,
            'google-research-identify-contrails-reduce-global-warming': False,
            'stanford-covid-vaccine': True,
            'tensorflow-speech-recognition-challenge': False,
            'AI4Code': False,
            'cdiscount-image-classification-challenge': False
    }

import os
from glob import glob
import pandas as pd
import codecs
import json
from pathlib import Path

def idToPath(id):
    f = str(id).zfill(10)
    prefix = '../.kaggle/meta-kaggle-code/'+f[0:4]+'/'+f[4:7]+'/'+str(id)+'.*'
    g = glob(prefix)
    if len(g) == 1:
        return g[0]
    return ""

def ftype(p):
    parts = os.path.splitext(p)
    if len(parts) != 2:
        return ""
    return parts[1]

def pathToId(p):
    parts = os.path.splitext(os.path.basename(p))
    if len(parts) != 2:
        return ""
    if parts[0] != "":
        return int(parts[0])
    return None

def rawSource(path):
    return Path(path).read_text()

def ipynbSource(path):
    f = codecs.open(path, 'r')
    source = f.read()
    
    jsource = json.loads(rawSource(path))
    cells = []
    for cell in jsource['cells']:
        cells.append(cell['source'])
    return cells

def sourceCode(path):
    t = ftype(path)
    if t == ".ipynb":
        return ipynbSource(path)
    return [Path(path).read_text()]

def sourceCodeById(id):
    p = idToPath(id)
    if p != "":
        return sourceCode(p)
    return None

def versionById(id):
    return versions.loc[id]
def kernelById(id):
    return kernels.loc[id]

src_kernel_version_id = pd.to_numeric(src["KernelVersionId"], errors="coerce")
def sourceDatasetByVersionId(id):
    kid = pd.to_numeric(pd.Series([id]), errors="coerce").iloc[0]
    return src.loc[src_kernel_version_id == kid, "SourceDatasetVersionId"].tolist()

def teamByCompetitionId(id):
    return teams.loc[teams["CompetitionId"] == id, "Id"].tolist()

def submissionByKernelId(ids):
    if not ids:
        return []

    # Vectorized numeric validation keeps retrieval fast on large tables.
    public_score = pd.to_numeric(submissions_df["PublicScoreFullPrecision"], errors="coerce")
    private_score = pd.to_numeric(submissions_df["PrivateScoreFullPrecision"], errors="coerce")

    mask = (
        submissions_df["TeamId"].isin(ids)
        & (public_score.notna() | private_score.notna())
    )

    kernel_version_ids = (
        pd.to_numeric(submissions_df.loc[mask, "SourceKernelVersionId"], errors="coerce")
        .dropna()
        .astype("int64")
        .unique()
    )

    return kernel_version_ids.tolist()

def competitionByName(name):
    return competitions_df.loc[competitions_df['Slug'] == name].iloc[0]

with open("scriptIds.json") as f:
    scriptIds_byCompt = json.load(f)

scriptIds_byCompetition = {}
for bfile in glob("../baseline/notebooks/*.ipynb"):
    compt = bfile.split("/")[-1].split("_")[0]
    fname = bfile.split("/")[-1]

    if compt not in scriptIds_byCompetition:
        scriptIds_byCompetition[compt] = {fname:scriptIds_byCompt[compt][fname]}
    scriptIds_byCompetition[compt][fname] = scriptIds_byCompt[compt][fname]

dataset_bound_kernel_ids = set(
    src_kernel_version_id
    .dropna()
    .astype("int64")
    .unique()
    .tolist()
)

submission_dates = pd.to_datetime(
    submissions_df["SubmissionDate"],
    format="%m/%d/%Y",   # example: 05/12/2014
    errors="coerce"
)
cutoff = pd.Timestamp("2025-05-31")  # May 31, 2025

# Pre-normalize versions columns once for speed.
version_ids = pd.to_numeric(versions["Id"], errors="coerce")
version_runtime_seconds = pd.to_numeric(
    versions["RunningTimeInMilliseconds"], errors="coerce"
) / 1000
script_language_ids = pd.to_numeric(versions["ScriptLanguageId"], errors="coerce")

all_script = 0
adding = 0
removal = 0
adding_byCompt = {}
removal_byCompt = {}
with open("../fetch/competitions.txt") as f:
    for line in f:
        collected_scriptIds = scriptIds_byCompetition.get(line.strip(), {}) 
        comptId = competitionByName(line.strip()).Id
        reported_submissions = competitions_df.loc[competitions_df["Id"] == comptId, "TotalSubmissions"].iloc[0]
        teamIds = teamByCompetitionId(comptId)
        matched_submissions = submissions_df.loc[submissions_df["TeamId"].isin(teamIds) & submission_dates.lt(cutoff)]

        has_score = submissionByKernelId(teamIds)
        sourceKernelIds = [
            sourceKernelId
            for sourceKernelId in has_score
            if sourceKernelId not in dataset_bound_kernel_ids
        ]

        kernelIds = (
            version_ids[
                version_ids.isin(sourceKernelIds)
                & version_runtime_seconds.le(600)
            ]
            .dropna()
            .astype("int64")
            .unique()
            .tolist()
        )

        # PythonKernelIds = (
        #     version_ids[
        #         version_ids.isin(kernelIds)
        #         & script_language_ids.isin([8, 9, 14])
        #     ]
        #     .dropna()
        #     .astype("int64")
        #     .unique()
        #     .tolist()
        # )

        PythonKernelIds = [p for p in kernelIds if ftype(idToPath(p)) == ".ipynb"]

        # print(
        #     f"{line.strip()}, "
        #     f"#Reported_Submissions: {reported_submissions}, #matched_submissions: {len(matched_submissions)}, #has_score: {len(has_score)}, #no_ext_ds: {len(sourceKernelIds)}, "
        #     f"#runtime<=600s: {len(kernelIds)}, #Notebooks: {len(PythonKernelIds)}"
        # )

        # python_kernel_ids_set = set(pd.to_numeric(pd.Series(PythonKernelIds), errors="coerce").dropna().astype("int64"))
        # version_id_series = pd.to_numeric(versions["Id"], errors="coerce")
        # matched_versions = versions.loc[version_id_series.isin(python_kernel_ids_set)].copy()
        # KernelScriptIds = KernelScriptIds = (
        #     pd.to_numeric(matched_versions["ScriptId"], errors="coerce")
        #     .dropna()
        #     .astype("int64")
        #     .unique()
        #     .tolist()
        # )

        python_kernel_ids_set = {str(x) for x in PythonKernelIds}  # from Kaggle tables
        collected_scriptIds_set = {str(v) for v in collected_scriptIds.values()}  # parsed from html

        # In Kaggle but not in collected
        new_in_python = python_kernel_ids_set - collected_scriptIds_set
        # In collected but not in Kaggle (reverse direction)
        missing_in_python = collected_scriptIds_set - python_kernel_ids_set

        print(
            f"{line.strip()}, "
            f"#Collection: {len(collected_scriptIds)}, "
            f"#Kaggle: {len(PythonKernelIds)}, "
            f"+{len(new_in_python)}, "
            f"-{len(missing_in_python)}"
        )
        
        adding_byCompt[line.strip()] = list(new_in_python)
        removal_byCompt[line.strip()] = list(missing_in_python)

        adding+=len(new_in_python)
        removal+=len(missing_in_python)
        all_script += len(PythonKernelIds)
print(f"Total scripts with no external dataset used, valid scores, runtime <= 600s, and Script is a notebook: {all_script}")
print(f"Total scripts to add: {(adding)}")
print(f"Total scripts to remove: {(removal)}")