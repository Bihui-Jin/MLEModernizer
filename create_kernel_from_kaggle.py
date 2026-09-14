from tqdm import tqdm
import re,os, time
import datetime
import ast
import html
import json
import warnings
import tempfile
import nbformat
import subprocess
from pathlib import Path
from nbformat.v4 import new_notebook, new_code_cell
from multiprocessing import Pool, Manager, RLock
import multiprocessing
warnings.filterwarnings("ignore", category=SyntaxWarning)
from glob import glob
import pandas as pd
import codecs

def init_pool(l):
    """Initialize the worker process with a shared lock for tqdm."""
    tqdm.set_lock(l)

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
# def kernelById(id):
#     return kernels.loc[id]

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
        & (private_score.notna())
        & (private_score.ne(0))
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

def kernel_key_from_version(version):
    author_user_id = pd.to_numeric(
        pd.Series([version.get("AuthorUserId")]),
        errors="coerce"
    ).iloc[0]
    script_id = pd.to_numeric(
        pd.Series([version.get("ScriptId")]),
        errors="coerce"
    ).iloc[0]
    version_number = pd.to_numeric(
        pd.Series([version.get("VersionNumber")]),
        errors="coerce"
    ).iloc[0]
    if pd.isna(author_user_id) or pd.isna(script_id) or pd.isna(version_number):
        return None

    user_ids = pd.to_numeric(users["Id"], errors="coerce")
    user_rows = users.loc[user_ids == author_user_id]
    

    kernel_ids = pd.to_numeric(kernels["Id"], errors="coerce")
    kernel_rows = kernels.loc[kernel_ids == script_id]
    if kernel_rows.empty:
        return None

    username = author_user_id if user_rows.empty else user_rows.iloc[0].get("UserName")  
    current_url_slug = kernel_rows.iloc[0].get("CurrentUrlSlug")
    if pd.isna(username) or pd.isna(current_url_slug):
        return None

    return f"{username}_{current_url_slug}_v{int(version_number)}_C1.html"

def kernel_id_from_entity_key(kernel_key):
    kernel_id = str(kernel_key).split("_", 1)[0].split(".", 1)[0]
    kernel_id_num = pd.to_numeric(pd.Series([kernel_id]), errors="coerce").iloc[0]
    if pd.isna(kernel_id_num):
        return None
    return int(kernel_id_num)


def kaggle_kernel_info():
    with open("./verification/scriptIds.json") as f:
        scriptIds_byCompt = json.load(f)

    scriptIds_byCompetition = {}
    for bfile in glob("./baseline/notebooks_2/*.ipynb"):
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

    adding_byCompt = {}
    with open("./fetch/competitions.txt") as f:
        for line in f:
            collected_scriptIds = scriptIds_byCompetition.get(line.strip(), {}) 
            comptId = competitionByName(line.strip()).Id
            teamIds = teamByCompetitionId(comptId)

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
                    & script_language_ids.isin([2, 8, 9, 14])
                ]
                .dropna()
                .astype("int64")
                .unique()
                .tolist()
            )

            PythonKernelIds = [p for p in kernelIds if ftype(idToPath(p)) == ".ipynb"]

            python_kernel_ids_set = {str(x) for x in PythonKernelIds}  # from Kaggle tables
            collected_scriptIds_set = {str(v) for v in collected_scriptIds.values()}  # parsed from html

            # In Kaggle but not in collected
            new_in_python = python_kernel_ids_set - collected_scriptIds_set
            # In collected but not in Kaggle (reverse direction)

            
            adding_byCompt[line.strip()] = list(new_in_python)
    nested_list = [v for v in adding_byCompt.values()]
    all_to_add = [item for sublist in nested_list for item in sublist]
    return all_to_add

def process_competition(args):
    """Process one kernel version using Meta Kaggle CSV relationships."""
    kernel_id = args
    kernel_id_num = pd.to_numeric(pd.Series([kernel_id]), errors="coerce").iloc[0]
    if pd.isna(kernel_id_num):
        return {}

    kernel_id_int = int(kernel_id_num)
    version_ids = pd.to_numeric(versions["Id"], errors="coerce")
    version_rows = versions.loc[version_ids == kernel_id_num]
    if version_rows.empty:
        return {}

    version = version_rows.iloc[0]
    script_language_id = pd.to_numeric(
        pd.Series([version.get("ScriptLanguageId")]),
        errors="coerce"
    ).iloc[0]
    if pd.isna(script_language_id) or int(script_language_id) not in {2, 8, 9, 14}:
        return {}

    runtime_ms = pd.to_numeric(
        pd.Series([version.get("RunningTimeInMilliseconds")]),
        errors="coerce"
    ).iloc[0]
    version_evaluation_dt = pd.to_datetime(
        pd.Series([version.get("EvaluationDate")]),
        errors="coerce"
    ).iloc[0]
    version_creation_dt = pd.to_datetime(
        pd.Series([version.get("CreationDate")]),
        errors="coerce"
    ).iloc[0]

    submission_kernel_ids = pd.to_numeric(
        submissions_df["SourceKernelVersionId"], errors="coerce"
    )
    kernel_submissions = submissions_df.loc[
        submission_kernel_ids == kernel_id_num
    ].copy()

    if kernel_submissions.empty:
        return {}

    kernel_submissions["_submission_dt"] = pd.to_datetime(
        kernel_submissions["SubmissionDate"],
        format="%m/%d/%Y",   # example: 05/12/2014
        errors="coerce"
    )
    if "ScoreDate" in kernel_submissions.columns:
        kernel_submissions["_score_dt"] = pd.to_datetime(
            kernel_submissions["ScoreDate"],
            errors="coerce"
        )
    else:
        kernel_submissions["_score_dt"] = pd.NaT
    kernel_submissions = kernel_submissions.sort_values(
        "_submission_dt", ascending=False, na_position="last"
    )

    def same_calendar_day(left, right):
        if pd.isna(left) or pd.isna(right):
            return False
        return left.date() == right.date()

    def submission_version_match_rank(submission):
        if same_calendar_day(submission.get("_score_dt"), version_evaluation_dt):
            return 2
        if same_calendar_day(submission.get("_submission_dt"), version_creation_dt):
            return 1
        return 0

    kernel_path = idToPath(kernel_id_int)
    file_deps = set()
    if kernel_path:
        try:
            if ftype(kernel_path) == ".ipynb":
                with open(kernel_path, "r", encoding="utf-8") as fp:
                    notebook = nbformat.read(fp, as_version=4)

                notebook.cells = [
                    cell
                    for cell in notebook.cells
                    if cell.get("cell_type") == "code"
                ]

                file_deps = get_imports_from_file(notebook, path=kernel_path)
        except Exception as e:
            print(f"Failed to inspect imports for {kernel_id}: {e}")

    datasets = []
    source_kernel_version_ids = pd.to_numeric(src["KernelVersionId"], errors="coerce")
    dataset_ids = src.loc[
        source_kernel_version_ids == kernel_id_num,
        "SourceDatasetVersionId"
    ]
    for dataset_id in dataset_ids:
        dataset_id_num = pd.to_numeric(pd.Series([dataset_id]), errors="coerce").iloc[0]
        datasets.append(int(dataset_id_num) if pd.notna(dataset_id_num) else str(dataset_id))
    datasets = sorted(set(datasets))

    entity = {}
    selected_match_ranks = {}
    for idx, submission in kernel_submissions.iterrows():
        team_id = pd.to_numeric(pd.Series([submission.get("TeamId")]), errors="coerce").iloc[0]
        if pd.isna(team_id):
            continue

        team_ids = pd.to_numeric(teams["Id"], errors="coerce")
        team = teams.loc[team_ids == team_id]
        if team.empty:
            continue

        competition_id = pd.to_numeric(
            pd.Series([team.iloc[0].get("CompetitionId")]),
            errors="coerce"
        ).iloc[0]
        if pd.isna(competition_id):
            continue

        competition_ids = pd.to_numeric(competitions_df["Id"], errors="coerce")
        competition = competitions_df.loc[competition_ids == competition_id]
        if competition.empty:
            continue

        competi = competition.iloc[0].get("Slug")

        kernel_key = kernel_key_from_version(version)
        if kernel_key is None:
            continue
        entity_key = (competi, kernel_key)
        match_rank = submission_version_match_rank(submission)
        if competi in entity and kernel_key in entity[competi]:
            if match_rank <= selected_match_ranks.get(entity_key, 0):
                continue

        dt = submission["_submission_dt"]
        kernel_entity = {
            "kernelId": kernel_id_int,
            "api": sorted(file_deps),
            "datasets": datasets,
        }

        if pd.notna(dt):
            kernel_entity["year"] = int(dt.year)
            kernel_entity["month"] = int(dt.month)
            kernel_entity["date"] = int(dt.day)
            kernel_entity["datetime"] = (
                dt.tz_localize(datetime.timezone.utc)
                if dt.tzinfo is None
                else dt.tz_convert(datetime.timezone.utc)
            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        if pd.notna(runtime_ms):
            kernel_entity["runtime"] = float(runtime_ms) / 1000

        private_score = pd.to_numeric(
            pd.Series([submission.get("PrivateScoreFullPrecision")]),
            errors="coerce"
        ).iloc[0]
        # public_score = pd.to_numeric(
        #     pd.Series([submission.get("PublicScoreFullPrecision")]),
        #     errors="coerce"
        # ).iloc[0]
        kernel_entity["ps"] = 0.0
        if pd.notna(private_score):
            kernel_entity["ps"] = float(private_score)
        if kernel_entity["ps"] == 0.0:
            continue
        # elif pd.notna(public_score):
        #     kernel_entity["ps"] = float(public_score)

        entity.setdefault(competi, {})[kernel_key] = kernel_entity
        selected_match_ranks[entity_key] = match_rank

    return entity
    
def get_imports_from_file(notebook, path=None):
    """Use pigar to detect dependencies"""
    deps = set()
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        
        with open(workdir / "script.ipynb", "w", encoding="utf-8") as file:
            nbformat.write(notebook, file)
        
        # Retry up to 3 times if pigar times out
        for attempt in range(3):
            try:
                # pigar generate <folder>
                # print("Running pigar...")
                result = subprocess.run(
                    ["pigar", "generate", "--auto-select", tmpdir],
                    # ["pipreqs", "--scan-notebooks", tmpdir],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    # input="*\n" + "y\n" + "*\n" * 20,
                    input="y\n" * 30,
                )
                # print("Pigar completed.")

                # print("STDOUT:", result.stdout)
                # print("STDERR:", result.stderr)
                # print("Return code:", result.returncode)

                # print(get_folders(tmpdir))
                # Parse requirements.txt output
                req_file = Path(tmpdir) / "requirements.txt"
                # print(req_file.exists())
                if req_file.exists():
                    for line in req_file.read_text().splitlines():
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        pkg = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                        if pkg:
                            deps.add(pkg)

                    return deps
                # If execution finished without timeout but no file was created, stop retrying
                # break
                time.sleep(0.5)
                continue
            except subprocess.TimeoutExpired:
                # If it timed out, loop will continue to next attempt
                if attempt == 2:
                    with open("kernel_timeout.json", "a", encoding="utf-8") as json_file:
                        json_file.write(f"{path}\n")
                continue

            except Exception as e:
                print(f"pigar failed: {e}")
                pass
    
    return deps

if __name__ == '__main__':

    filename = "../.kaggle/KernelVersions.csv"
    versions = pd.read_csv(filename,low_memory=False)

    filename = "../.kaggle/Kernels.csv"
    kernels = pd.read_csv(filename)

    filename = "../.kaggle/KernelVersionDatasetSources.csv"
    src = pd.read_csv(filename)
    src_kernel_version_id = pd.to_numeric(src["KernelVersionId"], errors="coerce")

    filename = "../.kaggle/Teams.csv"
    teams = pd.read_csv(filename,low_memory=False)

    filename = "../.kaggle/Submissions.csv"
    submissions_df = pd.read_csv(filename,low_memory=False)

    filename = "../.kaggle/Competitions.csv"
    competitions_df = pd.read_csv(filename)

    filename = "../.kaggle/Users.csv"
    users = pd.read_csv(filename,low_memory=False)

    all_to_add = kaggle_kernel_info()
    print(f"Total kernels to add: {len(all_to_add)}")
    # print(f"Total kernels to add: {len(set(all_to_add))}")

    # # Create a shared lock for tqdm
    # # This prevents workers from writing to the terminal simultaneously
    # tqdm_lock = RLock()
    # tqdm.set_lock(tqdm_lock)

    # # Create pool with n workers, and always join it before exiting.
    # pool = Pool(processes=32, initializer=init_pool, initargs=(tqdm_lock,))
    # try:
    #     # Use imap for progress bar
    #     results = list(tqdm(
    #         pool.imap(process_competition, all_to_add),
    #         total=len(all_to_add),
    #         desc="Competitions",
    #         position=0  # Force main bar to stay at the top
    #     ))
    # except BaseException:
    #     pool.terminate()
    #     raise
    # else:
    #     pool.close()
    # finally:
    #     pool.join()
    
    # # Merge all results
    # entity = {}
    # for result in results:
    #     for competi, kernels in result.items():
    #         entity.setdefault(competi, {}).update(kernels)
        
    # # Save the entity dictionary into a JSON file.
    # with open("kernels.json", "w", encoding="utf-8") as json_file:
    #     json.dump(entity, json_file, indent=4, ensure_ascii=False)