# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Provided with images of chemicals, predict the corresponding International Chemical Identifier (InChI) text string of the image.

## Metric
Mean [Levenshtein distance](http://en.wikipedia.org/wiki/Levenshtein_distance) between the InChi strings you submit and the ground truth InChi values.

## Submission Format
For each `image_id` in the test set, you must predict the InChi string of the molecule in the corresponding image. The file should contain a header and have the following format:

```
image_id,InChI
00000d2a601c,InChI=1S/H2O/h1H2
00001f7fc849,InChI=1S/H2O/h1H2
000037687605,InChI=1S/H2O/h1H2
etc.
```

## Dataset
- **train/** - the training images, arranged in a 3-level folder structure by `image_id`
- **test/** - the test images, arranged in the same folder structure as `train/`
- **train_labels.csv** - ground truth InChi labels for the training images
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (68 lines)
            extra_approved_InChIs.csv (9998712 lines)
            extra_approved_InChIs.csv.zip (428.7 MB)
            sample_submission.csv (484839 lines)
            sample_submission.csv.zip (4.1 MB)
            test.zip (883.2 MB)
            train.zip (3.5 GB)
            train_labels.csv (1939349 lines)
            train_labels.csv.zip (102.5 MB)
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
            test/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
            train/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
        input/
            description.md (68 lines)
            extra_approved_InChIs.csv (9998712 lines)
            extra_approved_InChIs.csv.zip (428.7 MB)
            sample_submission.csv (484839 lines)
            sample_submission.csv.zip (4.1 MB)
            test.zip (883.2 MB)
            train.zip (3.5 GB)
            train_labels.csv (1939349 lines)
            train_labels.csv.zip (102.5 MB)
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
            test/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
            train/
                0/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                1/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 14 other folders
                ... and 15 other folders
        working/
            bms-molecular-translation/
                description.md (68 lines)
                extra_approved_InChIs.csv (9998712 lines)
                ... and 7 other files
                bms-molecular-translation/
                test/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
                train/
                    0/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    1/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    ... and 15 other folders
```

-> data/bms-molecular-translation/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> data/bms-molecular-translation/sample_submission.csv has 484838 rows and 2 columns.
The columns are: image_id, InChI

-> data/bms-molecular-translation/train_labels.csv has 1939348 rows and 2 columns.
The columns are: image_id, InChI

-> data/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> data/sample_submission.csv has 484838 rows and 2 columns.
The columns are: image_id, InChI

-> data/train_labels.csv has 1939348 rows and 2 columns.
The columns are: image_id, InChI

-> input/bms-molecular-translation/extra_approved_InChIs.csv has 9998711 rows and 1 columns.
The columns are: InChI

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import random

try:
    from rapidfuzz.distance import Levenshtein as _RF_Levenshtein

    _HAVE_RAPIDFUZZ = True
except Exception:
    _RF_Levenshtein = None
    _HAVE_RAPIDFUZZ = False

from functools import lru_cache


def distance(a, b, _buf=np.empty(0, dtype=np.int32)):
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    if a == b:
        return 0

    la = len(a)
    lb = len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    if lb > la:
        a, b = b, a
        la, lb = lb, la

    need = lb + 1
    if _buf.size < need:
        _buf = np.empty(need, dtype=np.int32)

    row = _buf[:need]
    row[:need] = np.arange(need, dtype=np.int32)

    for i in range(1, la + 1):
        prev_diag = i - 1
        row0 = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            above = row[j]
            ins = row0 + 1
            dele = above + 1
            sub = prev_diag + (ca != b[j - 1])
            prev_diag = above
            m = ins if ins < dele else dele
            row0 = sub if sub < m else m
            row[j] = row0

    return int(row[lb])


def _distance_fast(a, b):
    a = "" if a is None else str(a)
    b = "" if b is None else str(b)
    if a == b:
        return 0
    if _HAVE_RAPIDFUZZ:
        return int(_RF_Levenshtein.distance(a, b))
    return distance(a, b)




## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)




## === cell 2
df = pd.read_csv(
    "../input/bms-molecular-translation/train_labels.csv",
    usecols=["InChI"],
    engine="c",
)
df




## === cell 3
TEST_POP_SIZE = 800

INIT_POP_SIZE = 500
MUTATIONS = 50
GENERATIONS = 100




## === cell 4
total_population = len(df)
print(f"total_population: {total_population}")

_sample = (
    df["InChI"]
    .sample(INIT_POP_SIZE + TEST_POP_SIZE, random_state=SEED)
    .reset_index(drop=True)
)

_sample = _sample.fillna("").astype(str)

initial_population = _sample.iloc[:INIT_POP_SIZE].tolist()
test_population = _sample.iloc[INIT_POP_SIZE:].tolist()
del _sample
del df




## === cell 5
def cross_over(parent1, parent2):
    """Splice two parents together at a random point to generate a child."""
    min_len = min(len(parent1), len(parent2))
    splice_idx = random.randint(0, min_len)
    child = parent1[:splice_idx] + parent2[splice_idx:]
    return child


def mutate(individual):
    """Mutate an individual by swapping two characters at a random point."""
    if len(individual) < 2:
        return individual
    mutation_idx = random.randint(0, len(individual) - 2)
    return (
        individual[:mutation_idx]
        + individual[mutation_idx + 1]
        + individual[mutation_idx]
        + individual[mutation_idx + 2 :]
    )


_TEST_STRINGS = tuple(test_population)
_TP_LEN = len(_TEST_STRINGS)


@lru_cache(maxsize=1_000_000)
def _cached_distance(a, b):
    if a <= b:
        return _distance_fast(a, b)
    return _distance_fast(b, a)


_DIST_TO_TEST = {}  # str -> np.ndarray[int32] of shape (TP_LEN,)


def _get_dist_row_to_test(s: str) -> np.ndarray:
    row = _DIST_TO_TEST.get(s)
    if row is not None:
        return row

    tp = _TEST_STRINGS
    tp_len = _TP_LEN

    if _HAVE_RAPIDFUZZ:
        row = np.asarray(_RF_Levenshtein.distance(s, tp), dtype=np.int32)
    else:
        dist_fn = _cached_distance
        row = np.empty(tp_len, dtype=np.int32)
        for j, t in enumerate(tp):
            row[j] = 0 if s == t else dist_fn(s, t)

    _DIST_TO_TEST[s] = row
    return row


def select_best(population, n_best):
    """Score the given population against a sample from the total training population."""
    pop = population

    scores = np.empty(len(pop), dtype=np.float32)
    for i, p in enumerate(pop):
        scores[i] = float(_get_dist_row_to_test(p).sum()) / _TP_LEN

    if n_best >= len(pop):
        order = np.argsort(scores, kind="mergesort")
        best = [pop[i] for i in order]
        return best, float(scores[order[0]])

    idx = np.argpartition(scores, n_best - 1)[:n_best]
    idx = idx[np.argsort(scores[idx], kind="mergesort")]
    best = [pop[i] for i in idx]
    return best, float(scores[idx[0]])




## === cell 6
print(cross_over("abcdef", "vwxyz"))
print(cross_over("abcdef", "vwxyz"))
print(cross_over("abcdef", "vwxyz"))
print(cross_over("vwxyz", "abcdef"))
print(cross_over("vwxyz", "abcdef"))
print(cross_over("vwxyz", "abcdef"))

print(mutate("abcdef"))
print(mutate("abcdef"))
print(mutate("abcdef"))
print(mutate("abcdef"))




## === cell 7
for s in initial_population:
    _get_dist_row_to_test(s)

select_best(initial_population, 1)




## === cell 8
def genetic_algorithm():
    """This is just meant to be a simple naive baseline. No need to make it more complex that it needs to be."""
    population = list(initial_population)

    half = INIT_POP_SIZE // 2

    randint = random.randint
    xo = cross_over
    mut = mutate

    for gen in range(GENERATIONS):
        population, best_score_half = select_best(population, half)

        children = [None] * half
        for i in range(half):
            p1 = population[randint(0, half - 1)]
            p2 = population[randint(0, half - 1)]
            c = xo(p1, p2)
            children[i] = c
            _get_dist_row_to_test(c)
        population.extend(children)

        for _ in range(MUTATIONS):
            mutant = randint(0, INIT_POP_SIZE - 1)
            m = mut(population[mutant])
            population[mutant] = m
            _get_dist_row_to_test(m)

        print(f"Generation: {gen} : {best_score_half}")

    return select_best(population, 1)


best_string, score = genetic_algorithm()

print(best_string[0], score)




## === cell 9
subm = pd.read_csv("../input/bms-molecular-translation/sample_submission.csv")
subm["InChI"] = best_string[0]
subm.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", subm.shape)
