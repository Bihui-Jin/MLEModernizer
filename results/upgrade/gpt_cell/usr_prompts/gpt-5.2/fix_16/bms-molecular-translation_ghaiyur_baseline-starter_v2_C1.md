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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
import random

try:
    from rapidfuzz.distance import Levenshtein as _RF_Levenshtein
    from rapidfuzz.process import cdist as _RF_cdist

    _HAVE_RAPIDFUZZ = True
except Exception:
    _RF_Levenshtein = None
    _RF_cdist = None
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

_TEST_LIST = list(_TEST_STRINGS)


@lru_cache(maxsize=1_000_000)
def _cached_distance(a, b):
    if a <= b:
        return _distance_fast(a, b)
    return _distance_fast(b, a)


def _mean_scores_batch(strings):
    if not strings:
        return np.empty(0, dtype=np.float32)

    if _HAVE_RAPIDFUZZ:
        mat = _RF_cdist(strings, _TEST_LIST, scorer=_RF_Levenshtein.distance)
        sums = mat.sum(axis=1, dtype=np.float64)
        return (sums / _TP_LEN).astype(np.float32, copy=False)

    dist_fn = _cached_distance
    out = np.empty(len(strings), dtype=np.float32)
    for i, s in enumerate(strings):
        total = 0
        for t in _TEST_STRINGS:
            total += 0 if s == t else dist_fn(s, t)
        out[i] = float(total) / _TP_LEN
    return out


@lru_cache(maxsize=200_000)
def _mean_score_cached(s: str) -> float:
    if _HAVE_RAPIDFUZZ:
        row = _RF_cdist([s], _TEST_LIST, scorer=_RF_Levenshtein.distance)
        return float(row.sum(dtype=np.float64)) / _TP_LEN
    else:
        dist_fn = _cached_distance
        total = 0
        for t in _TEST_STRINGS:
            total += 0 if s == t else dist_fn(s, t)
        return float(total) / _TP_LEN


@lru_cache(maxsize=20_000)
def _dist_row_to_test_cached(s: str) -> np.ndarray:
    if _HAVE_RAPIDFUZZ:
        row = _RF_cdist([s], _TEST_LIST, scorer=_RF_Levenshtein.distance).astype(
            np.int32, copy=False
        )[0]
        return row
    else:
        dist_fn = _cached_distance
        row = np.empty(_TP_LEN, dtype=np.int32)
        for j, t in enumerate(_TEST_STRINGS):
            row[j] = 0 if s == t else dist_fn(s, t)
        return row


def _ensure_scored(strings):
    if not strings:
        return
    seen = set()
    uniq = []
    for s in strings:
        if s in seen:
            continue
        seen.add(s)
        if s not in _mean_score_cached.cache_info().__dict__:
            uniq.append(s)

    if not uniq:
        return

    if _HAVE_RAPIDFUZZ:
        means = _mean_scores_batch(uniq)
        for s, m in zip(uniq, means):
            _mean_score_cached.__wrapped__(
                s
            )  # ensure function exists; no-op for cache population
    else:
        for s in uniq:
            _mean_score_cached(s)
    return


def _get_dist_row_to_test(s: str) -> np.ndarray:
    return _dist_row_to_test_cached(s)


def _get_mean_score(s: str) -> float:
    return _mean_score_cached(s)


def select_best(population, n_best):
    """Score the given population against a sample from the total training population."""
    pop = population
    n = len(pop)

    if n == 0:
        return [], float("inf")

    if _HAVE_RAPIDFUZZ:
        scores = _mean_scores_batch(pop)
    else:
        _ensure_scored(pop)
        scores = np.fromiter(
            (_mean_score_cached(p) for p in pop), dtype=np.float32, count=n
        )

    if n_best >= n:
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
select_best(initial_population, 1)




## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2253092885.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mselect_best[0m[0;34m([0m[0minitial_population[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/111521643.py[0m in [0;36mselect_best[0;34m(population, n_best)[0m
[1;32m    147[0m         [0mscores[0m [0;34m=[0m [0m_mean_scores_batch[0m[0;34m([0m[0mpop[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    148[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 149[0;31m         [0m_ensure_scored[0m[0;34m([0m[0mpop[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    150[0m         scores = np.fromiter(
[1;32m    151[0m             [0;34m([0m[0m_mean_score_cached[0m[0;34m([0m[0mp[0m[0;34m)[0m [0;32mfor[0m [0mp[0m [0;32min[0m [0mpop[0m[0;34m)[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m,[0m [0mcount[0m[0;34m=[0m[0mn[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/111521643.py[0m in [0;36m_ensure_scored[0;34m(strings)[0m
[1;32m    102[0m         [0mseen[0m[0;34m.[0m[0madd[0m[0;34m([0m[0ms[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    103[0m         [0;31m# Only warm if not in cache already[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 104[0;31m         [0;32mif[0m [0ms[0m [0;32mnot[0m [0;32min[0m [0m_mean_score_cached[0m[0;34m.[0m[0mcache_info[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__dict__[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    105[0m             [0;31m# cache_info doesn't expose keys; we just collect and batch compute then store via direct calls below[0m[0;34m[0m[0;34m[0m[0m
[1;32m    106[0m             [0muniq[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0ms[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'CacheInfo' object has no attribute '__dict__'

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

        pop_half = population  # already sorted best-half list
        children = [None] * half
        for i in range(half):
            p1 = pop_half[randint(0, half - 1)]
            p2 = pop_half[randint(0, half - 1)]
            children[i] = xo(p1, p2)
        population.extend(children)

        for _ in range(MUTATIONS):
            mutant = randint(0, INIT_POP_SIZE - 1)
            population[mutant] = mut(population[mutant])

        print(f"Generation: {gen} : {best_score_half}")

    return select_best(population, 1)


best_string, score = genetic_algorithm()

print(best_string[0], score)
