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
    from rapidfuzz.distance.Levenshtein import distance as _rf_distance  # type: ignore

    def distance(a, b):
        if a == b:
            return 0
        if a is None:
            a = ""
        if b is None:
            b = ""
        return int(_rf_distance(str(a), str(b)))

except Exception:
    try:
        from Levenshtein import distance  # type: ignore
    except ModuleNotFoundError:

        def distance(a, b):
            if a == b:
                return 0
            if a is None:
                a = ""
            if b is None:
                b = ""
            a = str(a)
            b = str(b)

            la, lb = len(a), len(b)
            if la == 0:
                return lb
            if lb == 0:
                return la

            if lb > la:
                a, b = b, a
                la, lb = lb, la

            prev = list(range(lb + 1))
            for i in range(1, la + 1):
                ca = a[i - 1]
                curr0 = i
                prev_diag = prev[0]
                prev[0] = curr0
                for j in range(1, lb + 1):
                    prev_up = prev[j]
                    cost = 0 if ca == b[j - 1] else 1
                    ins = prev[j - 1] + 1
                    dele = prev_up + 1
                    sub = prev_diag + cost
                    v = ins if ins < dele else dele
                    v = sub if sub < v else v
                    prev_diag = prev_up
                    prev[j] = v
            return prev[lb]




## === cell 1
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)



## === cell 2
df = pd.read_csv(
    "../input/bms-molecular-translation/train_labels.csv", usecols=["InChI"]
)
df



## === cell 3
TEST_POP_SIZE = 100
INIT_POP_SIZE = 500
MUTATIONS = 50
GENERATIONS = 100



## === cell 4
total_population = len(df)
print(f"total_population: {total_population}")

inchi_all = df["InChI"].astype(str).tolist()

initial_population = (
    pd.Series(inchi_all)
    .sample(INIT_POP_SIZE, random_state=SEED)
    .reset_index(drop=True)
    .tolist()
)

_rng = np.random.RandomState(SEED)


def _sample_test_population():
    idx = _rng.choice(total_population, size=TEST_POP_SIZE, replace=False)
    return [inchi_all[i] for i in idx]


n_select_calls = 1 + 2 * GENERATIONS + 1
TEST_POPULATIONS = [_sample_test_population() for _ in range(n_select_calls)]
_test_pop_ptr = 0

_dist_cache = None




## === cell 5
def cross_over(parent1, parent2):
    """Splice two parents together at a random point to generate a child."""
    min_len = min(len(parent1), len(parent2))
    splice_idx = random.randint(0, min_len)
    child = parent1[:splice_idx] + parent2[splice_idx:]
    return child


def mutate(individual):
    """Mutate an individual by swapping two characters at a random point."""
    mutation_idx = random.randint(0, len(individual) - 2)
    return (
        individual[:mutation_idx]
        + individual[mutation_idx + 1]
        + individual[mutation_idx]
        + individual[mutation_idx + 2 :]
    )


def select_best(population, n_best):
    """Score the given population against a sample from the total training population."""
    global _test_pop_ptr
    test_population = TEST_POPULATIONS[_test_pop_ptr]
    _test_pop_ptr += 1

    dist = distance
    inv_n = 1.0 / TEST_POP_SIZE

    row_cache = {}

    scores = np.empty(len(population), dtype=np.float64)

    tp = test_population

    for i, p in enumerate(population):
        s = 0
        cached = row_cache.get(p)
        if cached is None:
            ss = 0
            for t in tp:
                ss += dist(p, t)
            row_cache[p] = ss
            s = ss
        else:
            s = cached
        scores[i] = s * inv_n

    if n_best == 1:
        best_idx = int(scores.argmin())
        return [population[best_idx]], float(scores[best_idx])

    idx = np.argpartition(scores, n_best - 1)[:n_best]
    idx = idx[np.argsort(scores[idx], kind="mergesort")]
    best_population = [population[int(j)] for j in idx]
    return best_population, float(scores.min())




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




## === cell 8
def genetic_algorithm():
    """This is just meant to be a simple naive baseline. No need to make it more complex that it needs to be."""
    population = initial_population

    half = INIT_POP_SIZE // 2
    for gen in range(GENERATIONS):
        population, _ = select_best(population, half)

        children = [None] * half
        randint = random.randint
        pop = population
        for i in range(half):
            p1 = pop[randint(0, half - 1)]
            p2 = pop[randint(0, half - 1)]
            children[i] = cross_over(p1, p2)

        population.extend(children)

        for _ in range(MUTATIONS):
            mutant = randint(0, INIT_POP_SIZE - 1)
            population[mutant] = mutate(population[mutant])

        _, best_score = select_best(population, 1)
        print(f"Generation: {gen} : {best_score}")

    return select_best(population, 1)




## === cell 9
best_string, score = genetic_algorithm()
print(best_string[0], score)



## === cell 10
subm = pd.read_csv("../input/bms-molecular-translation/sample_submission.csv")
subm["InChI"] = best_string[0]
subm.to_csv("submission.csv.gz", compression="gzip", index=False)
