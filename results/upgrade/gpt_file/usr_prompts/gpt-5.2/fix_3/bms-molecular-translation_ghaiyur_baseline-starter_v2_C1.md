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


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    if la < lb:
        a, b = b, a
        la, lb = lb, la

    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = ins if ins < dele else dele
            if sub < cur[j]:
                cur[j] = sub
        prev = cur
    return prev[lb]


distance = levenshtein_distance



## === cell 1
SEED = 42

random.seed(SEED)
np.random.seed(SEED)



## === cell 2
df = pd.read_csv("../input/bms-molecular-translation/train_labels.csv")
df



## === cell 3
TEST_POP_SIZE = 100
INIT_POP_SIZE = 500
MUTATIONS = 50
GENERATIONS = 100



## === cell 4
total_population = len(df)
print(f"total_population: {total_population}")

initial_population = (
    df["InChI"].sample(INIT_POP_SIZE, random_state=SEED).reset_index(drop=True).tolist()
)




## === cell 5
def cross_over(parent1, parent2):
    """Splice two parents together at a random point to generate a child."""
    min_len = min(len(parent1), len(parent2))
    splice_idx = random.randint(0, min_len)
    child = parent1[:splice_idx] + parent2[splice_idx:]
    return child


def mutate(individual):
    """Mutate an individual by swapping two adjacent characters at a random point."""
    if len(individual) < 2:
        return individual
    mutation_idx = random.randint(0, len(individual) - 2)
    return (
        individual[:mutation_idx]
        + individual[mutation_idx + 1]
        + individual[mutation_idx]
        + individual[mutation_idx + 2 :]
    )


_TEST_POPULATION = (
    df["InChI"].sample(TEST_POP_SIZE, random_state=SEED).reset_index(drop=True).tolist()
)

_distance_cache = {}


def _cached_distance(a: str, b: str) -> int:
    key = (a, b)
    try:
        return _distance_cache[key]
    except KeyError:
        d = distance(a, b)
        _distance_cache[key] = d
        return d


def select_best(population, n_best):
    """Score the given population against a sample from the total training population."""
    test_population = _TEST_POPULATION

    scores = np.empty(len(population), dtype=np.float64)

    for i, p in enumerate(population):
        s = 0
        for t in test_population:
            s += _cached_distance(p, t)
        scores[i] = s / TEST_POP_SIZE

    if n_best >= len(population):
        best_idx = np.argsort(scores)
    else:
        part = np.argpartition(scores, n_best - 1)[:n_best]
        best_idx = part[np.argsort(scores[part])]

    sorted_population = [population[i] for i in best_idx]
    return sorted_population[:n_best], float(scores.min())




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
    population = list(initial_population)

    for gen in range(GENERATIONS):
        population, _ = select_best(population, INIT_POP_SIZE // 2)

        children = []
        for _ in range(INIT_POP_SIZE // 2):
            children.append(
                cross_over(
                    population[random.randint(0, INIT_POP_SIZE // 2 - 1)],
                    population[random.randint(0, INIT_POP_SIZE // 2 - 1)],
                )
            )
        population.extend(children)

        for _ in range(MUTATIONS):
            mutant = random.randint(0, INIT_POP_SIZE - 1)
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
subm.to_csv("submission.csv", index=False)
print(subm.head())
print("Wrote submission.csv with shape:", subm.shape)
