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

# 5. Target score

69.3

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import random

from Levenshtein import distance


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3148678864.py in <cell line: 0>()
      3 import random
      4 
----> 5 from Levenshtein import distance

ModuleNotFoundError: No module named 'Levenshtein'

## === cell 1
SEED = 2021

random.seed = SEED


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

initial_population = df["InChI"].sample(INIT_POP_SIZE, random_state=SEED).reset_index(drop=True)


## === cell 5

def cross_over(parent1, parent2):
    """ Splice two parents together at a random point to generate a child. """
    min_len = min(len(parent1), len(parent2))
    splice_idx = random.randint(0, min_len)
    child = parent1[:splice_idx] + parent2[splice_idx:]
    return child
    
def mutate(individual):
    """ Mutate an individual by swapping two characters at a random point."""
    mutation_idx = random.randint(0, len(individual)-2)
    return individual[:mutation_idx] + individual[mutation_idx+1] + individual[mutation_idx] + individual[mutation_idx+2:]

def select_best(population, n_best):
    """ Score the given population against a sample from the total training population. """
    test_population = df["InChI"].sample(TEST_POP_SIZE, random_state=SEED).reset_index(drop=True)
    scores = []
    for p in population:
        score = 0
        for t in test_population:
            score += distance(p,t)
        score = score/TEST_POP_SIZE
        scores.append(score)
    sorted_population = [p for _, p in sorted(zip(scores, population))]
    return sorted_population[:n_best], min(scores)


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
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4123870018.py in <cell line: 0>()
      1 # What is the fittest member of the population to start with?
----> 2 select_best(initial_population, 1)

/tmp/ipykernel_11/759150420.py in select_best(population, n_best)
     20         score = 0
     21         for t in test_population:
---> 22             score += distance(p,t)
     23         score = score/TEST_POP_SIZE
     24         scores.append(score)

NameError: name 'distance' is not defined

## === cell 8
def genetic_algorithm():
    """ This is just meant to be a simple naive baseline. No need to make it more complex that it needs to be. """
    population = initial_population

    for gen in range(GENERATIONS):
        population, _ = select_best(population, INIT_POP_SIZE//2)
        children = []
        for child in range(INIT_POP_SIZE//2):
            children.append(cross_over(population[random.randint(0, INIT_POP_SIZE//2 - 1)], population[random.randint(0, INIT_POP_SIZE//2 - 1)]))
        population.extend(children)
        
        for m in range(MUTATIONS):
            mutant = random.randint(0, INIT_POP_SIZE - 1)
            population[mutant] = mutate(population[mutant])
        
        _, best_score = select_best(population, 1)        
        print(f"Generation: {gen} : {best_score}")
    
    return select_best(population, 1)


## === cell 9
best_string, score = genetic_algorithm()

print(best_string[0], score)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3899327669.py in <cell line: 0>()
----> 1 best_string, score = genetic_algorithm()
      2 
      3 # What single string did we generate?
      4 print(best_string[0], score)

/tmp/ipykernel_11/3757253501.py in genetic_algorithm()
      5     for gen in range(GENERATIONS):
      6         # Select the top half of the population and then fill back to the original population limit by making children.
----> 7         population, _ = select_best(population, INIT_POP_SIZE//2)
      8         children = []
      9         for child in range(INIT_POP_SIZE//2):

/tmp/ipykernel_11/759150420.py in select_best(population, n_best)
     20         score = 0
     21         for t in test_population:
---> 22             score += distance(p,t)
     23         score = score/TEST_POP_SIZE
     24         scores.append(score)

NameError: name 'distance' is not defined

## === cell 10
subm = pd.read_csv('../input/bms-molecular-translation/sample_submission.csv')
subm['InChI'] = best_string[0]
subm.to_csv('submission.csv.gz', compression="gzip", index=False)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3467951095.py in <cell line: 0>()
      1 # Build the submission. The csv is huge and very repitative. .gz files can be submitted directly, so let's compress it.
      2 subm = pd.read_csv('../input/bms-molecular-translation/sample_submission.csv')
----> 3 subm['InChI'] = best_string[0]
      4 subm.to_csv('submission.csv.gz', compression="gzip", index=False)

NameError: name 'best_string' is not defined
