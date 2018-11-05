CS 7641 - Machine Learning - Fall 2018 (Unsupervised Learning and Dimensionality Reduction)
-------------------------------------------------------------------------------------------

## Overview
This repo contains the Analysis, Code, Visuals, and Datasets for the Unsupervised Learning and Dimensionality Reduction assignment as part of the Fall 2018 CS7641 course at Georgia Tech. The code for this assignment can be found on GitHub at https://github.com/levikanwischer/gatech-omsa-cs7641-assignment03. Below are instructions about how to run the code needed to replicate portions this analysis.


## Setup
Code is written in Python 3.7 using "pipenv" (https://pipenv.readthedocs.io/en/latest/) for versioning and reproducibility. All analysis was performed on MacOS High Sierra (Intel Core M). Analysis writeup is written in an R Notebook, and through knitr converted to LaTeX, then to PDF with Pandoc. To reproduce the analysis, you must install Python 3.7 locally, shell into the pipenv environment ($ pipenv shell), then execute each python file (bash$ python FILENAME.py). A "runall.py" has been added to allow for executing the bulk of the analysis. Additionally, each .py file contains a main function allowing just that portion of code to be run. Note running these filed may take some time to complete.


## Repo
At the top level of this repo lives the final analysis, the Rmarkdown & latex files used to generate the final analysis, and this README. Below that, the repo is split into four sub directories; code, logs, img, and data. code is where the python source code lives, as well as the pipenv files. You will need to "cd" into this folder to activate pipenv and run the analysis. Within logs you can find some misc outputs from code processing. Next, in the img repo lives outputs from running the analysis. Some of these visuals were used in the final report, but others were generated purely for personal analysis. Lastly, the data repo contains curated csv files of various datasets used during this assignment. Greater description of these can be found in the "Datasets" section below.


## Datasets
Five different datasets were initially included in this repo. Only two of them were used as the primary motivations for this assignment. Each dataset was sourced from the UCI Machine Learning Repository, and only slight modifications to the datasets were made. These modifications include cleaning up column names, and making response variables more human readable. Any other manipulation to that data can be found within the source code itself.

The two datasets used in this assignment are "Cardiotocography" (https://archive.ics.uci.edu/ml/datasets/Cardiotocography) referred to as "cardio", and "HTRU2" (https://archive.ics.uci.edu/ml/datasets/HTRU2) referred to as "pulsar".

Note, all datasets were extracted by hand with only light cleaning of column names and response values. No values were changed (flipped) in this process, merely reformatted for viewing purposes (ie. binary to proper names). All other changes/modifications/manipulation to datasets can be found within the code.
