# In the Search for Truth: Refining and Exploring Variability in Neuroimaging Pipelines
Youenn Merel Jourdan, Hege Spieker, Camille Maumet, Mathieu Acher

Preprint available at https://inria.hal.science/hal-05525807.

## Table of contents
   * [How to cite?](#how-to-cite)
   * [Contents overview](#contents-overview)
   * [Reproducing figures and tables](#reproducing-figures-and-tables)
      * [Table 1](#table-1)
      * [Fig. 1](#fig-1)
      * [Fig. 2](#fig-2)
   * [Reproducing full analysis](#reproducing-full-analysis)

## How to cite?

See [CITATION](CITATION).

# Contents overview

For `data`, `doc`, `figures`, `results`, `src` :
- `case_study_1` contains the data, results and code related to case study 1
- `case_study_2` contains the data, results and code related to case study 2

## src
Notebooks used for data post-processing analysis are stored in each directory
- `normalize.ipynb` is for data normalization
- `auditory_filtering.ipynb` is for filtering of valid / invalid configs (RQ2)
- `regression_analysis.ipynb` is for regression decision tree learning (RQ3)
- `classifier_analysis.ipynb` is for classifier decision tree learning (RQ4)
- `cost.ipynb` is for computational cost (RQ5)

## data

Task-fMRI dataset is available in data/auditory. It was downloaded from https://www.fil.ion.ucl.ac.uk/spm/data/auditory/

For the `data` directory in each `case_study_[1,2]` directory :
- `model/full_pipeline.uvl` is the UVL model used for sampling
- `configs` contains the sampled configs
  - produced by sampling command (see _Sample_ section below)
  - used by config runner (see _Pipelines execution & postprocessing_ section below)
- `regression` contains test and training subset with correlation to precompiled average images
  - used by `regression_analysis.ipynb`
- `dataset.csv` is the sample of 1000 configuration (+ 1 reference) with correlation to average image and to reference
- `normalized_dataset.csv` is the sample in which some categorical values (e.g., FWHM values) have been converted to continuous values (for decision tree learning)
  - produced by `normalize.ipynb`
- `correlations.csv` (zipped) is the pairwise Spearman correlation matrix for all configurations (+ average image)
- `invalid_dataset.csv` and `valid_dataset.csv` is the sample of 1000 configuration classified as valid or invalid
    - produced by `auditory_filtering.ipynb`

## results
The `results` directory contains intermediate results and figures.

## Reproducing figures and tables

### Fig. 3 (Pairwise correlations matrices)

Execute (run all cells) `src/correlations_analysis.ipynb` notebook, then see `Pairwise correlations matrix for both case studies` cell output

### Fig. 4 (Distributions of correlations)

Execute (run all cells) `src/correlations_analysis.ipynb` notebook, then see `Distributions of correlations` cell output 

### Fig. 5 (Regression decision tree learning curves)

Execute (run all cells) `src/regression_analysis.ipynb` notebook, then see `Regression decision tree learning curves` cell output 

### Table 2 (Feature importances)

Latex code for table can be generated with cell `Feature importances latex table code generator` of notebook `src/classifier_analysis.ipynb`
Set `case = 1` and `dataset = 'full'`

### Table 3 (Feature importances)

Latex code for table can be generated with cell `Feature importances latex table code generator` of notebook `src/classifier_analysis.ipynb`
Set `case = 1` and `dataset = 'valid'`

### Fig. 6 (Decision tree)


### Table 4 (Feature importance)

Latex code for table can be generated with cell `Feature importances latex table code generator` of notebook `src/classifier_analysis.ipynb`
Set `case = 2` and `dataset = 'full'`

### Table 5 (Feature importance)

Latex code for table can be generated with cell `Feature importances latex table code generator` of notebook `src/classifier_analysis.ipynb`
Set `case = 2` and `dataset = 'valid'`

### Fig. 7 (Decision tree)

### Fig. 8 (F1 score, clusters, features by clustering threshold)

Execute (run all cells) `src/classifier_analysis.ipynb` notebook, then see `F1-score, clusters, features by clustering threshold` cell output

### Fig. 9 (Feature importance)

Execute (run all cells) `src/classifier_analysis.ipynb` notebook, then see `Feature importance (valid configurations)` cell output

## Reproducing full analysis

### fMRI data

fMRI data used in this experiment can be downloaded at https://www.fil.ion.ucl.ac.uk/spm/data/auditory/
```
Raw functional and structural data (BIDS & NIfTI formats): ZIP archive: MoAEpilot.bids.zip (29Mb)
``` 

### Sampling configuration

Configurations sampled used in this experiment are in the `data/case_study_*/configs` folders of this repository.

To generate your own sample, see `README.md` of the https://github.com/Inria-Empenn/fmri_feature_model repository, also linked as submodule in the `src/case_study_*/fmri_feature_model` of this repository.
Use `splc_2025` tag for case study 1 and `sosym_2026` for case study 2.

### Running configuration

To run sampled configurations on fMRI data, see `README.md` of the https://github.com/Inria-Empenn/fmri-conf-runner repository, also linked as submodule in the `src/case_study_*/fmri_feature_model` of this repository.
Use `splc25` tag for case study 1 and `sosym_2026` for case study 2.

### Post-processing

To run post-processing on pipelines results, see `README.md` of the https://github.com/Inria-Empenn/fmri-conf-runner repository, also linked as submodule in the `src/case_study_*/fmri_feature_model` of this repository.
Use `splc25` tag for case study 1 and `sosym_2026` for case study 2.

### Analysis

Notebooks used for data post-processing analysis are stored in each directory
- `normalize.ipynb` is for data normalization
- `auditory_filtering.ipynb` is for filtering of valid / invalid configs (RQ2)
- `regression_analysis.ipynb` is for regression decision tree learning (RQ3)
- `classifier_analysis.ipynb` is for classifier decision tree learning (RQ4)
- `cost.ipynb` is for computational cost (RQ5)
