# In the Search for Truth: Refining and Exploring Variability in Neuroimaging Pipelines
Youenn Merel Jourdan, Hege Spieker, Camille Maumet, Mathieu Acher

## Repository structure
- `case_study_1` contains the data, results and code related to case study 1
- `case_study_2` contains the data, results and code related to case study 2
- `cross_case_study` contains the data, results and code related to cross case study analysis

### Code
Notebooks used for data post-processing analysis are stored in each directory
- `normalize.ipynb` is for data normalization
- `auditory_filtering.ipynb` is for filtering of valid / invalid configs (RQ2)
- `regression_analysis.ipynb` is for regression decision tree learning (RQ3)
- `classifier_analysis.ipynb` is for classifier decision tree learning (RQ4)
- `cost.ipynb` is for computational cost (RQ5)

### Data

Task-fMRI dataset is available in data/auditory. It was downloaded from https://www.fil.ion.ucl.ac.uk/spm/data/auditory/

For the `data` directory in each `case_study_[1,2]` directory :
- `model/full_` is the UVL model used for sampling
- `regression` contains test and training subset with correlation to precompiled average images
  - used by `regression_analysis.ipynb`
- `dataset.csv` is the sample of 1000 configuration (+ 1 reference) with correlation to average image and to reference
- `normalized_dataset.csv` is the sample in which some categorical values (e.g., FWHM values) have been converted to continuous values (for decision tree learning)
  - produced by `normalize.ipynb`
- `correlations.csv` (zipped) is the pairwise Spearman correlation matrix for all configurations (+ average image)
- `invalid_dataset.csv` and `valid_dataset.csv` is the sample of 1000 configuration classified as valid or invalid
    - produced by `auditory_filtering.ipynb`

### Results
The `results` directory in each `case_study_[1,2]` contains intermediate results and figures.

# Sampling

The code used for this part is available at https://github.com/Inria-Empenn/fmri_feature_model

## Pull & install project
``` shell
git clone https://github.com/Inria-Empenn/fmri_feature_model.git
cd fmri_feature_model
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Sample
Randomly sample 1000 configurations divided into 20 files (+ reference configuration)
``` sh
python sample.py --nconfig 1000 --parts 20
```

# Pipelines execution & postprocessing

The code used for this part is available at https://github.com/Inria-Empenn/fmri-conf-runner

## Pull & install project
``` shell
git clone https://github.com/Inria-Empenn/fmri-conf-runner.git
cd fmri-conf-runner
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Build docker image
``` shell
docker build . -t fmri-conf-runner
```
Final `fmri-conf-runner` image size is approximately 6.5 GB

Alternatively, you can directly pull latest image from GitHub : `docker pull ghcr.io/inria-empenn/fmri-confs-runner:latest`

## Pipelines execution

Change `/local/path/to/...` to your local paths

- `/local/path/to/data` : Will be mapped to `/data` in the container. This folder must contains
   - the `auditory` dataset/subfolder
   - `data_desc.json` file
- `/local/path/to/results` : This folder must exists. Will be mapped to `/results` in the container.
- `/local/path/to/workdir` : This folder must exists. Will be mapped to `/workdir` in the container.
- `/local/path/to/configs` : This folder must contains configuration CSV files (in this example `config.csv` and `config_ref.csv`). Will be mapped to `/configs` in the container.

``` sh
docker run -u root -v "/local/path/to/data:/data" -v "/local/path/to/results:/results" -v "/local/path/to/workdir:/work" -v "/local/path/to/configs:/configs" fmri-conf-runner python -u run.py --configs "/configs/config.csv" --data /data/data_desc.json --ref /configs/config_ref.csv
```

On Abaca (Inria cluster), use `run_configs.sh`
```sh
oarsub -S -n fmri-conf-runner ./run_configs.sh
```

## Postprocessing

Change `/local/path/to/...` to your local paths

- `/local/path/to/results` : This folder must contains the outputs of the pipeline execution step. Will be mapped to `/results` in the container.

``` sh
docker run -u root -v "/local/path/to/results:/results" fmri-conf-runner python -u postprocess.py --results "/results"
```

On Abaca (Inria cluster), use `postprocess.sh`
```sh
oarsub -S -n postprocess ./postprocess.sh
```





