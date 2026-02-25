fMRI pipeline Feature Model
===

# Install project
``` sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
Randomly sample 100 configurations (+ reference configuration)
``` sh
python sample.py --nconfig 100
```

Randomly sample 100 configurations divided into 10 files (+ reference configuration)
``` sh
python sample.py --nconfig 100 --parts 10
```