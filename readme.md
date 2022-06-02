# Xing-lab imaging analysis pipeline


## Environment installation guide
### 
- envrionment files are stored in envs/

```
conda create -n yourEnv python=3.7
conda activate yourEnv
pip install -U -r envs/tf1_requirements.txt
```
Note that python=3.7 is necesssary to support tensorflow 1.15 version. Higer versions of python do not exist when tensorflow==1.15.0 was released.

Create conda environment for cellprofiler4. Note that currently cellprofiler4 only supports 3.8, so it cannot be added to the tf1 environment we just created directly.
```
conda env create -f envs/cp4.yml 
```

## Slurm Script Usage
- Use script paths relative to this repository root
  - submit sample slurm jobs at root directory of this repository
- hacky tricks on PITT CSB cluster
  - please submit in conda **base** environment or without conda env activated, or the job conda env will not be activated environment correctly and your job is probably going to **fail**.
  - in slurm scripts, use **source** instead of **conda** cmd to activate environments. **source** activation method should have been deprecated and **conda** is preferred. However... Anyway this is our cluster status.


## Install Precommit Hook  
**You would like to ensure that the code you push has good code style**  
**This step enables pre-commit to check and auto-format your style before every git commit.**
### Install (once)  
`pip install pre-commit`  
`pre-commit install`  
### format all code (rare usage)  
`pre-commit run --all`


## Development guide
### Avoid copying code files and push identical code
### Avoid copying other package's python code directly