# Xing-lab imaging analysis pipeline


## Environment installation guide
### 
- envrionment files are stored in envs/

```
conda create -n yourEnv python=3.7
pip install -U -r envs/tf1_requirements.txt
```
Note that python=3.7 is necesssary to support tensorflow 1.15 version. Higer versions of python do not exist when tensorflow==1.15.0 was released.

Create conda environment for cellprofiler4. Note that currently cellprofiler4 only supports 3.8, so it cannot be added to the tf1 environment we just created directly.
```
conda env create -f envs/cp4.yml 
```


## Development guide