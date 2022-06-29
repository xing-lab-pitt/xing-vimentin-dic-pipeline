# Xing-lab imaging analysis pipeline


## How to use this repository

To clone this repository on cluster and setup your github account on CSB cluster, please follow the instructions here: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent  

Please follow the instructions [in guide.md](./guide.md)


## Slurm Script Conventions and Tricks
- Assume script paths relative to this repository root
  - submit sample slurm jobs at root directory of this repository
- hacky tricks on PITT CSB cluster
  - please submit in conda **base** environment or without conda env activated, or the job conda env will not be activated environment correctly and your job is probably going to **fail**.
  - in slurm scripts, use **source** instead of **conda** cmd to activate environments. **source** activation method should have been deprecated and **conda** is preferred according to the conda official website. However... Anyway this is our current cluster status.


## Install Precommit Hook (For developers/contributors only) 
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
