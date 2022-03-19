# CNN with pytorch

# STEPS

## Step 01 - Create a repository by using template repository
## step 02 - Clone the new repository
## step 03 - Create a conda environment after opening the repository in VSCODE
```
conda create --prefix ./env python=3.7 -y
```
## activate environment
```
conda activate ./env
```
### or
```
source activate ./env
```

## STEP 04- install the requirements
```
pip install -r requirements.txt
```
## step 05- install pytorch 11.3

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
## or use init_setup.sh if not want run step 01 to step 05
### in bash terminal use below command
```
bash init_setup.sh
```
## step 06- install setup.py
```
pip install -e .
``` 
