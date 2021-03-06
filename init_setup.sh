conda create --prefix ./env python=3.7 -y && source activate ./env 
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda env export > conda.yaml