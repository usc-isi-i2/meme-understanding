# Environment setup
```
conda create -n meme-understanding python=3.8 -y
conda activate meme-understanding
```

# Dependencies
```
conda install pytorch torchvision torchaudio cudatoolkit=11.4 -c pytorch
conda install -c huggingface transformers -y
conda install -c conda-forge scikit-learn=0.20.0 -y
pip3 install -U scikit-learn
pip install accelerate -y
pip install jupyter notebook -U
conda install -c conda-forge tabulate
pip install joblib
pip install timm
```
# Optional
* `pip install nvitop`