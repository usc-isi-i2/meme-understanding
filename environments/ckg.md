# Environment setup
```
conda create -n meme-understanding python=3.8 -y
conda activate meme-understanding
```

# Dependencies
```
conda install pytorch torchvision torchaudio cudatoolkit=11.4 -c pytorch
conda install -c huggingface transformers
conda install -c conda-forge scikit-learn
pip install accelerate
pip install jupyter notebook -U
conda install -c conda-forge tabulate
```
# Optional
* `pip install nvitop`