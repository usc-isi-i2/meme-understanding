```
!git clone https://<username>:<personal-access-token>@github.com/abhinav-kumar-thakur/meme-understanding
!git config --global user.email "<email>"
!git config --global user.name "<username>"
```

```
%cd meme-understanding
```

```
!pip install transformers[sentencepiece] > ./logs/install_transformers.log
!pip install accelerate > ./logs/install_accelerator.log
!apt install htop -y > ./logs/install_htop.log
!pip install nvitop > ./logs/install_htop.log
```

```
!bash scripts/extract_raw_data.sh ./../drive/MyDrive/MAMI\ DATASET
```
