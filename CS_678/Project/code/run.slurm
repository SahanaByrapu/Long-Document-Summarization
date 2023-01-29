#!/bin/bash
##SBATCH --account=cs678fl22
#SBATCH --job-name=bart
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=12

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=mbharadw@gmu.edu 

#SBATCH --mem=150GB
#SBATCH --time=0-02:30:00


#SBATCH --output=bart.%j.out
#SBATCH --error=bart.%j.err


## load modules
module load gnu10
module load python
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers
pip install datasets==1.0.2
pip install torchmetrics
pip install sentencepiece
pip install contractions
pip install seaborn
pip install rouge
rm seq2seq_trainer.py
wget https://raw.githubusercontent.com/huggingface/transformers/main/examples/legacy/seq2seq/seq2seq_trainer.py
pip install rouge_score
conda install -c pytorch torchvision cudatoolkit=10.1 pytorch
## Execute your script
#python main_roberta.py
#python main_backup.py
python text_rank.py
