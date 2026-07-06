# Afriprobe
To train probes first clone the github repo and the masakhane data repo


EXTRACT HIDDEN STATE

python extract/hidden_states.py   --model_name_or_path Davlan/afro-xlmr-large   --model_alias afro-xlmr-large   --languages yor ibo hau swa amh   --splits train validation test   --batch_size 36   --max_length 256   --output_dir /workspace/afriprobe/hidden_states 
  --device cuda   --data_dir masakhane-pos/data


PROBE TRAINING 
python probes/train_layer_sweep.py   --hidden_dir /workspace/afriprobe/hidden_states   --model_alias xlmr   --source_language yor   --target_languages yor ibo hau swa wol  --layers all   --train_split train   --eval_split test   --batch_size 4096   --lr 1e-3   --epochs 30   --output_dir /workspace/afriprobe/probes   --device cuda

CKA ANALYSIS
python analysis/cka.py   --hidden_dir /workspace/afriprobe/hidden_states   --model_alias xlmr   --languages yor ibo hau swa wol   --layers 3   --split test   --max_tokens 5000   --output_dir /workspace/afriprobe/analysis/cka   --device cuda

