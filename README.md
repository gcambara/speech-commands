# speech-commands
A benchmark for Speech Commands recognition with PyTorch-Lightning

## Training
Train a Speech Commands recognition module with:
```
python train.py --data ./data --run_dir ./experiments/train --num_workers 32
```

### Train Perceiver model
```
python train.py --data ./data --run_dir ./experiments/perceiver --num_workers 16 --num_labels 10 --max_epochs 1 --classifier perceiver
```

## Testing
A recommended step is to average the last N checkpoints (for instance, 10 checkpoints), with the following script:
```
python average_checkpoints.py --src <path/to/ckpts/folder> --dst <path/to/output/ckpt/folder> --n_ckpts 10
```

Then, get validation and test metrics with:
```
python test.py --data ./data --cfg <path/to/model/cfg.pt> --ckpt <path/to/model/ckpt> --run_dir ./experiments/test
```