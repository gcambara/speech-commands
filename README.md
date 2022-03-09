# speech-commands
A benchmark for Speech Commands recognition with PyTorch-Lightning

## Training
Train a Speech Commands recognition module with:
```
python train.py --data ./data --run_dir ./experiments/test/ --num_workers 32
```

### Train Perceiver model
```
python train.py --data ./data --run_dir ./experiments/perceiver --num_workers 16 --num_labels 10 --max_epochs 1 --classifier perceiver
```
