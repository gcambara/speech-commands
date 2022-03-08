import argparse
import os
import yaml

def parse_arguments(stage='train'):
    parser = argparse.ArgumentParser(description='Speech Commands Detection')
    # Basic options
    parser.add_argument('--config', default='', help='path to the config file')
    parser.add_argument('--data', default='', help='path to the data dir where Google Speech Commands is to be found')
    parser.add_argument('--run_dir', default='', help='path to the directory where output files shall be generated')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--precision', type=int, default=32, help='type precision of tensors: double precision (64), full precision (32), half precision (16)')
    parser.add_argument('--resume_from_ckpt', default=None, help='path to the checkpoint to resume training from')

    # Cuda options
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', help='use CUDA if GPUs are available')
    parser.add_argument('--use_cpu', dest='use_cuda', action='store_false', help='do not use CUDA, use CPU')
    parser.set_defaults(use_cuda=True)
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs to use, if available')
    parser.add_argument('--accelerator', default='ddp', help='name of the multi GPU accelerator to use: dp | ddp | ddp_spawn | ddp2 | horovod')
    parser.add_argument('--ddp_find_unused_parameters', dest='ddp_find_unused_parameters', action='store_true', help='find unused parameters in DDP accelerator')
    parser.add_argument('--ddp_dont_find_unused_parameters', dest='ddp_find_unused_parameters', action='store_false', help='do not find unused parameters in DDP accelerator')
    parser.set_defaults(ddp_find_unused_parameters=False)

    # Data loader options
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--batch_size_dev', type=int, default=32, help='development batch size')
    parser.add_argument('--batch_size_test', type=int, default=32, help='test batch size')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='shuffle train set at every epoch')
    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false', help='do not shuffle train set at every epoch')
    parser.set_defaults(shuffle=True)
    parser.add_argument('--shuffle_dev', dest='shuffle_dev', action='store_true', help='shuffle dev set at every epoch')
    parser.add_argument('--no_shuffle_dev', dest='shuffle_dev', action='store_false', help='do not shuffle dev set at every epoch')
    parser.set_defaults(shuffle_dev=False)
    parser.add_argument('--shuffle_test', dest='shuffle_test', action='store_true', help='shuffle test set at every epoch')
    parser.add_argument('--no_shuffle_test', dest='shuffle_test', action='store_false', help='do not shuffle test set at every epoch')
    parser.set_defaults(shuffle_test=False)
    parser.add_argument('--num_workers', type=int, default=0, help='how many subprocesses to use for data loading')
    parser.add_argument('--limit_train_batches', type=float, default=1.0, help='only use this percentage of samples for training')
    parser.add_argument('--num_labels', type=int, default=35, help='how many labels to use with Speech Commands: 10 | 20 | 35')

    # Audio processing options
    parser.add_argument('--waveform', dest='waveform', action='store_true', help='set waveform as the audio input type')
    parser.add_argument('--no_waveform', dest='waveform', action='store_false', help='do not set waveform as the audio input type')
    parser.set_defaults(waveform=True)
    parser.add_argument('--mfcc', dest='mfcc', action='store_true', help='set MFCCs as the audio input type')
    parser.add_argument('--no_mfcc', dest='mfcc', action='store_false', help='do not set MFCCs as the audio input type')
    parser.set_defaults(mfcc=False)
    parser.add_argument('--chunk_size', type=int, default=16000, help='audio chunk size in sample points for training')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate')
    parser.add_argument('--n_fft', type=int, default=1024, help='number of frequency bins in the STFT')
    parser.add_argument('--n_mels', type=int, default=128, help='number of mels')
    parser.add_argument('--win_length', type=float, default=0.025, help='window length in seconds, for STFT computation')
    parser.add_argument('--hop_length', type=float, default=0.010, help='hop length in seconds, for STFT computation')

    # Model options
    parser.add_argument('--wav_norm', default='layernorm', help='normalization to apply to waveforms: none | ')
    parser.add_argument('--featurizer', default='log-mfsc', help='name of the featurizer to use: mfsc | log-mfsc')
    parser.add_argument('--featurizer_post_norm', default='instancenorm2d', help='normalization to apply after feature extraction: instancenorm2d')
    parser.add_argument('--classifier', default='lenet', help='architecture name for the model to be trained: lenet |perceiver')
    parser.add_argument('--loss', default='cross-entropy', help='loss to use: cross-entropy')

    # Training options
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_scheduler', default='constant', help='learning rate scheduler, options: constant | step_lr')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='multiplicative factor for the learning rate')
    parser.add_argument('--lr_step_size', type=int, default=30, help='number of epochs to wait before applying gamma to the learning rate')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum number of epochs for training')
    parser.add_argument('--overfit_batches', type=float, default=0.0, help='number of batches to overfit')

    # Optimizer option
    parser.add_argument('--optimizer', default='adamw', help='optimizer: adam | adamw')
    parser.add_argument('--beta1', type=float, default=0.900, help='beta 1 for optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta 1 for optimizer')
    parser.add_argument('--optimizer_eps', type=float, default=1e-08, help='epsilon for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for optimizer')

    # Logger options
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='use tensorboard logger')
    parser.add_argument('--no_tensorboard', dest='tensorboard', action='store_false', help='do not use tensorboard logger')
    parser.set_defaults(tensorboard=True)
    parser.add_argument('--log_model_params', dest='log_model_params', action='store_true', help='log model params and gradients')
    parser.add_argument('--no_log_model_params', dest='log_model_params', action='store_false', help='do not log model params and gradients')
    parser.set_defaults(log_model_params=False)

    # Perceiver options
    parser.add_argument('--prc_input_channels', type=int, default=128, help='perceiver input channels')
    parser.add_argument('--prc_input_axis', type=int, default=1, help='perceiver input axis')
    parser.add_argument('--prc_num_freq_bands', type=int, default=6, help='perceiver num freq bands')
    parser.add_argument('--prc_max_freq', type=float, default=10., help='perceiver max freq')
    parser.add_argument('--prc_depth', type=int, default=6, help='perceiver depth')
    parser.add_argument('--prc_num_latents', type=int, default=256, help='perceiver num latents')
    parser.add_argument('--prc_latent_dim', type=int, default=512, help='perceiver latent dim')
    parser.add_argument('--prc_cross_heads', type=int, default=1, help='perceiver cross_heads')
    parser.add_argument('--prc_latent_heads', type=int, default=8, help='perceiver latent heads')
    parser.add_argument('--prc_cross_dim_head', type=int, default=64, help='perceiver cross dim head')
    parser.add_argument('--prc_latent_dim_head', type=int, default=64, help='perceiver latent dim head')
    parser.add_argument('--prc_attn_dropout', type=float, default=0.0, help='perceiver attention dropout')
    parser.add_argument('--prc_ff_dropout', type=float, default=0.0, help='perceiver feedforward dropout')
    parser.add_argument('--prc_weight_tie_layers', type=int, default=0, help='boolean perceiver weight tie layers')
    parser.add_argument('--prc_fourier_encode_data', type=int, default=1, help='boolean perceiver fourier encode data')
    parser.add_argument('--prc_self_per_cross_attn', type=int, default=2, help='perceiver self per cross attention')

    args = parser.parse_args()
    if args.config != '':
        return yaml_to_args(args, stage)
    else:
        return args

def yaml_to_args(args, stage):
    assert os.path.isfile(args.config), f"Config yaml file not found at: {args.config}"
    params = yaml.safe_load(open(args.config))[stage]

    for key, value in params.items():
        vars(args)[key] = value

    return args