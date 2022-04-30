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
    parser.add_argument('--batch_size', type=int, default=192, help='training batch size')
    parser.add_argument('--batch_size_dev', type=int, default=128, help='development batch size')
    parser.add_argument('--batch_size_test', type=int, default=128, help='test batch size')
    parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='shuffle train set at every epoch')
    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false', help='do not shuffle train set at every epoch')
    parser.set_defaults(shuffle=True)
    parser.add_argument('--shuffle_dev', dest='shuffle_dev', action='store_true', help='shuffle dev set at every epoch')
    parser.add_argument('--no_shuffle_dev', dest='shuffle_dev', action='store_false', help='do not shuffle dev set at every epoch')
    parser.set_defaults(shuffle_dev=False)
    parser.add_argument('--shuffle_test', dest='shuffle_test', action='store_true', help='shuffle test set at every epoch')
    parser.add_argument('--no_shuffle_test', dest='shuffle_test', action='store_false', help='do not shuffle test set at every epoch')
    parser.set_defaults(shuffle_test=False)
    parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
    parser.add_argument('--limit_train_batches', type=float, default=1.0, help='only use this percentage of samples for training')
    parser.add_argument('--num_labels', type=int, default=35, help='how many labels to use with Speech Commands: 10 | 20 | 35')
    parser.add_argument('--only_noise_p', type=float, default=0.1, help='percentage of the samples in the batch that will be silence (only applicable to 10 labels case)')
    parser.add_argument('--weighted_sampler', type=int, default=0, help='apply weighted sampler')
    parser.add_argument('--weighted_sampler_unk_weight', type=float, default=0.01, help='apply weighted sampler fixed weight for unknown class')

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
    parser.add_argument('--n_mels', type=int, default=64, help='number of mels')
    parser.add_argument('--n_mfcc', type=int, default=40, help='number of MFCCs')
    parser.add_argument('--deltas', type=int, default=0, help='deltas to compute in feature extraction')
    parser.add_argument('--win_length', type=float, default=0.025, help='window length in seconds, for STFT computation')
    parser.add_argument('--hop_length', type=float, default=0.010, help='hop length in seconds, for STFT computation')
    parser.add_argument('--chop_size', type=int, default=160, help='size of chops in wave chops')

    # Model options
    parser.add_argument('--wav_norm', default='layernorm', help='normalization to apply to waveforms: none | ')
    parser.add_argument('--featurizer', default='log-mfsc', help='name of the featurizer to use: mfsc | log-mfsc | mfcc | log-mfcc | waveform | wavechops')
    parser.add_argument('--featurizer_post_norm', default='instancenorm2d', help='normalization to apply after feature extraction: instancenorm2d')
    parser.add_argument('--classifier', default='perceiver', help='architecture name for the model to be trained: kwt | lenet | perceiver | perceiver_w2v2')
    parser.add_argument('--loss', default='cross-entropy', help='loss to use: cross-entropy')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='amount of smoothing for cross-entropy loss')
    parser.add_argument('--class_weights', type=int, default=0, help='whether to use class weights at cross entropy loss or not')
    parser.add_argument('--class_weights_batches', type=int, default=20, help='for how many batches to compute the class weights')

    # Training options
    parser.add_argument('--model', default='base', help='type of model to train: base | distil')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--lr_scheduler', default='step_lr', help='learning rate scheduler, options: constant | step_lr | cosine')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='multiplicative factor for the learning rate')
    parser.add_argument('--lr_step_size', type=int, default=1, help='number of epochs to wait before applying gamma to the learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='minimum learning rate for cosine scheduler')
    parser.add_argument('--lr_max_epochs', type=int, default=400, help='final epoch for cosine scheduler')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0, help='number of epochs to perform warmup with')
    parser.add_argument('--max_epochs', type=int, default=400, help='maximum number of epochs for training')
    parser.add_argument('--overfit_batches', type=float, default=0.0, help='number of batches to overfit')

    # Testing option
    parser.add_argument('--ckpt', default='', help='path to the model file to load')
    parser.add_argument('--cfg', default='', help='path to the config model file to load')

    # Optimizer option
    parser.add_argument('--optimizer', default='adamw', help='optimizer: adam | adamw')
    parser.add_argument('--beta1', type=float, default=0.900, help='beta 1 for optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta 1 for optimizer')
    parser.add_argument('--optimizer_eps', type=float, default=1e-08, help='epsilon for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay for optimizer')

    # Logger options
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='use tensorboard logger')
    parser.add_argument('--no_tensorboard', dest='tensorboard', action='store_false', help='do not use tensorboard logger')
    parser.set_defaults(tensorboard=True)
    parser.add_argument('--log_model_params', dest='log_model_params', action='store_true', help='log model params and gradients')
    parser.add_argument('--no_log_model_params', dest='log_model_params', action='store_false', help='do not log model params and gradients')
    parser.set_defaults(log_model_params=False)

    # Perceiver options
    parser.add_argument('--prc_input_channels', default=64, help='perceiver input channels')
    parser.add_argument('--prc_input_axis', default=1, help='perceiver input axis')
    parser.add_argument('--prc_num_freq_bands', default=12, help='perceiver num freq bands')
    parser.add_argument('--prc_max_freq', default=10., help='perceiver max freq')
    parser.add_argument('--prc_depth', default=6, help='perceiver depth')
    parser.add_argument('--prc_num_latents', default=128, help='perceiver num latents')
    parser.add_argument('--prc_latent_dim', default=256, help='perceiver latent dim')
    parser.add_argument('--prc_cross_heads', default=1, help='perceiver cross_heads')
    parser.add_argument('--prc_latent_heads', default=8, help='perceiver latent heads')
    parser.add_argument('--prc_cross_dim_head', default=64, help='perceiver cross dim head')
    parser.add_argument('--prc_latent_dim_head', default=64, help='perceiver latent dim head')
    parser.add_argument('--prc_attn_dropout', default=0.0, help='perceiver attention dropout')
    parser.add_argument('--prc_ff_dropout',  default=0.0, help='perceiver feedforward dropout')
    parser.add_argument('--prc_weight_tie_layers', default=1, help='boolean perceiver weight tie layers')
    parser.add_argument('--prc_fourier_encode_data', default=1, help='boolean perceiver fourier encode data')
    parser.add_argument('--prc_self_per_cross_attn', default=1, help='perceiver self per cross attention')
    parser.add_argument('--prc_classification', default='mean', help='perceiver classification mode: mean | cls')

    # Perceiver wav2vec2.0 options
    parser.add_argument('--prc_freeze_latents', type=int, default=0, help='freezes perceiver latents since the beginning')
    parser.add_argument('--use_w2v2_latents', type=int, default=0, help='use latents from the wav2vec2.0 latent extractor')
    parser.add_argument('--latent_weight_norm', default='kaiming', help='normalization for the wav2vec2.0 quantizer weights that are used as perceiver latents: none | kaiming')
    parser.add_argument('--latent_process_mode', default='none', help='mode to process the latents: none | avg_pool | random_sample | pile_up')
    parser.add_argument('--clusterize_latents', type=int, default=0, help='whether to clusterize latents or not, if yes, will use prc_num_latents as the number of clusters')

    # KWT options
    parser.add_argument('--kwt_depth', type=int, default=12, help='KWT depth')
    parser.add_argument('--kwt_dim', type=int, default=64, help='KWT dim')
    parser.add_argument('--kwt_heads', type=int, default=1, help='KWT heads')
    parser.add_argument('--kwt_mlp_dim', type=int, default=256, help='KWT heads')
    parser.add_argument('--kwt_dropout', type=float, default=0.0, help='KWT dropout')
    parser.add_argument('--kwt_dim_head', type=int, default=64, help='KWT head dimension')
    parser.add_argument('--kwt_patch_x', type=int, default=1, help='KWT x patch dim')
    parser.add_argument('--kwt_patch_y', type=int, default=40, help='KWT y patch dim')
    parser.add_argument('--kwt_channels', type=int, default=1, help='KWT channels')
    parser.add_argument('--kwt_pool', default='cls', help='KWT pool')

    # Distil model options
    parser.add_argument('--teacher', default='facebook/wav2vec2-base', help='teacher model path or URL, set to none to skip it')
    parser.add_argument('--teacher_zoo', default='huggingface', help='zoo to take the teacher from: huggingface | none')
    parser.add_argument('--w2v2_target_state', type=int, default=-1, help='select the wav2vec2 hidden state that shall be used as teacher target, projected state will be used if set to -1')

    # Data augmentation options
    parser.add_argument('--time_shift_p', type=float, default=0.3, help='probability of applying time shift augment')
    parser.add_argument('--time_shift_range', type=float, default=0.1, help='time shift range in seconds')
    parser.add_argument('--resample_p', type=float, default=0.3, help='probability of applying resampling augment')
    parser.add_argument('--resample_min', type=float, default=0.85, help='minimum fraction of the original sampling rate')
    parser.add_argument('--resample_max', type=float, default=1.15, help='maximum fraction of the original sampling rate')
    parser.add_argument('--background_noise_p', type=float, default=0.8, help='probability of applying background noises')
    parser.add_argument('--background_snr_min', type=float, default=5.0, help='min SNR of the background noise')
    parser.add_argument('--background_snr_max', type=float, default=30.0, help='max SNR of the background noise')
    parser.add_argument('--specaugment_p', type=float, default=0.7, help='probability of applying specaugment')
    parser.add_argument('--time_masks', type=int, default=2, help='number of time masks')
    parser.add_argument('--time_mask_size', type=int, default=25, help='maximum size of the time mask')
    parser.add_argument('--freq_masks', type=int, default=2, help='number of time masks')
    parser.add_argument('--freq_mask_size', type=int, default=7, help='maximum size of the freq mask')

    # Performance measuring info
    parser.add_argument('--warmup_fwd', type=int, default=10, help='number of forward warmups before start measuring time')
    parser.add_argument('--n_fwd', type=int, default=100, help='number of forwards to measure time')

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