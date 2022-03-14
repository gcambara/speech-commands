'''
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 10 \
  --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
'''
import argparse
import collections
import os
import torch

def average_checkpoints(ckpt_paths):
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(ckpt_paths)

    for ckpt_path in ckpt_paths:
        state = torch.load(ckpt_path, map_location='cpu')
        model_params = state['state_dict']

        if new_state is None:
            new_state = state

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state['state_dict'] = averaged_params
    return new_state

def get_n_last_checkpoints(src, n_ckpts):
    ckpts = {}
    for dirpath, dirnames, filenames in os.walk(src):
        for filename in filenames:
            if filename.endswith('.ckpt') and filename != 'last.ckpt':
                ckpt_path = os.path.join(dirpath, filename)
                ckpt_info = filename.split('-')
                epoch = int(ckpt_info[0].replace('epoch=', ''))
                ckpts[ckpt_path] = epoch

    return dict(sorted(ckpts.items(), key=lambda item: item[1], reverse=True)[:n_ckpts])

def main():
    parser = argparse.ArgumentParser(
        description="Generates a new checkpoint by averaging N pretrained ones."
    )

    parser.add_argument('--src', required=True, help='Path to the directory containing the checkpoints')
    parser.add_argument('--dst', required=True, help='Path to the directory to write the output at')
    parser.add_argument('--n_ckpts', type=int, default=10, help='Number of checkpoints to average')

    args = parser.parse_args()
    print(args)

    assert os.path.isdir(args.src), f"Source directory not found! {args.src}"
    ckpt_paths = get_n_last_checkpoints(args.src, args.n_ckpts)

    new_ckpt = average_checkpoints(ckpt_paths)

    os.makedirs(args.dst, exist_ok=True)
    out_path = os.path.join(args.dst, f'avg_{args.n_ckpts}_ckpts.ckpt')
    torch.save(new_ckpt, out_path)

    # new_state = average_checkpoints(args.inputs)
    # with PathManager.open(args.output, "wb") as f:
    #     torch.save(new_state, f)
    # print("Finished writing averaged checkpoint to {}".format(args.output))


if __name__ == "__main__":
    main()