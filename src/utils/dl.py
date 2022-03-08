def compute_conv1d_out_dim(input_length, kernel_size, stride, dilation, padding):
    return int(((input_length + (2 * padding) - (dilation * (kernel_size - 1)) - 1)/stride) + 1)
