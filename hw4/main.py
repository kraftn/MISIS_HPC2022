import argparse
import os
import math

import cupy as cp
from PIL import Image


def create_kernel():
    kernel_str = r'''
        extern "C" __global__ void convolution(const float* input, const int input_height, const int input_width, 
                                               const float* conv, const int conv_height, const int conv_width, 
                                               float* output) {
            int block_idx = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
            int idx = block_idx * (blockDim.x * blockDim.y * blockDim.z);
            idx += threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
            
            if (idx < input_width * input_height) {
                int i_row = idx / input_width;
                int i_col = idx % input_width;
                
                int is_border = i_row < (conv_height / 2) || i_row >= (input_height - conv_height / 2);
                is_border = is_border || i_col < (conv_width / 2) || i_col >= (input_width - conv_width / 2);
                if (is_border) {
                    output[idx] = input[idx];
                } else {
                    float s = 0.0f;
                    for (int i_row_conv = 0; i_row_conv < conv_height; ++i_row_conv) {
                        for (int i_col_conv = 0; i_col_conv < conv_width; ++i_col_conv) {
                            s += conv[i_row_conv * conv_width + i_col_conv];
                        }
                    }
                    output[idx] = 0.0f;
                    for (int i_row_conv = 0; i_row_conv < conv_height; ++i_row_conv) {
                        int shift_row = i_row_conv - conv_height / 2;
                        for (int i_col_conv = 0; i_col_conv < conv_width; ++i_col_conv) {
                            int shift_col = i_col_conv - conv_width / 2;
                            
                            int i_input_flatten = (i_row + shift_row) * input_width + (i_col + shift_col);
                            int i_conv_flatten = i_row_conv * conv_width + i_col_conv;
                            
                            output[idx] += input[i_input_flatten] * conv[i_conv_flatten];
                        }
                    }
                    output[idx] /= s;
                }
            }
        }
    '''
    return cp.RawKernel(kernel_str, 'convolution')


def conv2d(image: Image, conv: cp.ndarray):
    image = image.convert('L')
    image = cp.asarray(image, dtype=cp.float32)
    kernel = create_kernel()
    output = cp.zeros_like(image, dtype=cp.float32)

    n_elements = image.shape[0] * image.shape[1]
    grid_dim = math.ceil((n_elements / 1024) ** (1 / 3))
    kernel((grid_dim, grid_dim, grid_dim), (1024,),
           (image, image.shape[0], image.shape[1], conv, conv.shape[0], conv.shape[1], output))

    output = output.astype(cp.uint8)
    return Image.fromarray(output.get(), mode='L')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    args = parser.parse_args()

    image = Image.open(args.image_path)
    conv1 = cp.ones((11, 11), dtype=cp.float32)
    conv2 = cp.full((3, 3), -1, dtype=cp.float32)
    conv2[1, 1] = 9

    if not os.path.exists('data'):
        os.mkdir('data')

    for i_conv, conv in enumerate([conv1, conv2]):
        conv2d(image, conv).save(f'data/result_{i_conv + 1}.png')


if __name__ == '__main__':
    cp.cuda.Stream.null.synchronize()
    main()
