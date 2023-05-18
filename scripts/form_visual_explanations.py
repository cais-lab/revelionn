import argparse
import os
import sys
import matplotlib.pyplot as plt
import torch

from revelionn.occlusion import perform_occlusion
from revelionn.utils.model import load_mapping_model

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)


def visualize_mapping_activations(mapping_model_filename, path_to_img, window_size, stride, threads):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    main_module, mapping_module, activation_extractor, transformation, img_size = load_mapping_model(
        os.path.join(root_path, 'trained_models', 'mapping_models', f'{mapping_model_filename}.rvl'),
        os.path.join(root_path, 'trained_models', 'main_models'), device)

    perform_occlusion(main_module, mapping_module, activation_extractor, transformation, img_size,
                      os.path.join(root_path, 'data', path_to_img), window_size, stride, threads)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracting concepts from an image')
    parser.add_argument('path_to_img', type=str, help='The path to the image relative to the \'data\' '
                                                      'directory')
    parser.add_argument('mapping_model_filename', type=str, help='File containing the parameters '
                                                                 'of the mapping network model. '
                                                                 'File must be located in the '
                                                                 '\'trained_models'
                                                                 '\\mapping_models\' directory.')
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--threads', type=int, default=0)
    cmd_args = parser.parse_args()
    visualize_mapping_activations(cmd_args.mapping_model_filename, cmd_args.path_to_img,
                                  cmd_args.window_size, cmd_args.stride, cmd_args.threads)
