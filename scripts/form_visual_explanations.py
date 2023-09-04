import argparse
import matplotlib.pyplot as plt
import torch

from revelionn.occlusion import perform_occlusion
from revelionn.utils.model import load_mapping_model


def visualize_mapping_activations(mapping_model_filepath, main_models_directory, main_net_modules_directory,
                                  path_to_img, window_size, stride, threads):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    main_module, mapping_module, activation_extractor, transformation, img_size = load_mapping_model(
        mapping_model_filepath, main_models_directory, main_net_modules_directory, device)

    perform_occlusion(main_module, mapping_module, activation_extractor, transformation, img_size,
                      path_to_img, window_size, stride, threads)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracting concepts from an image')
    parser.add_argument('path_to_img', type=str, help='The path to the image relative to the \'data\' '
                                                      'directory')
    parser.add_argument('mapping_model_filepath', type=str, help='File containing the parameters '
                                                                 'of the mapping network model.')
    parser.add_argument('main_models_directory', type=str)
    parser.add_argument('main_net_modules_directory', type=str)
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--threads', type=int, default=0)
    cmd_args = parser.parse_args()
    visualize_mapping_activations(cmd_args.mapping_model_filepath, cmd_args.main_models_directory,
                                  cmd_args.main_net_modules_directory, cmd_args.path_to_img,
                                  cmd_args.window_size, cmd_args.stride, cmd_args.threads)
