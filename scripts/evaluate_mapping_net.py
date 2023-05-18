import argparse
import os
import sys
import torch

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)

try:
    from revelionn.utils.model import load_mapping_model
    from revelionn.datasets import create_dataloader
except ModuleNotFoundError:
    raise


def evaluate_mapping_net(device, mapping_model_filename, path_to_images, path_to_test_csv, image_names_column,
                         batch_size, num_workers):
    device = torch.device(device)

    main_module, mapping_module, activation_extractor, transformation, img_size = load_mapping_model(
        os.path.join(root_path, 'trained_models', 'mapping_models', f'{mapping_model_filename}.rvl'),
        os.path.join(root_path, 'trained_models', 'main_models'), device)

    test_loader = create_dataloader(path_to_test_csv, path_to_images, image_names_column,
                                    mapping_module.get_class_labels(), batch_size, num_workers, transformation)

    mapping_module.evaluate_model(test_loader)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation of the mapping networks')
    parser.add_argument('path_to_images', type=str, help='The path to the folder with images')
    parser.add_argument('path_to_test_csv', type=str, help='The path to the csv file with test data')
    parser.add_argument('image_names_column', type=str, help='The name of the column containing the names and '
                                                             'extensions of the image files')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Tensor processing device, default=\'cpu\'')
    parser.add_argument('-m', '--mapping_model_filename', type=str, help='File containing the parameters '
                                                                                    'of the mapping network model. '
                                                                                    'File must be located in the '
                                                                                    '\'trained_models'
                                                                                    '\\mapping_models\' directory.',
                        required=True)
    parser.add_argument('--test_batch_size', type=int, default=100, help='Test batch size, default=250')
    parser.add_argument('--num_workers', type=int, default=6, help='The number of loader worker processes for '
                                                                   'multi-process data loading, default=6')
    args = parser.parse_args()
    evaluate_mapping_net(args.device, args.mapping_model_filename, args.path_to_images, args.path_to_test_csv,
                         args.image_names_column, args.test_batch_size, args.num_workers)
