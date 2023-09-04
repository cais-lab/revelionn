import argparse
import importlib
import importlib.util
import torch
import os
import sys
from torch.utils.data import Dataset

from revelionn.main_module import MainModelProcessing
from revelionn.datasets import MultiLabeledImagesDataset

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)


def evaluate_main_nets(device, main_net_modules_directory, main_model_filepaths, path_to_images, path_to_test_csv, image_names_column,
                       test_batch_size, num_workers):
    device = torch.device(device)

    for path in main_model_filepaths:
        main_net_data = torch.load(path, map_location=device)
        module_path = os.path.join(main_net_modules_directory, f"{main_net_data['main_net_module_name']}.py")
        spec = importlib.util.spec_from_file_location(main_net_data['main_net_module_name'], module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        main_net = getattr(module, main_net_data['main_net_class'])()
        main_module = MainModelProcessing(main_net, device)
        main_module.load_model(path)

        transformation = getattr(module, main_net_data['transformation_name'])
        test_data = MultiLabeledImagesDataset(os.path.join(root_path, 'data', path_to_test_csv),
                                              os.path.join(root_path, 'data', path_to_images),
                                              image_names_column, [main_module.get_class_labels()[1]],
                                              transformation)

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=True,
                                                  num_workers=num_workers)

        test_loss, test_acc, test_auc = main_module.evaluate_model(test_loader)

        torch.cuda.empty_cache()
        
        return test_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation of the main networks')
    parser.add_argument('path_to_images', type=str, help='The path to the folder with images relative to the \'data\' '
                                                         'directory')
    parser.add_argument('path_to_test_csv', type=str, help='The path to the csv file with test data relative to '
                                                           'the \'data\' directory')
    parser.add_argument('image_names_column', type=str, help='The name of the column containing the names and '
                                                             'extensions of the image files')
    parser.add_argument('main_net_modules_directory', type=str, help='Path to the folder containing classes of neural '
                                                                     'network models.')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Tensor processing device, default=\'cpu\'')
    parser.add_argument('-m', '--main_model_filepaths', nargs='+', type=str, help='Paths to files containing the '
                                                                                  'parameters of'
                                                                                  'the main network model.',
                        required=True)
    parser.add_argument('--test_batch_size', type=int, default=250, help='Test batch size, default=250')
    parser.add_argument('--num_workers', type=int, default=6, help='The number of loader worker processes for '
                                                                   'multi-process data loading, default=6')
    args = parser.parse_args()
    evaluate_main_nets(args.device, args.main_net_modules_directory, args.main_model_filepaths, args.path_to_images,
                       args.path_to_test_csv, args.image_names_column, args.test_batch_size, args.num_workers)
