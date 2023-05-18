import argparse
import importlib
import torch
import os
import sys
from torch.utils.data import Dataset

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)

try:
    from revelionn.main_module import MainModelProcessing
    from revelionn.datasets import MultiLabeledImagesDataset
except ModuleNotFoundError:
    raise


def evaluate_main_nets(device, main_model_filenames, path_to_images, path_to_test_csv, image_names_column,
                       test_batch_size, num_workers):
    device = torch.device(device)

    main_model_names = main_model_filenames

    for name in main_model_names:
        main_net_data = torch.load(
            os.path.join(root_path, 'trained_models', 'main_models', f'{name}.rvl'),
            map_location=device)
        module = importlib.import_module(f"main_net_classes."
                                         f"{main_net_data['main_net_module_name'].replace(os.sep, '.')}")
        main_net = getattr(module, main_net_data['main_net_class'])()
        main_module = MainModelProcessing(main_net, device)
        main_module.load_model(
            os.path.join(root_path, 'trained_models', 'main_models', f'{name}.rvl'))

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
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Tensor processing device, default=\'cpu\'')
    parser.add_argument('-m', '--main_model_filenames', nargs='+', type=str, help='Files containing the parameters of '
                                                                                  'the main network model. Files '
                                                                                  'must be located in the '
                                                                                  '\'trained_models\\main_models\' '
                                                                                  'directory.', required=True)
    parser.add_argument('--test_batch_size', type=int, default=250, help='Test batch size, default=250')
    parser.add_argument('--num_workers', type=int, default=6, help='The number of loader worker processes for '
                                                                   'multi-process data loading, default=6')
    args = parser.parse_args()
    evaluate_main_nets(args.device, args.main_model_filenames, args.path_to_images, args.path_to_test_csv,
                       args.image_names_column, args.test_batch_size, args.num_workers)
