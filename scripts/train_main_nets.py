import argparse
import importlib
import torch
import os
import sys

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)

try:
    from revelionn.main_module import MainModelProcessing
    from revelionn.datasets import MultiLabeledImagesDataset, create_dataloader
except ModuleNotFoundError:
    raise


def train_main_nets(args):
    device = torch.device(args.device)
    module = importlib.import_module(f'main_net_classes.{args.main_net_module_name.replace(os.sep, ".")}')

    transformation = getattr(module, args.transformation_name)
    for class_label in args.label_columns:

        train_loader = create_dataloader(os.path.join(root_path, 'data', args.path_to_train_csv),
                                         os.path.join(root_path, 'data', args.path_to_images), args.image_names_column,
                                         [class_label], args.train_batch_size, args.num_workers, transformation)
        valid_loader = create_dataloader(os.path.join(root_path, 'data', args.path_to_valid_csv),
                                         os.path.join(root_path, 'data', args.path_to_images), args.image_names_column,
                                         [class_label], args.valid_batch_size, args.num_workers, args.transformation)

        main_net = getattr(module, args.main_net_class)()
        main_module = MainModelProcessing(main_net, device)

        try:
            os.makedirs(os.path.join(root_path, 'trained_models', 'main_models'))
        except FileExistsError:
            pass

        if args.prefix_to_save is None:
            main_module.train_model(train_loader, valid_loader, args.patience, args.epochs,
                                    os.path.join(root_path, 'trained_models', 'main_models',
                                                 f'{class_label}_{args.main_net_class}'),
                                    class_label, args.main_net_module_name, args.main_net_class,
                                    args.transformation_name, getattr(module, args.img_size_name),
                                    getattr(module, args.num_channels_name))
        else:
            main_module.train_model(train_loader, valid_loader, args.patience, args.epochs,
                                    os.path.join(root_path, 'trained_models', 'main_models',
                                                 f'{args.prefix_to_save}_{class_label}'),
                                    class_label, args.main_net_module_name, args.main_net_class,
                                    args.transformation_name, getattr(module, args.img_size_name),
                                    getattr(module, args.num_channels_name))

        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of the main networks')
    parser.add_argument('main_net_module_name', type=str, help='The name of the file containing the main network '
                                                               'class, a variable storing transformations, a variable '
                                                               'storing the size of the image side, and a variable '
                                                               'storing the number of image channels. The file must '
                                                               'be located in the \'main_net_classes\' directory.')
    parser.add_argument('main_net_class', type=str, help='Name of the main network class')
    parser.add_argument('transformation_name', type=str, help='Name of the variable storing transformations')
    parser.add_argument('img_size_name', type=str, help='Name of the variable storing the size of the image side')
    parser.add_argument('num_channels_name', type=str, help='The name of the variable storing the number of image '
                                                            'channels')
    parser.add_argument('path_to_images', type=str, help='The path to the folder with images relative to the \'data\' '
                                                         'directory')
    parser.add_argument('path_to_train_csv', type=str, help='The path to the csv file with training data relative to '
                                                            'the \'data\' directory')
    parser.add_argument('path_to_valid_csv', type=str, help='The path to the csv file with validation data relative to '
                                                            'the \'data\' directory')
    parser.add_argument('image_names_column', type=str, help='The name of the column containing the names and '
                                                             'extensions of the image files')
    parser.add_argument('-l', '--label_columns', nargs='+', type=str, help='Column names with class labels for '
                                                                           'training', required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Tensor processing device, default=\'cpu\'')
    parser.add_argument('--train_batch_size', type=int, default=250, help='Training batch size, default=250')
    parser.add_argument('--valid_batch_size', type=int, default=250, help='Validation batch size, '
                                                                          'default=250')
    parser.add_argument('--num_workers', type=int, default=6, help='The number of loader worker processes for '
                                                                   'multi-process data loading, default=6')
    parser.add_argument('--patience', type=int, default=4, help='How many epochs to wait after last time validation '
                                                                'loss improved, default=4')
    parser.add_argument('--epochs', type=int, default=20, help='Number of network learning epochs, default=20')
    parser.add_argument('--prefix_to_save', type=str, help='Prefix of the file name in which the parameters of the '
                                                           'trained model will be saved')
    cmd_args = parser.parse_args()
    train_main_nets(cmd_args)
