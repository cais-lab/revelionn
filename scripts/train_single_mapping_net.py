import argparse
import torch
import os
import sys

from revelionn.mapping_trainer import MappingTrainer

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)


def train_single_mapping_net(device, main_net_modules_directory, main_model_filepath, path_to_images, path_to_train_csv,
                             path_to_valid_csv, image_names_column, label_column, layer_types, layers, num_neurons,
                             batch_size, num_workers, patience, epochs):
    device = torch.device(device)

    trainer = MappingTrainer(main_model_filepath, main_net_modules_directory, layer_types, patience, epochs,
                             os.path.join(root_path, 'trained_models', 'mapping_models'), device, path_to_images,
                             path_to_train_csv, path_to_valid_csv, image_names_column, batch_size, num_workers, None)
    trainer.train_single_model(num_neurons, label_column, layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of the single mapping network')
    parser.add_argument('main_model_filepath', type=str, help='Path to file containing the parameters of the main '
                                                              'network model.')
    parser.add_argument('main_net_modules_directory', type=str, help='Path to the folder containing classes of neural '
                                                                     'network models.')
    parser.add_argument('path_to_images', type=str, help='The path to the folder with images')
    parser.add_argument('path_to_train_csv', type=str, help='The path to the csv file with training data')
    parser.add_argument('path_to_valid_csv', type=str, help='The path to the csv file with validation data')
    parser.add_argument('image_names_column', type=str, help='The name of the column containing the names and '
                                                             'extensions of the image files')
    parser.add_argument('label_column', type=str, help='Column name with class label for training')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Tensor processing device, default=\'cpu\'')
    parser.add_argument('--layer_types', nargs='+', type=str, help="Types of layers to be identified "
                                                                   "('bn', 'fc', 'conv')", required=True)
    parser.add_argument('--layers', nargs='+', type=str, help='The keys of the layers in the dictionary, from which '
                                                              'you need to get activations for training the mapping '
                                                              'network. To get a dictionary of the layers of the main '
                                                              'network, use script \'print_layers_dict.py\'',
                        required=True)
    parser.add_argument('--num_neurons', nargs='+', type=int, help='The number of neurons in consecutive fully '
                                                                   'connected layers. The last layer always '
                                                                   'has 1 output neuron.', required=True)
    parser.add_argument('--batch_size', type=int, default=100, help='Training and validation batch size, '
                                                                    'default=100')
    parser.add_argument('--num_workers', type=int, default=6, help='The number of loader worker processes for '
                                                                   'multi-process data loading, default=6')
    parser.add_argument('--patience', type=int, default=6, help='How many epochs to wait after last time validation '
                                                                'loss improved, default=6')
    parser.add_argument('--epochs', type=int, default=50, help='Number of network learning epochs, default=50')
    args = parser.parse_args()
    train_single_mapping_net(args.device, args.main_net_modules_directory, args.main_model_filepath,
                             args.path_to_images, args.path_to_train_csv, args.path_to_valid_csv,
                             args.image_names_column, args.label_column, args.layer_types,
                             args.layers, args.num_neurons, args.batch_size, args.num_workers, args.patience,
                             args.epochs)
