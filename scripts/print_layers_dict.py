import argparse
import os
import torch
import importlib
import importlib.util

try:
    from revelionn.main_module import MainModelProcessing
    from revelionn.activation_extraction import ActivationExtractor
except ModuleNotFoundError:
    raise


def print_layers_dict(args):
    device = torch.device(args.device)

    module_path = os.path.join(args.main_net_modules_directory, f"{args.main_net_module_name}.py")
    spec = importlib.util.spec_from_file_location(args.main_net_module_name, module_path)
    main_net_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_net_module)

    main_net = getattr(main_net_module, args.main_net_class)()
    main_module = MainModelProcessing(main_net, device)

    activation_extractor = ActivationExtractor(main_module, args.layers_types, None)

    layers_dict = activation_extractor.get_layers_dict()
    for key, value in layers_dict.items():
        print("{0}: {1}".format(key, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Printing a dictionary of layers of the main neural network')
    parser.add_argument('main_net_modules_directory', type=str, help='Path to the folder containing classes of neural '
                                                                     'network models.')
    parser.add_argument('main_net_module_name', type=str, help='The name of the file containing the main network class')
    parser.add_argument('main_net_class', type=str, help='Name of the main network class')
    parser.add_argument('-l', '--layers_types', nargs='+', type=str, help="Types of layers to be identified ('bn', "
                                                                          "'fc', 'mp')", required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Tensor processing device, default=\'cpu\'')
    cmd_args = parser.parse_args()
    print_layers_dict(cmd_args)
