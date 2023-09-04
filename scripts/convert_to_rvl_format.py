import argparse
import importlib
import importlib.util
import os
import torch

from revelionn.utils.model import convert_to_rvl_format


def convert_model(args):
    module_path = os.path.join(args.main_net_modules_directory, f"{args.main_net_module_name}.py")
    spec = importlib.util.spec_from_file_location(args.main_net_module_name, module_path)
    main_net_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_net_module)

    main_net = getattr(main_net_module, args.main_net_class)()
    main_net.load_state_dict(torch.load(args.model_filepath))
    convert_to_rvl_format(main_net, args.rvl_filename, args.class_label, args.main_net_module_name, args.main_net_class,
                          args.transformation_name, getattr(main_net_module, args.img_size_name),
                          getattr(main_net_module, args.num_channels_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converting the main network to the RevelioNN format')
    parser.add_argument('model_filepath', type=str, help='Path to the model file containing only state_dict.')
    parser.add_argument('main_net_modules_directory', type=str, help='Path to the folder containing classes of neural '
                                                                     'network models.')
    parser.add_argument('main_net_module_name', type=str, help='The name of the file containing the main network '
                                                               'class, a variable storing transformations, a variable '
                                                               'storing the size of the image side, and a variable '
                                                               'storing the number of image channels.')
    parser.add_argument('main_net_class', type=str, help='Name of the main network class')
    parser.add_argument('transformation_name', type=str, help='Name of the variable storing transformations')
    parser.add_argument('img_size_name', type=str, help='Name of the variable storing the size of the image side')
    parser.add_argument('num_channels_name', type=str, help='The name of the variable storing the number of image '
                                                            'channels')
    parser.add_argument('rvl_filename', type=str, help='Filename for saving the converted model')
    parser.add_argument('class_label', type=str, help='Classification label name')

    cmd_args = parser.parse_args()
    convert_model(cmd_args)
