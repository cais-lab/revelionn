import argparse
import importlib
import torch
import os
import sys

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)

try:
    from revelionn.utils.model import convert_to_rvl_format
except ModuleNotFoundError:
    raise


def convert_model(args):
    module = importlib.import_module(f"main_net_classes.{args.main_net_module_name}")
    main_net = getattr(module, args.main_net_class)()
    main_net.load_state_dict(torch.load(args.model_filepath))
    convert_to_rvl_format(main_net, args.rvl_filename, args.class_label, args.main_net_module_name, args.main_net_class,
                          args.transformation_name, getattr(module, args.img_size_name),
                          getattr(module, args.num_channels_name))

    print('Successfully converted')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converting the main network to the RevelioNN format')
    parser.add_argument('model_filepath', type=str, help='')
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
    parser.add_argument('rvl_filename', type=str, help='')
    parser.add_argument('class_label', type=str, help='')

    cmd_args = parser.parse_args()
    convert_model(cmd_args)
