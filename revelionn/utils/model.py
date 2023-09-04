import importlib
import importlib.util
import os
import torch
from revelionn.activation_extraction import ActivationExtractor
from revelionn.main_module import MainModelProcessing
from revelionn.mapping_module import MappingModelProcessing
from revelionn.mapping_nets.simultaneous_mapping_net import SimultaneousMappingNet
from revelionn.mapping_nets.single_mapping_net import SingleMappingNet


def convert_to_rvl_format(main_model, filename, class_label, module_name, main_net_class,
                          transformation_name, img_size, num_channels):
    """
    Converts the pre-trained main network model to RevelioNN format. Creates the converted model as an RVL file.

    Parameters
    ----------
    main_model : torch.nn.Module
        Main network model with loaded weights.
    filename : str
        Filename (path) to save the converted model.
    class_label : str
        Name of the output class label of the main network.
    module_name : str
        Name of the module (.py file name) containing the class of the main network.
    main_net_class : str
        Name of the main network class.
    transformation_name : str
        Name of the variable storing transformations.
    img_size : int
        Size of the image side.
    num_channels : int
        Number of image channels.
    """

    classes = {1: class_label,
               0: f'Not{class_label}'}

    torch.save({'classes': classes,
                'model_state_dict': main_model.state_dict(),
                'main_net_module_name': module_name,
                'main_net_class': main_net_class,
                'transformation_name': transformation_name,
                'img_size': img_size,
                'num_channels': num_channels
                }, f'{filename}.rvl')

    msg = 'The model was successfully converted to .rvl format.'
    print(msg)

    return msg


def load_main_model(main_model_filepath, main_net_modules_directory, device):
    """
    Loads the main network model in RevelioNN format from a file. Initializes and returns a class to work with
    the main net, as well as a transformation object and image size.

    Parameters
    ----------
    main_model_filepath : str
        File path containing the parameters of the main network model.
    main_net_modules_directory : str
        Directory containing .py files with classes of the main networks.
    device : torch.device
        Tensor processing device.

    Returns
    -------
    main_module : MainModelProcessing
        Class for training, evaluation and processing the main network model.
    transformation : torchvision.transforms
        A transform to apply to the images.
    img_size : int
        Size of the image side.
    """

    main_net_data = torch.load(main_model_filepath, map_location=device)
    module_path = os.path.join(main_net_modules_directory, f"{main_net_data['main_net_module_name']}.py")
    spec = importlib.util.spec_from_file_location(main_net_data['main_net_module_name'], module_path)
    main_net_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_net_module)

    main_net = getattr(main_net_module, main_net_data['main_net_class'])()

    # main_net_module = importlib.import_module(f"{main_net_modules_directory}."
    #                                           f"{main_net_data['main_net_module_name'].replace(os.sep, '.')}")
    # main_net = getattr(main_net_module, main_net_data['main_net_class'])()

    main_module = MainModelProcessing(main_net, device)
    main_module.load_model(main_model_filepath)

    img_size = main_net_data['img_size']
    transformation = getattr(main_net_module, main_net_data['transformation_name'])
    return main_module, transformation, img_size


def load_mapping_model(mapping_model_filepath, main_models_directory, main_net_modules_directory, device):
    """
    Loads the mapping network model from a file. Initializes and returns a class to work with the main net,
    as well as a transformation object and image size.

    Parameters
    ----------
    mapping_model_filepath : str
        File path containing the parameters of the mapping network model.
    main_models_directory : str
        Directory containing files with parameters of the main network models.
    main_net_modules_directory : str
        Directory containing .py files with classes of the main networks.
    device : torch.device
        Tensor processing device.

    Returns
    -------
    main_module : MainModelProcessing
        Class for training, evaluation and processing the main network model.
    mapping_module : MappingModelProcessing
        Class for training, evaluation and processing the mapping network model.
    activation_extractor : ActivationExtractor
        Class for identifying layers of the main network and for extracting activations produced during
        network inference.
    transformation : torchvision.transforms
        A transform to apply to the images.
    img_size : int
        Size of the image side.
    """

    mapping_model_data = torch.load(mapping_model_filepath, map_location=device)
    main_module, transformation, img_size = load_main_model(os.path.join
                                                            (main_models_directory,
                                                             f"{mapping_model_data['main_model_filename']}.rvl"),
                                                            main_net_modules_directory,
                                                            device)

    if 'decoder_channels' in mapping_model_data:
        activation_extractor = ActivationExtractor(main_module, mapping_model_data['layers_types'],
                                                   is_concatenate=False)
        activation_extractor.set_layers_for_research(mapping_model_data['layers'])
        mapping_net = SimultaneousMappingNet(activation_extractor,
                                             mapping_model_data['decoder_channels'],
                                             mapping_model_data['num_shared_neurons'],
                                             mapping_model_data['num_output_neurons'],
                                             mapping_model_data['num_outs'])
    else:
        activation_extractor = ActivationExtractor(main_module, mapping_model_data['layers_types'], is_concatenate=True)
        activation_extractor.set_layers_for_research(mapping_model_data['layers'])
        mapping_net = SingleMappingNet(activation_extractor.count_num_activations(mapping_model_data['num_channels'],
                                                                                  mapping_model_data['img_size'],
                                                                                  mapping_model_data['img_size']),
                                       mapping_model_data['num_neurons_list'])
    mapping_module = MappingModelProcessing(activation_extractor, mapping_net, device)
    mapping_module.load_model(mapping_model_filepath)

    return main_module, mapping_module, activation_extractor, transformation, img_size
