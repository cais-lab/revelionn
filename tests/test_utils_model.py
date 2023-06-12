import os
import sys

import torch
from torchvision import transforms

from revelionn.activation_extraction import ActivationExtractor
from revelionn.main_module import MainModelProcessing
from revelionn.mapping_module import MappingModelProcessing
from revelionn.utils.model import convert_to_rvl_format, load_main_model, load_mapping_model
from main_net_classes.resnet18 import ResNet18, NUM_CHANNELS, IMG_SIDE_SIZE, transformation

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)


def test_convert_to_rvl_format():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_net = ResNet18()
    main_net.load_state_dict(torch.load(os.path.join(root_path, 'tests', 'data', 'main_models', 'TypeA_ResNet18.pt'),
                                        map_location=device))

    result = convert_to_rvl_format(main_net, os.path.join(root_path, 'tests', 'data', 'main_models', 'TypeA_ResNet18'),
                                   'TypeA', 'resnet18', 'ResNet18', 'transformation', IMG_SIDE_SIZE, NUM_CHANNELS)

    assert result == 'The model was successfully converted to .rvl format.'
    assert os.path.isfile(os.path.join(root_path, 'tests', 'data', 'main_models', 'TypeA_ResNet18.rvl'))


def test_load_main_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_module, transformation, img_size = load_main_model(os.path.join(
        root_path, 'tests', 'data', 'main_models', 'TypeA_ResNet18.rvl'), device)

    # Perform assertions to check if the returned values are correct
    assert isinstance(main_module, MainModelProcessing)
    assert isinstance(transformation, transforms.Compose)
    assert isinstance(img_size, int)


def test_load_mapping_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_module, mapping_module, activation_extractor, transformation, img_size = load_mapping_model(
        os.path.join(root_path, 'tests', 'data', 'mapping_models', 'TypeA_10_[10, 5]_[5, 1].rvl'),
        os.path.join(root_path, 'tests', 'data', 'main_models'),
        device
    )

    # Perform assertions to check if the returned values are correct
    assert isinstance(main_module, MainModelProcessing)
    assert isinstance(mapping_module, MappingModelProcessing)
    assert isinstance(activation_extractor, ActivationExtractor)
    assert isinstance(transformation, transforms.Compose)
    assert isinstance(img_size, int)
