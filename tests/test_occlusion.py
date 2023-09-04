import os
import sys

import matplotlib
import torch

from revelionn.occlusion import perform_occlusion
from revelionn.utils.model import load_mapping_model

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)


def test_perform_occlusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_module, mapping_module, activation_extractor, transformation, img_size = load_mapping_model(
        os.path.join(root_path, 'tests', 'data', 'mapping_models', 'C1_20_[160, 80, 40, 20]_[20, 1].rvl'),
        os.path.join(root_path, 'tests', 'data', 'main_models'),
        os.path.join(root_path, 'tests', 'data', 'main_net_classes'),
        device
    )

    path_to_img = os.path.join(root_path, 'tests', 'data', '001236.png')
    plt = perform_occlusion(main_module, mapping_module, activation_extractor, transformation, img_size, path_to_img,
                            100, 50, 0)

