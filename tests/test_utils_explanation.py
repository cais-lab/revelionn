import os
import sys

import torch
from PIL import Image

from .data.ontologies.xtrains_ontology import concepts_map
from revelionn.utils.explanation import extract_concepts_from_img, form_observations
from revelionn.utils.model import load_mapping_model

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)


def test_extract_concepts_from_img():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_module, mapping_module, activation_extractor, transformation, img_size = load_mapping_model(
        os.path.join(root_path, 'tests', 'data', 'mapping_models', 'TypeA_10_[10, 5]_[5, 1].rvl'),
        os.path.join(root_path, 'tests', 'data', 'main_models'), os.path.join(
            root_path, 'tests', 'data', 'main_net_classes'),
        device
    )

    img = Image.open(os.path.join(root_path, 'data',
                                  os.path.join(root_path, 'tests', 'data', 'images', '0000376.png')))
    main_concept, extracted_concepts, mapping_probabilities = extract_concepts_from_img(
        main_module, mapping_module, img, transformation
    )

    # Perform assertions to check if the returned values are correct
    assert isinstance(main_concept, list)
    assert isinstance(extracted_concepts, list)
    assert isinstance(mapping_probabilities, list)


def test_form_observations():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_module, mapping_module, activation_extractor, transformation, img_size = load_mapping_model(
        os.path.join(root_path, 'tests', 'data', 'mapping_models', 'TypeA_10_[10, 5]_[5, 1].rvl'),
        os.path.join(root_path, 'tests', 'data', 'main_models'), os.path.join(
            root_path, 'tests', 'data', 'main_net_classes'),
        device
    )

    img = Image.open(os.path.join(root_path, 'data',
                                  os.path.join(root_path, 'tests', 'data', 'images', '0000376.png')))
    main_concept, extracted_concepts, mapping_probabilities = extract_concepts_from_img(
        main_module, mapping_module, img, transformation
    )

    path_to_temp_files = os.path.join(root_path, 'tests', 'data', 'temp')
    if not os.path.exists(path_to_temp_files):
        os.makedirs(path_to_temp_files)

    observations_filepath = os.path.join(path_to_temp_files, "observations.txt")
    target_concept = "TypeA"

    form_observations(observations_filepath, concepts_map, target_concept, extracted_concepts, mapping_probabilities)

    # Check if the observations file exists
    assert os.path.isfile(observations_filepath)
