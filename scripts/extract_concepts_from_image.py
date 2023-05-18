import argparse
import os
import sys

import torch
from PIL import Image

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)

try:
    from revelionn.utils.model import load_mapping_model
    from revelionn.utils.explanation import extract_concepts_from_img
except ModuleNotFoundError:
    raise


def extract_concepts(path_to_img, mapping_model_filenames, device):
    device = torch.device(device)

    img = Image.open(os.path.join(root_path, 'data', path_to_img))
    main_concepts = []
    extracted_concepts = []
    mapping_probabilities = []

    for name in mapping_model_filenames:
        main_module, mapping_module, activation_extractor, transformation, img_size = load_mapping_model(
            os.path.join(root_path, 'trained_models', 'mapping_models', f'{name}.rvl'),
            os.path.join(root_path, 'trained_models', 'main_models'), device)

        target_concepts, mapping_concepts, mapping_concept_probabilities = extract_concepts_from_img(main_module,
                                                                                                     mapping_module,
                                                                                                     img,
                                                                                                     transformation)
        main_concepts += target_concepts
        extracted_concepts += mapping_concepts
        mapping_probabilities += mapping_concept_probabilities

    print(f'\nThe image is classified as {main_concepts}.')
    print('\nThe following concepts were extracted from the image:')
    print(extracted_concepts)
    print('with the following probabilities:')
    print(f'{mapping_probabilities}\n')

    return extracted_concepts, mapping_probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracting concepts from an image')
    parser.add_argument('path_to_img', type=str, help='The path to the image relative to the \'data\' '
                                                      'directory')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Tensor processing device, default=\'cpu\'')
    parser.add_argument('-m', '--mapping_model_filenames', nargs='+', type=str, help='Files containing the parameters '
                                                                                     'of the mapping network model. '
                                                                                     'Files must be located in the '
                                                                                     '\'trained_models'
                                                                                     '\\mapping_models\' directory.',
                        required=True)
    cmd_args = parser.parse_args()
    extract_concepts(cmd_args.path_to_img, cmd_args.mapping_model_filenames, cmd_args.device)
