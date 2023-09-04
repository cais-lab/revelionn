import argparse
import torch
from PIL import Image

from revelionn.utils.model import load_mapping_model
from revelionn.utils.explanation import extract_concepts_from_img


def extract_concepts(path_to_img, mapping_model_filepaths, main_models_directory, main_net_modules_directory, device):
    device = torch.device(device)

    img = Image.open(path_to_img)
    main_concepts = []
    extracted_concepts = []
    mapping_probabilities = []

    for path in mapping_model_filepaths:
        main_module, mapping_module, activation_extractor, transformation, img_size = load_mapping_model(
            path, main_models_directory, main_net_modules_directory, device)

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
    parser.add_argument('path_to_img', type=str, help='The path to the image')
    parser.add_argument('main_models_directory', type=str)
    parser.add_argument('main_net_modules_directory', type=str)
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Tensor processing device, default=\'cpu\'')
    parser.add_argument('-m', '--mapping_model_filepaths', nargs='+', type=str, help='Files containing the parameters '
                                                                                     'of the mapping network model.',
                        required=True)
    cmd_args = parser.parse_args()
    extract_concepts(cmd_args.path_to_img, cmd_args.mapping_model_filepaths, cmd_args.main_models_directory,
                     cmd_args.main_net_modules_directory, cmd_args.device)
