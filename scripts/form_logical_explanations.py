import argparse
import importlib
import importlib.util
import os
import sys

from revelionn.utils.explanation import explain_target_concept
from scripts.extract_concepts_from_image import extract_concepts


def explain_class(args):
    extracted_concepts, mapping_probabilities = extract_concepts(
        args.path_to_img, args.mapping_model_filepaths, args.main_models_directory, args.main_net_modules_directory,
        args.device)

    module_path = os.path.join(args.concepts_map_directory, f"{args.concepts_map_module_name}.py")
    spec = importlib.util.spec_from_file_location(args.concepts_map_module_name, module_path)
    concept_map_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(concept_map_module)

    concepts_map = getattr(concept_map_module, args.concepts_map_name)

    current_script_path = os.path.abspath(sys.argv[0])
    current_script_directory = os.path.dirname(current_script_path)

    try:
        os.makedirs(os.path.join(current_script_directory, 'temp'))
    except FileExistsError:
        pass

    justifications = explain_target_concept(extracted_concepts, mapping_probabilities, concepts_map,
                                            args.target_concept,
                                            args.path_to_ontology,
                                            os.path.join(current_script_directory, 'temp'))
    print(justifications)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Formation of an ontology-oriented explanation')
    parser.add_argument('path_to_img', type=str, help='The path to the image')
    parser.add_argument('path_to_ontology', type=str, help='The path to the ontology')
    parser.add_argument('concepts_map_directory', type=str)
    parser.add_argument('concepts_map_module_name', type=str, help='The name of the file containing the dictionary of '
                                                                   'relations of concepts relative to the superclass '
                                                                   'of the ontology. The file must be located in the '
                                                                   '\'ontologies\' directory.')
    parser.add_argument('concepts_map_name', type=str, help='The name of the variable that stores the dictionary of '
                                                            'relations of concepts relative to the superclass '
                                                            'of the ontology.')
    parser.add_argument('target_concept', type=str, help='The concept that needs to be obtained by ontological inference')
    parser.add_argument('main_models_directory', type=str)
    parser.add_argument('main_net_modules_directory', type=str, help='Path to the folder containing classes of neural '
                                                                     'network models.')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Tensor processing device, default=\'cpu\'')
    parser.add_argument('-m', '--mapping_model_filepaths', nargs='+', type=str, help='Files containing the parameters '
                                                                                     'of the mapping network model. '
                                                                                     'Files must be located in the '
                                                                                     '\'trained_models'
                                                                                     '\\mapping_models\' directory.',
                        required=True)
    cmd_args = parser.parse_args()
    explain_class(cmd_args)
