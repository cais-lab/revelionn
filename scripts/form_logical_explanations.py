import argparse
import importlib
import os
import sys

from revelionn.utils.explanation import explain_target_concept

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)

try:
    from scripts.extract_concepts_from_image import extract_concepts
except ModuleNotFoundError:
    raise


def explain_class(args):
    extracted_concepts, mapping_probabilities = extract_concepts(args.path_to_img, args.mapping_model_filenames,
                                                                 args.device)

    module = importlib.import_module(f'ontologies.{args.concepts_map_module_name.replace(os.sep, ".")}')
    concepts_map = getattr(module, args.concepts_map_name)

    try:
        os.makedirs(os.path.join(root_path, 'temp'))
    except FileExistsError:
        pass

    jar_filepath = os.path.join(root_path, 'scripts', 'onto_justify.jar')
    justifications = explain_target_concept(extracted_concepts, mapping_probabilities, concepts_map, args.main_concept,
                                            jar_filepath, os.path.join(root_path, 'ontologies', args.path_to_ontology),
                                            os.path.join(root_path, 'temp'))
    print(justifications)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Formation of an ontology-oriented explanation')
    parser.add_argument('path_to_img', type=str, help='The path to the image relative to the \'data\' '
                                                      'directory')
    parser.add_argument('path_to_ontology', type=str, help='The path to the ontology relative to the \'ontologies\' '
                                                           'directory')
    parser.add_argument('concepts_map_module_name', type=str, help='The name of the file containing the dictionary of '
                                                                   'relations of concepts relative to the superclass '
                                                                   'of the ontology. The file must be located in the '
                                                                   '\'ontologies\' directory.')
    parser.add_argument('concepts_map_name', type=str, help='The name of the variable that stores the dictionary of '
                                                            'relations of concepts relative to the superclass '
                                                            'of the ontology.')
    parser.add_argument('target_concept', type=str, help='The concept that needs to be obtained by ontological inference')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Tensor processing device, default=\'cpu\'')
    parser.add_argument('-m', '--mapping_model_filenames', nargs='+', type=str, help='Files containing the parameters '
                                                                                     'of the mapping network model. '
                                                                                     'Files must be located in the '
                                                                                     '\'trained_models'
                                                                                     '\\mapping_models\' directory.',
                        required=True)
    cmd_args = parser.parse_args()
    explain_class(cmd_args)
