import os
import subprocess
import pkg_resources
import torch

from revelionn.mapping_nets.simultaneous_mapping_net import SimultaneousMappingNet


def extract_concepts_from_img(main_module, mapping_module, img, transformation):
    """
    Extracts a set of concepts present in a given image.

    Parameters
    ----------
    main_module : MainModelProcessing
        Class for training, evaluation and processing the main network model.
    mapping_module : MappingModelProcessing
        Class for training, evaluation and processing the mapping network model.
    img : PIL.Image
        Class that represents a PIL image.
    transformation : torchvision.transforms
        A transform to apply to the image.

    Returns
    -------
    main_concept : list[str]
        Target concept extracted by the convolutional network.
    extracted_concepts : list[str]
        Concepts relevant to the target concept, which are extracted by the mapping network.
    mapping_probabilities : list[float]
        The probabilities, obtained from the output of the sigmoid, of each of the extracted concepts relevant to the
        target concept.
    """

    device = main_module.get_device()
    image = transformation(img)

    main_net = main_module.get_main_net()
    mapping_net = mapping_module.get_mapping_net()

    main_concept = []
    extracted_concepts = []
    mapping_probabilities = []

    with torch.no_grad():
        main_net.eval()
        mapping_net.eval()

        image = image.to(device)
        main_class_labels = main_module.get_class_labels()
        output = main_net(image.unsqueeze(0))
        if output > 0.5:
            main_concept.append(main_class_labels[1])
        else:
            main_concept.append(main_class_labels[0])

        mapping_class_labels = mapping_module.get_class_labels()
        mapping_output = mapping_net(mapping_module.get_activation_extractor().get_activations(1))

        if isinstance(mapping_net, SimultaneousMappingNet):
            for i in range(len(mapping_output)):
                if mapping_output[i] > 0.5:
                    extracted_concepts.append(mapping_class_labels[i])
                    mapping_probabilities.append(mapping_output[i].cpu().detach().numpy()[0][0])
                else:
                    extracted_concepts.append(f'Not{mapping_class_labels[i]}')
                    mapping_probabilities.append(1 - mapping_output[i].cpu().detach().numpy()[0][0])
        else:
            if mapping_output > 0.5:
                extracted_concepts.append(mapping_class_labels[0])
                mapping_probabilities.append(mapping_output.cpu().detach().numpy()[0][0])
            else:
                extracted_concepts.append(f'Not{mapping_class_labels[0]}')
                mapping_probabilities.append(1 - mapping_output.cpu().detach().numpy()[0][0])

        torch.cuda.empty_cache()

    return main_concept, extracted_concepts, mapping_probabilities


def to_main_observation(concept):
    """
    Formats a string from the name of the target concept to be parsed by the justifier.

    Parameters
    ----------
    concept : str
        Name of the target concept.

    Returns
    -------
    str
        String from the name of the target concept to be parsed by the justifier.
    """

    return f'__input__ Type: {concept}\n'


def to_mapping_observation(concept, probability):
    """
    Formats a string from the name of the concept relevant to the target concept, which will be parsed by the justifier.

    Parameters
    ----------
    concept : str
        Name of the concept relevant to the target concept.
    probability : float
        The probability of the concept obtained at the output of the sigmoid.

    Returns
    -------
    str
        String from the name of the concept relevant to the target concept, which will be parsed by the justifier.
    """

    return f'__input__ Type: {concept}, {str(probability)}\n'


def form_observations(observations_filepath, concepts_map, target_concept, extracted_concepts, mapping_probabilities):
    with open(observations_filepath, 'w') as observations_file:
        observations_file.write(to_main_observation(concepts_map[target_concept]))

        for i in range(len(extracted_concepts)):
            concept = extracted_concepts[i]
            probability = mapping_probabilities[i]
            if concept.startswith('Not'):
                concept = concept[3:]
                observations_file.write(to_mapping_observation(f'not {concepts_map[concept]}', probability))
            else:
                observations_file.write(to_mapping_observation(concepts_map[concept], probability))

        observations_file.close()


def explain_target_concept(extracted_concepts, mapping_probabilities, concepts_map, target_concept, ontology_filepath,
                           path_to_temp_files):
    """

    Parameters
    ----------
    extracted_concepts : list[str]
        Concepts relevant to the target concept, which are extracted by the mapping network.
    mapping_probabilities : list[float]
        The probabilities, obtained from the output of the sigmoid, of each of the extracted concepts relevant to the
        target concept.
    concepts_map : dict
        Dictionary whose keys are the names of the attributes of the dataset, and the values are the corresponding
        concepts of the ontology.
    target_concept : str
        The concept of ontology, which should be obtained by ontological inference from the extracted concepts.
    ontology_filepath : str
        Path to the OWL ontology file.
    path_to_temp_files
        Temporary files directory for storing observations and explanations.

    Returns
    -------
    justifications : str
        A set of obtained justifications of the target class.
    """

    observations_filepath = os.path.join(path_to_temp_files, 'observations.txt')
    form_observations(observations_filepath, concepts_map, target_concept, extracted_concepts, mapping_probabilities)

    justifications_filepath = os.path.join(path_to_temp_files, 'justifications.txt')
    jar_filepath = pkg_resources.resource_filename(__name__, 'onto_justify.jar')

    subprocess.call(["java", "-Dsun.stdout.encoding=UTF-8", "-Dsun.err.encoding=UTF-8", "-jar",
                     jar_filepath, ontology_filepath, observations_filepath, justifications_filepath])

    with open(justifications_filepath, 'r') as f:
        justifications = f.read()

    return justifications
