Usage Scenarios
===============

Script-Only
-----------

This scenario basically does not require programming and reveals only the basic capabilities of the library.

To use it, the appropriate environment must be initialized (see `Installation <intro.html#installation>`_). Further, being in the terminal in the project
directory in a prepared virtual environment, you can interact with RevelioNN through the following scripts.

- Converting a pre-trained model to the RevelioNN format

.. code-block:: shell

  $ python convert_to_rvl_format.py <model_filepath> <main_net_modules_directory> <main_net_module_name> <main_net_class> <transformation_name> <img_size_name> <num_channels_name> <rvl_filename> <class_label>

- Printing a dictionary of layers of the main neural network

.. code-block:: shell

  $ python print_layers_dict.py <main_net_modules_directory> <main_net_module_name> <main_net_class> -l <layer_types>

- Training of the main networks

.. code-block:: shell

  $ python train_main_nets.py <main_net_modules_directory> <module_name> <main_net_class> <transformation_name> <img_size_name> <num_channels_name> <path_to_images> <path_to_train_csv> <path_to_valid_csv> <image_names_column> -l <label_columns> -d <device>

- Evaluation of the main networks

.. code-block:: shell

  $ python evaluate_main_nets.py <path_to_images> <path_to_test_csv> <image_names_column> <main_net_modules_directory> -m <main_model_filenames> -d <device>

- Training a single mapping network

.. code-block:: shell

  $ python train_single_mapping_net.py <main_model_filepath> <main_net_modules_directory> <path_to_images> <path_to_train_csv> <path_to_valid_csv> <image_names_column> <label_column> --layers_types <layer_types> --layers <layers> --num_neurons <num_neurons> -d <device>

- Training a simultaneous mapping network

.. code-block:: shell

  $ python train_simultaneous_mapping_net.py <main_model_filename> <main_net_modules_directory> <path_to_images> <path_to_train_csv> <path_to_valid_csv> <image_names_column> --label_columns <label_columns> --layers_types <layer_types> --decoder_channels <decoder_channels> --num_shared_neurons <num_shared_neurons> --num_output_neurons <num_output_neurons> -d <device>

- Evaluation of the mapping network

.. code-block:: shell

  $ python evaluate_mapping_net.py <path_to_images> <path_to_test_csv> <image_names_column> main_net_modules_directory -m <mapping_model_filename> -d <device>

- Extracting concepts from an image

.. code-block:: shell

  $ python extract_concepts_from_image.py <path_to_img> <main_models_directory> <main_net_modules_directory> -m <mapping_model_filepaths> -d <device>

- Formation of logical explanations based on ontology

.. code-block:: shell

  $ python form_logical_explanations.py <path_to_img> <path_to_ontology> <concepts_map_directory> <concepts_map_module_name> <concepts_map_name> <target_concept> <main_models_directory> <main_net_modules_directory> -m <mapping_model_filenames> -d <device>

- Formation of visual explanations

.. code-block:: shell

  $ python form_visual_explanations.py <path_to_img> <mapping_model_filepath> <main_models_directory> <main_net_modules_directory> --window_size <window_size> --stride <stride>

To get detailed information on each of the scripts, you need to run:

.. code-block:: shell

  $ python <script_name>.py --help

Program-Level
-------------

To use the API, follow these steps:

#. Import ``convert_to_rvl_format()`` function:

    .. code-block:: python

       from revelionn.utils.model import convert_to_rvl_format

   Call this function by passing the data of the previously declared network model as parameters:

    .. code-block:: python

       convert_to_rvl_format(main_model, filename, class_label, module_name, main_net_class, transformation_name, img_size, num_channels)

#. Import ``MappingTrainer`` class:

    .. code-block:: python

       from revelionn.mapping_trainer import MappingTrainer

#. Initialize the MappingTrainer object and define a list of layer types to be identified in the convolutional network.
   It provides a training/evaluation interface:

    - ``MappingTrainer.train_single_model()`` trains a single mapping network for a given concept based on the activations of given layers;
    - ``MappingTrainer.train_simultaneous_model()`` trains a simultaneous mapping network for a given set of concepts based on the activations of layers of previously defined types;
    - ``MappingTrainer.train_simultaneous_model_semisupervised()`` trains a simultaneous mapping network for a given set of concepts using semi-supervised learning, in which a semantic loss is calculated for unlabeled samples, taking into account the relationships between the concepts;
    - ``MappingTrainer.evaluate_model()`` evaluates the mapping network model on the test set using the ROC AUC.

#. Once the mapping network is trained, you can form logical and visual explanations. To do this, you must first load
   the trained network model via ``load_mapping_model()``.

    .. code-block:: python

       from revelionn.utils.model import load_mapping_model

       main_module, mapping_module, activation_extractor, transformation, img_size =
       load_mapping_model(mapping_model_filepath, main_models_directory, main_net_modules_directory, device)

#. To form logical explanations using an ontology, one must first extract the concepts relevant to the target concept
   from the image, and then transfer the extracted concepts and their probabilities to the reasoning module along with the
   ontology. This can be done as follows:

    .. code-block:: python

       from revelionn.utils.explanation import extract_concepts_from_img, explain_target_concept

       image = Image.open(image_path)
       main_concepts, extracted_concepts, mapping_probabilities = extract_concepts_from_img(main_module,
                                                                                         mapping_module,
                                                                                         image,
                                                                                         transformation)
       justifications = explain_target_concept(extracted_concepts, mapping_probabilities, concepts_map, target_concept,
                                            jar_filepath, owl_ontology, temp_files_path)
       print(justifications)

#. Visual explanations can be formed as follows:

    .. code-block:: python

       import matplotlib.pyplot as plt
       from revelionn.occlusion import perform_occlusion

       perform_occlusion(main_module, mapping_module, activation_extractor, transformation, img_size,
                      image_path, window_size, stride, threads)
       plt.show()
