import importlib
import os
import torch
from torch import optim

from .activation_extraction import ActivationExtractor
from .early_stopping import EarlyStopping
from .main_module import MainModelProcessing
from .mapping_module import MappingModelProcessing
from .mapping_nets.simultaneous_mapping_net import SimultaneousMappingNet
from .mapping_nets.single_mapping_net import SingleMappingNet
from .datasets import create_dataloader


class MappingTrainer:
    """
    Mapping Trainer class provides an interface for learning/evaluating mapping networks

    Methods
    -------
    train_single_model(mapping_neurons, concept, layer_names)
        Trains a single mapping network for a given concept based on the activations of given layers.
    train_simultaneous_model(concepts, decoder_channels, num_shared_neurons, num_output_neurons)
        Trains a simultaneous mapping network for a given set of concepts based on the activations of layers of
        previously defined types.
    train_simultaneous_model_semisupervised(concepts, decoder_channels, num_shared_neurons, num_output_neurons,
    semantic_loss, sem_loss_weight, unlabeled_samples)
        Trains a simultaneous mapping network for a given set of concepts using semi-supervised learning, in which a
        semantic loss is calculated for unlabeled samples, taking into account the relationships between the concepts.
    evaluate_model()
        Evaluates the mapping network model on the test set using the ROC AUC.
    """

    def __init__(self, main_model_filepath, layers_types, patience, epochs, path_to_save, device,
                 path_to_images, path_to_train_csv, path_to_valid_csv, image_names_column, batch_size, num_workers,
                 path_to_test_csv):
        """

        Parameters
        ----------
        main_model_filepath : str
            File path containing the parameters of the main network model.
        layers_types : list[str]
            Types of layers to be identified ('bn', 'fc', 'conv').
        patience : int
            How many epochs to wait after last time validation loss improved.
        epochs : int
            Number of network learning epochs.
        path_to_save : str
            Path for saving models.
        device : torch.device
            Tensor processing device.
        path_to_images : str
            The path to the folder with images.
        path_to_train_csv : str
            The path to the csv file with training data.
        path_to_valid_csv : str
            The path to the csv file with validation data.
        image_names_column : str
            The name of the column containing the names and extensions of the image files.
        batch_size : int
            Batch size.
        num_workers : int
            The number of loader worker processes for multi-process data loading.
        path_to_test_csv : str
            The path to the csv file with test data.
        """

        main_model_data = torch.load(main_model_filepath, map_location=device)
        module = importlib.import_module(f"main_net_classes."
                                         f"{main_model_data['main_net_module_name'].replace(os.sep, '.')}")
        main_net = getattr(module, main_model_data['main_net_class'])()
        self.main_module = MainModelProcessing(main_net, device)
        self.main_module.load_model(os.path.join(main_model_filepath))
        self.transformation = getattr(module, main_model_data['transformation_name'])

        self.main_net_module_name = main_model_data['main_net_module_name']
        self.main_net_class = main_model_data['main_net_class']
        self.main_model_filepath = main_model_filepath
        self.transformation_name = main_model_data['transformation_name']
        self.img_size = main_model_data['img_size']
        self.num_channels = main_model_data['num_channels']

        self.layers_types = layers_types
        self.device = device
        self.mapping_module = None
        self.activation_extractor = None

        self.patience = patience
        self.epochs = epochs
        self.path_to_save = path_to_save
        self.path_to_images = path_to_images
        self.path_to_train_csv = path_to_train_csv
        self.path_to_valid_csv = path_to_valid_csv
        self.image_names_column = image_names_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path_to_test_csv = path_to_test_csv

    def train_single_model(self, mapping_neurons, concept, layer_names):
        """
        Trains a single mapping network for a given concept based on the activations of given layers.

        Parameters
        ----------
        mapping_neurons : list[int]
            The number of neurons in consecutive fully connected layers. The output layer should always have 1 neuron.
        concept : str
            The target concept for which to train the mapping network.
        layer_names : list[str]
            A list of layer names to consider for training and evaluation.
        """

        train_loader = create_dataloader(self.path_to_train_csv, self.path_to_images, self.image_names_column,
                                         [concept], self.batch_size, self.num_workers, self.transformation)
        valid_loader = create_dataloader(self.path_to_valid_csv, self.path_to_images, self.image_names_column,
                                         [concept], self.batch_size, self.num_workers, self.transformation)
        self.activation_extractor = ActivationExtractor(self.main_module, self.layers_types, is_concatenate=True)
        self.activation_extractor.set_layers_for_research(layer_names)
        mapping_net = SingleMappingNet(self.activation_extractor.count_num_activations(self.num_channels,
                                                                                       self.img_size,
                                                                                       self.img_size),
                                       mapping_neurons)
        self.mapping_module = MappingModelProcessing(self.activation_extractor, mapping_net, self.device)

        optimizer = optim.Adam(mapping_net.parameters(), lr=0.001)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=0.001)

        self.mapping_module.train_model_single(train_loader, valid_loader, optimizer, early_stopping, self.epochs,
                                               os.path.join(self.path_to_save,
                                                            f'{concept}_{layer_names}_{mapping_neurons}_'
                                                            f'{self.main_module.get_class_labels()[1]}'),
                                               concept, self.main_net_module_name, self.main_net_class,
                                               self.main_model_filepath,
                                               self.transformation_name, self.img_size, self.num_channels)

    def train_simultaneous_model(self, concepts, decoder_channels, num_shared_neurons, num_output_neurons):
        """
        Trains a simultaneous mapping network for a given set of concepts based on the activations of layers of
        previously defined types.

        Parameters
        ----------
        concepts : list[str]
            The target concepts for which to train the mapping network.
        decoder_channels : int
            The number of decoder channels. The output number of channels of the convolutional layer of the decoder or
            the output number of neurons of the decoder of the fully connected layer.
        num_shared_neurons : list[int]
            The number of neurons in consecutive fully connected layers of the common part of the network
            (internal representation of the simultaneous extraction network).
        num_output_neurons : list[int]
            The number of neurons in consecutive fully connected layers of each of the concept blocks.
        """

        train_loader = create_dataloader(self.path_to_train_csv, self.path_to_images, self.image_names_column,
                                         concepts, self.batch_size, self.num_workers, self.transformation)
        valid_loader = create_dataloader(self.path_to_valid_csv, self.path_to_images, self.image_names_column,
                                         concepts, self.batch_size, self.num_workers, self.transformation)
        self.activation_extractor = ActivationExtractor(self.main_module, self.layers_types, is_concatenate=False)
        self.activation_extractor.set_layers_for_research(list(self.activation_extractor.get_layers_dict().keys()))
        mapping_net = SimultaneousMappingNet(self.activation_extractor, decoder_channels, num_shared_neurons,
                                             num_output_neurons, len(concepts))
        self.mapping_module = MappingModelProcessing(self.activation_extractor, mapping_net, self.device)

        optimizer = optim.Adam(mapping_net.parameters(), lr=0.001)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=0.001)

        self.mapping_module.train_model_simultaneous(train_loader, valid_loader, optimizer, early_stopping, self.epochs,
                                                     os.path.join(self.path_to_save,
                                                                  f'{self.main_module.get_class_labels()[1]}_'
                                                                  f'{decoder_channels}_'
                                                                  f'{num_shared_neurons}_'
                                                                  f'{num_output_neurons}'),
                                                     concepts, self.main_net_module_name, self.main_net_class,
                                                     self.main_model_filepath, self.transformation_name, self.img_size,
                                                     self.num_channels)

    def train_simultaneous_model_semisupervised(self, concepts, decoder_channels, num_shared_neurons,
                                                num_output_neurons, semantic_loss, sem_loss_weight, unlabeled_samples):
        """
        Trains a simultaneous mapping network for a given set of concepts using semi-supervised learning, in which a
        semantic loss is calculated for unlabeled samples, taking into account the relationships between the concepts.

        Parameters
        ----------
        concepts : list[str]
            The target concepts for which to train the mapping network.
        decoder_channels : int
            The number of decoder channels. The output number of channels of the convolutional layer of the decoder or
            the output number of neurons of the decoder of the fully connected layer.
        num_shared_neurons : list[int]
            The number of neurons in consecutive fully connected layers of the common part of the network
            (internal representation of the simultaneous extraction network).
        num_output_neurons : list[int]
            The number of neurons in consecutive fully connected layers of each of the concept blocks.
        semantic_loss : semantic_loss_pytorch.SemanticLoss
            An object of the semantic loss class, for initialization of which it is necessary to use the generated .sdd
            and .vtree.
        sem_loss_weight : float
            The contribution of semantic loss to the overall loss function.
        unlabeled_samples : int or float
            The number of unlabeled samples to include. If float, it represents the fraction of unlabeled samples.
        """

        train_loader = create_dataloader(self.path_to_train_csv, self.path_to_images, self.image_names_column,
                                         concepts, self.batch_size, self.num_workers, self.transformation,
                                         unlabeled_samples)
        valid_loader = create_dataloader(self.path_to_valid_csv, self.path_to_images, self.image_names_column,
                                         concepts, self.batch_size, self.num_workers, self.transformation)
        self.activation_extractor = ActivationExtractor(self.main_module, self.layers_types, is_concatenate=False)
        self.activation_extractor.set_layers_for_research(list(self.activation_extractor.get_layers_dict().keys()))
        mapping_net = SimultaneousMappingNet(self.activation_extractor, decoder_channels, num_shared_neurons,
                                             num_output_neurons, len(concepts))
        self.mapping_module = MappingModelProcessing(self.activation_extractor, mapping_net, self.device)

        optimizer = optim.Adam(mapping_net.parameters(), lr=0.001)
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=0.001)

        self.mapping_module.train_model_semisupervised(train_loader, valid_loader, optimizer, early_stopping,
                                                       self.epochs, semantic_loss, sem_loss_weight,
                                                       os.path.join(self.path_to_save,
                                                                    f'{self.main_module.get_class_labels()[1]}_'
                                                                    f'{sem_loss_weight}_'
                                                                    f'{decoder_channels}_'
                                                                    f'{num_shared_neurons}_'
                                                                    f'{num_output_neurons}'),
                                                       concepts, self.main_net_module_name, self.main_net_class,
                                                       self.main_model_filepath, self.transformation_name,
                                                       self.img_size,
                                                       self.num_channels)

    def evaluate_model(self):
        """
        Evaluates the mapping network model on the test set using the ROC AUC.

        Returns
        -------
        float
            The ROC AUC value of a single mapping network or the ROC AUC value for all labels of a simultaneous mapping
            network.
        """

        test_loader = create_dataloader(self.path_to_test_csv, self.path_to_images, self.image_names_column,
                                        self.mapping_module.get_class_labels(), self.batch_size, self.num_workers,
                                        self.transformation)
        concepts_auc, all_auc = self.mapping_module.evaluate_model(test_loader)

        if not concepts_auc:
            return all_auc
        else:
            return concepts_auc, all_auc
