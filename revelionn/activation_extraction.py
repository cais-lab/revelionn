import operator
from functools import reduce
import torch

from .main_module import MainModelProcessing


class ActivationExtractor:
    """
    Class for identifying layers of a convolutional neural network and for extracting activations produced during
    network inference from a selected set of layers.

    It has two modes of operation:
    1. Activations of the given layers are concatenated and transformed to a one-dimensional tensor.
    It is used to train a single mapping network.
    2. Activations of the specified layers are returned as a tuple without transformations.
    It is used for training simultaneous mapping network.

    Attributes
    ----------
    device : torch.device
        Tensor processing device.
    main_net : torch.nn.Module
        The model of the main neural network.
    layers_types_dict : dict
        The dictionary contains the names of layer types as keys, and the corresponding values represent
        the layer class in PyTorch.
    layers_types : list
        Types of layers to be found in the hierarchy of network layers. This list should contain only the names of the
        layer types that are in 'layers_types_dict' dictionary.
    layers_dict : dict
        Dictionary of neural network layers. The keys of this dictionary represent the unique names of each of the
        layers of the convolutional network.
    activation : dict
        Activation values of the specified layers.
    layers_for_research : list
        A list of the studied convolutional network layers. Set by the user with keys in the 'layers_dict' dictionary.
    is_concatenate : bool
        Logical parameter that sets the mode of operation of ActivationExtractor. If True, the activations of the given
        layers are concatenated and transformed to a one-dimensional tensor. If False, the activations of the specified
        layers are returned as a tuple without transformations.

    Methods
    -------
    get_layers_types()
        Returns user-defined types of layers to be found in the hierarchy of network layers.
    find_layer_predicate_recursive(model, predicate)
        Recursively searches through a PyTorch model and returns a list of all layers that satisfy a given predicate.
    find_layers_types_recursive(model, layers_types)
        Recursively searches through a PyTorch model and returns a list of all layers that are of a given type or types.
    create_layers_dict(model, cur_layers_types)
        Creates a dictionary of PyTorch layers of the given types from a PyTorch model.
    get_layers_dict()
        Returns a dictionary of neural network layers.
    get_layer_name_by_number(number)
        Returns the layer name in the layers dictionary by its number.
    get_activation(name)
        Saves the values of layer activations.
    register_hooks()
        Registers the interception of activations of the studied layers.
    count_num_activations(num_channels, width_img, height_img)
        Returns the number of activations of neurons of the studied layers.
    get_activations(mapping_batch_size)
        Returns the activation tensor of the studied layers.
    set_layers_for_research(layers)
        Sets the list of layers to be examined.
    get_layers_for_research()
        Returns a list of the layers under study.
    get_main_net()
        Returns the main neural network.
    """

    def __init__(self, main_module, layers_types, is_concatenate):
        """
        Sets all the necessary attributes for the MainNetExplanation object.

        Parameters
        ----------
        main_module : MainModelProcessing
            The model of the main neural network.
        layers_types : list
            Types of layers to be found in the hierarchy of network layers.
        is_concatenate : bool
            Logical parameter that sets the mode of operation of ActivationExtractor. If True, the activations of the
            given layers are concatenated and transformed to a one-dimensional tensor. If False, the activations of the
            specified layers are returned as a tuple without transformations.
        """

        self.device = main_module.get_device()
        self.main_net = main_module.get_main_net().to(self.device)
        self.layers_types = layers_types
        self.layers_types_dict = {'bn': torch.nn.BatchNorm2d,
                                  'conv': torch.nn.Conv2d,
                                  'fc': torch.nn.Linear}
        self.layers_dict = self.create_layers_dict(self.main_net, self.layers_types)
        self.activation = {}
        self.is_concatenate = is_concatenate
        self.layers_for_research = None

    def get_layers_types(self):
        """
        Returns user-defined types of layers to be found in the hierarchy of network layers.

        Returns
        -------
        list[str]
            Layers types.
        """

        return self.layers_types

    def find_layer_predicate_recursive(self, model, predicate):
        """
        Recursively searches through a PyTorch model and returns a list of all layers that satisfy a given predicate.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to search through.
        predicate : function
            A function that takes a PyTorch layer as input and returns a boolean indicating whether or not the layer
            satisfies the predicate.

        Returns
        -------
        list
            A list of all layers in the model that satisfy the given predicate.
        """

        layers = []
        for name, layer in model._modules.items():
            if predicate(layer):
                layers.append(layer)
            layers.extend(self.find_layer_predicate_recursive(layer, predicate))
        return layers

    def find_layers_types_recursive(self, model, layers_types):
        """
        Recursively searches through a PyTorch model and returns a list of all layers that are of a given type or types.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to search through.
        layers_types : list
            A list of PyTorch layer types to search for.

        Returns
        -------
        list
            A list of all layers in the model that are of one of the given layer types.
        """

        def predicate(layer):
            return type(layer) in layers_types

        return self.find_layer_predicate_recursive(model, predicate)

    def create_layers_dict(self, model, cur_layers_types):
        """
        Creates a dictionary of PyTorch layers of the given types from a PyTorch model.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to extract layers from.
        cur_layers_types : list
            A list of strings representing the types of layers to extract.

        Returns
        -------
        dict
            A dictionary mapping layer names to PyTorch layers of the corresponding type.
        """

        cur_layers_types_dict = {}
        for cur_layer_type in cur_layers_types:
            cur_layers_types_dict[cur_layer_type] = self.layers_types_dict[cur_layer_type]

        layers = self.find_layers_types_recursive(model, list(cur_layers_types_dict.values()))
        layer_names = []
        count = 0
        for layer in layers:
            for name, layer_type in cur_layers_types_dict.items():
                if isinstance(layer, layer_type):
                    layer_names.append(f'{name}{count}')
                    count += 1
        layers_dict = dict(zip(layer_names, layers))
        return layers_dict

    def get_layers_dict(self):
        """
        Returns a dictionary of neural network layers.

        Returns
        -------
        layers_dict : dict
            A dictionary of neural network layers.
        """

        return self.layers_dict

    def get_layer_name_by_number(self, number):
        """
        Returns the layer name in the layers dictionary by its number.

        Parameters
        ----------
        number : int
            Layer number.

        Returns
        -------
        str
            Layer name.
        """
        for name, layer in self.layers_dict.items():
            if str(number) == ''.join(i for i in name if not i.isalpha()):
                return name
        raise f'Layer {number} not found'

    def get_activation(self, name):
        """
        Saves the values of layer activations.

        Parameters
        ----------
        name : str
            User-defined name of the neural network layer.

        Returns
        -------
        hook
        """

        def hook(model, input, output):
            self.activation[name] = output

        return hook

    def register_hooks(self):
        """
        Registers the interception of activations of the studied layers.
        """

        for layer in self.layers_for_research:
            self.layers_dict.get(layer).register_forward_hook(self.get_activation(layer))

    @staticmethod
    def prod(iterable):
        return reduce(operator.mul, iterable, 1)

    def count_num_activations(self, num_channels, width_img, height_img):
        """
        Returns the number of activations of neurons of the studied layers.

        Parameters
        ----------
        num_channels : int
            Number of channels in the input image.
        width_img : int
            Width of the input image.
        height_img : int
            Height of the input image.

        Returns
        -------
        num_activations : int
            Number of neuron activations.
        """

        self.main_net.eval()
        inp = torch.randn(1, num_channels, width_img, height_img).to(self.device)
        with torch.no_grad():
            self.main_net(inp)

        num_activations = 0
        for layer in self.layers_for_research:
            num_activations += self.prod(self.activation[layer].shape)
        return num_activations

    def get_activations(self, mapping_batch_size):
        """
        Returns the activation tensor of the studied layers.

        Parameters
        ----------
        mapping_batch_size : int
            The size of the data batch for training the mapping network.

        Returns
        -------
        cur_acts : torch.Tensor or tuple[torch.Tensor]
            Activations of the studied layers.
        """
        if self.is_concatenate:
            cur_acts = None
            for layer in self.layers_for_research:
                if 'fc' in layer:
                    act = self.activation[layer]
                elif 'bn' in layer or 'conv' in layer:
                    act = torch.reshape(self.activation[layer], (mapping_batch_size, -1))
                else:
                    continue
                if cur_acts is None:
                    cur_acts = act.T
                else:
                    cur_acts = torch.cat((cur_acts, act.T), 0)
            return cur_acts.T
        else:
            acts = []
            for layer in self.layers_for_research:
                acts.append(self.activation[layer])
            return tuple(acts)

    def set_layers_for_research(self, layers):
        """
        Sets the list of layers to be examined.

        Parameters
        ----------
        layers : list[str] or list[int]
            Contains layer names or layer numbers (indexing from 0) available in the dictionary of layers.
        """

        all_elements_are_int = all(isinstance(layer, int) for layer in layers)

        if all_elements_are_int:
            for i in range(len(layers)):
                layers[i] = self.get_layer_name_by_number(layers[i])
        else:
            for layer in layers:
                if layer not in self.layers_dict.keys():
                    raise f'Not found {layer} in layers_dict'

        self.layers_for_research = layers
        self.register_hooks()

    def get_layers_for_research(self):
        """
        Returns a list of the layers under study.

        Returns
        -------
        layers_for_research : list
            The list of layers to be examined.
        """

        return self.layers_for_research

    def get_main_net(self):
        """
        Returns the main neural network.

        Returns
        -------
        main_net : torch.nn.Module
            The main neural network.
        """

        return self.main_net
