from copy import deepcopy
import torch
from torch import nn


class MappingModule(nn.Module):
    """
    A module representing a common fully connected part of a simultaneous mapping network and blocks of concepts.

    Attributes
    ----------
    common_layers : nn.Sequential
        The shared layers.
    output_layers_list : nn.ModuleList
        A list of output layers, each of which maps the input tensor to an output tensor.
    sigmoid : nn.Sigmoid
        The sigmoid function used to transform the output tensor(s).

    Methods
    -------
    generate_layers(num_neurons)
        Generates a list of PyTorch layers based on the number of neurons in each layer.

    forward(x)
        Forward pass through the module.
    """

    def __init__(self, in_features, num_shared_neurons, num_output_neurons, num_outs=1):
        """
        Sets all the necessary attributes for the MappingModule object.

        Parameters
        ----------
        in_features : int
            The number of input features.
        num_shared_neurons : list[int]
            The number of neurons in consecutive fully connected layers of the common part of the network
            (internal representation of the simultaneous extraction network).
        num_output_neurons : list[int]
            The number of neurons in consecutive fully connected layers of each of the concept blocks.
        num_outs : int
            The number of outputs of the simultaneous extraction network. It is determined by the number of extracted
            concepts.
        """

        super(MappingModule, self).__init__()

        num_shared_neurons = deepcopy(num_shared_neurons)
        num_output_neurons = deepcopy(num_output_neurons)

        if len(num_shared_neurons) != 0 and num_shared_neurons[-1] != num_output_neurons[0]:
            raise ValueError('The last element of num_shared_neurons list must have the same value as the first '
                             'element of num_output_neurons list.')

        if len(num_shared_neurons) != 0:
            num_shared_neurons.insert(0, in_features)
            common_layers = self.generate_layers(num_shared_neurons)
            common_layers.append(nn.ReLU())
            self.common_layers = nn.Sequential(*tuple(common_layers))
        else:
            num_output_neurons.insert(0, in_features)

        output_layers = self.generate_layers(num_output_neurons)

        self.output_layers_list = nn.ModuleList()
        for i in range(num_outs):
            self.output_layers_list.append(deepcopy(nn.Sequential(*tuple(output_layers))))

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def generate_layers(num_neurons):
        """
        Generates a list of PyTorch layers based on the number of neurons in each layer.

        Parameters
        ----------
        num_neurons : list[int]
            The number of neurons in consecutive fully connected layers.

        Returns
        -------
        list[nn.Module]
            A list of PyTorch layers.
        """

        layers = []
        for i in range(len(num_neurons)):
            if i != 0 and i != (len(num_neurons) - 1):
                layers.append(nn.ReLU())
            if i + 1 < len(num_neurons):
                layers.append(nn.Linear(num_neurons[i], num_neurons[i + 1]))
        return layers

    def forward(self, x):
        """
        Forward pass through the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        tuple[torch.Tensor]
            The output tensor(s).
        """

        x = self.common_layers(x)
        outs = []
        for i, output_layers in enumerate(self.output_layers_list):
            outs.append(output_layers(x))
            outs[i] = self.sigmoid(outs[i])
        return tuple(outs)


class LayerDecoder(nn.Module):
    """
    Module consisting of a 1x1 convolution layer, followed by a ReLU activation function, a global average pooling
    layer, and a flattening layer.

    Parameters
    ----------
    in_channels : int
        The number of input channels to the 1x1 convolution layer.
    out_channels : int
        The number of output channels from the 1x1 convolution layer.

    Attributes
    ----------
    layers : nn.Sequential
        A sequential container of the layers that make up this module.

    Methods
    -------
    forward(x)
        Forward pass through the module.
    """

    def __init__(self, in_channels, out_channels):
        """
        Sets all the necessary attributes for the LayerDecoder object.

        Parameters
        ----------
        in_channels : int
            The number of input channels to the 1x1 convolution layer.
        out_channels : int
            The number of output channels from the 1x1 convolution layer.
        """
        super(LayerDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        """
        Forward pass through the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, out_channels).
        """

        x = self.layers(x)
        return x


class SimultaneousMappingNet(nn.Module):
    """
    Simultaneous Mapping Network for RevelioNN.

    Receives an input tuple of activations of the specified convolutional network layers, after which the input tensors
    are processed by decoder blocks. The output tensors of each of the decoders are concatenated and fed into a common
    fully connected part of the network. This is followed by blocks of concepts (one for each of the concepts), which
    are sets of fully connected layers having 1 neuron and a sigmoid at the output.

    Attributes
    ----------
    decoder_channels : int
        The number of decoder channels. The output number of channels of the convolutional layer of the decoder or the
        output number of neurons of the decoder of the fully connected layer.
    num_shared_neurons : list[int]
        The number of neurons in consecutive fully connected layers of the common part of the network
        (internal representation of the simultaneous extraction network).
    num_output_neurons : list[int]
        The number of neurons in consecutive fully connected layers of each of the concept blocks.
    num_outs : int
        The number of outputs of the simultaneous extraction network. It is determined by the number of extracted
        concepts.
    decoders : torch.nn.ModuleList
        Contains the generated decoder blocks in the list.

    Methods
    -------
    forward(x)
        Forward pass through the network.
    get_decoder_channels()
        Returns the number of decoder channels.
    get_num_shared_neurons()
        Returns the number of neurons in consecutive fully connected layers of the common part of the network.
    get_num_output_neurons()
        Returns the number of neurons in consecutive fully connected layers of each of the concept blocks.
    get_num_outs()
        Returns the number of outputs of the simultaneous extraction network.
    """

    def __init__(self, activation_extractor, decoder_channels, num_shared_neurons, num_output_neurons, num_outs):
        """
        Sets all the necessary attributes for the SimultaneousMappingNet object.

        Parameters
        ----------
        activation_extractor : ActivationExtractor
            Class for identifying layers of a convolutional neural network and for extracting activations produced
            during network inference from a selected set of layers.
        decoder_channels : int
            The number of decoder channels. The output number of channels of the convolutional layer of the decoder or
            the output number of neurons of the decoder of the fully connected layer.
        num_shared_neurons : list[int]
            The number of neurons in consecutive fully connected layers of the common part of the network
            (internal representation of the simultaneous extraction network).
        num_output_neurons : list[int]
            The number of neurons in consecutive fully connected layers of each of the concept blocks.
        num_outs : int
            The number of outputs of the simultaneous extraction network. It is determined by the number of extracted
            concepts.
        """

        super(SimultaneousMappingNet, self).__init__()

        self.decoder_channels = decoder_channels
        self.num_shared_neurons = num_shared_neurons
        self.num_output_neurons = num_output_neurons
        self.num_outs = num_outs
        self.decoders = nn.ModuleList()

        if not activation_extractor.is_concatenate:
            layers_dict = activation_extractor.get_layers_dict()
            layers_for_research = activation_extractor.get_layers_for_research()

            for layer_name in layers_for_research:
                if isinstance(layers_dict[layer_name], torch.nn.BatchNorm2d):
                    self.decoders.append(LayerDecoder(layers_dict[layer_name].num_features, decoder_channels))
                if isinstance(layers_dict[layer_name], torch.nn.Conv2d):
                    self.decoders.append(LayerDecoder(layers_dict[layer_name].out_channels, decoder_channels))
                if isinstance(layers_dict[layer_name], torch.nn.Linear):
                    self.decoders.append(nn.Linear(layers_dict[layer_name].out_features, decoder_channels))

            self.mapping_module = MappingModule(decoder_channels * len(self.decoders), num_shared_neurons,
                                                num_output_neurons, num_outs)
        else:
            raise ValueError("ActivationExtractor.is_concatenate must be set to False for its use in "
                             "SimultaneousMappingNet.")

    def forward(self, activations):
        """
        Forward pass through the network.

        Parameters
        ----------
        activations : tuple[torch.Tensor]
            A list of input activations.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """

        outs = []
        if len(self.decoders) != 0:
            for i, decoder in enumerate(self.decoders):
                outs.append(decoder(activations[i]))

        outs = torch.cat(tuple(outs), dim=1)
        outs = self.mapping_module(outs)
        return outs

    def get_decoder_channels(self):
        """
        Return the number of decoder channels.

        Returns
        -------
        int
            The number of decoder channels. The output number of channels of the convolutional layer of the decoder or
            the output number of neurons of the decoder of the fully connected layer.
        """

        return self.decoder_channels

    def get_num_shared_neurons(self):
        """
        Return the number of shared neurons.

        Returns
        -------
        list[int]
            The number of neurons in consecutive fully connected layers of the common part of the network
            (internal representation of the simultaneous extraction network).
        """

        return self.num_shared_neurons

    def get_num_output_neurons(self):
        """
        Return the number of output neurons.

        Returns
        -------
        list[int]
            The number of neurons in consecutive fully connected layers of each of the concept blocks.
        """

        return self.num_output_neurons

    def get_num_outs(self):
        """
        Return the number of outputs of the simultaneous extraction network.

        Returns
        -------
        int
            The number of outputs of the simultaneous extraction network.
            It is determined by the number of extracted concepts.
        """

        return self.num_outs
