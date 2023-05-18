from torch import nn


class SingleMappingNet(nn.Module):
    """
    Single Mapping Network for RevelioNN.

    It is a fully connected network that receives as input the layer activations reduced to a single dimension or the
    concatenation of activations of convolutional network layers. It has a ReLU activation function in its hidden
    layers and a sigmoid in its output. In connection with this there must be 1 neuron in the output layer.

    Attributes
    ----------
    in_features : str
        Input number of neuron activations.
    num_neurons_list : list[int]
        The number of neurons in consecutive fully connected layers.

    Methods
    -------
    forward(x)
        Determines how the data will pass through the neural network.
    get_num_neurons_list()
        Returns the number of neurons in consecutive fully connected layers.
    """

    def __init__(self, in_features, num_neurons_list):
        """
        Sets all the necessary attributes for the SingleMappingNet object.

        Parameters
        ----------
        in_features : int
            Input number of neuron activations. Can be calculated by the count_num_activations() method of the
            ActivationExtractor class.
        num_neurons_list : list[int]
            The number of neurons in consecutive fully connected layers. The output layer should always have 1 neuron.
        """

        if len(num_neurons_list) == 0:
            raise ValueError("Parameter 'num_neurons_list' is empty.")

        self.in_features = in_features
        self.num_neurons_list = num_neurons_list
        super(SingleMappingNet, self).__init__()

        layers = []
        if len(num_neurons_list) != 0:
            layers.append(nn.Linear(in_features, num_neurons_list[0]))
            for i in range(len(num_neurons_list)):
                if num_neurons_list[i] != 1:
                    layers.append(nn.ReLU())
                if i + 1 < len(num_neurons_list):
                    layers.append(nn.Linear(num_neurons_list[i], num_neurons_list[i + 1]))
                else:
                    assert num_neurons_list[i] == 1

        self.layers = nn.Sequential(*tuple(layers))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Determines how the data will pass through the neural network. Returns the data received after processing by
        the neural network.

        Parameters
        ----------
        x : torch.tensor
            The input activations tensor reduced to one dimension.

        Returns
        -------
        x
            Output tensor.
        """

        x = self.layers(x)
        x = self.sigmoid(x)
        return x

    def get_in_features(self):
        """
        Returns the input number of neuron activations.

        Returns
        -------
        in_features : int
            Input number of neuron activations.
        """

        return self.in_features

    def get_num_neurons_list(self):
        """
        Returns the number of neurons in consecutive fully connected layers.

        Returns
        -------
        num_neurons_list : list[int]
            The number of neurons in consecutive fully connected layers.
        """

        return self.num_neurons_list
