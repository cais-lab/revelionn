import torch
from revelionn.mapping_nets.single_mapping_net import SingleMappingNet


def test_single_mapping_net_forward():
    model = SingleMappingNet(in_features=10, num_neurons_list=[20, 10, 1])
    x = torch.randn(5, 10)
    output = model(x)
    assert output.shape == torch.Size([5, 1])


def test_single_mapping_net_get_in_features():
    model = SingleMappingNet(in_features=10, num_neurons_list=[20, 10, 1])
    in_features = model.get_in_features()
    assert in_features == 10


def test_single_mapping_net_get_num_neurons_list():
    model = SingleMappingNet(in_features=10, num_neurons_list=[20, 10, 1])
    num_neurons_list = model.get_num_neurons_list()
    assert num_neurons_list == [20, 10, 1]


def test_single_mapping_net_empty_num_neurons_list():
    try:
        SingleMappingNet(in_features=10, num_neurons_list=[])
    except ValueError:
        assert True
    else:
        assert False
