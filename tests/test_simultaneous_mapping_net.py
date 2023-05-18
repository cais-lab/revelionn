import torch

from tests.data.main_net_classes.resnet18 import ResNet18
from revelionn.activation_extraction import ActivationExtractor
from revelionn.main_module import MainModelProcessing
from revelionn.mapping_nets.simultaneous_mapping_net import MappingModule, LayerDecoder, SimultaneousMappingNet


def test_mapping_module():
    # Test for initialization of the MappingModule class
    mapping_module = MappingModule(20, [160, 80, 40, 20], [20, 1], 1)
    assert len(mapping_module.output_layers_list) == 1
    assert mapping_module.sigmoid is not None

    # Test for the generate_layers method
    layers = MappingModule.generate_layers([10, 5, 1])
    assert isinstance(layers, list)
    assert len(layers) == 3

    # Forward method test
    x = torch.randn(60, 20)
    out = mapping_module(x)
    assert isinstance(out, tuple)
    assert len(out) == 1
    assert isinstance(out[0], torch.Tensor)
    assert out[0].shape == (60, 1)


def test_layer_decoder():
    # LayerDecoder class initialization test
    layer_decoder = LayerDecoder(20, 10)
    assert layer_decoder.layers is not None

    # Forward method test
    x = torch.randn(2, 20, 10, 10)
    out = layer_decoder(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 10)


def test_simultaneous_mapping_net():
    main_net = ResNet18()
    device = torch.device('cpu')
    main_module = MainModelProcessing(main_net, device)
    activation_extractor = ActivationExtractor(main_module, ['bn', 'conv', 'fc'], is_concatenate=False)
    activation_extractor.set_layers_for_research(list(activation_extractor.get_layers_dict().keys()))

    # Test for initialization of the SimultaneousMappingNet class
    mapping_net = SimultaneousMappingNet(activation_extractor, 32, [64, 32], [32, 16, 1], 2)
    assert mapping_net.num_outs == 2
    assert mapping_net.decoders is not None

    # Test for methods get_decoder_channels, get_num_shared_neurons, get_num_output_neurons, get_num_outs
    assert mapping_net.get_decoder_channels() == 32
    assert mapping_net.get_num_shared_neurons() == [64, 32]
    assert mapping_net.get_num_output_neurons() == [32, 16, 1]
    assert mapping_net.get_num_outs() == 2

    # Forward method test
    main_net(torch.randn(2, 3, 224, 224))
    x = activation_extractor.get_activations(2)
    out = mapping_net(x)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], torch.Tensor)
    assert out[0].shape == (2, 1)
    assert isinstance(out[1], torch.Tensor)
    assert out[1].shape == (2, 1)