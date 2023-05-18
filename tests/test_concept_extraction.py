import pytest
from unittest.mock import MagicMock

from ontologies.xtrains_ontology import create_xtrains_nxo
from revelionn.concept_extraction import ConceptExtractor


class TestConceptExtractor:
    @pytest.fixture
    def trainer(self):
        return MagicMock()

    @pytest.fixture
    def ontology(self):
        return create_xtrains_nxo()

    @pytest.fixture
    def extractor(self, trainer, ontology):
        return ConceptExtractor(trainer, ontology)

    def test_create_subgraph(self, extractor, ontology):
        subgraph = extractor.create_subgraph(ontology.graph, 'WarTrain')
        assert sorted(subgraph.nodes) == ['PassengerCar', 'ReinforcedCar', 'WarTrain']
        assert sorted(subgraph.edges) == [('WarTrain', 'PassengerCar'), ('WarTrain', 'ReinforcedCar')]

    def test_order_concepts(self, extractor, ontology):
        ordered_concepts = extractor.order_concepts('TypeA', ontology)
        assert ordered_concepts == [("TypeA", "WarTrain"), ("TypeA", "EmptyTrain"), ("WarTrain", "ReinforcedCar"),
                                    ("WarTrain", "PassengerCar"), ("EmptyTrain", "EmptyWagon")]

        ordered_concepts = extractor.order_concepts('WarTrain', ontology)
        assert ordered_concepts == [("WarTrain", "ReinforcedCar"), ("WarTrain", "PassengerCar")]

    def test_exhaustive_search(self, extractor, trainer):

        def evaluate_model_mock():
            return evaluate_model_mock.return_values.pop(0)

        evaluate_model_mock.return_values = [0.8, 0.9, 0.7]
        extractor.trainer.evaluate_model = MagicMock(side_effect=evaluate_model_mock)

        expected_best_layer = "layer2"
        expected_best_value = 0.9

        best_layer, best_value = extractor.exhaustive_search('concept', ['layer1', 'layer2', 'layer3'], [10, 5, 1])

        assert best_layer == expected_best_layer
        assert best_value == expected_best_value

    def test_linear_search(self, extractor, trainer):

        def evaluate_model_mock():
            return evaluate_model_mock.return_values.pop(0)

        evaluate_model_mock.return_values = [0.8, 0.9, 0.95, 0.9, 0.8, 0.7]
        extractor.trainer.evaluate_model = MagicMock(side_effect=evaluate_model_mock)

        expected_best_layer_num = 3
        expected_best_value = 0.95

        best_layer_num, best_value = extractor.linear_search('concept', 5, 1, [10, 5, 1])

        assert best_layer_num == expected_best_layer_num
        assert best_value == expected_best_value

    def test_heuristic_search(self, extractor, trainer):

        def evaluate_model_mock():
            return evaluate_model_mock.return_values.pop(0)

        evaluate_model_mock.return_values = [0.8, 0.97, 0.7, 0.93, 0.8, 0.95, 0.8, 0.87, 0.96, 0.83, 0.93]
        extractor.trainer.evaluate_model = MagicMock(side_effect=evaluate_model_mock)

        best_layers = extractor.heuristic_search('TypeA', 2, 1, [10, 5, 1])

        assert best_layers == {'WarTrain': [1, 0.97], 'EmptyTrain': [0, 0.95], 'ReinforcedCar': [0, 0.87],
                               'PassengerCar': [1, 0.96], 'EmptyWagon': [0, 0.93]}
