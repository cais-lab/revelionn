import os
import sys

import pytest
import torch
from semantic_loss_pytorch import SemanticLoss

from revelionn.mapping_trainer import MappingTrainer

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)


@pytest.fixture
def mapping_trainer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = MappingTrainer(os.path.join(root_path, 'tests', 'data', 'main_models', 'TypeA_ResNet18.rvl'),
                             ['bn'], 2, 1,
                             os.path.join(root_path, 'tests', 'data', 'mapping_models'), device,
                             os.path.join(root_path, 'tests', 'data', 'images'),
                             os.path.join(root_path, 'tests', 'data', 'partition', 'typeA_mapping_train.csv'),
                             os.path.join(root_path, 'tests', 'data', 'partition', 'typeA_mapping_val.csv'),
                             'name', 50, 6,
                             os.path.join(root_path, 'tests', 'data', 'partition', 'typeA_mapping_test.csv'))

    return trainer


def test_train_single_model(mapping_trainer):
    mapping_trainer.train_single_model([10, 5, 1], 'WarTrain', ['bn19'])

    assert os.path.exists(os.path.join(root_path, 'tests', 'data', 'mapping_models',
                                       "WarTrain_['bn19']_[10, 5, 1]_TypeA.rvl"))

    auc = mapping_trainer.evaluate_model()

    assert auc is not None and 0 < auc <= 1


def test_train_simultaneous_model(mapping_trainer):
    mapping_trainer.train_simultaneous_model(['WarTrain', 'EmptyTrain', 'ReinforcedCar', 'PassengerCar', 'EmptyWagon'],
                                             10, [10, 5], [5, 1])
    main_concept = 'TypeA'

    assert os.path.exists(os.path.join(root_path, 'tests', 'data', 'mapping_models',
                                       f"{main_concept}_10_[10, 5]_[5, 1].rvl"))

    concepts_auc, all_auc = mapping_trainer.evaluate_model()

    assert len(concepts_auc) == 5
    assert all_auc is not None and 0 < all_auc <= 1


def test_train_simultaneous_model_semisupervised(mapping_trainer):
    sl = SemanticLoss(os.path.join(root_path, 'tests', 'data', 'XTRAINS_with_target.sdd'),
                      os.path.join(root_path, 'tests', 'data', 'XTRAINS_with_target.vtree'))

    mapping_trainer.train_simultaneous_model_semisupervised(
        ['WarTrain', 'EmptyTrain', 'ReinforcedCar', 'PassengerCar', 'EmptyWagon'], 10, [10, 5], [5, 1], sl, 0.1, 300)

    assert os.path.exists(os.path.join(root_path, 'tests', 'data', 'mapping_models',
                                       f"TypeA_0.1_10_[10, 5]_[5, 1].rvl"))

    concepts_auc, all_auc = mapping_trainer.evaluate_model()

    assert len(concepts_auc) == 5
    assert all_auc is not None and 0 < all_auc <= 1

