import os
import sys

import pytest
import torch
from torchvision import transforms

from revelionn.datasets import MultiLabeledImagesDataset, SemiSupervisedImagesDataset
from torch.utils.data import Dataset

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
sys.path.append(root_path)


class TestMultiLabeledImagesDataset:
    @pytest.fixture
    def dataset(self):
        return MultiLabeledImagesDataset(
            annotations_file=os.path.join(root_path, 'tests', 'data', 'partition', 'typeA_all.csv'),
            img_dir=os.path.join(root_path, 'tests', 'data', 'images'),
            name_column='name',
            target_columns=['WarTrain', 'EmptyTrain'],
            transform=transforms.Compose([transforms.Resize(size=(224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
        )

    def test_len(self, dataset):
        assert len(dataset) == 1000

    def test_getitem(self, dataset):
        image, labels = dataset[0]
        assert image.shape == (3, 224, 224)
        assert labels.shape == (2,)

    def test_labels(self, dataset):
        assert dataset.labels() == ['WarTrain', 'EmptyTrain']


class TestSemiSupervisedImagesDataset:
    @pytest.fixture
    def dataset(self):
        return SemiSupervisedImagesDataset(
            annotations_file=os.path.join(root_path, 'tests', 'data', 'partition', 'typeA_all.csv'),
            img_dir=os.path.join(root_path, 'tests', 'data', 'images'),
            name_column='name',
            target_columns=['WarTrain', 'EmptyTrain'],
            unlabeled_samples=0.2,
            transform=transforms.Compose([transforms.Resize(size=(224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
        )

    def test_len(self, dataset):
        assert len(dataset) == 1000

    def test_getitem(self, dataset):
        image, labels, is_unlabeled = dataset[0]
        assert image.shape == (3, 224, 224)
        assert labels.shape == (2,)
        assert is_unlabeled.item() in [0, 1]

    def test_separate_unlabeled(self, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=True)
        for x_raw, y_raw, is_unlabeled in dataloader:
            x, y, x_unlab, y_unlab = dataset.separate_unlabeled(x_raw, y_raw, is_unlabeled)
            assert x.shape == (800, 3, 224, 224)
            assert y.shape == (800, 2)
            assert x_unlab.shape == (200, 3, 224, 224)
            assert y_unlab.shape == (200, 2)
