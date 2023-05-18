import os
from typing import Iterable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader


class MultiLabeledImagesDataset(Dataset):
    """
    A PyTorch dataset class for multi-labeled image data.

    Attributes
    ----------
    img_labels : pd.DataFrame
        A pandas DataFrame containing the image annotations.
    img_dir : str
        The directory path containing the images.
    transform : torchvision.transforms
        A transform to apply to the image data.

    Methods
    -------
    __len__()
        Returns the total number of samples in the dataset.
    __getitem__(idx)
        Returns the image and corresponding labels at the given index.
    labels()
        Returns a list of the target labels.
    """

    def __init__(self, annotations_file, img_dir, name_column, target_columns, transform=None):
        """
        Initialize the MultiLabeledImagesDataset.

        Parameters
        ----------
        annotations_file : str
            The file path to the annotations file in CSV format.
        img_dir : str
            The directory path containing the images.
        name_column : str
            The name of the column in the annotations file that contains the image names.
        target_columns : str or list[str]
            The column name(s) of the target labels in the annotations file.
        transform : torchvision.transforms
            A transform to apply to the image data. Default is None.
        """

        self.img_labels = pd.read_csv(annotations_file, dtype={name_column: str})
        if isinstance(target_columns, Iterable):
            selected_columns = [name_column] + list(target_columns)
        else:
            selected_columns = [name_column, target_columns]
        self.img_labels = self.img_labels[selected_columns]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            The total number of samples.
        """

        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Get the image and corresponding labels at the given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the image and corresponding labels.
        """

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = pil_loader(img_path)
        labels = torch.from_numpy(self.img_labels.iloc[idx, 1:].to_numpy(dtype=np.int8))
        if self.transform:
            image = self.transform(image)
        return image, labels

    def labels(self):
        """
        Return a list of the target labels.

        Returns
        -------
        list
            A list of target labels.
        """

        return list(self.img_labels.columns[1:])


class SemiSupervisedImagesDataset(MultiLabeledImagesDataset):
    """
    A PyTorch dataset class for semi-supervised multi-labeled image data, inheriting from MultiLabeledImagesDataset.

    Attributes
    ----------
    img_labels : pd.DataFrame
        A pandas DataFrame containing the image annotations.
    img_dir : str
        The directory path containing the images.
    transform : torchvision.transforms
        A transform to apply to the image data.
    unlabeled_idx : numpy.ndarray
        An array containing the indices of unlabeled samples.

    Methods
    -------
    __init__(annotations_file, img_dir, name_column, target_columns, unlabeled_samples, transform=None)
        Initialize the SemiSupervisedImagesDataset.
    __getitem__(idx)
        Get the image, corresponding labels, and unlabeled flag at the given index.
    separate_unlabeled(x_raw, y_raw, is_unlabeled)
        Separate the labeled and unlabeled samples from the given data.
    """

    def __init__(self, annotations_file, img_dir, name_column, target_columns, unlabeled_samples, transform=None):
        """
        Initialize the SemiSupervisedImagesDataset.

        Parameters
        ----------
        annotations_file : str
            The file path to the annotations file in CSV format.
        img_dir : str
            The directory path containing the images.
        name_column : str
            The name of the column in the annotations file that contains the image names.
        target_columns : str or list[str]
            The column name(s) of the target labels in the annotations file.
        unlabeled_samples : int or float
            The number of unlabeled samples to include. If float, it represents the fraction of unlabeled samples.
        transform : torchvision.transforms
            A transform to apply to the image data. Default is None.

        Raises
        ------
        ValueError
            If the value of the parameter 'unlabeled_samples' is invalid.
        """

        super().__init__(annotations_file, img_dir, name_column, target_columns, transform=transform)

        if isinstance(unlabeled_samples, int):
            self.unlabeled_idx = np.random.permutation(np.arange(0, len(self.img_labels)))[:unlabeled_samples]
        elif isinstance(unlabeled_samples, float) and unlabeled_samples <= 1.0:
            self.unlabeled_idx = np.random.permutation(np.arange(0, len(self.img_labels)))[:int(len(self.img_labels) *
                                                                                                unlabeled_samples)]
        else:
            raise ValueError("Invalid value of the parameter: unlabeled samples.")
        self.img_labels.loc[:, 'Unlabeled'] = 0
        self.img_labels.loc[self.unlabeled_idx, 'Unlabeled'] = 1

    def __getitem__(self, idx):
        """
        Get the image, corresponding labels, and unlabeled flag at the given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the image, corresponding labels, and unlabeled flag.
        """

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = pil_loader(img_path)
        labels = torch.from_numpy(self.img_labels.iloc[idx, 1:-1].to_numpy(dtype=np.int8))
        is_unlabeled = torch.from_numpy(self.img_labels.iloc[idx, -1:].to_numpy(dtype=np.int8))
        if self.transform:
            image = self.transform(image)
        return image, labels, is_unlabeled

    @staticmethod
    def separate_unlabeled(x_raw, y_raw, is_unlabeled):
        """
        Separate the labeled and unlabeled samples from the given data.

        Parameters
        ----------
        x_raw : torch.Tensor
            The input data.
        y_raw : torch.Tensor
            The target labels.
        is_unlabeled : torch.Tensor
            The unlabeled flags indicating whether a sample is labeled (0) or unlabeled (1).

        Returns
        -------
        tuple
            A tuple containing the labeled data, labeled target labels, unlabeled data, and unlabeled target labels.
        """

        unlabeled_idx = torch.where(is_unlabeled == 1)
        labeled_idx = torch.where(is_unlabeled == 0)
        x, y = x_raw[labeled_idx[0]], y_raw[labeled_idx[0]]
        x_unlab, y_unlab = x_raw[unlabeled_idx[0]], y_raw[unlabeled_idx[0]]
        return x, y, x_unlab, y_unlab


def create_dataloader(path_to_csv, path_to_images, image_names_column, target_columns,
                      batch_size, num_workers, transformation, unlabeled_samples=None):
    """
    Create a PyTorch DataLoader for loading the multi-labeled image dataset.

    Parameters
    ----------
    path_to_csv : str
        The file path to the annotations file in CSV format.
    path_to_images : str
        The directory path containing the images.
    image_names_column : str
        The name of the column in the annotations file that contains the image names.
    target_columns : str or list[str]
        The column name(s) of the target labels in the annotations file.
    batch_size : int
        The batch size for the DataLoader.
    num_workers : int
        The number of worker processes to use for data loading.
    transformation : torchvision.transforms
        A transform to apply to the image data.
    unlabeled_samples : int or float, optional
        The number of unlabeled samples to include. If float, it represents the fraction of unlabeled samples.
        Default is None.

    Returns
    -------
    torch.utils.data.DataLoader
        A PyTorch DataLoader for the multi-labeled image dataset.

    Raises
    ------
    ValueError
        If the value of the parameter 'unlabeled_samples' is invalid.
    """

    if unlabeled_samples is None:
        data = MultiLabeledImagesDataset(path_to_csv, path_to_images, image_names_column, target_columns,
                                         transformation)
    else:
        data = SemiSupervisedImagesDataset(path_to_csv, path_to_images, image_names_column, target_columns,
                                           unlabeled_samples, transformation)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader
