import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

from .datasets import SemiSupervisedImagesDataset


class MappingModelProcessing:
    """
    Class for training, evaluation and processing the mapping network model.

    Attributes
    ----------
    device : torch.device
        Tensor processing device.
    activation_extractor : MainNetExplanation
        Class for identifying layers of a convolutional neural network and for extracting activations produced during
        network inference from a selected set of layers.
    mapping_net : MappingNet(nn.Module)
        The model of the mapping neural network.
    class_labels : dict
        Names of mapping network output classes.


    Methods
    -------
    train_model_single(train_loader, valid_loader, optimizer, early_stopping, epochs, filename, class_label,
    main_net_module_name, main_net_class, main_model_filename, transformation_name, img_size, num_channels)
        Trains a single mapping network for a given concept.
    train_model_simultaneous(train_loader, valid_loader, optimizer, early_stopping, epochs, filename, class_labels,
    main_net_module_name, main_net_class, main_model_filename, transformation_name, img_size, num_channels)
        Trains a simultaneous mapping network for a given set of concepts.
    train_model_semisupervised(train_loader, valid_loader, optimizer, early_stopping, epochs, semantic_loss,
    sem_loss_weight, filename, class_labels, main_net_module_name, main_net_class, main_model_filename,
    transformation_name, img_size, num_channels)
        Trains a simultaneous mapping network for a given set of concepts using semi-supervised learning, in which a
        semantic loss is calculated for unlabeled samples, taking into account the relationships between the concepts.
    evaluate_model(self, test_loader)
        Evaluates the mapping network model on the test set.
    get_mapping_net()
        Returns the mapping network.
    get_class_labels()
        Returns names of mapping network output classes.
    get_activation_extractor()
        Returns the ActivationExtractor object.
    load_model(path_to_model_dict)
        Loads weights and class labels of the mapping network model from a file.
    evaluate_model(test_loader)
        Evaluation of the model on the test set.
    """

    def __init__(self, activation_extractor, mapping_net, device):
        """
        Sets all the necessary attributes for the MappingModelProcessing object.

        Parameters
        ----------
        activation_extractor : ActivationExtractor
            Class for identifying layers of a convolutional neural network and for extracting activations produced
            during network inference from a selected set of layers.
        mapping_net : torch.nn.Module
            The model of the mapping network.
        device : torch.device
            Tensor processing device.
        """

        self.activation_extractor = activation_extractor
        self.device = device
        self.mapping_net = mapping_net.to(self.device)
        self.class_labels = None

    def get_mapping_net(self):
        """
        Returns the mapping network.

        Returns
        -------
        mapping_net : torch.nn.Module
            The mapping network.
        """

        return self.mapping_net

    def get_activation_extractor(self):
        """
        Returns the ActivationExtractor object.

        Returns
        -------
        ActivationExtractor
            Class for identifying layers of a convolutional neural network and for extracting activations produced
            during network inference.
        """
        return self.activation_extractor

    def get_class_labels(self):
        """
        Returns names of mapping network output classes.

        Returns
        -------
        classes : dict
            Names of mapping network output classes.
        """

        return self.class_labels

    def load_model(self, path_to_model):
        """
        Loads weights and class labels of the mapping network model from a file.

        Parameters
        ----------
        path_to_model : str
            The path to the file containing weights.
        """

        checkpoint = torch.load(path_to_model, map_location=self.device)
        self.mapping_net.load_state_dict(checkpoint['model_state_dict'])
        self.class_labels = checkpoint['classes']

    def train_model_single(self, train_loader, valid_loader, optimizer, early_stopping, epochs, filename, class_label,
                           main_net_module_name, main_net_class, main_model_filename, transformation_name,
                           img_size, num_channels):
        """
        Trains a single mapping network for a given concept.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Training data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        valid_loader : torch.utils.data.DataLoader
            Validation data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        optimizer : torch.optim.Optimizer
            The used weight optimizer of the mapping network.
        early_stopping : EarlyStopping
            Class to stop training when validation loss stops improving.
        epochs : int
            The number of training epochs of the mapping neural network.
        filename : str
            The name of the file in which the parameters of the trained model will be saved.
        class_label : str
            The name of the label of the class used for training.
        main_net_module_name : str
            The name of the file containing the main network class.
        main_net_class : str
            Name of the main network class.
        main_model_filename : str
            The file containing the parameters of the main network model.
        transformation_name : str
            Name of the variable storing transformations.
        img_size : int
            The size of the image side.
        num_channels : int
            The number of image channels.
        """

        self.class_labels = [class_label]
        self.mapping_net.to(self.device)

        criterion = nn.BCELoss()

        for e in range(epochs):
            num_train_batches_without_auc = 0
            num_valid_batches_without_auc = 0
            train_loss = 0
            train_acc = 0
            train_auc = 0

            main_net = self.activation_extractor.get_main_net()
            main_net.eval()
            self.mapping_net.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    output = main_net(images)
                logits = self.mapping_net(self.activation_extractor.get_activations(train_loader.batch_size))
                labels = labels.float()
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                predictions = (logits > 0.5).long()
                train_acc += accuracy_score(labels.cpu(), predictions.cpu())

                try:
                    auc = roc_auc_score(labels.cpu(), logits.cpu().detach().numpy())
                    train_auc += auc
                except ValueError:
                    num_train_batches_without_auc += 1

            valid_loss = 0
            val_acc = 0
            val_auc = 0
            self.mapping_net.eval()
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = main_net(images)
                    logits = self.mapping_net(self.activation_extractor.get_activations(valid_loader.batch_size))
                    labels = labels.float()
                    batch_loss = criterion(logits, labels)
                    valid_loss += batch_loss.item()

                    predictions = (logits > 0.5).long()
                    val_acc += accuracy_score(labels.cpu(), predictions.cpu())

                    try:
                        auc = roc_auc_score(labels.cpu(), logits.cpu().detach().numpy())
                        val_auc += auc
                    except ValueError:
                        num_valid_batches_without_auc += 1

            result = f"Epoch {e + 1}/{epochs}.. " \
                     f"Train loss: {train_loss / len(train_loader):.3f}.. " \
                     f"Valid loss: {valid_loss / len(valid_loader):.3f}.. " \
                     f"Train acc: {train_acc / len(train_loader):.3f}.. " \
                     f"Valid acc: {val_acc / len(valid_loader):.3f}.. " \
                     f"Train AUC: {train_auc / (len(train_loader) - num_train_batches_without_auc):.3f}.. " \
                     f"Val AUC: {val_auc / (len(valid_loader) - num_valid_batches_without_auc):.3f}.. "

            print(result)

            with open(f'{filename}.txt', "a") as file:
                file.write(result + '\n')

            valid_loss_decrease = early_stopping(valid_loss / len(valid_loader))

            if early_stopping.early_stop:
                print("Early stopping")
                break
            elif valid_loss_decrease is not None:
                with open(f'{filename}.txt', "a") as file:
                    file.write(valid_loss_decrease + '\n')

                torch.save({'classes': self.class_labels,
                            'model_state_dict': self.mapping_net.state_dict(),
                            'main_net_module_name': main_net_module_name,
                            'main_net_class': main_net_class,
                            'main_model_filename': main_model_filename,
                            'transformation_name': transformation_name,
                            'img_size': img_size,
                            'num_channels': num_channels,
                            'layers_types': self.activation_extractor.get_layers_types(),
                            'layers': self.activation_extractor.get_layers_for_research(),
                            'num_neurons_list': self.mapping_net.get_num_neurons_list()
                            }, f'{filename}.rvl')

    def train_model_simultaneous(self, train_loader, valid_loader, optimizer, early_stopping, epochs, filename,
                                 class_labels, main_net_module_name, main_net_class, main_model_filename,
                                 transformation_name, img_size, num_channels):
        """
        Trains a simultaneous mapping network for a given set of concepts.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Training data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        valid_loader : torch.utils.data.DataLoader
            Validation data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        optimizer : torch.optim.Optimizer
            The used weight optimizer of the mapping network.
        early_stopping : EarlyStopping
            Class to stop training when validation loss stops improving.
        epochs : int
            The number of training epochs of the mapping neural network.
        filename : str
            The name of the file in which the parameters of the trained model will be saved.
        class_labels : list[str]
            Names of class labels used for training.
        main_net_module_name : str
            The name of the file containing the main network class.
        main_net_class : str
            Name of the main network class.
        main_model_filename : str
            The file containing the parameters of the main network model.
        transformation_name : str
            Name of the variable storing transformations.
        img_size : int
            The size of the image side.
        num_channels : int
            The number of image channels.
        """

        self.class_labels = class_labels
        self.mapping_net.to(self.device)

        criterion = nn.BCELoss()

        for e in range(epochs):
            num_train_batches_without_auc = 0
            num_valid_batches_without_auc = 0
            train_loss = 0
            train_acc = 0
            train_auc = 0

            train_concepts_auc = [0] * len(class_labels)
            num_train_batches_without_concepts_auc = [0] * len(class_labels)

            main_net = self.activation_extractor.get_main_net()
            main_net.eval()
            self.mapping_net.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                with torch.no_grad():
                    output = main_net(images)

                logits = self.mapping_net(self.activation_extractor.get_activations(train_loader.batch_size))
                logits = torch.cat(logits, dim=1)

                labels = labels.float()
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                predictions = (logits > 0.5).long()
                train_acc += accuracy_score(labels.cpu(), predictions.cpu())

                try:
                    auc = roc_auc_score(labels.cpu(), logits.cpu().detach().numpy())
                    train_auc += auc
                except ValueError:
                    num_train_batches_without_auc += 1

                assert len(class_labels) == logits.shape[1]

                for i, output in enumerate(logits.T.cpu().detach().numpy()):
                    try:
                        auc = roc_auc_score(labels.T[i].cpu(), output)
                        train_concepts_auc[i] += auc
                    except ValueError:
                        num_train_batches_without_concepts_auc[i] += 1

            valid_loss = 0
            val_acc = 0
            val_auc = 0

            valid_concepts_auc = [0] * len(class_labels)
            num_valid_batches_without_concepts_auc = [0] * len(class_labels)

            self.mapping_net.eval()

            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = main_net(images)

                    logits = self.mapping_net(self.activation_extractor.get_activations(valid_loader.batch_size))
                    logits = torch.cat(logits, dim=1)

                    labels = labels.float()
                    batch_loss = criterion(logits, labels)
                    valid_loss += batch_loss.item()

                    predictions = (logits > 0.5).long()
                    val_acc += accuracy_score(labels.cpu(), predictions.cpu())

                    try:
                        auc = roc_auc_score(labels.cpu(), logits.cpu().detach().numpy())
                        val_auc += auc
                    except ValueError:
                        num_valid_batches_without_auc += 1

                    for i, output in enumerate(logits.T.cpu().detach().numpy()):
                        try:
                            auc = roc_auc_score(labels.T[i].cpu(), output)
                            valid_concepts_auc[i] += auc
                        except ValueError:
                            num_valid_batches_without_concepts_auc[i] += 1

            res_train_concepts_auc = ""
            res_val_concepts_auc = ""
            for i in range(len(class_labels)):
                res_train_concepts_auc += \
                    f'Train AUC {class_labels[i]}: ' \
                    f'{train_concepts_auc[i] / (len(train_loader) - num_train_batches_without_concepts_auc[i]):.3f}.. '
                res_val_concepts_auc += \
                    f'Val AUC {class_labels[i]}: ' \
                    f'{valid_concepts_auc[i] / (len(valid_loader) - num_valid_batches_without_concepts_auc[i]):.3f}.. '

            result = f"Epoch {e + 1}/{epochs}.. " \
                     f"Train loss: {train_loss / len(train_loader):.3f}.. " \
                     f"Valid loss: {valid_loss / len(valid_loader):.3f}.. " \
                     f"Train acc: {train_acc / len(train_loader):.3f}.. " \
                     f"Valid acc: {val_acc / len(valid_loader):.3f}.. " \
                     f"Train AUC: {train_auc / (len(train_loader) - num_train_batches_without_auc):.3f}.. " \
                     f"Valid AUC: {val_auc / (len(valid_loader) - num_valid_batches_without_auc):.3f}.. \n" \
                     f"{res_train_concepts_auc} \n" \
                     f"{res_val_concepts_auc}"

            print(result)

            with open(f'{filename}.txt', "a") as file:
                file.write(result + '\n')

            valid_loss_decrease = early_stopping(valid_loss / len(valid_loader))

            if early_stopping.early_stop:
                print("Early stopping")
                break
            elif valid_loss_decrease is not None:
                with open(f'{filename}.txt', "a") as file:
                    file.write(valid_loss_decrease + '\n')

                torch.save({'classes': class_labels,
                            'model_state_dict': self.mapping_net.state_dict(),
                            'main_net_module_name': main_net_module_name,
                            'main_net_class': main_net_class,
                            'main_model_filename': main_model_filename,
                            'transformation_name': transformation_name,
                            'img_size': img_size,
                            'num_channels': num_channels,
                            'layers_types': self.activation_extractor.get_layers_types(),
                            'layers': self.activation_extractor.get_layers_for_research(),
                            'decoder_channels': self.mapping_net.get_decoder_channels(),
                            'num_shared_neurons': self.mapping_net.get_num_shared_neurons(),
                            'num_output_neurons': self.mapping_net.get_num_output_neurons(),
                            'num_outs': self.mapping_net.get_num_outs()
                            }, f'{filename}.rvl')

    def train_model_semisupervised(self, train_loader, valid_loader, optimizer, early_stopping, epochs, semantic_loss,
                                   sem_loss_weight, filename, class_labels, main_net_module_name, main_net_class,
                                   main_model_filename, transformation_name, img_size, num_channels):
        """
        Trains a simultaneous mapping network for a given set of concepts using semi-supervised learning, in which a
        semantic loss is calculated for unlabeled samples, taking into account the relationships between the concepts.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            Training data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        valid_loader : torch.utils.data.DataLoader
            Validation data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        optimizer : torch.optim.Optimizer
            The used weight optimizer of the mapping network.
        early_stopping : EarlyStopping
            Class to stop training when validation loss stops improving.
        epochs : int
            The number of training epochs of the mapping neural network.
        semantic_loss : semantic_loss_pytorch.SemanticLoss
            An object of the semantic loss class, for initialization of which it is necessary to use the generated .sdd
            and .vtree.
        sem_loss_weight : float
            The contribution of semantic loss to the overall loss function.
        filename : str
            The name of the file in which the parameters of the trained model will be saved.
        class_labels : list[str]
            Names of class labels used for training.
        main_net_module_name : str
            The name of the file containing the main network class.
        main_net_class : str
            Name of the main network class.
        main_model_filename : str
            The file containing the parameters of the main network model.
        transformation_name : str
            Name of the variable storing transformations.
        img_size : int
            The size of the image side.
        num_channels : int
            The number of image channels.
        """

        self.class_labels = class_labels
        self.mapping_net.to(self.device)

        criterion = nn.BCELoss()

        for e in range(epochs):
            num_train_batches_without_auc = 0
            num_valid_batches_without_auc = 0
            train_loss = 0
            train_acc = 0
            train_auc = 0
            train_semantic_loss = 0

            train_concepts_auc = [0] * len(class_labels)
            num_train_batches_without_concepts_auc = [0] * len(class_labels)

            main_net = self.activation_extractor.get_main_net()
            main_net.eval()
            self.mapping_net.train()
            for images, labels, is_unlabeled in train_loader:
                images, labels, is_unlabeled = images.to(self.device), labels.to(self.device), is_unlabeled.to(
                    self.device)
                images_lab, labels_lab, images_unlab, labels_unlab = SemiSupervisedImagesDataset.separate_unlabeled(
                    images, labels, is_unlabeled)
                images_lab, labels_lab = images_lab.to(self.device), labels_lab.to(self.device)
                images_unlab, labels_unlab = images_unlab.to(self.device), labels_unlab.to(self.device)
                optimizer.zero_grad()

                with torch.no_grad():
                    output = main_net(images_lab)

                logits = self.mapping_net(self.activation_extractor.get_activations(len(images_lab)))
                logits = torch.cat(logits, dim=1)

                labels_lab = labels_lab.float()
                bce_loss = criterion(logits, labels_lab)

                with torch.no_grad():
                    output = main_net(images_unlab)

                logits = self.mapping_net(self.activation_extractor.get_activations(len(images_unlab)))
                logits = torch.cat(logits, dim=1)
                semantic_logits = torch.cat((output, logits), dim=1)

                sem_loss, wmc, wmc_per_sample = semantic_loss(probabilities=semantic_logits.cpu(), output_wmc=True,
                                                              output_wmc_per_sample=True)

                sem_loss = sem_loss_weight * sem_loss
                loss = bce_loss + sem_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_semantic_loss += sem_loss.item()

                with torch.no_grad():
                    output = main_net(images)

                logits = self.mapping_net(self.activation_extractor.get_activations(train_loader.batch_size))
                logits = torch.cat(logits, dim=1)

                predictions = (logits > 0.5).long()
                train_acc += accuracy_score(labels.cpu(), predictions.cpu())

                try:
                    auc = roc_auc_score(labels.cpu(), logits.cpu().detach().numpy())
                    train_auc += auc
                except ValueError:
                    num_train_batches_without_auc += 1

                assert len(class_labels) == logits.shape[1]

                for i, output in enumerate(logits.T.cpu().detach().numpy()):
                    try:
                        auc = roc_auc_score(labels.T[i].cpu(), output)
                        train_concepts_auc[i] += auc
                    except ValueError:
                        num_train_batches_without_concepts_auc[i] += 1

            valid_loss = 0
            val_acc = 0
            val_auc = 0
            val_semantic_loss = 0

            valid_concepts_auc = [0] * len(class_labels)
            num_valid_batches_without_concepts_auc = [0] * len(class_labels)

            self.mapping_net.eval()

            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = main_net(images)

                    logits = self.mapping_net(self.activation_extractor.get_activations(valid_loader.batch_size))
                    logits = torch.cat(logits, dim=1)
                    semantic_logits = torch.cat((output, logits), dim=1)

                    labels = labels.float()
                    bce_loss = criterion(logits, labels)

                    sem_loss, wmc, wmc_per_sample = semantic_loss(probabilities=semantic_logits.cpu(), output_wmc=True,
                                                                  output_wmc_per_sample=True)
                    sem_loss = sem_loss_weight * sem_loss

                    loss = bce_loss + sem_loss
                    valid_loss += loss.item()
                    val_semantic_loss += sem_loss.item()

                    predictions = (logits > 0.5).long()
                    val_acc += accuracy_score(labels.cpu(), predictions.cpu())

                    try:
                        auc = roc_auc_score(labels.cpu(), logits.cpu().detach().numpy())
                        val_auc += auc
                    except ValueError:
                        num_valid_batches_without_auc += 1

                    for i, output in enumerate(logits.T.cpu().detach().numpy()):
                        try:
                            auc = roc_auc_score(labels.T[i].cpu(), output)
                            valid_concepts_auc[i] += auc
                        except ValueError:
                            num_valid_batches_without_concepts_auc[i] += 1

            res_train_concepts_auc = ""
            res_val_concepts_auc = ""
            for i in range(len(class_labels)):
                res_train_concepts_auc += \
                    f'Train AUC {class_labels[i]}: ' \
                    f'{train_concepts_auc[i] / (len(train_loader) - num_train_batches_without_concepts_auc[i]):.3f}.. '
                res_val_concepts_auc += \
                    f'Val AUC {class_labels[i]}: ' \
                    f'{valid_concepts_auc[i] / (len(valid_loader) - num_valid_batches_without_concepts_auc[i]):.3f}..'

            result = f"Epoch {e + 1}/{epochs}.. " \
                     f"Train loss: {train_loss / len(train_loader):.3f}.. " \
                     f"Valid loss: {valid_loss / len(valid_loader):.3f}.. " \
                     f"Train semantic loss: {train_semantic_loss / len(train_loader):.3f}.. " \
                     f"Valid semantic loss: {val_semantic_loss / len(valid_loader):.3f}.. \n" \
                     f"Train acc: {train_acc / len(train_loader):.3f}.. " \
                     f"Valid acc: {val_acc / len(valid_loader):.3f}.. " \
                     f"Train AUC: {train_auc / (len(train_loader) - num_train_batches_without_auc):.3f}.. " \
                     f"Valid AUC: {val_auc / (len(valid_loader) - num_valid_batches_without_auc):.3f}.. \n" \
                     f"{res_train_concepts_auc} \n" \
                     f"{res_val_concepts_auc}"

            print(result)

            with open(f'{filename}.txt', "a") as file:
                file.write(result + '\n')

            valid_loss_decrease = early_stopping(valid_loss / len(valid_loader))

            if early_stopping.early_stop:
                print("Early stopping")
                break
            elif valid_loss_decrease is not None:
                with open(f'{filename}.txt', "a") as file:
                    file.write(valid_loss_decrease + '\n')

                torch.save({'classes': class_labels,
                            'model_state_dict': self.mapping_net.state_dict(),
                            'main_net_module_name': main_net_module_name,
                            'main_net_class': main_net_class,
                            'main_model_filename': main_model_filename,
                            'transformation_name': transformation_name,
                            'img_size': img_size,
                            'num_channels': num_channels,
                            'layers_types': self.activation_extractor.get_layers_types(),
                            'layers': self.activation_extractor.get_layers_for_research(),
                            'decoder_channels': self.mapping_net.get_decoder_channels(),
                            'num_shared_neurons': self.mapping_net.get_num_shared_neurons(),
                            'num_output_neurons': self.mapping_net.get_num_output_neurons(),
                            'num_outs': self.mapping_net.get_num_outs()
                            }, f'{filename}.rvl')

    def evaluate_model(self, test_loader):
        """
        Evaluates the mapping network model on the test set.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            Training data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.

        Returns
        -------
        res_test_concepts_auc : list[float]
            ROC AUC values for each of the concepts.
        test_auc : float
            The ROC AUC value of a single mapping network or the ROC AUC value for all labels of a simultaneous mapping
            network.
        """

        self.mapping_net.to(self.device)

        criterion = nn.BCELoss()

        main_net = self.activation_extractor.get_main_net()
        main_net.eval()
        self.mapping_net.eval()

        num_test_batches_without_auc = 0
        test_loss = 0
        test_acc = 0
        test_auc = 0

        test_concepts_auc = [0] * len(self.class_labels)
        num_test_batches_without_concepts_auc = [0] * len(self.class_labels)

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = main_net(images)

                logits = self.mapping_net(self.activation_extractor.get_activations(test_loader.batch_size))

                if len(self.class_labels) > 1:
                    logits = torch.cat(logits, dim=1)

                labels = labels.float()

                batch_loss = criterion(logits, labels)
                test_loss += batch_loss.item()

                predictions = (logits > 0.5).long()
                test_acc += accuracy_score(labels.cpu(), predictions.cpu())

                try:
                    auc = roc_auc_score(labels.cpu(), logits.cpu().detach().numpy())
                    test_auc += auc
                except ValueError:
                    num_test_batches_without_auc += 1

                if len(self.class_labels) > 1:
                    for i, output in enumerate(logits.T.cpu().detach().numpy()):
                        try:
                            auc = roc_auc_score(labels.T[i].cpu(), output)
                            test_concepts_auc[i] += auc
                        except ValueError:
                            num_test_batches_without_concepts_auc[i] += 1

        test_loss = test_loss / len(test_loader)
        test_acc = test_acc / len(test_loader)
        test_auc = test_auc / (len(test_loader) - num_test_batches_without_auc)

        res_test_concepts_auc = []
        if len(self.class_labels) > 1:
            for i in range(len(self.class_labels)):
                res_test_concepts_auc.append(
                    test_concepts_auc[i] / (len(test_loader) - num_test_batches_without_concepts_auc[i]))

        print(f"Test loss: {test_loss:.4f}.. "
              f"Test acc: {test_acc:.4f}.. "
              f"Test AUC: {test_auc:.4f}.. \n"
              f"Test concepts AUC: {res_test_concepts_auc}")

        return res_test_concepts_auc, test_auc
