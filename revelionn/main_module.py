import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn, optim

from .early_stopping import EarlyStopping


class MainModelProcessing:
    """
    Class for training, evaluation and processing the main network model.

    Attributes
    ----------
    device : torch.device
        Tensor processing device.
    main_net : torch.nn.Module
        The model of the main neural network.
    classes : dict
        Names of neural network output classes.

    Methods
    -------
    load_model(path_to_model_dict)
        Loads the weights of the neural network model from a file.
    train_model(patience, epochs, file_name, class_label_name, module_name,
                main_net_class, transformation_name, img_size, num_channels)
        Training and validation of the main neural network.
    evaluate_model(test_loader)
        Evaluation of the model on the test set.
    get_main_net()
        Returns the main neural network.
    get_class_labels()
        Returns names of neural network output classes.
    get_device()
        Returns the current tensor processing device.
    """

    def __init__(self, main_net, device):
        """
        Sets all the necessary attributes for the MainModelProcessing object.

        Parameters
        ----------
        main_net : torch.nn.Module
            The model of the main neural network.
        device : torch.device
            Tensor processing device.
        """

        self.device = device
        self.main_net = main_net.to(self.device)
        self.classes = None

    def get_main_net(self):
        """
        Returns the main neural network.

        Returns
        -------
        main_net : MainNet(nn.Module)
            The main neural network.
        """

        return self.main_net

    def get_class_labels(self):
        """
        Returns names of neural network output classes.

        Returns
        -------
        classes : dict
            Names of neural network output classes.
        """

        return self.classes

    def get_device(self):
        """
        Returns the current tensor processing device.

        Returns
        -------
        device : torch.device
            Tensor processing device.
        """

        return self.device

    def load_model(self, path_to_model):
        """
        Loads the weights of the neural network model from a file.

        Parameters
        ----------
        path_to_model : str
            The path to the file containing weights.

        Returns
        -------
        None
        """

        checkpoint = torch.load(path_to_model, map_location=self.device)
        self.main_net.load_state_dict(checkpoint['model_state_dict'])
        self.classes = checkpoint['classes']

    def train_model(self, train_loader, valid_loader, patience, epochs, filename, class_label,
                    module_name, main_net_class, transformation_name, img_size, num_channels):
        """
        Training and validation of the main neural network.

        Parameters
        ----------
        patience : int
            How many epochs to wait after last time validation loss improved.
        epochs : int
            The number of training epochs of the main neural network.
        filename : str
            The name of the file in which the parameters of the trained model will be saved.
        class_label : str
            The name of the label of the class used for training.
        module_name : str
            The name of the file containing the main network class.
        main_net_class : str
            Name of the main network class.
        transformation_name : str
            Name of the variable storing transformations.
        img_size : int
            The size of the image side.
        num_channels : int
            The number of image channels.

        Returns
        -------
        None
        """

        self.main_net.to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.main_net.parameters(), lr=0.001)

        early_stopping = EarlyStopping(patience=patience, verbose=True)

        for e in range(epochs):
            num_train_batches_without_auc = 0
            num_valid_batches_without_auc = 0
            train_loss = 0
            train_acc = 0
            train_auc = 0

            self.main_net.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.main_net(images)
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
            valid_acc = 0
            valid_auc = 0
            self.main_net.eval()
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = self.main_net(images)
                    labels = labels.float()
                    batch_loss = criterion(logits, labels)
                    valid_loss += batch_loss.item()

                    predictions = (logits > 0.5).long()
                    valid_acc += accuracy_score(labels.cpu(), predictions.cpu())

                    try:
                        auc = roc_auc_score(labels.cpu(), logits.cpu().detach().numpy())
                        valid_auc += auc
                    except ValueError:
                        num_valid_batches_without_auc += 1

            result = f"Epoch {e + 1}/{epochs}.. " \
                     f"Train loss: {train_loss / len(train_loader):.3f}.. " \
                     f"Valid loss: {valid_loss / len(valid_loader):.3f}.. " \
                     f"Train acc: {train_acc / len(train_loader):.3f}.. " \
                     f"Valid acc: {valid_acc / len(valid_loader):.3f}.. " \
                     f"Train AUC: {train_auc / (len(train_loader) - num_train_batches_without_auc):.3f}.. " \
                     f"Valid AUC: {valid_auc / (len(valid_loader) - num_valid_batches_without_auc):.3f}.. "

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

                classes = {1: class_label,
                           0: f'Not{class_label}'}

                torch.save({'classes': classes,
                            'model_state_dict': self.main_net.state_dict(),
                            'main_net_module_name': module_name,
                            'main_net_class': main_net_class,
                            'transformation_name': transformation_name,
                            'img_size': img_size,
                            'num_channels': num_channels
                            }, f'{filename}.rvl')

    def evaluate_model(self, test_loader):
        """
        Evaluation of the model on the test set.

        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            Training data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.

        Returns
        -------
        test_loss : float
            Test loss.
        test_acc : float
            Accuracy on the test set.
        test_auc : float
            ROC AUC on the test set.
        """

        criterion = nn.BCELoss()

        self.main_net.eval()

        num_test_batches_without_auc = 0
        test_loss = 0
        test_acc = 0
        test_auc = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.main_net(images)
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

        test_loss = test_loss / len(test_loader)
        test_acc = test_acc / len(test_loader)
        test_auc = test_auc / (len(test_loader) - num_test_batches_without_auc)

        print(f"Test loss: {test_loss:.4f}.. "
              f"Test acc: {test_acc:.4f}.. "
              f"Test AUC: {test_auc:.4f}.. ")

        return test_loss, test_acc, test_auc
