import numpy as np


class EarlyStopping:
    """
    Early stopping class to stop training when validation loss stops improving.

    Attributes
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping.
    verbose : bool
        If True, prints a message when validation loss decreases and the model is saved.
    counter : int
        Counter to track the number of epochs without improvement.
    best_score : float or None
        Best score (negative validation loss) obtained so far.
    early_stop : bool
        Flag indicating whether to stop the training.
    val_loss_min : float
        Minimum validation loss observed so far.
    delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    trace_func : function
        A function used to trace the output message.

    Methods
    -------
    __call__(val_loss)
        Call the early stopping class and determine whether to stop the training.
    save_checkpoint(val_loss)
        Save the model checkpoint when the validation loss decreases.
    """

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Initialize the EarlyStopping class.

        Parameters
        ----------
        patience : int, optional
            Number of epochs to wait for improvement before stopping. Default is 7.
        verbose : bool, optional
            If True, prints a message when validation loss decreases and the model is saved. Default is False.
        delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
        trace_func : function, optional
            A function used to trace the output message. Default is the built-in `print` function.
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss):
        """
        Call the EarlyStopping class and determine whether to stop the training.

        Parameters
        ----------
        val_loss : float
            The validation loss value to evaluate.

        Returns
        -------
        str
            A message indicating that the validation loss decreased and the model is saved, or None.
        """

        score = -val_loss
        valid_loss_decrease = None
        if self.best_score is None:
            self.best_score = score
            valid_loss_decrease = self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            valid_loss_decrease = self.save_checkpoint(val_loss)
            self.counter = 0
        return valid_loss_decrease

    def save_checkpoint(self, val_loss):
        """
        Save the model checkpoint when the validation loss decreases.

        Parameters
        ----------
        val_loss : float
            The current validation loss value.

        Returns
        -------
        str
            A message indicating that the validation loss decreased and the model is saved, or None.
        """

        valid_loss_decrease = None
        if self.verbose:
            valid_loss_decrease = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  ' \
                                  f'Saving model ...'
            self.trace_func(valid_loss_decrease)

        self.val_loss_min = val_loss

        return valid_loss_decrease
