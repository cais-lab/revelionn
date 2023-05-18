import numpy as np
from pytest import fixture

from revelionn.early_stopping import EarlyStopping


@fixture
def early_stopping():
    return EarlyStopping(patience=2, verbose=True)


def test_early_stopping_init(early_stopping):
    assert early_stopping.patience == 2
    assert early_stopping.verbose == True
    assert early_stopping.counter == 0
    assert early_stopping.best_score is None
    assert early_stopping.early_stop == False
    assert early_stopping.val_loss_min == np.Inf
    assert early_stopping.delta == 0
    assert early_stopping.trace_func == print


def test_early_stopping_call(early_stopping):
    val_loss = 10.0
    assert early_stopping(val_loss) == 'Validation loss decreased (inf --> 10.000000).  Saving model ...'
    assert early_stopping.best_score == -10.0
    assert early_stopping.counter == 0

    val_loss = 9.5
    assert early_stopping(val_loss) == 'Validation loss decreased (10.000000 --> 9.500000).  Saving model ...'
    assert early_stopping.best_score == -9.5
    assert early_stopping.counter == 0

    val_loss = 9.6
    assert early_stopping(val_loss) is None
    assert early_stopping.best_score == -9.5
    assert early_stopping.counter == 1

    val_loss = 9.7
    assert early_stopping(val_loss) is None
    assert early_stopping.best_score == -9.5
    assert early_stopping.counter == 2

    val_loss = 9.8
    assert early_stopping(val_loss) is None
    assert early_stopping.best_score == -9.5
    assert early_stopping.counter == 3
    assert early_stopping.early_stop == True


def test_early_stopping_save_checkpoint(early_stopping):
    val_loss = 9.5
    assert early_stopping.save_checkpoint(val_loss) == 'Validation loss decreased (inf --> 9.500000).  Saving model ...'
    assert early_stopping.val_loss_min == 9.5
