import numpy as np

from rakhshai_graph_nlp.metrics import accuracy, macro_f1, confusion_matrix


def test_classification_metrics():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 2, 1, 1])

    acc = accuracy(y_true, y_pred)
    f1 = macro_f1(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    assert acc == 0.5
    assert np.isclose(f1, 0.5)
    assert np.array_equal(
        cm,
        np.array(
            [
                [1, 0, 0],
                [0, 1, 1],
                [0, 1, 0],
            ]
        ),
    )
