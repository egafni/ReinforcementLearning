import numpy

from rl.algos import cross_entropy


def test_cross_entropy():
    loss = cross_entropy.main(max_epochs=3, seed=1)
    # getting some precision error with git hub's CI
    # since we're switching to gitlab eventually, numpy.isclose is good enough
    assert numpy.isclose(float(loss), 0.6615647673606873)
