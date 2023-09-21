import pytest
from optimization import methods


def test_should_return_correct_root():
    """
    Test if the method returns the correct root
    """
    # TODO: Add test input arguments
    assert methods.classical_newton() == pytest.approx()

    
def test_should_hang():
    """
    Test if the method detects that a function has landed in an infinite cycle
    """
    with pytest.raises(RuntimeError):
        # TODO: Add test input arguments
        methods.classical_newton() # etc, f(x) = X^3 - 2x + 2 should hang (stuck in 2-cycle).

def test_should_raise_value_error():
    """
    Test if the method raises a ValueError when the derivative is zero
    """
    with pytest.raises(ValueError):
        # TODO: Add test input arguments
        methods.classical_newton()
