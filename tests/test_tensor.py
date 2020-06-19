import numpy as np
import leptonic_tensor.lorentz_tensor as lt


def test_tensor_addition():
    # Test tensor + tensor without transposes required
    array = np.array([[0, 1, 0, 0], [2, 0, 0, 0], [0, 0, 0, 1], [0, 0, 2, 0]])
    tensor1 = lt.Tensor(array,
                        (lt.SpinIndex(0), lt.SpinIndex(1)))
    tensor2 = lt.Tensor(-array,
                        (lt.SpinIndex(0), lt.SpinIndex(1)))
    zeros = np.zeros_like(array)
    result = lt.Tensor(zeros, (lt.SpinIndex(0), lt.SpinIndex(1)))
    assert(tensor1 + tensor2 == result)

    # Test tensor + tensor with transposes required
    tensor2 = lt.Tensor(np.transpose(-array),
                        (lt.SpinIndex(1), lt.SpinIndex(0)))
    assert(tensor1 + tensor2 == result)

    # Test tensor + constant and constant + tensor for floats
    ones = np.ones_like(array)
    twos = 2*np.ones_like(array)
    tensor3 = lt.Tensor(ones, (lt.SpinIndex(0), lt.SpinIndex(1)))
    result = lt.Tensor(twos, (lt.SpinIndex(0), lt.SpinIndex(1)))
    assert(tensor3 + 1 == 1 + tensor3 == result)

    # Test tensor + constant and constant + tensor for complex
    one_complex = (1+1j)*np.ones_like(array)
    result = lt.Tensor(one_complex, (lt.SpinIndex(0), lt.SpinIndex(1)))
    assert(tensor3 + 1j == 1j + tensor3 == result)


def test_tensor_subtraction():
    # Test tensor - tensor without transposes required
    array = np.array([[0, 1, 0, 0], [2, 0, 0, 0], [0, 0, 0, 1], [0, 0, 2, 0]])
    tensor1 = lt.Tensor(array,
                        (lt.SpinIndex(0), lt.SpinIndex(1)))
    tensor2 = lt.Tensor(array,
                        (lt.SpinIndex(0), lt.SpinIndex(1)))
    zeros = np.zeros_like(array)
    result = lt.Tensor(zeros, (lt.SpinIndex(0), lt.SpinIndex(1)))
    assert(tensor1 - tensor2 == result)

    # Test tensor - tensor with transposes required
    tensor2 = lt.Tensor(np.transpose(array),
                        (lt.SpinIndex(1), lt.SpinIndex(0)))
    result = lt.Tensor(zeros, (lt.SpinIndex(0), lt.SpinIndex(1)))
    assert(tensor1 - tensor2 == result)

    # Test tensor - constant and constant - tensor for floats
    twos = 2*np.ones_like(array)
    threes = 3*np.ones_like(array)
    tensor3 = lt.Tensor(threes, (lt.SpinIndex(0), lt.SpinIndex(1)))
    result = lt.Tensor(twos, (lt.SpinIndex(0), lt.SpinIndex(1)))
    assert(tensor3 - 1 == -(1 - tensor3) == result)

    # Test tensor - constant and constant - tensor for complex
    ones = np.ones_like(array)
    one_complex = (3-1j)*np.ones_like(array)
    result = lt.Tensor(one_complex, (lt.SpinIndex(0), lt.SpinIndex(1)))
    assert(tensor3 - 1j == -(1j - tensor3) == result)


def test_tensor_multiplication():
    # Test tensor * tensor without transposes required
    array = np.array([[0, 1, 0, 0], [2, 0, 0, 0], [0, 0, 0, 1], [0, 0, 2, 0]])
    tensor1 = lt.Tensor(array,
                        (lt.SpinIndex(0), lt.SpinIndex(1)))
    tensor2 = lt.Tensor(array,
                        (lt.SpinIndex(0), lt.SpinIndex(1)))
    result_array = np.einsum('ij,ij->', array, array)
    result = lt.Tensor(result_array, ())
    assert(tensor1 * tensor2 == result)

    # Test tensor * tensor with transposes required
    tensor2 = lt.Tensor(np.transpose(array),
                        (lt.SpinIndex(1), lt.SpinIndex(0)))
    assert(tensor1 * tensor2 == result)

    # Test tensor * constant and constant * tensor for floats
    ones = np.ones_like(array)
    twos = 2*np.ones_like(array)
    tensor3 = lt.Tensor(ones, (lt.SpinIndex(0), lt.SpinIndex(1)))
    result = lt.Tensor(twos, (lt.SpinIndex(0), lt.SpinIndex(1)))
    assert(tensor3*2 == 2*tensor3 == result)

    # Test tensor * constant and constant * tensor for complex
    one_complex = 1j*np.ones_like(array)
    result = lt.Tensor(one_complex, (lt.SpinIndex(0), lt.SpinIndex(1)))
    assert(tensor3*1j == 1j*tensor3 == result)


def test_tensor_division():
    # Test tensor / constant for floats
    ones = np.ones((4, 4))
    twos = 2*np.ones((4, 4))
    tensor = lt.Tensor(twos, (lt.SpinIndex(0), lt.SpinIndex(1)))
    result = lt.Tensor(ones, (lt.SpinIndex(0), lt.SpinIndex(1)))
    assert(tensor/2 == result)

    # Test tensor / constant for complex
    twos_complex = twos/1j
    result = lt.Tensor(twos_complex, (lt.SpinIndex(0), lt.SpinIndex(1)))
    assert(tensor/1j == result)


def test_tensor_reduce():
    # Test tensor reduction
    identity = np.diag([1, 1, 1, 1])
    tensor = lt.Tensor(identity, (lt.SpinIndex(0), lt.SpinIndex(0)))
    assert(tensor.reduce() == 4)
