import numpy as np


class Tensor:
    def __init__(self, array, keys):
        self._array = np.array(array)
        self._keys = tuple(keys)
        self._scalar = (True if self._keys is None or self._keys == tuple()
                        else False)
        if self._array.ndim != len(self._keys) and self._array.size != 1:
            raise IndexError(f"Tensor not properly indexed. Needs "
                             f"{self._array.ndim} indices, got "
                             f"{len(self._keys)}")

    def __repr__(self):
        return "Tensor({}, {})".format(self._array, self._keys)

    def __str__(self):
        if self._scalar:
            return str(self._array)
        else:
            return "indices: {}\ndata:\n{}".format(self._keys, self._array)

    def __eq__(self, other):
        if self._scalar:
            if isinstance(other, (int, float, complex)):
                return self._array == other
            elif other._scalar:
                return self._array[0] == other._array[0]
        return (np.array_equal(self._array, other._array)
                and self._keys == other._keys)

    def __ne__(self, other):
        return not self == other

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            result = self._array * other
            return Tensor(result, self._keys)

        if not isinstance(other, Tensor):
            raise TypeError(f"Tensor multiplication is not valid for type "
                            f"{type(other)}")

        return self.contract(other)

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            result = self._array * other
            return Tensor(result, self._keys)
        return other*self

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            return self*(1/other)
        raise TypeError(f"Tensor division is not valid for type {type(other)}")

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            result = self._array + other
            return Tensor(result, self._keys)

        if self._keys != other._keys:
            raise ValueError(f"Inconsistent keys in addition: {self._keys} "
                             f"and {other._keys}")
        result = self._array + other._array
        return Tensor(result, self._keys)

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            result = self._array - other
            return Tensor(result, self._keys)

        if self._keys != other._keys:
            raise ValueError(f"Inconsistent keys in addition: {self._keys} "
                             f"and {other._keys}")
        result = self._array - other._array
        return Tensor(result, self._keys)

    def __neg__(self):
        return Tensor(-self._array, self._keys)

    def __pos__(self):
        return Tensor(self._array, self._keys)

    def contract(self, other):
        out_array = np.einsum(self._array, self._keys,
                              other._array, other._keys)
        out_keys = tuple(i.item() for i in np.setxor1d(self._keys, other._keys))
        return Tensor(out_array, out_keys)

    def __getitem__(self, indices):
        if self._scalar:
            return self._array[0]
        else:
            return self._array[indices]

    def __setitem__(self, indices, value):
        if self._scalar:
            self._array[0] = value
        else:
            self._array[indices] = value

    def reduce(self):
        out_array = np.einsum(self._array, self._keys)
        out_keys, out_counts = np.unique(self._keys, return_counts=True)
        out_keys = tuple(key.item() for i, key in enumerate(out_keys) if out_counts[i] == 1)
        return Tensor(out_array, out_keys)
