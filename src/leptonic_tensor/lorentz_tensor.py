import numpy as np
import collections


class Index:
    def __init__(self, index, lorentz):
        self.index = index
        self.lorentz = lorentz

    def __eq__(self, rhs):
        if isinstance(rhs, Index):
            if self.lorentz != rhs.lorentz:
                return False
            return self.index == rhs.index
        return False

    def __ne__(self, rhs):
        return not self == rhs

    def __hash__(self):
        return self.index.__hash__()

    def __repr__(self):
        return f"Index({self.index}, {self.lorentz})"

    def __str__(self):
        if self.lorentz:
            return f'mu{self.index}'
        else:
            return f'i{self.index}'

    def __int__(self):
        return self.index


class LorentzIndex(Index):
    def __init__(self, index, raised=True):
        super().__init__(index, True)
        self.raised = raised


class SpinIndex(Index):
    def __init__(self, index):
        super().__init__(index, False)


class Tensor:
    def __init__(self, array, indices):
        self._array = np.array(array)
        self._indices = tuple(indices)
        self._scalar = (True if self._indices is None
                        or self._indices == tuple()
                        else False)
        if self._array.ndim != len(self._indices) and self._array.size != 1:
            raise IndexError(f"Tensor not properly indexed. Needs "
                             f"{self._array.ndim} indices, got "
                             f"{len(self._indices)}")

    def __repr__(self):
        return "Tensor({}, {})".format(self._array, self._indices)

    def __str__(self):
        if self._scalar:
            return str(self._array)
        else:
            return "indices: {}\ndata:\n{}".format(self._indices, self._array)

    def __eq__(self, other):
        if self._scalar:
            if isinstance(other, (int, float, complex)):
                return self._array == other
            elif other._scalar:
                return self._array[0] == other._array[0]
        return (np.array_equal(self._array, other._array)
                and self._indices == other._indices)

    def __ne__(self, other):
        return not self == other

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            result = self._array * other
            return Tensor(result, self._indices)

        if not isinstance(other, Tensor):
            raise TypeError(f"Tensor multiplication is not valid for type "
                            f"{type(other)}")

        return self.contract(other)

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            result = self._array * other
            return Tensor(result, self._indices)
        return other*self

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            return self*(1/other)
        raise TypeError(f"Tensor division is not valid for type {type(other)}")

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            result = self._array + other
            return Tensor(result, self._indices)

        if not isinstance(other, Tensor):
            raise TypeError("Invalid argument")

        if set(self._indices) != set(other._indices):
            raise ValueError(f"Inconsistent indices in addition: "
                             f"{self._indices} and {other._indices}")
        result = self._array + other._array
        return Tensor(result, self._indices)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return Tensor(-self._array, self._indices)

    def __pos__(self):
        return Tensor(self._array, self._indices)

    def contract(self, rhs):
        # Get indices
        lorentz_lhs = [idx for idx in self._indices if idx.lorentz]
        spin_lhs = [idx for idx in self._indices if not idx.lorentz]

        lorentz_rhs = [idx for idx in rhs._indices if idx.lorentz]
        spin_rhs = [idx for idx in rhs._indices if not idx.lorentz]

        out_lorentz = tuple(elm for elm, count
                            in collections.Counter(lorentz_lhs
                                                   + lorentz_rhs).items()
                            if count == 1)
        out_spin = tuple(elm for elm, count
                         in collections.Counter(spin_lhs
                                                + spin_rhs).items()
                         if count == 1)
        out_indices = out_lorentz + out_spin
        lhs_indices = [int(x) for x in self._indices]
        rhs_indices = [int(x) for x in rhs._indices]

        out_array = np.einsum(self._array, lhs_indices,
                              rhs._array, rhs_indices)
        indices = list(self._indices) + list(rhs._indices)
        out_indices = tuple(elm for elm, count
                            in collections.Counter(indices).items()
                            if count == 1)
        return Tensor(out_array, out_indices)

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
        out_array = np.einsum(self._array, self._indices)
        out_keys, out_counts = np.unique(self._indices, return_counts=True)
        out_keys = tuple(key.item() for i, key in enumerate(out_keys) if out_counts[i] == 1)
        return Tensor(out_array, out_keys)
