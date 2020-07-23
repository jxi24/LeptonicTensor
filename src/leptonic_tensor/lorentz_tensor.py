import numpy as np
import collections


LKEYS = list(map(chr, range(65, 91)))
SKEYS = list(map(chr, range(97, 123)))


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

    def __lt__(self, rhs):
        return self.index < rhs.index

    def __hash__(self):
        return self.index.__hash__()

    def __repr__(self):
        return f"Index({self.index}, {self.lorentz})"

    def __str__(self):
        if self.lorentz:
            return LKEYS[self.index]
        return SKEYS[self.index]
        # if self.lorentz:
        #     return f'mu{self.index}'
        # else:
        #     return f'i{self.index}'

    def __int__(self):
        if isinstance(self.index, np.int64):
            return self.index.item()
        return self.index


class LorentzIndex(Index):
    def __init__(self, index, raised=True):
        super().__init__(index, True)
        self.raised = raised


class SpinIndex(Index):
    def __init__(self, index):
        super().__init__(index, False)


class Tensor:
    def __init__(self, array, indices=None):
        self._array = np.array(array)
        self._indices = tuple(indices) if indices is not None else None
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
                return np.array_equal(self._array, other._array)
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
        elif other._scalar:
            return self*(1/float(other._array))
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
        return self.sum(other)

    def __radd__(self, other):
        if isinstance(other, (int, float, complex)):
            result = self._array + other
            return Tensor(result, self._indices)

        if not isinstance(other, Tensor):
            raise TypeError("Invalid argument")

        if set(self._indices) != set(other._indices):
            raise ValueError(f"Inconsistent indices in addition: "
                             f"{self._indices} and {other._indices}")
        return __add__(other, self)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return Tensor(-self._array, self._indices)

    def __pos__(self):
        return Tensor(self._array, self._indices)

    def __float__(self):
        if self._scalar:
            return float(self._array)
        raise TypeError("Tensor is not a scalar")

    def __complex__(self):
        if self._scalar:
            return complex(self._array)
        raise TypeError("Tensor is not a scalar")

    def conjugate(self):
        return Tensor(self._array.conjugate(), self._indices)

    def copy(self):
        return Tensor(self._array, self._indices)

    def real(self):
        if self._scalar:
            return self._array.real
        raise TypeError("Tensor is not a scalar")

    @staticmethod
    def _merge(arr, temp, left, mid, right):
        nswaps = 0
        i = left
        j = mid
        k = left
        while (i <= mid - 1) and (j <= right):
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                k += 1
                i += 1
            else:
                temp[k] = arr[j]
                k += 1
                j += 1

                nswaps = nswaps + (mid - i)

        while i <= mid - 1:
            temp[k] = arr[i]
            k += 1
            i += 1

        while j <= right:
            temp[k] = arr[j]
            k += 1
            j += 1

        for i in range(left, right+1, 1):
            arr[i] = temp[i]

        return nswaps

    @staticmethod
    def _merge_sort(arr, temp, left, right):
        nswaps = 0
        if right > left:
            mid = int((right + left)/2)
            nswaps = Tensor._merge_sort(arr, temp, left, mid)
            nswaps += Tensor._merge_sort(arr, temp, mid+1, right)
            nswaps += Tensor._merge(arr, temp, left, mid+1, right)

        return nswaps

    @staticmethod
    def _swaps(idx):
        n = len(idx)
        if n == 0:
            return 0
        arr = [int(x) for x in idx]
        temp = [0 for i in range(n)]
        return Tensor._merge_sort(arr, temp, 0, n-1)

    def sum(self, rhs):
        # Perform sum with transposes
        arg_indices = np.argsort(np.array(self._indices))
        # rhs_indices = np.argsort(np.array([int(x) for x in rhs._indices]))
        rhs_indices = np.argsort(np.array([x for x in rhs._indices]))
        rhs_array = np.transpose(rhs._array, rhs_indices)
        rhs_array = np.transpose(rhs_array, np.argsort(arg_indices))

        return Tensor(self._array + rhs_array, self._indices)

    def contract(self, rhs):
        # Get indices
        lorentz_lhs = [idx for idx in self._indices if idx.lorentz]
        spin_lhs = [idx for idx in self._indices if not idx.lorentz]

        lorentz_rhs = [idx for idx in rhs._indices if idx.lorentz]
        spin_rhs = [idx for idx in rhs._indices if not idx.lorentz]

        # Get output indices
        out_lorentz = list(elm for elm, count
                           in collections.Counter(lorentz_lhs
                                                  + lorentz_rhs).items()
                           if count == 1)
        out_spin = list(elm for elm, count
                        in collections.Counter(spin_lhs
                                               + spin_rhs).items()
                        if count == 1)
        out_indices = ''.join([str(x) for x in out_lorentz + out_spin])
        lhs_indices = ''.join([str(x) for x in self._indices])
        rhs_indices = ''.join([str(x) for x in rhs._indices])
        einsum = f'{lhs_indices},{rhs_indices}->{out_indices}'

        out_array = np.einsum(einsum, self._array, rhs._array)

        return Tensor(out_array, out_lorentz + out_spin)

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
        # Get indices
        lorentz_lhs = [idx for idx in self._indices if idx.lorentz]
        spin_lhs = [idx for idx in self._indices if not idx.lorentz]

        # Get output indices
        out_lorentz = list(elm for elm, count
                           in collections.Counter(lorentz_lhs).items()
                           if count == 1)
        out_spin = list(elm for elm, count
                        in collections.Counter(spin_lhs).items()
                        if count == 1)
        out_indices = ''.join([str(x) for x in out_lorentz + out_spin])
        lhs_indices = ''.join([str(x) for x in self._indices])
        einsum = f'{lhs_indices}->{out_indices}'

        out_array = np.einsum(einsum, self._array)

        return Tensor(out_array, out_lorentz + out_spin)
