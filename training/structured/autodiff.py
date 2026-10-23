"""Small reverse-mode NumPy graph with explicit straight-through truncation.

The straight-through derivative is a surrogate, not the derivative of integer arithmetic.
Broadcasting, indexing and reductions accumulate adjoints back into parameter arrays.
"""
import numpy as np


class Tensor:
    __array_priority__ = 1000

    def __init__(self, value, parents=()):
        self.value = np.asarray(value, dtype=np.float64)
        self.parents = parents
        self.grad = np.zeros_like(self.value)

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    def _binary(self, other, operation, left, right):
        other = self.wrap(other)
        a, b = self.value, other.value
        return Tensor(
            operation(a, b),
            (
                (self, lambda g: unbroadcast(left(g, a, b), a.shape)),
                (other, lambda g: unbroadcast(right(g, a, b), b.shape)),
            ),
        )

    def __add__(self, other):
        return self._binary(other, np.add, lambda g, a, b: g, lambda g, a, b: g)

    __radd__ = __add__

    def __neg__(self):
        return Tensor(-self.value, ((self, lambda g: -g),))

    def __sub__(self, other):
        return self + -self.wrap(other)

    def __rsub__(self, other):
        return self.wrap(other) + -self

    def __mul__(self, other):
        return self._binary(
            other, np.multiply, lambda g, a, b: g * b, lambda g, a, b: g * a
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(
            other, np.divide, lambda g, a, b: g / b, lambda g, a, b: -g * a / (b * b)
        )

    def pow(self, exponent):
        exponent = np.asarray(exponent)
        value = np.power(self.value, exponent)
        derivative = np.where(
            exponent == 0, 0, exponent * np.power(self.value, exponent - 1)
        )
        return Tensor(
            value, ((self, lambda g: unbroadcast(g * derivative, self.value.shape)),)
        )

    def __getitem__(self, index):
        def pullback(g):
            out = np.zeros_like(self.value)
            np.add.at(out, index, g)
            return out

        return Tensor(self.value[index], ((self, pullback),))

    def reshape(self, *shape):
        return Tensor(
            self.value.reshape(*shape), ((self, lambda g: g.reshape(self.value.shape)),)
        )

    def sum(self, axis=None):
        def pullback(g):
            if axis is not None:
                g = np.expand_dims(g, axis)
            return np.broadcast_to(g, self.value.shape)

        return Tensor(self.value.sum(axis=axis), ((self, pullback),))

    def trunc(self):
        return Tensor(np.trunc(self.value), ((self, lambda g: g),))

    def clip(self, low, high):
        active = (self.value > low) & (self.value < high)
        return Tensor(np.clip(self.value, low, high), ((self, lambda g: g * active),))

    def backward(self, seed=None):
        order, visited = [], set()

        def visit(tensor):
            if id(tensor) in visited:
                return
            visited.add(id(tensor))
            for parent, _ in tensor.parents:
                visit(parent)
            order.append(tensor)

        visit(self)
        for node in order:
            node.grad.fill(0)
        self.grad[...] = np.ones_like(self.value) if seed is None else seed
        for node in reversed(order):
            for parent, pullback in node.parents:
                parent.grad += pullback(node.grad)


def unbroadcast(gradient, shape):
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, length in enumerate(shape):
        if length == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)
