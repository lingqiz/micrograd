import numpy as np
import torch

class Tensor:
    '''
    A simple vector generalization for micrograd
    with autodiff for matrix-vector product
    '''

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0

        self._children = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # gradient pass through
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __sub__(self, other):
        neg = Tensor(np.ones_like(other.data) * -1)
        return self + (other * neg)

    def __matmul__(self, other):
        '''
        Matrix-vector product
        The backward function computes the gradient
        (Jacobian-vector product)
        '''
        out = Tensor(self.data @ other.data, (self, other), 'x')

        def _backward():
            other.grad += self.data.T @ out.grad
            self.grad += out.grad @ other.data.T

        out._backward = _backward

        return out

    def relu(self):
        ind = self.data > 0
        out = Tensor(self.data * ind, (self,), 'ReLU')

        def _backward():
            # gradient flows back if ind > 0
            self.grad += ind * out.grad

        out._backward = _backward

        return out

    def mss(self):
        val = np.mean((self.data ** 2))
        out = Tensor(val, (self, ), 'MSS')

        def _backward():
            n = float(self.data.size)
            partial = 2/n * self.data
            self.grad += partial * out.grad

        out._backward = _backward

        return out

    def backward(self):
        '''
        The backward call is the same as scalar micrograd
        '''
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

class MLP:
    '''
    MLP with two hidden layers
    '''

    def __init__(self, n_in, n_hidden, n_out):
        self.W1 = Tensor(np.random.randn(n_hidden, n_in))
        self.b1 = Tensor(np.zeros((n_hidden, 1)))

        self.W2 = Tensor(np.random.randn(n_hidden, n_hidden))
        self.b2 = Tensor(np.zeros((n_hidden, 1)))

        self.W3 = Tensor(np.random.randn(n_out, n_hidden))
        self.b3 = Tensor(np.zeros((n_out, 1)))

        self.params = [self.W1, self.b1,
                       self.W2, self.b2,
                       self.W3, self.b3]

    def forward(self, x):
        x = self.W1 @ x + self.b1
        x = x.relu()
        x = self.W2 @ x + self.b2
        x = x.relu()
        x = self.W3 @ x + self.b3
        return x

    def predict(self, X):
        y_pred = []
        for idx in range(X.shape[0]):
            x = Tensor(X[idx].reshape(2, 1))
            y_p = self.forward(x)
            y_pred.append(y_p.data)

        y_pred = np.array(y_pred).flatten()
        y_pred = (y_pred > 1.0) * 2 - 1

        return y_pred

    def zero_grad(self):
        for p in self.params:
            p.grad = 0