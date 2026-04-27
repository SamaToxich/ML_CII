import numpy as np


class Tensor(object):
    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if id is None:
            id = np.random.randint(1, 100_000)
        self.id = id
        if isinstance(data, np.ndarray):
            self.shape = data.shape
        else:
            self.shape = np.array(data).shape

        if creators is not None:
            for x in creators:
                if self.id not in x.children:
                    x.children[self.id] = 1
                else:
                    x.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))

            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("не можем выполнить обратное распространение более одного раза")
                else:
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
                if self.creation_op == "add":
                    self.creators[0].backward(grad)
                    self.creators[1].backward(grad)

                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())

                if self.creation_op == "sub":
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)

                if self.creation_op == "mul":
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new,self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new,self)

                if self.creation_op == "mm":
                    act = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mm(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)

                if self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())

                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    ds = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim,ds))

                if "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

                if self.creation_op == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))

                if self.creation_op == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self. grad * (ones - (self * self)))

                if self.creation_op == "softmax":
                    grad_input = np.zeros_like(self.data)
                    for i in range(self.data.shape[0]):
                        s = self.data[i].reshape(-1, 1)
                        jacobian = np.diagflat(s) - s @ s.T
                        grad_input[i] = self.grad.data[i] @ jacobian

                    self.creators[0].backward(Tensor(grad_input))

                if(self.creation_op == "index_select"):
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))

    def __repr__(self):
            return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, creators=[self,other], autograd=True, creation_op="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, autograd=True, creators=[self], creation_op="neg")
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, autograd=True, creators=[self, other], creation_op="sub")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        # Проверяем, является ли other тензором
        if isinstance(other, Tensor):
            if self.autograd and other.autograd:
                return Tensor(self.data * other.data, autograd=True, creators=[self, other], creation_op="mul")
            return Tensor(self.data * other.data)
        else:
            # Если other - число или numpy массив
            return Tensor(self.data * other)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),autograd=True,creators=[self],creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data, autograd=True, creators=[self], creation_op="expand_" + str(dim))
        return Tensor(new_data)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), autograd=True, creators = [self], creation_op="transpose")
        return Tensor(self.data.transpose())

    def mm(self, x):
        if self.autograd:
            return Tensor(self.data @ x.data, autograd=True, creators=[self, x], creation_op="mm")
        return Tensor(self.data @ x.data)

    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)), autograd=True, creators=[self], creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), autograd=True, creators=[self], creation_op="tanh")
        return Tensor(np.tanh(self.data))

    def softmax(self):
        if self.autograd:
            return Tensor(np.exp(self.data) / np.sum(np.exp(self.data), axis=1, keepdims=True),autograd=True, creators=[self], creation_op="softmax")
        return Tensor(np.exp(self.data) / np.sum(np.exp(self.data), axis=1, keepdims=True))

    def index_select(self, indices):
        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

class SGD(object):
    def __init__(self, tensors, alpha=0.1):
        self.tensors = tensors
        self.alpha = alpha

    def zero(self):
        for t in self.tensors:
            t.grad.data *= 0

    def step(self, zero=True):
        for t in self.tensors:

            t.data -= t.grad.data * self.alpha

            if zero:
                t.grad.data *= 0


class Layer(object):
    def __init__(self):
        self.tensors = list()

    def get_tensors(self):
        return self.tensors


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0/n_inputs)

        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.tensors.append(self.weight)
        self.tensors.append(self.bias)

    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))


class Sequential(Layer):
    def __init__(self, layers = list()):
        super().__init__()

        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_tensors()
        return params


class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target)*(pred-target)).sum(0)


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()


class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.softmax()


class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim

        weight = (np.random.rand(vocab_size, dim) - 0.5) / dim
        self.weight = Tensor(weight, autograd=True)
        self.tensors.append(self.weight)

    def forward(self, input):
        return self.weight.index_select(input)