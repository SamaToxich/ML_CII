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
            id = np.random.randint(0, 1_000_000_000)
        self.id = id


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

            if(grad_origin is not None):
                if(self.children[grad_origin.id] == 0):
                    return
                else:
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            assert grad.autograd == False

            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
                if self.creation_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())

                if self.creation_op == "sub":
                    self.creators[0].backward(Tensor(self.grad.data), self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

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

                if(self.creation_op == "index_select"):
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))

                if(self.creation_op == "cross_entropy"):
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))

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
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, autograd=True, creators=[self, other], creation_op="mul")
        return Tensor(self.data * other.data)

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
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,axis=len(self.data.shape)-1,keepdims=True)
        return softmax_output

    def index_select(self, indices):
        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,axis=len(self.data.shape)-1,keepdims=True)

        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t),-1)

        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()

        if(self.autograd):
            out = Tensor(loss,autograd=True,creators=[self],creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out

        return Tensor(loss)

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
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()

        self.use_bias = bias

        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0/n_inputs)

        self.weight = Tensor(W, autograd=True)

        if self.use_bias:
            self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.tensors.append(self.weight)

        if self.use_bias:
            self.tensors.append(self.bias)

    def forward(self, input):
        if self.use_bias:
            return input.mm(self.weight)+self.bias.expand(0,len(input.data))
        return input.mm(self.weight)


class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim

        self.weight = Tensor((np.random.rand(vocab_size, dim) - 0.5) / dim, autograd=True)

        self.tensors.append(self.weight)

    def forward(self, input):
        return self.weight.index_select(input)


class RNNCell(Layer):
    def __init__(self, n_in, n_hide, n_out, activation='sigmoid'):
        super().__init__()

        self.n_in = n_in
        self.n_hide = n_hide
        self.n_out = n_out

        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            raise Exception("Функция активации не найдена")

        self.w_ih = Linear(n_in, n_hide)
        self.w_hh = Linear(n_hide, n_hide)
        self.w_ho = Linear(n_hide, n_out)

        self.tensors += self.w_ih.get_tensors()
        self.tensors += self.w_hh.get_tensors()
        self.tensors += self.w_ho.get_tensors()

    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)

        combined = self.w_ih.forward(input) + from_prev_hidden

        new_hidden = self.activation.forward(combined)

        output = self.w_ho.forward(new_hidden)

        return output, new_hidden

    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.n_hide)), autograd=True)


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


class CrossEntropyLoss(object):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.cross_entropy(target)


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
