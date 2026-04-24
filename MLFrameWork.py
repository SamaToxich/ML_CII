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

    def __sum__(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),autograd=True,creators=[self],creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


a = Tensor([1,2,3,4,5], autograd=True)
b = Tensor([2,2,2,2,2], autograd=True)
c = Tensor([5,4,3,2,1], autograd=True)
d = a + b
e = b + c
f = d + e
f.backward(Tensor(np.array([1,1,1,1,1])))
print(b.grad.data == np.array([2,2,2,2,2]))


