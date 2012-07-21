import theano

class DoubleOp(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        #print type(inputs[0])
        x = inputs[0]
        z = output_storage[0]
        z[0] = x * 2

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        print type(output_grads[0])
        return [output_grads[0] * 2]

    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)



from theano import tensor

x = theano.tensor.matrix()
f = theano.function([x], DoubleOp()(x))
l = DoubleOp()(x).sum()
g = theano.function([x], tensor.grad(l, x))
import numpy
inp = numpy.random.rand(5, 4).astype('f')
out = f(inp)
print g(inp)
assert numpy.allclose(inp * 2, out)
print inp
print out
