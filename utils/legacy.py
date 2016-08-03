class Permute(Layer):
    def __init__(self, output_dim, permutations, batch_size, **kwargs):
        self.output_dim = output_dim
        self.permutations = permutations
        self.num_p = len(permutations)
        self.batch_size = batch_size
        super(Permute, self).__init__(**kwargs)

    def build(self, input_shape):
        self.transforms = [E(p) for p in self.permutations]
        self.T = K.variable(self.transforms)
        self.trainable = False

    def call(self, x, mask=None):

        skip = x.shape[0]
        temp = K.concatenate([K.dot(x, self.T[i].transpose()) for i in xrange(self.num_p)], axis=0)

        """
        def shift(x, y, i, step, skip):
            return T.set_subtensor(x[i*step:(i+1)*step], y[i::skip])

        out, _ = theano.scan(lambda b, prior, step: shift(prior, temp, b, step, skip),
                             n_steps=skip,
                             outputs_info=temp,
                             sequences=[T.arange(skip)],
                             non_sequences=[self.num_p])
        return out[-1]"""
        return temp

    def get_output_shape_for(self, input_shape):
        return (self.num_p, self.output_dim)

    def compute_mask(self, input, input_mask=None):
        return None


def clean_outputs(x):
    skip = x.shape[0] // len(perms)
    out, _ = theano.scan(lambda b, x, step: K.reshape(x[b::skip], (1, x.shape[1] * step)),
                         n_steps=skip,
                         sequences=[T.arange(skip)],
                         non_sequences=[x, len(perms)])
    return T.flatten(out, outdim=2)


o = Lambda(clean_outputs,
           output_shape=lambda s: (s[0] // len(perms), 20 * len(perms)),
           name="Filter")(o)