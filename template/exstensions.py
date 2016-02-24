from pylearn2.train_extensions import TrainExtension
import scipy as sp
import scipy.stats as stat

class ObserveWeights(TrainExtension):

    def on_monitor(self, model, dataset, algorithm):
        try:
            total = 0
            nweights = 0
            for layer in model.layers:
                weight = layer.get_weights()
                indexer = []
                absweight = abs(weight)
                means = absweight.mean(axis=0)
                stds = absweight.std(axis=0)
                for w, mu, sigma in zip(weight, means, stds):
                    nweights += len(w)
                    gauss = stat.norm(loc=mu, scale=sigma)
                    prob = gauss.cdf(0)
                    #print "Prob: {}".format(prob)
                    n = int(len(w)*prob)
                    total += n
                    bad = sorted(abs(w))[:n]
                    index = sp.array([True if t in bad else False for t in abs(w)], dtype=bool)
                    #print w[index]
                    indexer.append(index)
            print "Total Bad Weights: {}\nOut of: {}\nPercent: {}".format(total, nweights, total*100.0/nweights)
        except IndexError as e:
            print e