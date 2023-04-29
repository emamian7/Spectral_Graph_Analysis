## The purpose of this project was twofold. The first was to implement spectral clustering for 
#unsupervised learning, and the second was to explore the use of multiple kernel learning as 
#applied to spectral clustering. The algorithm was applied to novel data to gauge its efficacy and 
#compared to a k-Means implementation. 
import numpy as np
import random
from numpy import linalg
from numpy import random
from scipy import spatial
import matplotlib.pyplot as plt
import idx2numpy

"""
--------------- K-MEANs Clustering --------------------------------------------
"""

def calcEuclidean(x, y):
    sum = 0;
    for i in range(0, len(x)):
        sum += (abs(x[i] - y[i]))**2
    return np.sqrt(sum);

def calcManhattan(x, y):
    sum = 0;
    for i in range(0, len(x)):
        sum += abs(x[i] - y[i])
    return sum;

def stoppingCondition(X, Y):
    for i in range(0, len(X)):
        if (X[i] != Y[i]):
            return True;
    return False;

def calcQuality(k, clusters, lbls, distribution, classes):
    entropies = []
    purities = []
    cluster_Distribution = np.zeros((k, classes));
    for i in range(0, len(lbls)):
        cluster = clusters[i]
        class_ = lbls[i]
        cluster_Distribution[int(cluster)][int(class_)] += 1;
    for i in range(0, k):
        sum = 0;
        max = 0;
        current_cluster = cluster_Distribution[i]
        for j in range(0, classes):
            ratio = current_cluster[j] / distribution[j]
            print("ratio: ", ratio);
            if ratio > max:
                max = ratio
            if (ratio > 0):
                sum += ratio * np.log2(ratio);
                print("entered sum: ", sum);
        purities.append(max);
        entropies.append(-1*sum);

    return entropies, purities;

def calcVars(k, means, clusters, features, sizes, size):
    #size = 784
    vars = np.zeros((k, size, size));
    for i in range(0, features.shape[0]):
        cluster = int(clusters[i]);
        diff = np.array((size, 1));
        diff = np.subtract(features[i], means[cluster]);
        #print(diff.shape)
        diff_shape = diff.reshape((-1, 1))
        #print(diff_shape.shape)
        mult = np.matmul(diff_shape, diff_shape.reshape((1, size)));
        #print(mult.shape)
        vars[cluster] += mult;
    for i in range(0, len(sizes)):
        vars[i] = vars[i]/sizes[i];
    return vars;

def calcDistribution(lbls, classes=10):
    distribution = np.zeros((classes, ));
    for i in range(0, len(lbls)):
        index = lbls[i];
        distribution[index] += 1;
    return distribution;

def applyKMeans(k, d, trainSamples, features, lbls, size, classes):
    centroids = np.ndarray((k, size));
    choices = np.arange(1, trainSamples + 1, 1);
    clusters = np.ndarray((trainSamples,));
    sizes = np.zeros((k,));
    temp = features
    didChange = True

    # Get initial centroids
    for i in range(0, k):
        choice = np.random.choice(choices);
        sample = temp[choice]
        centroids[i] = sample;


    iteration = 1
    while (didChange):
        # Compute Cluster membership
        min_distances = np.full((trainSamples,), np.inf);
        for i in range(0, centroids.shape[0]):
            for j in range(0, trainSamples):
                dist = d(centroids[i], features[j]);
                #print(dist)
                if min_distances[j] > dist:
                    clusters[j] = i;
                    min_distances[j] = dist;

        new_centroids = np.zeros((k, size));

        #Calc new centroids
        new_sizes = np.zeros((k,));
        for j in range(0, trainSamples):
            index = int(clusters[j])
            feature = features[j];
            new_centroids[index] += feature;
            new_sizes[index] += 1;

        new_centroids = [new_centroids[x]/new_sizes[x] for x in range(0, sizes.shape[0])];
        didChange = stoppingCondition(sizes, new_sizes);
        new_centroids = np.asarray(new_centroids)
        centroids = new_centroids;
        sizes = new_sizes;
        print('iterations: ', iteration);
        print('sizes: ', sizes);
        iteration += 1

    means = centroids.astype(int);
    vars = calcVars(k, means, clusters, features, sizes, size);
    distribution = calcDistribution(lbls);
    entropies, purities = calcQuality(k, clusters, lbls, distribution, classes);

    return means, vars, entropies, purities;
�Final_project.pyimport urllib as url
import idx2numpy
import numpy as np
from numpy import genfromtxt
import random
from numpy import linalg as la
from numpy import random
from scipy import spatial
import matplotlib.pyplot as plt
import HW_6 as hw6
from _functools import partial

"""

-------------------- Spectral Clustering and Multiple Kernel Learning -----------------
"""


def getData(file, numSamples):
    data = genfromtxt(file, delimiter=',')
    features = np.ndarray((numSamples, data.shape[1] - 3))
    lbls = np.ndarray((numSamples, 1))

    for i in range(0, numSamples):
        features[i] = data[i][1:data.shape[1] - 2];
        lbls[i] = data[i][data.shape[1]-1]
    features = features.astype(int)
    lbls = lbls.astype(int)
    return features, lbls;

def guassian_rbfKernel(sigma, X, Y, size,):
    X = X.reshape((1, size))
    Y = Y.reshape((1, size))
    diff = X - Y;
    sim = np.matmul(diff, diff.reshape((size, 1)));
    sim = np.exp((-1*sim)/(2*(sigma**2)));
    return sim[0][0];

    return 0

def guassian_rbfKernel_2(sigma, X, Y, size):
    X = X.reshape((1, size))
    Y = Y.reshape((1, size))
    diff = X - Y;
    sim = np.matmul(diff, diff.reshape((size, 1)));
    sim =  np.exp((-1*sim)/(2*(sigma**2)));
    return -1 * sim;

    return 0

def polynomialKernel(power, X, Y, size):
    X, Y = X.reshape((size, )), Y.reshape((size, ));
    dot = np.dot(X, Y) + 1
    return abs(dot)**power;

    return 0
def laplace_rbfKernel(sigma, X, Y, size):
    X = X.reshape((1, size))
    Y = Y.reshape((1, size))
    diff = X - Y;
    sim = np.sqrt(np.matmul(diff, diff.reshape((size, 1))));
    sim = np.exp((-1*sim)/sigma)
    return sim[0][0];

def getSim(X, Y, size, weights, functions):
    sum = 0
    for i in range(0, len(functions)):
        sum += functions[i](X, Y, size) * weights[i]
    return sum;

def calculateGradient(means, kernels, last, size):
    total_sim = [];
    print(len(kernels))
    for i in range(0, len(kernels)):
        sum = 0;
        #print("TEST1")
        for j in range(0, means.shape[0]):
        #    print("TEST2")
            mean = means[j]
            #print("mean: ", mean);
            for k in range(j + 1, means.shape[0]):
            #    print("test");
                sum += kernels[i](mean, means[k], size);
                print("sum ", sum)
        total_sim.append(sum);
    print("In gradient: similarities ", total_sim)
    print("In gradientL: last ", last)
    grad = abs(np.asarray(total_sim) - np.asarray(last));
    return grad, total_sim;

def buildGraph(features, k, weights, kernels):
    size = features.shape[0];
    feature_size =  features.shape[1];
    edgeMatrix = np.zeros((size, size));
    for i in range(0, features.shape[0]):
        neighbors = [];
        similarities = [];
        for j in range(0, features.shape[0]):
            if (j != i):
                sim = getSim(features[i], features[j], feature_size, weights, kernels);
                if (len(similarities) < k):
                    similarities.append(sim);
                    neighbors.append(j);
                else:
                    min = np.amin(similarities);
                    argmin = np.argmin(similarities);
                    if (sim > min):
                        similarities[argmin] = sim
                        neighbors[argmin] = j;

        for l in range(0, len(neighbors)):
            sim = similarities[l]
            neighbor = neighbors[l]
            edgeMatrix[i][neighbor] = sim
            edgeMatrix[neighbor][i] = sim

    return edgeMatrix;

def calcQuality(k, clusters, lbls, distribution, classes):
    entropies = []
    purities = []
    cluster_Distribution = np.zeros((k, classes));
    for i in range(0, len(lbls)):
        cluster = clusters[i]
        class_ = lbls[i]
        cluster_Distribution[int(cluster)][int(class_)] += 1;
    for i in range(0, k):
        sum = 0;
        max = 0;
        current_cluster = cluster_Distribution[i]
        for j in range(0, classes):
            ratio = current_cluster[j] / distribution[j]
            #print("ratio: ", ratio);
            if ratio > max:
                max = ratio
            if (ratio > 0):
                sum += ratio * np.log2(ratio);
            #    print("entered sum: ", sum);
        purities.append(max);
        entropies.append(-1*sum);

    return entropies, purities;

def calcDistribution(lbls, classes):
    distribution = np.zeros((classes, ));
    for i in range(0, len(lbls)):
        index = lbls[i];
        distribution[index] += 1;
    return distribution;

def mkl_Spectral_Clustering(features, lbls, k, f, lr, kernels, ascent):
    sample_size = features.shape[0];
    feature_size = features.shape[1];
    weights = np.asarray([random.random() for i in range(0, f)]);
    print("intital weights: ", weights);
    last = [0.0 for i in range(0, f)];
    didChange = True
    epoch = 1
    while (didChange):
        graph = buildGraph(features, k, weights, kernels);

        diagonal_Matrix = np.zeros((graph.shape[0], graph.shape[0]));
        for i in range(0, graph.shape[0]):
            diagonal_Matrix[i][i] = k;

        Laplacian = diagonal_Matrix - graph;
        values, vectors = la.eig(Laplacian);
        print("eigen values shape: ", values.shape)
        print("eigen vectors shape: ", vectors.shape)

        sorted_args = values.argsort()[::-1]
        values = values[sorted_args]
        vectors = vectors[:,sorted_args]

        f_vector = vectors[:, vectors.shape[1] - 2]
        print("Shape: ", f_vector.shape);
        clusters = []
        means = np.zeros((2, feature_size));
        size = [0, 0]

        for i in range(0, sample_size):
            sign = f_vector[i];
            if (sign > 0):
                clusters.append(0);
                means[0] += features[i]
                size[0] += 1;
            else:
                clusters.append(1);
                means[1] += features[i]
                size[1] += 1

        means[0] = means[0] / size[0];
        means[1] = means[1] / size[1];

        grad, last = calculateGradient(means, kernels, last, feature_size);
        if (-0.0001 < np.sum(grad) < 0.0001):
            didChange = False;
        print(grad)
        if (ascent):
            new_weights = weights + lr * grad;
        else:
            new_weights = weights - lr * grad;
        weights = new_weights
        print(weights)
        print("epoch: ", epoch)
        epoch += 1

    distribution = calcDistribution(lbls, 2)
    entropies, purities = calcQuality(2, clusters, lbls, distribution, 2);
    print("stopped");
    return entropies, purities


def z_score_Normalization(features):
    for i in range(0, features.shape[0]):
        mean = np.sum(features[i])/(features.shape[1] - 1);
        std = 0;
        for j in range(0, features.shape[1]):
            std += (features[i, j] - mean)**2
        std = np.sqrt(std/(features.shape[1] - 1));
        for k in range(0, features.shape[1]):
            features[i, k] = (features[i, k] - mean)/std
    return features;

num_samples = 1000

features, lbls = getData("seizure_data.csv", num_samples)
print(features.shape)

#Relabel the data
lbls[lbls != 5] = 0
lbls[lbls == 5] = 1

means, vars, entropies, purities = hw6.applyKMeans(2, hw6.calcEuclidean, num_samples, features, lbls, features.shape[1], 2)
print("k-Means Entropies", entropies)
print("k-Means Purities", purities)

kernels_ascent = []
# Gradient Ascent
kernels.append([partial(guassian_rbfKernel, (1)), partial(laplace_rbfKernel, (0.5)), partial(polynomialKernel, (1/2))])
kernels.append([partial(guassian_rbfKernel, (1)), partial(laplace_rbfKernel, (0.5)), partial(polynomialKernel, 3)])
kernels.append([partial(guassian_rbfKernel, (1)), partial(laplace_rbfKernel, (0.5)), partial(polynomialKernel, 5)])
kernels.append([partial(guassian_rbfKernel, (1)), partial(laplace_rbfKernel, (0.5)), partial(polynomialKernel, 13)])
kernels.append([partial(guassian_rbfKernel, (1)), partial(laplace_rbfKernel, (0.5)), partial(polynomialKernel, 13), partial(polynomialKernel, 1)])
kernels.append([partial(guassian_rbfKernel, (1)), partial(laplace_rbfKernel, (0.5)), partial(polynomialKernel, 13), partial(polynomialKernel, 6)])
kernels.append([partial(guassian_rbfKernel, (1)), partial(polynomialKernel, (1))])

#kernels = [laplace_rbfKernel, guassian_rbfKernel]
features = z_score_Normalization(features);
for kernel in kernels_ascent:
    entropies, purities = mkl_Spectral_Clustering(features, lbls, 10, len(kernel), .03, kernel, ascent)
    print("GA Entropies", entropies)
    print("GA Purities", purities)

kernels_descent = []
# Gradient Ascent
kernels.append([partial(guassian_rbfKernel, (1)), partial(laplace_rbfKernel, (0.5)), partial(polynomialKernel, (2))])
kernels.append([partial(guassian_rbfKernel, (1)), partial(laplace_rbfKernel, (0.5)), partial(polynomialKernel, 4), partial(polynomialKernel, 2)])
kernels.append([partial(polynomialKernel, 4), partial(polynomialKernel, 2)])
kernels.append([partial(guassian_rbfKernel, (1)), partial(laplace_rbfKernel, (0.5)), partial(polynomialKernel, 8), partial(polynomialKernel, 2)])
kernels.append([partial(guassian_rbfKernel, (1)), partial(laplace_rbfKernel, (0.5)), partial(polynomialKernel, 14), partial(polynomialKernel, 5)])
kernels.append([partial(guassian_rbfKernel, (1)), partial(laplace_rbfKernel, (0.5)), partial(polynomialKernel, 13), partial(polynomialKernel, 6)])
kernels.append([partial(guassian_rbfKernel, (1)), partial(polynomialKernel, (1))])

for kernel in kernels_descent:
    entropies, purities = mkl_Spectral_Clustering(features, lbls, 10, len(kernel), .03, kernel, ascent)
    print("GD Entropies", entropies)
    print("GD Purities", purities)
