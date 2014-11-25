import numpy as np
import scipy.stats
from sklearn.mixture import GMM

# Parse data
data = []
with open('./gaussmix.csv') as f:
    for line in f:
        data.append([float(x) for x in line.strip().split(',')])
data = np.asarray(data)

# Initialize data
num_iter = 1000
d = 3 # Dimension of data
n = 200
mu1 = np.random.standard_normal(d)
mu2 = np.random.standard_normal(d)
sigma1 = np.identity(d)
sigma2 = np.identity(d)
pi = 0.9

# EM
for i in range(num_iter):

    # E-step
    norm1 = scipy.stats.multivariate_normal(mu1, sigma1)
    norm2 = scipy.stats.multivariate_normal(mu2, sigma2)
    tau1 = pi*norm1.pdf(data)
    tau2 = (1-pi)*norm2.pdf(data)
    gamma = tau1 / (tau1 + tau2)

    # M-step
    mu1 = np.dot(gamma, data)/sum(gamma)
    mu2 = np.dot((1-gamma), data)/sum((1-gamma))
    if i % 10 == 0:
        print mu1, mu2
    sigma1 = np.dot(np.array([x*gamma for x in (data-mu1).T]), (data-mu1))/sum(gamma)
    sigma2 = np.dot(np.array([x*(1-gamma) for x in (data-mu2).T]), (data-mu2))/sum((1-gamma))
    pi = sum(gamma)/n

print mu1, mu2
print pi
print sigma1
print sigma2
print '\n'

g = GMM(n_components=2, covariance_type='full', n_iter=1000)
g.fit(data)
print g.means_
print g.covars_
