"""
=========================================
SVM: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a Support Vector Machines classifier with
linear kernel.
"""
print __doc__

import numpy as np
import pylab as pl
from sklearn import svm
import math

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# we create n*2 separable points in d dimensions and write to data.dat
n = 2
d = 3
np.random.seed(1)
X = np.r_[np.random.randn(n, 2) - [2, 2], np.random.randn(n, 2) + [2, 2]]
Y = [0] * n + [1] * n

for i in range(0, 2*n):
    if (i < n): print np.random.randn(1, d) - [2] * d
    else: print np.random.randn(1, d) + [2] * d

f = open('data.dat', 'w')
for i in X:
    print i
    for j in i:
        f.write(str(j)+" ")
    f.write("\n")
f.close()

# handles a file with one ordered pair (space between x y z) per line
f = open('data.dat')
print f.read()
f.close()

print X
print Y

ave0 = [sum(X[:n, 0]) / n, sum(X[:n, 1]) / n]
ave1 = [sum(X[n:, 0]) / n, sum(X[n:, 1]) / n]

print "ave0", ave0, "\nave1", ave1

cloudsm = (ave1[1] - ave0[1]) / (ave1[0] - ave0[0])


# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5, 2)
yy = a * xx - (clf.intercept_[0]) / w[1]

print "clf.coef_", clf.coef_
print "w", w
print "a", a
print "xx", xx
print "clf.intercept_", (clf.intercept_[0]) / w[1]
print "yy", yy
print "a * cloudsm =", a, "*", cloudsm, "=", a*cloudsm


# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

print "clf.support_vectors_", clf.support_vectors_

b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


# plot the line, the points, and the nearest vectors to the plane
pl.plot(xx, yy)
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')
pl.plot([0],[0],'g.',markersize=5.0)
pl.plot([0], -clf.intercept_[0],'g.',markersize=5.0)

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.plot([ave0[0], ave1[0]], [ave0[1], ave1[1]])
#pl.plot([ave0[0], ave1[0]], [ave0[1], ((ave1[0]-ave0[0])*cloudsm+ave0[1])])
pl.plot([ave0[0], ave1[0]], [ave0[1], ((ave1[0]-ave0[0])*-1/a+ave0[1])])
pl.show()
"""
fig = pl.figure()
ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=pl.cm.Paired)
pl.show()
"""
