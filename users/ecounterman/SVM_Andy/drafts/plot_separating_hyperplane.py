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

# we create 80 separable points
np.random.seed(0)
X = np.r_[np.random.randn(40, 2) - [1, 1], np.random.randn(40, 2) + [1, 1]]
Y = [0] * 40 + [1] * 40

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5, 2)
yy = a * xx - (clf.intercept_[0]) / w[1]
binter = (clf.intercept_[0]) / w[1]

print "clf.coef_", clf.coef_
print "w", w
print "a", a
print "xx", xx
print "clf.intercept_", (clf.intercept_[0]) / w[1]
print "yy", yy

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
pl.plot([0], -binter,'g.',markersize=5.0)

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=160, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
