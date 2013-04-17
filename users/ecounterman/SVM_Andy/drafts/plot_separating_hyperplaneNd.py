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

# number of points in clouds 0 and 1 (only works if generateData)
n0 = 20
n1 = 20
# dimensions (only works if generateData)
d = 3
# generate random data or use existing data?
generateData = True
# FileNames
fileData = 'data.dat'
fileDataClass = 'dataClass.dat'
# plot dimension x versus dimension y starting at 0
xdim = 1
ydim = 2

# we create n0+n1 separable points in d dimensions and write to files
if (generateData):
    print "Generating Data\n"
    np.random.seed(1)
    data = np.r_[np.random.randn(n0, d) - [2]*d, np.random.randn(n1, d) + [2]*d]
    dataClass = [0] * n0 + [1] * n1

    # writes data
    f = open(fileData, 'w')
    for i in data:
        for j in i:
            f.write(str(j)+" ")
        f.write("\n")
    f.close()

    # writes category (class) info
    f = open(fileDataClass, 'w')
    for i in dataClass:
        f.write(str(i)+"\n")
    f.close()

# Reads files with one ordered pair (space between x y z) per line
f = open(fileData)
data = [] # resets "data"
for line in f.readlines():
    data.append([])
    for i in line.split():
        data[-1].append(float(i))
f.close()
data = np.asarray(data)
d = data.shape[1] # sets dimension to reflect this dataset

# Reads class data (0s and 1s must match corresponding data)
f = open(fileDataClass)
dataClass = [] # resets "dataClass"
for line in f.readlines():
    for i in line.split():
        dataClass.append(int(i))
f.close()

# gets the average tuple in a list of tuples
def arrayMean(X):
    dim = X.shape
    Xmean = [0]*dim[1] # creates and populates "Xmean"
    for i in range(0, dim[1]):
        Xmean[i] = np.average(X[:, i])
    Xmean = np.asarray(Xmean)
    return Xmean

# returns array with only one class of data point (from dataClass)
def cloud(X, Y, n): # n = 0 or 1 (for each cloud)
    dim = X.shape
    Xcloud = []
    for i in range(0, dim[0]):
        if Y[i]==n: Xcloud.append(X[i])
    Xcloud = np.asarray(Xcloud)
    return Xcloud

ave0 = arrayMean(cloud(data, dataClass, 0))
ave1 = arrayMean(cloud(data, dataClass, 1))
print "ave0", ave0, "\nave1", ave1

#vector from center of 0s to center 1s
deltavector = ave1 - ave0
center = (ave0 + ave1)/2
print "deltavector", deltavector
print "center", center

print "\nEquation of plane is: ", deltavector[0], "(x -", center[0], ") + "
print deltavector[1], "(y -", center[1], ") + "
if d==3: print deltavector[2], "(z -", center[2], ") = 0"

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(data, dataClass)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1] # finds slope
xx = np.linspace(-5, 5, 3) # creates equidistant x values
yy = a * xx - (clf.intercept_[0]) / w[1] # finds y at those values

# separating hyperplane for average method
mAvePlane = -deltavector[0]/deltavector[1]
yyAve = mAvePlane * xx + (center[1] - mAvePlane*center[0])

print "\nclf.coef_", clf.coef_
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
pl.plot(xx, yyAve)
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')
pl.plot([0],[0],'g.',markersize=8.0, color='red')
pl.plot(center[xdim], center[ydim],'g.',markersize=8.0, color='green')

pl.scatter(clf.support_vectors_[:, xdim], clf.support_vectors_[:, ydim],
           s=80, facecolors='none')
pl.scatter(data[:, xdim], data[:, ydim], c=dataClass, cmap=pl.cm.Paired)

pl.axis('tight')
pl.plot([ave0[0], ave1[0]], [ave0[1], ave1[1]])
svmNoAve = ave1[1] - ((ave1[0]-ave0[0])*-1/a + ave0[1])
pl.plot([ave0[0], ave1[0]], [ave0[1]+svmNoAve/2, ave1[1]-svmNoAve/2])

if d==3:
    fig = pl.figure()
    ax = Axes3D(fig)
    
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=dataClass, cmap=pl.cm.Paired)

pl.show()

