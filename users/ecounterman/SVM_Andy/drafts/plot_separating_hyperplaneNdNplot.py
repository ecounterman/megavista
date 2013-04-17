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
d = 5
# generate random data or use existing data?
generateData = True
# FileNames
fileData = 'data.dat'
fileDataClass = 'dataClass.dat'
# plot dimension y versus dimension x starting at 0
xdim = 0
ydim = 1
# -OR- specify any plane to project onto
specifyPlane = False # if "True", xdim and ydim will be rewritten later
plotByClassifier = False # plots so that classifier plane is ortho to window
plotByAverage = False # plots so that average plane is ortho to window
# if (specifyPlane && !plotByClassifier && !plotByAverage)==True,
# the following vectors must be filled in and they must be orthogonal
# specify1 =
# specify2 =
if (plotByClassifier & plotByAverage):
    print "Error: Only one of plotByClassifier and plotByAverage can be \"True\""
    exit()

# plot both "up" and "down" support planes?
plotSupportPlanes = False

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

# Makes sure data and dataClass have the same number of rows
if data.shape[0] != np.asarray(dataClass).shape[0]:
    print "Error: data and dataClass are not the same size."
    exit()

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

# Define dot_prod
def dot_prod(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

# Normalize a vector
def norm(v):
    return v / math.sqrt(dot_prod(v,v))

# Create ~random orthogonal vector to v
def makeOrthoVector(v):
    randnumber = .52099932 # Just because I feel like it
    w = [randnumber] * v.shape[0] # Supposes odds of ( w || v ) to be ~0
    return w - v * dot_prod(w, v) / dot_prod(v, v)

# Project a vector v and scale it to a dimension s
def projVector(v, s):
    s = norm(s) # normalizes s
    return dot_prod(v, s)/dot_prod(s, s) #* s # "* s" makes this a std projection

# Projects dataset onto two dimensions
def projectArray(X, v): # dataset X, v is in plane; other basis will be created
    w = makeOrthoVector(v)
    projX = [0]*X.shape[0] # creates and populates "projX"
    i = 0
    for x in X:
        projX[i]=(projVector(x, v), projVector(x, w))
        i += 1
    return np.asarray(projX)

projData = projectArray(data, deltavector)

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(data, dataClass)

projSupVecs = projectArray(clf.support_vectors_, deltavector)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[xdim] / w[ydim] # finds slope
xx = np.linspace(-5, 5, 3) # creates equidistant x values
yy = a * xx - (clf.intercept_[0]) / w[ydim] # finds y at those values

# separating hyperplane for average method
mAvePlane = -deltavector[xdim]/deltavector[ydim]
yyAve = mAvePlane * xx + (center[ydim] - mAvePlane*center[xdim])

print "\nclf.coef_", clf.coef_
print "w", w
print "a", a
print "xx", xx
print "clf.intercept_", (clf.intercept_[0]) / w[ydim]
print "yy", yy

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[ydim] - a * b[xdim])

print "clf.support_vectors_", clf.support_vectors_

b = clf.support_vectors_[-1]
yy_up = a * xx + (b[ydim] - a * b[xdim])


# plot the line, the points, and the nearest vectors to the plane
pl.figure(1)
pl.plot(xx, yy)
pl.plot(xx, yyAve)
if plotSupportPlanes:
    pl.plot(xx, yy_down, 'k--')
    pl.plot(xx, yy_up, 'k--')
pl.plot([0],[0],'g.',markersize=8.0, color='red') # plots origin
pl.plot(center[xdim], center[ydim],'g.',markersize=8.0, color='green') # plots average of clouds

pl.scatter(clf.support_vectors_[:, xdim], clf.support_vectors_[:, ydim],
           s=80, facecolors='none')
pl.scatter(data[:, xdim], data[:, ydim], c=dataClass, cmap=pl.cm.Paired)

pl.axis('tight')
pl.plot([ave0[xdim], ave1[xdim]], [ave0[ydim], ave1[ydim]])
svmNoAve = ave1[ydim] - ((ave1[xdim]-ave0[xdim])*-1/a + ave0[ydim])
pl.plot([ave0[xdim], ave1[xdim]], [ave0[ydim]+svmNoAve/2, ave1[ydim]-svmNoAve/2])

if d==3:
    fig = pl.figure()
    ax = Axes3D(fig)
    
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=dataClass, cmap=pl.cm.Paired)

#pl.show()
pl.figure(2)
pl.scatter(projSupVecs[:, 0], projSupVecs[:, 1], s=80, facecolors='none')
pl.scatter(projData[:, 0], projData[:, 1], c=dataClass, cmap=pl.cm.Paired)

pl.show()
