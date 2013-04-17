"""
=========================================
SVM: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a Support Vector Machines classifier with
linear kernel.

Initial source:
http://scikit-learn.org/dev/auto_examples/svm/plot_separating_hyperplane.html
"""

print __doc__

import numpy as np
import pylab as pl
from sklearn import svm
import math
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# generate random data or use existing data?
generateData = True
# run test and train routine?
testAndTrain = True
testSetNumber = 0 # 0 thru numsets
# number of points in clouds 0 and 1 (only works if generateData)
n0 = 20
n1 = 20
# dimensions (only works if generateData)
d = 5
# FileNames (for !generateData)
fileData = 'data.dat'
fileDataClass = 'dataClass.dat'
# specify plots
plotByClassifier = True # plots so that classifier plane is ortho to window
plotByAverage = True # plots so that average plane is ortho to window
plotXversusY = True # plots dimension x versus dimension y starting at 0
xdim = 0
ydim = 1

# plot both "up" and "down" support planes?
plotSupportPlanes = True

# we create n0+n1 separable points in d dimensions and write to files
if (generateData):
    print "Generating Data\n"
    np.random.seed(1)
    data = np.r_[np.random.randn(n0, d) - [.2]*d, np.random.randn(n1, d) + [.2]*d]
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
def readData(fileName):
    f = open(fileName)
    data = [] # resets "data"
    for line in f.readlines():
        data.append([])
        for i in line.split():
            data[-1].append(float(i))
    f.close()
    data = np.asarray(data)
    d = data.shape[1] # sets dimension to reflect this dataset
    return data
data = readData(fileData)

# Reads class data (0s and 1s must match corresponding data)
def readDataClass(fileName):
    f = open(fileName)
    dataClass = [] # resets "dataClass"
    for line in f.readlines():
        for i in line.split():
            dataClass.append(int(i))
    f.close()
    return dataClass
dataClass = readDataClass(fileDataClass)

# Makes sure data and dataClass have the same number of rows
if data.shape[0] != np.asarray(dataClass).shape[0]:
    print "Error: data and dataClass are not the same size."
    exit()

# Split up data for learning and testing
numsets = 5
if data.shape[0]/numsets != data.shape[0]/(1.0*numsets):
    print "Error: data is not divisible by 5 and cannot be split into training/test sets."
else:
    # writes data into 5 pairs of test + training files (+ class files)
    for k in range(0, numsets):
        f = open('test_'+str(k)+'.dat', 'w') # creates file
        g = open('train_'+str(k)+'.dat', 'w') # creates file
        l = numsets
        for i in data:
            if (l+k)/numsets == (l+k)/(1.0*numsets): # writes every 5th line to the kth test file
                for j in i:
                    f.write(str(j)+" ")
                f.write("\n")
            else: # writes all the other lines to the kth training file
                for j in i:
                    g.write(str(j)+" ")
                g.write("\n")
            l -= 1
        f.close()
        g.close()

    for k in range(0, numsets):
        f = open('test_'+str(k)+'_class.dat', 'w') # creates file
        g = open('train_'+str(k)+'_class.dat', 'w') # creates file
        l = numsets
        for i in dataClass:
            if (l+k)/numsets == (l+k)/(1.0*numsets): # writes every 5th line to the kth test class file
                f.write(str(i)+"\n")
            else: # writes all the other lines to the kth training class file
                g.write(str(i)+"\n")
            l -= 1
        f.close()
        g.close()

# make use of test and train files
if (testAndTrain):
    data = readData('train_'+str(testSetNumber)+'.dat') # sets "data" to a training set
    dataClass = readDataClass('train_'+str(testSetNumber)+'_class.dat') # gets class info for that set

# gets the average tuple in a list of tuples
def arrayMean(X):
    dim = X.shape
    Xmean = [0]*dim[1] # creates and populates "Xmean"
    for i in range(0, dim[1]):
        Xmean[i] = np.average(X[:, i])
    Xmean = np.asarray(Xmean)
    return Xmean

# returns array with only one class of data point (from dataClass)
def cloud(X, Y, n): # n = 0 or 1 (for the two clouds, respectively)
    dim = X.shape
    Xcloud = []
    for i in range(0, dim[0]):
        if Y[i]==n: Xcloud.append(X[i])
    Xcloud = np.asarray(Xcloud)
    return Xcloud

ave0 = arrayMean(cloud(data, dataClass, 0))
ave1 = arrayMean(cloud(data, dataClass, 1))
print "ave0", ave0, "\nave1", ave1

# vector from center of 0s to center of 1s
deltavector = ave1 - ave0
center = (ave0 + ave1)/2
print "deltavector", deltavector
print "center", center

# Define dot product
def dot_prod(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

# Normalize a vector
def norm(v):
    return v / math.sqrt(dot_prod(v,v))

# Create ~random orthogonal vector to vector v
def makeOrthoVector(v):
    randnumber = .52099932 # Just because I feel like it
    w = [randnumber] * v.shape[0] # This will break later if w is parallel to v
    ortho = w - v * dot_prod(w, v) / dot_prod(v, v)
    if dot_prod(ortho, v) == 0: # I may have confused the math up here... at any rate, this is just to catch an extremely unlikely error
        print "Error: The very unlikely has happened! A randomly produced vector \nis parallel to a specified vector."
    return ortho

# Project a vector v and scale it to a dimension s
def projVector(v, s):
    s = norm(s) # normalizes s
    return dot_prod(v, s)/dot_prod(s, s) #* s # "* s" makes this a std projection

# Projects dataset onto two dimensions
def projectArray(X, v): # dataset X, v is in plane; other basis will be created
    w = makeOrthoVector(v) # this is the other basis
    projX = [0]*X.shape[0] # creates and populates "projX"
    i = 0
    for x in X:
        projX[i]=(projVector(x, v), projVector(x, w))
        i += 1
    return np.asarray(projX)

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(data, dataClass)

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
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[ydim] - a * b[xdim])


### plot the line, the points, and the nearest vectors to the plane
pl.figure(1)
pl.plot(xx, yy)
pl.plot(xx, yyAve)
if plotSupportPlanes:
    pl.plot(xx, yy_down, 'k--')
    pl.plot(xx, yy_up, 'k--')
pl.plot(center[xdim], center[ydim],'g.',markersize=8.0, color='green') # plots average of clouds

pl.scatter(clf.support_vectors_[:, xdim], clf.support_vectors_[:, ydim],
           s=80, facecolors='none')
pl.scatter(data[:, xdim], data[:, ydim], c=dataClass, cmap=pl.cm.Paired)

pl.axis('tight')
pl.plot([ave0[xdim], ave1[xdim]], [ave0[ydim], ave1[ydim]])
svmNoAve = ave1[ydim] - ((ave1[xdim]-ave0[xdim])*-1/a + ave0[ydim])
pl.plot([ave0[xdim], ave1[xdim]], [ave0[ydim]+svmNoAve/2, ave1[ydim]-svmNoAve/2])
### ^ old plot ^
### v new plots v

yyy = np.linspace(-2, 2, 3)
xxxave = [projVector(center, deltavector)]*3

# Finds the location of the vertical classifier line
clfarray=[]
wnorm=norm(w)
points=np.linspace(-1, 1, 10000) # creates many points from -1 to 1
for i in points:
    clfarray.append(wnorm*i) # creates an array of vectors parallel to wnorm
clfarray = np.asarray(clfarray)
check1 = clf.predict(clfarray[0])
check2 = clf.predict(clfarray[0])
for i in range(0, clfarray.shape[0]): # works down the array and finds first 
    check1 = clf.predict(clfarray[i]) # vector which flips predicted class
    if check1 != check2:
        svmInflection = arrayMean(np.asarray([clfarray[i], clfarray[i-1]])) # approximates inflection point
    check2 = check1
xxxsvm = [projVector(svmInflection, w)]*3

testData = readData('test_'+str(testSetNumber)+'.dat')
testDataClass = readDataClass('test_'+str(testSetNumber)+'_class.dat')

def classifierAccuracy(testSet, testSetClass):
    aveCounter = 0
    for i in range(0, testSet.shape[0]):
        if testSetClass[i] == 0:
            if np.linalg.norm(testSet[i]-ave0) < np.linalg.norm(testSet[i]-ave1):
                aveCounter+=1
        elif testSetClass[i] == 1:
            if np.linalg.norm(testSet[i]-ave0) > np.linalg.norm(testSet[i]-ave1):
                aveCounter+=1
    print "Classifying by Average gives " + str(1.0*aveCounter/testSet.shape[0]*100) + "% accuracy."

    svmCounter = 0
    for i in range(0, testSet.shape[0]):
        if clf.predict(testSet[i]) == testSetClass[i]:
            svmCounter+=1
    print "Classifying by SVM gives " + str(1.0*svmCounter/testSet.shape[0]*100) + "% accuracy."
    
classifierAccuracy(testData, testDataClass)

if plotByAverage:
    pl.figure(2)
    pl.title("Plot by Averages")
    pl.plot(xxxave, yyy)
    projData = projectArray(data, deltavector)
    projSupVecs = projectArray(clf.support_vectors_, deltavector)
    projTestData = projectArray(testData, deltavector)
    pl.scatter(projSupVecs[:, 0], projSupVecs[:, 1], s=80, facecolors='none')
    pl.scatter(projData[:, 0], projData[:, 1], c=dataClass, cmap=pl.cm.Paired)
    pl.scatter(projTestData[:, 0], projTestData[:, 1],
               s=80, facecolors='yellow')
    pl.scatter(projTestData[:, 0], projTestData[:, 1],
               c=testDataClass, cmap=pl.cm.Paired)
pl.savefig('byaverage.png')

if plotByClassifier:
    pl.figure(3)
    pl.title("Plot by Classifier")
    pl.plot(xxxsvm, yyy)
    projData2 = projectArray(data, w)
    projSupVecs2 = projectArray(clf.support_vectors_, w)
    projTestData2 = projectArray(testData, w)
    pl.scatter(projSupVecs2[:, 0], projSupVecs2[:, 1], s=80, facecolors='none')
    pl.scatter(projData2[:, 0], projData2[:, 1], c=dataClass, cmap=pl.cm.Paired)
    pl.scatter(projTestData2[:, 0], projTestData2[:, 1],
               s=80, facecolors='yellow')
    pl.scatter(projTestData2[:, 0], projTestData2[:, 1],
               c=testDataClass, cmap=pl.cm.Paired)
pl.savefig('byclassifier.png')

if plotXversusY:
    pl.figure(4)
    pl.title("Plot dimension X versus dimension Y")
    pl.scatter(clf.support_vectors_[:, xdim], clf.support_vectors_[:, ydim], s=80, facecolors='none')
    pl.scatter(data[:, xdim], data[:, ydim], c=dataClass, cmap=pl.cm.Paired)
    pl.scatter(testData[:, xdim], testData[:, ydim],
               s=80, facecolors='yellow')
    pl.scatter(testData[:, xdim], testData[:, ydim],
               c=testDataClass, cmap=pl.cm.Paired)
pl.savefig('xversusy.png')

pl.show()
