from MNISTHandwriting import readimgs
import numpy as np
import numpy.linalg as LA

images = readimgs('./data/train-images-idx3-ubyte')[0].astype('float')
labels = readimgs('./data/train-labels-idx1-ubyte')[0].astype('float')
imgs = readimgs('./data/t10k-images-idx3-ubyte')[0].astype('float')
imgs = np.reshape(imgs, (10000,784))
lbls = readimgs('./data/t10k-labels-idx1-ubyte')[0].astype('float')
images = np.reshape(images, (60000,784))
ones = np.ones((60000, 1))
cols = images[:,np.nonzero(np.sum(images, axis = 0))[0]]
A = np.concatenate((cols, ones), axis = 1)


#matches between output vector and correct vector
for i in range(10):
    labels = readimgs('./data/train-labels-idx1-ubyte')[0].astype('float')
    labels[labels != i] = -1
    labels[labels == i] = 1
    cols1 = imgs[:,np.nonzero(np.sum(images, axis = 0))[0]]
    ones = np.ones((10000, 1))
    A18 = np.concatenate((cols1, ones), axis = 1)
    lbls[lbls != i] = -1
    lbls[lbls == i] = 1
    mat18 = LA.lstsq(A,labels)[0]
    newA18 = A18@mat18
    newA18[newA18 > 0] = 1
    newA18[newA18 <= 0] = -1
    right = np.sum(newA18==lbls)
    return right

#error rate calculation for test data
for i in range(10):
    labels = readimgs('./data/train-labels-idx1-ubyte')[0].astype('float')
    labels[labels != i] = -1
    labels[labels == i] = 1
    cols1 = imgs[:,np.nonzero(np.sum(images, axis = 0))[0]]
    ones = np.ones((10000, 1))
    A18 = np.concatenate((cols1, ones), axis = 1)
    lbls[lbls != i] = -1
    lbls[lbls == i] = 1
    mat18 = LA.lstsq(A,labels)[0]
    newA18 = A18@mat18
    newA18[newA18 > 0] = 1
    newA18[newA18 <= 0] = -1
    print((10000-np.sum(newA18==lbls))/10000)
    
#error rate for training data
for i in range(10):
    labels = readimgs('./data/train-labels-idx1-ubyte')[0].astype('float')
    labels[labels != i] = -1
    labels[labels == i] = 1
    cols1 = images[:,np.nonzero(np.sum(images, axis = 0))[0]]
    ones = np.ones((60000, 1))
    A18 = np.concatenate((cols1, ones), axis = 1)
    mat18 = LA.lstsq(A,labels)[0]
    newA18 = A18@mat18
    newA18[newA18 > 0] = 1
    newA18[newA18 <= 0] = -1
    print((60000-np.sum(newA18==labels))/60000)
 
uniques = np.zeros((10000,10))
for i in range(10):
    labels = readimgs('./data/train-labels-idx1-ubyte')[0].astype('float')
    labels[labels != i] = -1
    labels[labels == i] = 1
    cols1 = imgs[:,np.nonzero(np.sum(images, axis = 0))[0]]
    ones = np.ones((10000, 1))
    A18 = np.concatenate((cols1, ones), axis = 1)
    lbls[lbls != i] = -1
    lbls[lbls == i] = 1
    mat18 = LA.lstsq(A,labels)[0]
    newA18 = A18@mat18
    newA18[newA18 > 0] = 1
    newA18[newA18 <= 0] = 0
    uniques[:,i] = newA18

#matches = np.zeros((10000,1))
matches = []
for j in range(10000):
    matches.append(sum(uniques[j]))

print(matches.count(0))

#creates a vector that shows output of all unique matches and correct matches. Used to calculate answers in part 4.
for i in range(10):
    labels = readimgs('./data/train-labels-idx1-ubyte')[0].astype('float')
    labels[labels != i] = -1
    labels[labels == i] = 1
    cols1 = imgs[:,np.nonzero(np.sum(images, axis = 0))[0]]
    ones = np.ones((10000, 1))
    A18 = np.concatenate((cols1, ones), axis = 1)
    lbls[lbls != i] = -1
    lbls[lbls == i] = 1
    mat18 = LA.lstsq(A,labels)[0]
    newA18 = A18@mat18
    newA18[newA18 > 0] = 1
    newA18[newA18 <= 0] = -1
    for i in range(10000):
        if newA18[i] == lbls[i] and matches[i] == 1:
            correct[i] = 1
        else: 
            correct[i] = 0

#finds false positives based on our classifier
falpos = np.zeros((10000,1))
for i in range(10):
    labels = readimgs('./data/train-labels-idx1-ubyte')[0].astype('float')
    labels[labels != i] = -1
    labels[labels == i] = 1
    cols1 = imgs[:,np.nonzero(np.sum(images, axis = 0))[0]]
    ones = np.ones((10000, 1))
    A18 = np.concatenate((cols1, ones), axis = 1)
    lbls[lbls != i] = -1
    lbls[lbls == i] = 1
    mat18 = LA.lstsq(A,labels)[0]
    newA18 = A18@mat18
    newA18[newA18 > 0] = 1
    newA18[newA18 <= 0] = -1
    for i in range(10000):
        if newA18[i] != lbls[i] and matches[i] == 1:
            falpos[i] = 1
        else: 
            falpos[i] = 0 
falposcount = np.sum(falpos)
print(falposcount/10000)

#calculates false negatives for our classifier
falneg = np.zeros((10000,1))
for i in range(10):
    labels = readimgs('./data/train-labels-idx1-ubyte')[0].astype('float')
    labels[labels != i] = -1
    labels[labels == i] = 1
    cols1 = imgs[:,np.nonzero(np.sum(images, axis = 0))[0]]
    ones = np.ones((10000, 1))
    A18 = np.concatenate((cols1, ones), axis = 1)
    lbls[lbls != i] = -1
    lbls[lbls == i] = 1
    mat18 = LA.lstsq(A,labels)[0]
    newA18 = A18@mat18
    newA18[newA18 > 0] = 1
    newA18[newA18 <= 0] = -1
    for i in range(10000):
        if matches[i] != 1:
            falneg[i] = 1
        else: 
            falneg[i] = 0 
falnegc = np.sum(falneg)
print(falnegc/10000)
