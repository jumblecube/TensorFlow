# Script to make hdf5 files from training and test set
import h5py
import numpy
import time
import matplotlib.pyplot as plt

hf1 = h5py.File('X.h5', 'r')
X1 = numpy.array(hf1['imgs'])
Y1 = numpy.array(hf1['labels'])

print("X1 shape ---> ", X1.shape)

plt.imshow(X1[0])
plt.gray()
plt.show()
print("Showed hd5f image 0 of type ", Y1[0])
time.sleep(5)

plt.imshow(X1[111])
plt.gray()
plt.show()
print("Showed hd5f image 11 of type ", Y1[111])
time.sleep(5)

hf2 = h5py.File('X_test.h5', 'r')
X2 = numpy.array(hf2['imgs'])
Y2 = numpy.array(hf2['labels'])

print("X2 test shape ---> ", X2.shape)

plt.imshow(X2[0])
plt.gray()
plt.show()
print("Showed hd5f image 0 of type ", Y2[0])
time.sleep(5)

plt.imshow(X2[111])
plt.gray()
plt.show()
print("Showed hd5f image 1 of type ", Y2[111])
time.sleep(5)

print("Y2 --- >", Y2)
