from PIL import Image
import h5py
import numpy

# Load the data set
hf1 = h5py.File('X.h5', 'r')
#hf2 = h5py.File('X_test.h5', 'r')

X1 = numpy.array(hf1['imgs'])
print("X1 shape ---> ", X1.shape)
X = numpy.rollaxis(X1, 1, 4)
print("X shape after roll ---> ", X.shape)

img = Image.fromarray(X[0], 'RGB')

img.show()

'''
Y1 = numpy.array(hf1['labels'])
Y = numpy.zeros(shape=(len(Y1), 43))

print("Y1 Length ---> ", len(Y1))
print("Y1 shape ---> ", Y1.shape)
print("Y shape ---> ", Y.shape)
print("Y1 ---- > ", Y1[1])
print("Y[1][:] before ---> ", Y[1])

for i in range(0, 3):
    print("Y1 ---- > ", i, "   ", Y1[i])
    print("loop ---> ", int(Y1[i]))
    Y[i][int(Y1[i])] = 1

print("Y[1][:] after ---> ", Y[1])
'''
#len(Y1)-1)
#print("Image --- > ", X[0][0])
'''print("Label --- >", Y[0])
print("shape label --- >", len(Y))

for DS1 in hf2:
    print(DS1, "----->", hf2[DS1])

X_test = hf2['imgs'][:]
Y1_test = numpy.array(hf2['labels'])
Y2_test = hf2['labels']

Y_test = Y1_test.reshape((-1, 1))

#print("Image --- > ", X_test[0][0])
print("Label --- >", Y_test[0])
print("shape label --- >", Y_test.shape)
print("Label --- >", Y2_test)
print("shape label --- >", Y2_test.shape)'''
