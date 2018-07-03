import numpy as numpy

#sigmoid function
def nonlinear(x, derivative=False):
	if (derivative == True):
		return x*(1-x)
	else:
		return 1/(1+numpy.exp(-x))

# Manually create dataset for input
X = numpy.array([
[0,0,1],
[0,1,1],
[1,0,1],
[1,1,1]
])

# output dataset
y = numpy.array([[0,0,1,1]]).T

# random number seeder
numpy.random.seed(1)

# initialize weights randomly with mean 0
synapse0 = 2 * numpy.random.random((3,1)) - 1

for iter in xrange(10000):
	# forward propogation
	layer0 = X
	layer1 = nonlinear(numpy.dot(layer0, synapse0))

	# How much did it miss by
	layer1Err = y - layer1

	# how much we missed * slope of sigmoid at layer1 values
	layer1Delta = layer1Err * nonlinear(layer1, True)

	# update weight
	synapse0 += numpy.dot(layer0.T, layer1Delta)

print ("Output after training")
print (layer1)
