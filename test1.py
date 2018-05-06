import numpy as np
def __computeDistanceSq(d1, d2):
	s = 0
	for i in range(len(d1)):
		s += (d1[i] - d2[i])**2
	return s



data1 = [1,2,3,4,5]
data2 = [2,3,4,5,6]
s = __computeDistanceSq(data1, data2)
print(np.sqrt(s))


def computeKernelWidth(data):
	dist = []
	for i in xrange(len(data)):
		for j in range(i+1, len(data)):
			s = __computeDistanceSq(data[i], data[j])
			dist.append(math.sqrt(s))
	return numpy.median(numpy.array(dist))