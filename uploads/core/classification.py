from __future__ import division
import operator
import numpy as np
import sys

sys.path.insert(0, "/home/msd/.local/install/caffe/python")

import caffe
import lmdb



#########################ALEXNET FLOWER
alexnet_flower_model_def = "/home/msd/Desktop/alexnetflower/deploy.prototxt"
alexnet_flower_model_weights = '/home/msd/Desktop/alexnetflower/alexnet.caffemodel'
alexnet_flower_net = caffe.Net(alexnet_flower_model_def, alexnet_flower_model_weights, caffe.TEST)

alexnet_flower_mu = np.load("/home/msd/Desktop/alexnetflower/out.npy" )
alexnet_flower_mu = alexnet_flower_mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
alexnet_flower_transformer = caffe.io.Transformer({'data': alexnet_flower_net.blobs['data'].data.shape})
alexnet_flower_transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
alexnet_flower_transformer.set_mean('data', alexnet_flower_mu)                # subtract the dataset-mean value in each channel
alexnet_flower_transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
alexnet_flower_transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR

# batch size, 3-channel (BGR) images, image size is 227x227 
alexnet_flower_net.blobs['data'].reshape(1, 3, 227, 227)  

#########################ALEXNET ENTIRE
alexnet_entire_model_def = "/home/msd/Desktop/alexnetentire/deploy.prototxt"
alexnet_entire_model_weights = '/home/msd/Desktop/alexnetentire/alexnet.caffemodel'
alexnet_entire_net = caffe.Net(alexnet_entire_model_def, alexnet_entire_model_weights, caffe.TEST)

alexnet_entire_mu = np.load("/home/msd/Desktop/alexnetentire/out.npy" )
alexnet_entire_mu = alexnet_entire_mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
alexnet_entire_transformer = caffe.io.Transformer({'data': alexnet_entire_net.blobs['data'].data.shape})
alexnet_entire_transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
alexnet_entire_transformer.set_mean('data', alexnet_entire_mu)                # subtract the dataset-mean value in each channel
alexnet_entire_transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
alexnet_entire_transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR

# batch size, 3-channel (BGR) images, image size is 227x227 
alexnet_flower_net.blobs['data'].reshape(1, 3, 227, 227)  

#########################ALEXNET LEAF
alexnet_leaf_model_def = "/home/msd/Desktop/alexnetleaf/deploy.prototxt"
alexnet_leaf_model_weights = '/home/msd/Desktop/alexnetleaf/alexnet.caffemodel'
alexnet_leaf_net = caffe.Net(alexnet_leaf_model_def, alexnet_leaf_model_weights, caffe.TEST)

alexnet_leaf_mu = np.load("/home/msd/Desktop/alexnetleaf/out.npy" )
alexnet_leaf_mu = alexnet_leaf_mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
alexnet_leaf_transformer = caffe.io.Transformer({'data': alexnet_leaf_net.blobs['data'].data.shape})
alexnet_leaf_transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
alexnet_leaf_transformer.set_mean('data', alexnet_leaf_mu)                # subtract the dataset-mean value in each channel
alexnet_leaf_transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
alexnet_leaf_transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR

# batch size, 3-channel (BGR) images, image size is 227x227 
alexnet_flower_net.blobs['data'].reshape(1, 3, 227, 227)  

#########################GOOGLENET FLOWER
googlenet_flower_model_def = "/home/msd/Desktop/googlenetflower/deploy.prototxt"
googlenet_flower_model_weights = '/home/msd/Desktop/googlenetflower/googlenet.caffemodel'
googlenet_flower_net = caffe.Net(googlenet_flower_model_def, googlenet_flower_model_weights, caffe.TEST)

googlenet_flower_mu = np.load("/home/msd/Desktop/googlenetflower/out.npy" )
googlenet_flower_mu = googlenet_flower_mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
googlenet_flower_transformer = caffe.io.Transformer({'data': googlenet_flower_net.blobs['data'].data.shape})
googlenet_flower_transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
googlenet_flower_transformer.set_mean('data', googlenet_flower_mu)                # subtract the dataset-mean value in each channel
googlenet_flower_transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
googlenet_flower_transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR


# batch size, 3-channel (BGR) images, image size is 227x227 
googlenet_flower_net.blobs['data'].reshape(1, 3, 227, 227)  

#########################GOOGLENET ENTIRE
googlenet_entire_model_def = "/home/msd/Desktop/googlenetentire/deploy.prototxt"
googlenet_entire_model_weights = '/home/msd/Desktop/googlenetentire/googlenet.caffemodel'
googlenet_entire_net = caffe.Net(googlenet_entire_model_def, googlenet_entire_model_weights, caffe.TEST)

googlenet_entire_mu = np.load("/home/msd/Desktop/googlenetentire/out.npy" )
googlenet_entire_mu = googlenet_entire_mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
googlenet_entire_transformer = caffe.io.Transformer({'data': googlenet_entire_net.blobs['data'].data.shape})
googlenet_entire_transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
googlenet_entire_transformer.set_mean('data', googlenet_entire_mu)                # subtract the dataset-mean value in each channel
googlenet_entire_transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
googlenet_entire_transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR

# batch size, 3-channel (BGR) images, image size is 227x227 
googlenet_entire_net.blobs['data'].reshape(1, 3, 227, 227) 
 
#########################GOOGLENET LEAF
googlenet_leaf_model_def = "/home/msd/Desktop/googlenetleaf/deploy.prototxt"
googlenet_leaf_model_weights = '/home/msd/Desktop/googlenetleaf/googlenet.caffemodel'
googlenet_leaf_net = caffe.Net(googlenet_leaf_model_def, googlenet_leaf_model_weights, caffe.TEST)

googlenet_leaf_mu = np.load("/home/msd/Desktop/googlenetleaf/out.npy" )
googlenet_leaf_mu = googlenet_leaf_mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
googlenet_leaf_transformer = caffe.io.Transformer({'data': googlenet_leaf_net.blobs['data'].data.shape})
googlenet_leaf_transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
googlenet_leaf_transformer.set_mean('data', googlenet_leaf_mu)                # subtract the dataset-mean value in each channel
googlenet_leaf_transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
googlenet_leaf_transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR

# batch size, 3-channel (BGR) images, image size is 227x227 
googlenet_leaf_net.blobs['data'].reshape(1, 3, 227, 227)  





#########################CAFFENET FLOWER
caffenet_flower_model_def = "/home/msd/Desktop/caffenetflower/deploy.prototxt"
caffenet_flower_model_weights = '/home/msd/Desktop/caffenetflower/caffenet.caffemodel'
caffenet_flower_net = caffe.Net(caffenet_flower_model_def, caffenet_flower_model_weights, caffe.TEST)

caffenet_flower_mu = np.load("/home/msd/Desktop/caffenetflower/out.npy" )
caffenet_flower_mu = caffenet_flower_mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
caffenet_flower_transformer = caffe.io.Transformer({'data': caffenet_flower_net.blobs['data'].data.shape})
caffenet_flower_transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
caffenet_flower_transformer.set_mean('data', caffenet_flower_mu)                # subtract the dataset-mean value in each channel
caffenet_flower_transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
caffenet_flower_transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR


# batch size, 3-channel (BGR) images, image size is 227x227 
caffenet_flower_net.blobs['data'].reshape(1, 3, 227, 227)  

#########################CAFFENET ENTIRE
caffenet_entire_model_def = "/home/msd/Desktop/caffenetentire/deploy.prototxt"
caffenet_entire_model_weights = '/home/msd/Desktop/caffenetentire/caffenet.caffemodel'
caffenet_entire_net = caffe.Net(caffenet_entire_model_def, caffenet_entire_model_weights, caffe.TEST)

caffenet_entire_mu = np.load("/home/msd/Desktop/caffenetentire/out.npy" )
caffenet_entire_mu = caffenet_entire_mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
caffenet_entire_transformer = caffe.io.Transformer({'data': caffenet_entire_net.blobs['data'].data.shape})
caffenet_entire_transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
caffenet_entire_transformer.set_mean('data', caffenet_entire_mu)                # subtract the dataset-mean value in each channel
caffenet_entire_transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
caffenet_entire_transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR

# batch size, 3-channel (BGR) images, image size is 227x227 
caffenet_entire_net.blobs['data'].reshape(1, 3, 227, 227) 
 
#########################CAFFENET LEAF
caffenet_leaf_model_def = "/home/msd/Desktop/caffenetleaf/deploy.prototxt"
caffenet_leaf_model_weights = '/home/msd/Desktop/caffenetleaf/caffenet.caffemodel'
caffenet_leaf_net = caffe.Net(caffenet_leaf_model_def, caffenet_leaf_model_weights, caffe.TEST)

caffenet_leaf_mu = np.load("/home/msd/Desktop/caffenetleaf/out.npy" )
caffenet_leaf_mu = caffenet_leaf_mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
caffenet_leaf_transformer = caffe.io.Transformer({'data': caffenet_leaf_net.blobs['data'].data.shape})
caffenet_leaf_transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
caffenet_leaf_transformer.set_mean('data', caffenet_leaf_mu)                # subtract the dataset-mean value in each channel
caffenet_leaf_transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
caffenet_leaf_transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR

# batch size, 3-channel (BGR) images, image size is 227x227 
caffenet_leaf_net.blobs['data'].reshape(1, 3, 227, 227)  






flowerFile = open('/home/msd/Downloads/flowerClass.txt')
leafFile = open('/home/msd/Downloads/leafClass.txt')
entireFile = open('/home/msd/Downloads/entireClass.txt')

flower = {}
leaf = {}
entire = {}
for i in range(0,967):
	token = flowerFile.readline()
	path = token.split()[0]
	label = token.split()[1]
	flower[label] = path
	#print path,label
for i in range(0,899):
	token = leafFile.readline()
	path = token.split()[0]
	label = token.split()[1]
	leaf[label] = path
	#print path,label
for i in range(0,993):
	token = entireFile.readline()
	path = token.split()[0]
	label = token.split()[1]
	entire[label] = path
	#print path,label


def giveAlexnetFlowerResults(path):

	labels_file = '/home/msd/Desktop/alexnetflower/classDocuments.txt'
	labels = np.loadtxt(labels_file, str, delimiter='\t')

	image = caffe.io.load_image(path) 
	transformed_image = alexnet_flower_transformer.preprocess('data', image)
	alexnet_flower_net.blobs['data'].data[...] = transformed_image
	output = alexnet_flower_net.forward()
	output_prob = output['prob'][0]


	result = output_prob.argmax()

	top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
	#print len(top_inds)

	classIds = []
	probs = []

	for i in zip(output_prob[top_inds], labels[top_inds]):
		classIds.append(flower.get(str(i[1].split()[0])))
		probs.append(str( output_prob[int(i[1].split()[0])] ))
		#print flower.get(str(i[1].split()[0])) + " " + str( output_prob[int(i[1].split()[0])] )  + " " + flower.get(str(label))
	
	return classIds, probs


def giveAlexnetEntireResults(path):

	labels_file = '/home/msd/Desktop/alexnetentire/classDocuments.txt'
	labels = np.loadtxt(labels_file, str, delimiter='\t')



	image = caffe.io.load_image(path) 
	transformed_image = alexnet_entire_transformer.preprocess('data', image)
	alexnet_entire_net.blobs['data'].data[...] = transformed_image
	output = alexnet_entire_net.forward()
	output_prob = output['prob'][0]


	result = output_prob.argmax()

	top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

	classIds = []
	probs = []
	for i in zip(output_prob[top_inds], labels[top_inds]):
		classIds.append(entire.get(str(i[1].split()[0])))
		probs.append(str( output_prob[int(i[1].split()[0])] ))
		#print entire.get(str(i[1].split()[0])) + " " + str( output_prob[int(i[1].split()[0])] )  + " " + entire.get(str(label))
	return classIds, probs

def giveAlexnetLeafResults(path):

	labels_file = '/home/msd/Desktop/alexnetleaf/classDocuments.txt'
	labels = np.loadtxt(labels_file, str, delimiter='\t')



	image = caffe.io.load_image(path) 
	transformed_image = alexnet_leaf_transformer.preprocess('data', image)
	alexnet_leaf_net.blobs['data'].data[...] = transformed_image
	output = alexnet_leaf_net.forward()
	output_prob = output['prob'][0]

	result = output_prob.argmax()

	top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

	classIds = []
	probs = []
	for i in zip(output_prob[top_inds], labels[top_inds]):
		classIds.append(leaf.get(str(i[1].split()[0])))
		probs.append(str( output_prob[int(i[1].split()[0])] ))
		#print leaf.get(str(i[1].split()[0])) + " " + str( output_prob[int(i[1].split()[0])] )  + " " + leaf.get(str(label))
	return classIds, probs

def giveGooglenetFlowerResults(path):

	labels_file = '/home/msd/Desktop/googlenetflower/classDocuments.txt'
	labels = np.loadtxt(labels_file, str, delimiter='\t')



	image = caffe.io.load_image(path) 
	transformed_image = googlenet_flower_transformer.preprocess('data', image)
	googlenet_flower_net.blobs['data'].data[...] = transformed_image
	output = googlenet_flower_net.forward()
	output_prob = output['prob'][0]


	result = output_prob.argmax()

	top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

	classIds = []
	probs = []
	for i in zip(output_prob[top_inds], labels[top_inds]):
		classIds.append(flower.get(str(i[1].split()[0])))
		probs.append(str( output_prob[int(i[1].split()[0])] ))
		#print flower.get(str(i[1].split()[0])) + " " + str( output_prob[int(i[1].split()[0])] )  + " " + flower.get(str(label))
	return classIds, probs

def giveGooglenetEntireResults(path):

	labels_file = '/home/msd/Desktop/googlenetentire/classDocuments.txt'
	labels = np.loadtxt(labels_file, str, delimiter='\t')

	
	image = caffe.io.load_image(path) 
	transformed_image = googlenet_entire_transformer.preprocess('data', image)
	googlenet_entire_net.blobs['data'].data[...] = transformed_image
	output = googlenet_entire_net.forward()
	output_prob = output['prob'][0]


	result = output_prob.argmax()

	top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

	classIds = []
	probs = []
	for i in zip(output_prob[top_inds], labels[top_inds]):
		classIds.append(entire.get(str(i[1].split()[0])))
		probs.append(str( output_prob[int(i[1].split()[0])] ))
		#print entire.get(str(i[1].split()[0])) + " " + str( output_prob[int(i[1].split()[0])] )  + " " + entire.get(str(label))
	return classIds, probs

def giveGooglenetLeafResults(path):

	labels_file = '/home/msd/Desktop/googlenetleaf/classDocuments.txt'
	labels = np.loadtxt(labels_file, str, delimiter='\t')

	

	image = caffe.io.load_image(path) 
	transformed_image = googlenet_leaf_transformer.preprocess('data', image)
	googlenet_leaf_net.blobs['data'].data[...] = transformed_image
	output = googlenet_leaf_net.forward()
	output_prob = output['prob'][0]

	result = output_prob.argmax()

	top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

	classIds = []
	probs = []
	for i in zip(output_prob[top_inds], labels[top_inds]):
		classIds.append(leaf.get(str(i[1].split()[0])))
		probs.append(str( output_prob[int(i[1].split()[0])] ))
		#print leaf.get(str(i[1].split()[0])) + " " + str( output_prob[int(i[1].split()[0])] )  + " " + leaf.get(str(label))
	return classIds, probs




def giveCaffenetFlowerResults(path):

	labels_file = '/home/msd/Desktop/caffenetflower/classDocuments.txt'
	labels = np.loadtxt(labels_file, str, delimiter='\t')



	image = caffe.io.load_image(path) 
	transformed_image = caffenet_flower_transformer.preprocess('data', image)
	caffenet_flower_net.blobs['data'].data[...] = transformed_image
	output = caffenet_flower_net.forward()
	output_prob = output['prob'][0]


	result = output_prob.argmax()

	top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

	classIds = []
	probs = []
	for i in zip(output_prob[top_inds], labels[top_inds]):
		classIds.append(flower.get(str(i[1].split()[0])))
		probs.append(str( output_prob[int(i[1].split()[0])] ))
		#print flower.get(str(i[1].split()[0])) + " " + str( output_prob[int(i[1].split()[0])] )  + " " + flower.get(str(label))
	return classIds, probs

def giveCaffenetEntireResults(path):

	labels_file = '/home/msd/Desktop/caffenetentire/classDocuments.txt'
	labels = np.loadtxt(labels_file, str, delimiter='\t')

	
	image = caffe.io.load_image(path) 
	transformed_image = caffenet_entire_transformer.preprocess('data', image)
	caffenet_entire_net.blobs['data'].data[...] = transformed_image
	output = caffenet_entire_net.forward()
	output_prob = output['prob'][0]


	result = output_prob.argmax()

	top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

	classIds = []
	probs = []
	for i in zip(output_prob[top_inds], labels[top_inds]):
		classIds.append(entire.get(str(i[1].split()[0])))
		probs.append(str( output_prob[int(i[1].split()[0])] ))
		#print entire.get(str(i[1].split()[0])) + " " + str( output_prob[int(i[1].split()[0])] )  + " " + entire.get(str(label))
	return classIds, probs

def giveCaffenetLeafResults(path):

	labels_file = '/home/msd/Desktop/caffenetleaf/classDocuments.txt'
	labels = np.loadtxt(labels_file, str, delimiter='\t')

	

	image = caffe.io.load_image(path) 
	transformed_image = caffenet_leaf_transformer.preprocess('data', image)
	caffenet_leaf_net.blobs['data'].data[...] = transformed_image
	output = caffenet_leaf_net.forward()
	output_prob = output['prob'][0]

	result = output_prob.argmax()

	top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

	classIds = []
	probs = []
	for i in zip(output_prob[top_inds], labels[top_inds]):
		classIds.append(leaf.get(str(i[1].split()[0])))
		probs.append(str( output_prob[int(i[1].split()[0])] ))
		#print leaf.get(str(i[1].split()[0])) + " " + str( output_prob[int(i[1].split()[0])] )  + " " + leaf.get(str(label))
	return classIds, probs




#print(giveAlexnetFlowerResults("/home/msd/Desktop/alexnetflower/flowerTest.txt"))
#print str(" ")
#giveGooglenetFlowerResults("/home/msd/Desktop/googlenetflower/flowerTest.txt")
#print str(" ")
#giveAlexnetEntireResults("/home/msd/Desktop/alexnetentire/entireTestMain.txt")
#print str(" ")
#giveGooglenetEntireResults("/home/msd/Desktop/googlenetentire/entireTestMain.txt")
#print str(" ")
#giveAlexnetLeafResults("/home/msd/Desktop/alexnetleaf/leafTestMain.txt")
#print str(" ")
#giveGooglenetLeafResults("/home/msd/Desktop/googlenetleaf/leafTestMain.txt")
#print str(" ")

alexnetFlowerClassId = []
alexnetFlowerProbs = [] 
googlenetFlowerClassId = []
googlenetFlowerProbs = []
caffenetFlowerClassId = []
caffenetFlowerProbs = [] 

alexnetLeafClassId = []
alexnetLeafProbs = []
googlenetLeafClassId = []
googlenetLeafProbs = []
caffenetLeafClassId = []
caffenetLeafProbs = [] 

alexnetEntireClassId = []
alexnetEntireProbs = []
googlenetEntireClassId = []
googlenetEntireProbs = []
caffenetEntireClassId = []
caffenetEntireProbs = [] 


def dataFusionMethod1():
	global alexnetFlowerClassId 
	global alexnetFlowerProbs 
	global googlenetFlowerClassId 
	global googlenetFlowerProbs 
	global alexnetLeafClassId 
	global alexnetLeafProbs 
	global googlenetLeafClassId 
	global googlenetLeafProbs 
	global alexnetEntireClassId 
	global alexnetEntireProbs 
	global googlenetEntireClassId 
	global googlenetEntireProbs 
	global caffenetFlowerClassId
	global caffenetFlowerProbs
	global caffenetLeafClassId
	global caffenetLeafProbs
	global caffenetEntireClassId
	global caffenetEntireProbs


	new_dict = {}
	for i in range(0,5):
		if alexnetFlowerClassId[i] in new_dict.keys():
			temp = new_dict[ alexnetFlowerClassId[i] ]
			new_dict[ alexnetFlowerClassId[i] ] = float(temp) +  float(alexnetFlowerProbs[i])
		else:
			new_dict[ alexnetFlowerClassId[i] ] = float(alexnetFlowerProbs[i])
		if googlenetFlowerClassId[i] in new_dict.keys():
			temp = new_dict[ googlenetFlowerClassId[i] ]
			new_dict[ googlenetFlowerClassId[i] ] = float(temp) +  float(googlenetFlowerProbs[i])
		else:
			new_dict[ googlenetFlowerClassId[i] ] = float(googlenetFlowerProbs[i])
		if caffenetFlowerClassId[i] in new_dict.keys():
			temp = new_dict[ caffenetFlowerClassId[i] ]
			new_dict[ caffenetFlowerClassId[i] ] = float(temp) +  float(caffenetFlowerProbs[i])
		else:
			new_dict[ caffenetFlowerClassId[i] ] = float(caffenetFlowerProbs[i])



		if alexnetLeafClassId[i] in new_dict.keys():
			temp = new_dict[ alexnetLeafClassId[i] ]
			new_dict[ alexnetLeafClassId[i] ] = float(temp) +  float(alexnetLeafProbs[i])
		else:
			new_dict[ alexnetLeafClassId[i] ] = float(alexnetLeafProbs[i])
		if googlenetLeafClassId[i] in new_dict.keys():
			temp = new_dict[ googlenetLeafClassId[i] ]
			new_dict[ googlenetLeafClassId[i] ] = float(temp) +  float(googlenetLeafProbs[i])
		else:
			new_dict[ googlenetLeafClassId[i] ] = float(googlenetLeafProbs[i])
		if caffenetLeafClassId[i] in new_dict.keys():
			temp = new_dict[ caffenetLeafClassId[i] ]
			new_dict[ caffenetLeafClassId[i] ] = float(temp) +  float(caffenetLeafProbs[i])
		else:
			new_dict[ caffenetLeafClassId[i] ] = float(caffenetLeafProbs[i])


		if alexnetEntireClassId[i] in new_dict.keys():
			temp = new_dict[ alexnetEntireClassId[i] ]
			new_dict[ alexnetEntireClassId[i] ] = float(temp) +  float(alexnetEntireProbs[i])
		else:
			new_dict[ alexnetEntireClassId[i] ] = float(alexnetEntireProbs[i])
		if googlenetEntireClassId[i] in new_dict.keys():
			temp = new_dict[ googlenetEntireClassId[i] ]
			new_dict[ googlenetEntireClassId[i] ] = float(temp) +  float(googlenetEntireProbs[i])
		else:
			new_dict[ googlenetEntireClassId[i] ] = float(googlenetEntireProbs[i])
		if caffenetEntireClassId[i] in new_dict.keys():
			temp = new_dict[ caffenetEntireClassId[i] ]
			new_dict[ caffenetEntireClassId[i] ] = float(temp) +  float(caffenetEntireProbs[i])
		else:
			new_dict[ caffenetEntireClassId[i] ] = float(caffenetEntireProbs[i])


	
	#print str(' ')	
	#print new_dict

	#print str(' ')
	sorted_dict = sorted(new_dict.items(), key=operator.itemgetter(1))


	#print sorted_x
	#print str(' ')
	#print sorted_x[::-1][:5]
	return sorted_dict[::-1][:5]

def dataFusionMethod2():
	global alexnetFlowerClassId 
	global alexnetFlowerProbs 
	global googlenetFlowerClassId 
	global googlenetFlowerProbs 
	global alexnetLeafClassId 
	global alexnetLeafProbs 
	global googlenetLeafClassId 
	global googlenetLeafProbs 
	global alexnetEntireClassId 
	global alexnetEntireProbs 
	global googlenetEntireClassId 
	global googlenetEntireProbs 
	global caffenetFlowerClassId
	global caffenetFlowerProbs
	global caffenetLeafClassId
	global caffenetLeafProbs
	global caffenetEntireClassId
	global caffenetEntireProbs
	alexnet_dict = {}
	googlenetnet_dict = {}
	caffenet_dict = {}
	for i in range(0,5):
		if alexnetFlowerClassId[i] in alexnet_dict.keys():
			temp = alexnet_dict[ alexnetFlowerClassId[i] ]
			alexnet_dict[ alexnetFlowerClassId[i] ] = float(temp) +  float(alexnetFlowerProbs[i])
		else:
			alexnet_dict[ alexnetFlowerClassId[i] ] = float(alexnetFlowerProbs[i])
		if googlenetFlowerClassId[i] in googlenetnet_dict.keys():
			temp = googlenetnet_dict[ googlenetFlowerClassId[i] ]
			googlenetnet_dict[ googlenetFlowerClassId[i] ] = float(temp) +  float(googlenetFlowerProbs[i])
		else:
			googlenetnet_dict[ googlenetFlowerClassId[i] ] = float(googlenetFlowerProbs[i])
		if caffenetFlowerClassId[i] in caffenet_dict.keys():
			temp = caffenet_dict[ caffenetFlowerClassId[i] ]
			caffenet_dict[ caffenetFlowerClassId[i] ] = float(temp) +  float(caffenetFlowerProbs[i])
		else:
			caffenet_dict[ caffenetFlowerClassId[i] ] = float(caffenetFlowerProbs[i])


		if alexnetLeafClassId[i] in alexnet_dict.keys():
			temp = alexnet_dict[ alexnetLeafClassId[i] ]
			alexnet_dict[ alexnetLeafClassId[i] ] = float(temp) +  float(alexnetLeafProbs[i])
		else:
			alexnet_dict[ alexnetLeafClassId[i] ] = float(alexnetLeafProbs[i])
		if googlenetLeafClassId[i] in googlenetnet_dict.keys():
			temp = googlenetnet_dict[ googlenetLeafClassId[i] ]
			googlenetnet_dict[ googlenetLeafClassId[i] ] = float(temp) +  float(googlenetLeafProbs[i])
		else:
			googlenetnet_dict[ googlenetLeafClassId[i] ] = float(googlenetLeafProbs[i])
		if caffenetLeafClassId[i] in caffenet_dict.keys():
			temp = caffenet_dict[ caffenetLeafClassId[i] ]
			caffenet_dict[ caffenetLeafClassId[i] ] = float(temp) +  float(caffenetLeafProbs[i])
		else:
			caffenet_dict[ caffenetLeafClassId[i] ] = float(caffenetLeafProbs[i])


		if alexnetEntireClassId[i] in alexnet_dict.keys():
			temp = alexnet_dict[ alexnetEntireClassId[i] ]
			alexnet_dict[ alexnetEntireClassId[i] ] = float(temp) +  float(alexnetEntireProbs[i])
		else:
			alexnet_dict[ alexnetEntireClassId[i] ] = float(alexnetEntireProbs[i])
		if googlenetEntireClassId[i] in googlenetnet_dict.keys():
			temp = googlenetnet_dict[ googlenetEntireClassId[i] ]
			googlenetnet_dict[ googlenetEntireClassId[i] ] = float(temp) +  float(googlenetEntireProbs[i])
		else:
			googlenetnet_dict[ googlenetEntireClassId[i] ] = float(googlenetEntireProbs[i])
		if caffenetEntireClassId[i] in caffenet_dict.keys():
			temp = caffenet_dict[ caffenetEntireClassId[i] ]
			caffenet_dict[ caffenetEntireClassId[i] ] = float(temp) +  float(caffenetEntireProbs[i])
		else:
			caffenet_dict[ caffenetEntireClassId[i] ] = float(caffenetEntireProbs[i])


	
	#print str(' ')	
	#print new_dict

	#print str(' ')
	sorted_alexnet = sorted(alexnet_dict.items(), key=operator.itemgetter(1))[::-1][:5]
	sorted_googlenet = sorted(googlenetnet_dict.items(), key=operator.itemgetter(1))[::-1][:5]
	sorted_caffenet = sorted(caffenet_dict.items(), key=operator.itemgetter(1))[::-1][:5]
	#print str(' ')
	#print sorted_alexnet
	#print str(' ')
	#print sorted_googlenet
	#print str(' ')
	#print sorted_alexnet[0][0] 
	#print sorted_alexnet[0][1] 

	if sorted_alexnet[0][1] > sorted_googlenet[0][1] and sorted_alexnet[0][1] > sorted_caffenet[0][1] :
		return sorted_alexnet
	elif sorted_googlenet[0][1] > sorted_alexnet[0][1] and sorted_googlenet[0][1] > sorted_caffenet[0][1] :
		return sorted_googlenet
	elif sorted_caffenet[0][1] > sorted_googlenet[0][1] and sorted_caffenet[0][1] > sorted_alexnet[0][1] :
		return sorted_caffenet
	else:
		return sorted_alexnet
	#return sorted_x[::-1][:5]

def dataFusionMethod3():
	global alexnetFlowerClassId 
	global alexnetFlowerProbs 
	global googlenetFlowerClassId 
	global googlenetFlowerProbs 
	global alexnetLeafClassId 
	global alexnetLeafProbs 
	global googlenetLeafClassId 
	global googlenetLeafProbs 
	global alexnetEntireClassId 
	global alexnetEntireProbs 
	global googlenetEntireClassId 
	global googlenetEntireProbs 
	global caffenetFlowerClassId
	global caffenetFlowerProbs
	global caffenetLeafClassId
	global caffenetLeafProbs
	global caffenetEntireClassId
	global caffenetEntireProbs

	flower_dict = {}
	leaf_dict = {}
	entire_dict = {}
	for i in range(0,5):
		if alexnetFlowerClassId[i] in flower_dict.keys():
			temp = flower_dict[ alexnetFlowerClassId[i] ]
			flower_dict[ alexnetFlowerClassId[i] ] = float(temp) +  float(alexnetFlowerProbs[i])
		else:
			flower_dict[ alexnetFlowerClassId[i] ] = float(alexnetFlowerProbs[i])
		if googlenetFlowerClassId[i] in flower_dict.keys():
			temp = flower_dict[ googlenetFlowerClassId[i] ]
			flower_dict[ googlenetFlowerClassId[i] ] = float(temp) +  float(googlenetFlowerProbs[i])
		else:
			flower_dict[ googlenetFlowerClassId[i] ] = float(googlenetFlowerProbs[i])
		if caffenetFlowerClassId[i] in flower_dict.keys():
			temp = flower_dict[ caffenetFlowerClassId[i] ]
			flower_dict[ caffenetFlowerClassId[i] ] = float(temp) +  float(caffenetFlowerProbs[i])
		else:
			flower_dict[ caffenetFlowerClassId[i] ] = float(caffenetFlowerProbs[i])

		if alexnetLeafClassId[i] in leaf_dict.keys():
			temp = leaf_dict[ alexnetLeafClassId[i] ]
			leaf_dict[ alexnetLeafClassId[i] ] = float(temp) +  float(alexnetLeafProbs[i])
		else:
			leaf_dict[ alexnetLeafClassId[i] ] = float(alexnetLeafProbs[i])
		if googlenetLeafClassId[i] in leaf_dict.keys():
			temp = leaf_dict[ googlenetLeafClassId[i] ]
			leaf_dict[ googlenetLeafClassId[i] ] = float(temp) +  float(googlenetLeafProbs[i])
		else:
			leaf_dict[ googlenetLeafClassId[i] ] = float(googlenetLeafProbs[i])
		if caffenetLeafClassId[i] in leaf_dict.keys():
			temp = leaf_dict[ caffenetLeafClassId[i] ]
			leaf_dict[ caffenetLeafClassId[i] ] = float(temp) +  float(caffenetLeafProbs[i])
		else:
			leaf_dict[ caffenetLeafClassId[i] ] = float(caffenetLeafProbs[i])



		if alexnetEntireClassId[i] in entire_dict.keys():
			temp = entire_dict[ alexnetEntireClassId[i] ]
			entire_dict[ alexnetEntireClassId[i] ] = float(temp) +  float(alexnetEntireProbs[i])
		else:
			entire_dict[ alexnetEntireClassId[i] ] = float(alexnetEntireProbs[i])
		if googlenetEntireClassId[i] in entire_dict.keys():
			temp = entire_dict[ googlenetEntireClassId[i] ]
			entire_dict[ googlenetEntireClassId[i] ] = float(temp) +  float(googlenetEntireProbs[i])
		else:
			entire_dict[ googlenetEntireClassId[i] ] = float(googlenetEntireProbs[i])
		if caffenetEntireClassId[i] in entire_dict.keys():
			temp = entire_dict[ caffenetEntireClassId[i] ]
			entire_dict[ caffenetEntireClassId[i] ] = float(temp) +  float(caffenetEntireProbs[i])
		else:
			entire_dict[ caffenetEntireClassId[i] ] = float(caffenetEntireProbs[i])


	
	#print str(' ')	
	#print new_dict

	#print str(' ')
	flower_dict = sorted(flower_dict.items(), key=operator.itemgetter(1))[::-1][:5]
	leaf_dict = sorted(leaf_dict.items(), key=operator.itemgetter(1))[::-1][:5]
	entire_dict = sorted(entire_dict.items(), key=operator.itemgetter(1))[::-1][:5]

	#print str(' ')	
	#print flower_dict
	#print str(' ')	
	#print leaf_dict
	#print str(' ')	
	#print entire_dict
	#print str(' ')

	#print str(' ')
	#print sorted_alexnet
	#print str(' ')
	#print sorted_googlenet
	#print str(' ')
	#print sorted_alexnet[0][0] 
	#print sorted_alexnet[0][1] 

	if flower_dict[0][1] > leaf_dict[0][1]  and flower_dict[0][1] > entire_dict[0][1]    :
		return flower_dict
	elif leaf_dict[0][1] > flower_dict[0][1]  and leaf_dict[0][1] > entire_dict[0][1]    :
		return leaf_dict
	elif entire_dict[0][1] > leaf_dict[0][1]  and entire_dict[0][1] > flower_dict[0][1]    :
		return entire_dict
	else:
		return flower_dict
	#return sorted_x[::-1][:5]
 

#print(dataFusionMethod1())
#print(dataFusionMethod2())
#print(dataFusionMethod3())

#print(giveAlexnetEntireResults('/home/msd/Desktop/EntireTest/359.jpg'))	
def dataFusionController(flowerPath,leafPath,entirePath):
	global alexnetFlowerClassId 
	global alexnetFlowerProbs 
	global googlenetFlowerClassId 
	global googlenetFlowerProbs 
	global alexnetLeafClassId 
	global alexnetLeafProbs 
	global googlenetLeafClassId 
	global googlenetLeafProbs 
	global alexnetEntireClassId 
	global alexnetEntireProbs 
	global googlenetEntireClassId 
	global googlenetEntireProbs 
	global caffenetFlowerClassId
	global caffenetFlowerProbs
	global caffenetLeafClassId
	global caffenetLeafProbs
	global caffenetEntireClassId
	global caffenetEntireProbs
	alexnetFlowerClassId, alexnetFlowerProbs = giveAlexnetFlowerResults(flowerPath)
	googlenetFlowerClassId, googlenetFlowerProbs = giveGooglenetFlowerResults(flowerPath)
	alexnetLeafClassId, alexnetLeafProbs = giveAlexnetLeafResults(leafPath)
	googlenetLeafClassId, googlenetLeafProbs = giveGooglenetLeafResults(leafPath)
	alexnetEntireClassId, alexnetEntireProbs = giveAlexnetEntireResults(entirePath)
	googlenetEntireClassId, googlenetEntireProbs = giveGooglenetEntireResults(entirePath)
	caffenetFlowerClassId, caffenetFlowerProbs = giveCaffenetEntireResults(flowerPath)
	caffenetLeafClassId, caffenetLeafProbs = giveCaffenetEntireResults(leafPath)
	caffenetEntireClassId, caffenetEntireProbs = giveCaffenetEntireResults(entirePath)
	

	result1 = dataFusionMethod1()
	result2 = dataFusionMethod2()
	result3 = dataFusionMethod3() 

	
	#if float(result1[0][1]) > float(result2[0][1]) and float(result1[0][1]) > float(result3[0][1]):
	#	return result1
	#elif float(result2[0][1]) > float(result1[0][1]) and float(result2[0][1]) > float(result3[0][1]):
#		return result2
##	elif float(result3[0][1]) > float(result1[0][1]) and float(result3[0][1]) > float(result2[0][1]):
#		return result3
#	else:
#		return result1
	allResults = {}
	for i in range(0,5):
		if result1[i][0] in allResults.keys():
			temp = allResults[ result1[i][0] ]
			allResults[ result1[i][0] ] = float(temp) +  float(result1[i][1])
		else:
			allResults[ result1[i][0] ] = float(result1[i][1])
		if result2[i][0] in allResults.keys():
			temp = allResults[ result2[i][0] ]
			allResults[ result2[i][0] ] = float(temp) +  float(result2[i][1])
		else:
			allResults[ result2[i][0] ] = float(result2[i][1])
		if result3[i][0] in allResults.keys():
			temp = allResults[ result3[i][0] ]
			allResults[ result3[i][0] ] = float(temp) +  float(result3[i][1])
		else:
			allResults[ result3[i][0] ] = float(result3[i][1])


	sorted_results = sorted(allResults.items(), key=operator.itemgetter(1))[::-1][:5]
	return sorted_results
	#return dataFusionMethod1(),dataFusionMethod2(),dataFusionMethod3()



def findTops():
	test = open("/home/msd/Downloads/qwert.txt")
	count = 0
	accuracy = 0
	for i in range(0,638):

		token = test.readline()
		expected = token.split()[0]
		flower = token.split()[1] + str('.jpg')
		leaf = token.split()[2] + str('.jpg')
		entire = token.split()[3] + str('.jpg')
		

		result = dataFusionController('/home/msd/Desktop/DatasetPrepairing/AllSnaps/' +str(flower),'/home/msd/Desktop/DatasetPrepairing/AllSnaps/'+str(leaf),'/home/msd/Desktop/DatasetPrepairing/AllSnaps/'+str(entire))

		count = count +  1

		#for i in result:
		#	if(str(i[0]) == str(expected)):
		#	    accuracy += 1
		if str(result[0][0]) == str(expected):
			accuracy += 1		
		print "Accuracy", "%" + str((accuracy / count)*100), str(count)




	print "Accuracy", "%" + str((accuracy / 638)*100)

findTops()

#print(dataFusionController('/home/msd/Desktop/DatasetPrepairing/AllSnaps/3568.jpg','/home/msd/Desktop/DatasetPrepairing/AllSnaps/13484.jpg','/home/msd/Desktop/DatasetPrepairing/AllSnaps/4626.jpg'))
