from GradCAM import GradCAM

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121

from PIL import Image
from skimage.transform import resize

def load_data(IMAGE_SIZE = (224, 224), NUM_CLASSES = 14, ZEROS = True):
	testCSV = np.loadtxt("./../CheXpert-v1.0-small/valid.csv", delimiter=",", dtype=str)
	testPaths = testCSV[1:, 0]
	test_labels = testCSV[1:, 5:]
	label_names = testCSV[0, 5:]
	print(label_names)
	for i in range(len(label_names)):
		label_names[i] = label_names[i].replace(" ", "_")
	print(label_names)
	x_test = np.zeros((testPaths.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

	for i in range(testPaths.shape[0]):
		image = Image.open("./../" + testPaths[i])
		image_array = np.asarray(image.convert("RGB"))
		image_array = image_array / 255.
		x_test[i] = resize(image_array, IMAGE_SIZE)

		test_labels[test_labels == '1.0'] = '1'
		test_labels[test_labels == ''] = '0'
		test_labels[test_labels == '0.0'] = '0'
		if ZEROS:
			test_labels[test_labels == '-1.0'] = '0'
		else:
			test_labels[test_labels == '-1.0'] = '1'
		y_test = np.asarray(test_labels, dtype = int)
	print(testPaths.shape, test_labels.shape, x_test.shape, y_test.shape)
	return testPaths, label_names, x_test, y_test

def transparent_cmap(cmap, N=255):
	"Copy colormap and set alpha values"

	mycmap = cmap
	mycmap._init()
	mycmap._lut[:,-1] = np.linspace(0, 0.4, N+4)
	return mycmap

def plot_GradCAM(heatmap, x1, path):
	w, h = heatmap.shape
	y, x = np.mgrid[0:h, 0:w]   
	mycmap = transparent_cmap(plt.cm.Reds)
	
	'''
	fig, ax = plt.subplots(1, 1)
	ax.imshow(x1, cmap='gray')
	cb = ax.contourf(x, y, heatmap, 4, cmap=mycmap)
	plt.colorbar(cb)
	plt.savefig(path)
	plt.close()
	'''

	fig, ax = plt.subplots(1, 1)
	x2 = np.ones(x1.shape)
	ax.imshow(x2, cmap='gray')
	cb = ax.contourf(x, y, heatmap, 4, cmap=mycmap)
	plt.colorbar(cb)
	plt.savefig(path)
	plt.close()
	
import os
if __name__ == "__main__":
	num_classes = 14
	inp = layers.Input(shape=(224, 224, 3))
	model = tf.keras.applications.DenseNet121(include_top=True, weights="./Grad_CAM/weightsBest.h5", input_tensor=inp, input_shape=(224, 224, 3), classes=num_classes)
	#model.summary()

	grad_cam = GradCAM(model)

	testPaths, label_names, x_test, y_test = load_data()

	script_dir = os.path.dirname(__file__)
	

	for i in range(x_test.shape[0]):
		path = testPaths[i].replace("CheXpert-v1.0-small/valid/", "").replace("/study1","").replace("/", "_").replace(".jpg", "") + "/"
		results_dir = os.path.join(script_dir, 'Results/' + path)
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)

		x1 = x_test[[i]]
		predictions = model.predict(x1)[0]
		print(i+1, path)
		for j in range(len(predictions)):
			heatmap = grad_cam.get_heatmap(j, x1)
			save_dir =  results_dir + label_names[j] + "_heatmap.png"
			plot_GradCAM(heatmap, x1[0], save_dir)