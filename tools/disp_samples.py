import numpy as np
import matplotlib.pyplot as plt
import sys

def show_images(images, cols = 1, titles = None):
	"""Display a list of images in a single figure with matplotlib.

	Parameters
	---------
	images: List of np.arrays compatible with plt.imshow.

	cols (Default = 1): Number of columns in figure (number of rows is
						set to np.ceil(n_images/float(cols))).

	titles: List of titles corresponding to each image. Must have
			the same length as titles.
	"""
	assert((titles is None) or (len(images) == len(titles)))
	n_images = len(images)
	if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
	fig = plt.figure()
	for n, (image, title) in enumerate(zip(images, titles)):
		a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
		if image.ndim == 2:
			plt.gray()
		plt.imshow(image, vmin=0, vmax=255)
		a.set_title(title)
	fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
	plt.show()

idx = int(sys.argv[1])
a1 = np.load('../outputs/preds.npy')
b1 = a1[idx, :, :, :]
c1 = [b1[:, :, i] for i in range(4)]
# a2 = np.load('../outputs/next_states.npy')
# b2 = a2[idx, :, :, :]
# c2 = [b2[:, :, i] for i in range(4)]
# c = c1+c2
show_images(c1)
