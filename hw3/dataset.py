import os
from glob import glob
from keras.preprocessing import image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

num_classes = 2

root = os.path.join(os.getcwd(), 'dataset')

px256_folder = '256/woman-man-256'
px32_folder = '32'

'''
dataset_man_px256 = []

f_path = os.path.join(px256_path, 'man')
for img_path in glob(os.path.join(f_path, img_mask)):
	img = image.load_img(img_path, target_size=(256, 256))
	dataset_man_px256.append(image.img_to_array(img))

print(np.array(dataset_man_px256).shape)

'''

def get_folders(i):
	if i == 256:
		f = os.path.join(root, px256_folder)
		folders = [os.path.join(f, d) for d in sorted(os.listdir(f))] 
	elif i == 32:
		f = os.path.join(root, px32_folder)
		folders = [os.path.join(f, d) for d in sorted(os.listdir(f))]
	else:
		print("E: Wrong arg. dataset() in dataset.py")
		folders=[]
	return folders # path of folders

def load_imgdataset(folder, px_num):
	print('load_imgdata arg folder:', folder)
	image_files = os.listdir(folder)
	dataset = [image.img_to_array(image.load_img(os.path.join(folder, img), \
							target_size=(px_num, px_num), \
							grayscale=False)) for img in image_files]
#	dataset = image.img_to_array(image.load_img(os.path.join(folder, '*.png'), target_size=(px_num, px_num)))
#	print("Shape of dataset: ", np.ndarray(dataset).shape)
#	print(dataset)
	return dataset

def load_xydata(px_num):
	folders = get_folders(px_num)
	x_data = []
	y_data = []
	label = 0
	for f in folders:
		#label = os.path.basename(f)
		#x_data += load_dataset(f))
		imgdataset = load_imgdataset(f, px_num)
		x_data.extend(imgdataset)
		for j in range(len(imgdataset)):
			y_data.append(label)
		label += 1
	#for i in range(len(folders)-1):
	#	if i == 0: x_data = np.concatenate((x_data[0], x_data[1]), axis=0)
	#	else: x_data = np.concatenate((x_data, x_data[i+1]), axis=0)
#	print("x_data.shape:", np.ndarray(x_data).shape, "y_data.shape:", np.ndarray(y_data).shape)
	return x_data, y_data

def train_test_dataset(px_num):
	x, y = load_xydata(px_num)
	return train_test_split(x, to_categorical(y, num_classes), test_size=0.2) # x_train, x_test, y_train, y_test



# datagen = image.ImageDataGenerator(rescale=1.255, ) I want to use this class but! this doesn't provide random_train_test_split! FUCK FUCK! But next time I will use this!!


