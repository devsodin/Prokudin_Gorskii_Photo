import cv2
import glob
import os
import time as t
import numpy as np


def load_dataset(folder):
	images = {}
	files = glob.glob(folder+"/*.jpg")
	for file in files:
		image = cv2.imread(os.path.join(folder,os.path.basename(file)),cv2.IMREAD_GRAYSCALE)
		if image is not None:
			images[os.path.basename(file)] = image

	return images

def get_color_image(image):

	h = int(image.shape[0]/3)
	w = image.shape[1]

	color_img = np.zeros((h,w,3))
	
	for i in range(0,3):
		color_img[:,:,2-i] = image[h*i:h*(i+1),:]

	#return as RGB
	return color_img

def remove_margins(image,x_margin,y_margin):
	[w,h,d] = image.shape
	y_margin = int(h * y_margin_percent)
	x_margin = int(w * x_margin_percent)
	return image[ x_margin : w-x_margin, y_margin : h-y_margin, : ]

def align_image2template(i,t,displacement):

	bestNCC = -1
	align = [0,0]
	for x in range(-displacement,displacement+1):
		for y in range(-displacement,displacement+1):
			#displace image
			moved_i= np.roll(i,[x,y],axis=(0,1))
			moved_i = moved_i - moved_i.mean(axis=0)
			
			t = t - t.mean(axis=0)

			NCC = np.sum( (moved_i/np.linalg.norm(moved_i)) * (t/np.linalg.norm(t)) )
			
			if NCC > bestNCC:
				bestNCC = NCC
				align = [x,y]

	return np.roll(i,align,axis=(0,1))

	

def photoplate2color(image,displacement,margin_percent):
	
	r = image[:,:,0]
	g = image[:,:,1]
	b = image[:,:,2]

	color = np.zeros_like(img_channels)
	print(color.dtype)

	color[:,:,0] = align_image2template(r,g,displacement)
	color[:,:,1] = g
	color[:,:,2] = align_image2template(b,g,displacement)



	return color

def write_image(name, subname, image, out_folder):
	fname = os.path.splitext(name)
	savename = out_folder + fname[0] + subname + fname[1]
	print(savename)
	cv2.imwrite(savename,image)

	return

ds_folder = "images"
displacement = 25
x_margin_percent = 0.03
y_margin_percent = 0.05

images = load_dataset(ds_folder+"/input")

for name, image in images.items():
	time = t.time()
	
	img_3channel = get_color_image(image)
	img_channels = remove_margins(img_3channels,x_margin_percent,y_margin_percent)
	color_img = photoplate2color(img_channels,displacement,margin_percent)
	time = t.time() - time

	write_image(name,"_cn", img_channels, ds_folder + "/output/")
	write_image(name,"_color", color_img, ds_folder + "/output/")

	print("Elapsed time: ", time)

#ADD METRICS ¿?
