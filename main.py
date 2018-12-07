import os
import random

from IPython.display import display, HTML
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import cv2

blur_kernel_size = 15
canny_low_threshold = 20
canny_high_threshold = 100
rho = 1
theta = np.pi / 180
threshold = 10
min_line_length = 20
max_line_gap = 1
images_directory = 'images'
videos_directory = 'videos'
output_gray_images = 'output_gray_images'
output_blur_images = 'output_blur_images'
output_canny_images = 'output_canny_images'
output_region_images = 'output_region_images'
output_hough_images = 'output_hough_images'
output_merge_images = 'output_merge_images'
videos_output_directory = 'output_videos'


def gray_scale(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
	return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
	mask = np.zeros_like(img)
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image


def draw_line(img, x, y, color = [255, 0, 0], thickness = 20):
	if len(x) == 0:
		return
	line_parameters = np.polyfit(x, y, 1)

	m = line_parameters[0]
	b = line_parameters[1]

	max_y = img.shape[0]
	maxX = img.shape[1]
	y1 = max_y
	x1 = int((y1 - b) / m)
	y2 = int((max_y / 2)) + 60
	x2 = int((y2 - b) / m)
	cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 4)


def draw_lines(img, lines, color = [255, 0, 0], thickness = 20):
	left_points_x = []
	left_points_y = []
	right_points_x = []
	right_points_y = []

	for line in lines:
		for x1, y1, x2, y2 in line:
			m = (y1 - y2) / (x1 - x2)
			if m < 0:
				left_points_x.append(x1)
				left_points_y.append(y1)
				left_points_x.append(x2)
				left_points_y.append(y2)
			else:
				right_points_x.append(x1)
				right_points_y.append(y1)
				right_points_x.append(x2)
				right_points_y.append(y2)

	draw_line(img, left_points_x, left_points_y, color, thickness)

	draw_line(img, right_points_x, right_points_y, color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
	draw_lines(line_img, lines)
	return line_img


def weighted_img(img, initial_img, α = 0.8, β = 1., λ = 0.):
	return cv2.addWeighted(initial_img, α, img, β, λ)


def showImagesInHtml(images, dir):
	random_number = random.randint(1, 100000)
	buffer = "<div>"
	for img in images:
		imgSource = dir + '/' + img + "?" + str(random_number)
		buffer += """<img src="{0}" width="300" height="110" style="float:left; margin:1px"/>""".format(imgSource)
	buffer += "</div>"
	display(HTML(buffer))


def saveImages(images, outputDir, imageNames, isGray = 0):
	if not os.path.exists(outputDir):
		os.makedirs(outputDir)

	zipped = list(map(lambda imgZip: (outputDir + '/' + imgZip[1], imgZip[0]), zip(images, imageNames)))
	for imgPair in zipped:
		if isGray:
			plt.imsave(imgPair[0], imgPair[1], cmap = 'gray')
		else:
			plt.imsave(imgPair[0], imgPair[1])


def doSaveAndDisplay(images, outputDir, imageNames, somethingToDo, isGray = 0):
	outputImages = list(map(somethingToDo, images))
	saveImages(outputImages, outputDir, imageNames, isGray)
	showImagesInHtml(imageNames, outputDir)
	return outputImages


def grayAction(img):
	return gray_scale(img)


def maskAction(img):
	ysize = img.shape[0]
	xsize = img.shape[1]
	region = np.array([[0, ysize], [xsize / 2, (ysize / 2) + 10], [xsize, ysize]], np.int32)
	return region_of_interest(img, [region])


def finding_lanes_on_images():
	image_names = os.listdir(images_directory)
	showImagesInHtml(image_names, images_directory)
	images = list(map(lambda img: plt.imread(images_directory + '/' + img), image_names))

	gray_images = doSaveAndDisplay(images, output_gray_images, image_names, grayAction, 1)

	blur_action = lambda img: gaussian_blur(img, blur_kernel_size)

	blur_images = doSaveAndDisplay(gray_images, output_blur_images, image_names, blur_action, 1)

	canny_action = lambda img: canny(img, canny_low_threshold, canny_high_threshold)
	canny_images = doSaveAndDisplay(blur_images, output_canny_images, image_names, canny_action)

	region_images = doSaveAndDisplay(canny_images, output_region_images, image_names, maskAction)

	hough_action = lambda img: hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap)
	hough_images = doSaveAndDisplay(region_images, output_hough_images, image_names, hough_action)

	merge_images = list(map(lambda imgs: weighted_img(imgs[0], imgs[1]), zip(images, hough_images)))
	imagesMerged = doSaveAndDisplay(merge_images, output_merge_images, image_names, lambda img: img)


def process_image(image):
	withLines = hough_lines(maskAction(canny(gaussian_blur(grayAction(image), blur_kernel_size), canny_low_threshold, canny_high_threshold)), rho, theta, threshold, min_line_length, max_line_gap)
	return weighted_img(image, withLines)


def processVideo(videoFileName, inputVideoDir, outputVideoDir):
	if not os.path.exists(outputVideoDir):
		os.makedirs(outputVideoDir)
	clip = VideoFileClip(inputVideoDir + '/' + videoFileName)
	outputClip = clip.fl_image(process_image)
	outVideoFile = outputVideoDir + '/' + videoFileName
	outputClip.write_videofile(outVideoFile, audio = False)
	display(
		HTML("""
		<video width="960" height="540" controls>
          <source src="{0}">
        </video>
        """.format(outVideoFile))
	)


def finding_lanes_on_videos():
	processVideo('solidWhiteRight.mp4', videos_directory, videos_output_directory)
	processVideo('solidYellowLeft.mp4', videos_directory, videos_output_directory)
	processVideo('challenge.mp4', videos_directory, videos_output_directory)


if __name__ == '__main__':
	finding_lanes_on_images()
	finding_lanes_on_videos()
