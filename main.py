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
output_directory = 'output'
output_gray_images = 'images_gray'
output_blur_images = 'images_blur'
output_canny_images = 'images_canny'
output_region_images = 'images_region'
output_hough_images = 'images_hough'
output_merge_images = 'images_merge'
videos_output_directory = 'videos'


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


def show_images_in_html(images, dir):
	random_number = random.randint(1, 100000)
	buffer = "<div>"
	for img in images:
		img_src = dir + '/' + img + "?" + str(random_number)
		buffer += """<img src="{0}" width="300" height="110" style="float:left; margin:1px"/>""".format(img_src)
	buffer += "</div>"
	display(HTML(buffer))


def save_images(images, output_dir, image_names, is_gray = 0):
	if not os.path.exists(output_directory + '/' + output_dir):
		os.makedirs(output_directory + '/' + output_dir)

	zipped = list(map(lambda imgZip: (output_directory + '/' + output_dir + '/' + imgZip[1], imgZip[0]), zip(images, image_names)))
	for imgPair in zipped:
		if is_gray:
			plt.imsave(imgPair[0], imgPair[1], cmap = 'gray')
		else:
			plt.imsave(imgPair[0], imgPair[1])


def save_file_and_display(images, output_dir, image_names, something_to_do, is_gray = 0):
	output_images = list(map(something_to_do, images))
	save_images(output_images, output_dir, image_names, is_gray)
	show_images_in_html(image_names, output_dir)
	return output_images


def gray_action(img):
	return gray_scale(img)


def mask_action(img):
	y_size = img.shape[0]
	x_size = img.shape[1]
	region = np.array([[0, y_size], [x_size / 2, (y_size / 2) + 10], [x_size, y_size]], np.int32)
	return region_of_interest(img, [region])


def finding_lanes_on_images():
	image_names = os.listdir(images_directory)
	show_images_in_html(image_names, images_directory)
	images = list(map(lambda img: plt.imread(images_directory + '/' + img), image_names))

	gray_images = save_file_and_display(images, output_gray_images, image_names, gray_action, 1)

	blur_action = lambda img: gaussian_blur(img, blur_kernel_size)

	blur_images = save_file_and_display(gray_images, output_blur_images, image_names, blur_action, 1)

	canny_action = lambda img: canny(img, canny_low_threshold, canny_high_threshold)
	canny_images = save_file_and_display(blur_images, output_canny_images, image_names, canny_action)

	region_images = save_file_and_display(canny_images, output_region_images, image_names, mask_action)

	hough_action = lambda img: hough_lines(img, rho, theta, threshold, min_line_length, max_line_gap)
	hough_images = save_file_and_display(region_images, output_hough_images, image_names, hough_action)

	merge_images = list(map(lambda img: weighted_img(img[0], img[1]), zip(images, hough_images)))
	imagesMerged = save_file_and_display(merge_images, output_merge_images, image_names, lambda img: img)


def process_image(image):
	white_lines = hough_lines(mask_action(canny(gaussian_blur(gray_action(image), blur_kernel_size), canny_low_threshold, canny_high_threshold)), rho, theta, threshold, min_line_length, max_line_gap)
	return weighted_img(image, white_lines)


def process_video(video_file_name, input_video_dir, output_video_dir):
	if not os.path.exists(output_directory + '/' + output_video_dir):
		os.makedirs(output_directory + '/' + output_video_dir)
	clip = VideoFileClip(input_video_dir + '/' + video_file_name)
	output_clip = clip.fl_image(process_image)
	video_file_output = output_directory + '/' + output_video_dir + '/' + video_file_name
	output_clip.write_videofile(video_file_output, audio = False)
	display(
		HTML("""<video width="960" height="540" controls><source src="{0}"></video>""".format(video_file_output))
	)


def finding_lanes_on_videos():
	process_video('solidWhiteRight.mp4', videos_directory, videos_output_directory)
	process_video('solidYellowLeft.mp4', videos_directory, videos_output_directory)
	process_video('challenge.mp4', videos_directory, videos_output_directory)


if __name__ == '__main__':
	finding_lanes_on_images()
	finding_lanes_on_videos()
