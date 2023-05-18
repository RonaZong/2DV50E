import os
import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import warnings
from multiprocessing import Pool
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_input(input, output_size=None):
    # Size of the image in pixels (size of original image)
    width, height = input.size  # (4752, 3168)

    # Setting the points for cropped image
    left = width * 1 / 6
    top = height * 1 / 30
    right = width * 5 / 6
    bottom = height * 200 / 199

    # Cropped image of above dimension
    resized_input = input.crop((left, top, right, bottom))
    resized_input = resized_input.resize(output_size)
    # Shows the image in image viewer
    # resized_input.show()
    return resized_input


def save_image(args):
    input, input_path, output_path, output_size = args
    # Open image in RGB mode
    input_original = Image.open(os.path.join(input_path, input))
    input_resize = resize_input(input_original, output_size)
    input_resize.save(os.path.join(output_path + input))


def fast_image_resize(input_path, output_path, output_size=None):
    """
    Uses multiprocessing to make it fast
    """
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=100")
        exit()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input = [(file, input_path, output_path, output_size) for file in os.listdir(input_path)]

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_image, input), total=len(input)))


if __name__ == "__main__":
    input_path = "../Data/train1/"
    output_path = "../Data/train1_resized1_150/"
    output_size = (150, 150)
    fast_image_resize(input_path, output_path, output_size=output_size)