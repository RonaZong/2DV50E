import os
import numpy as np
from PIL import Image, ImageFile
import cv2
from tqdm import tqdm
import warnings
from multiprocessing import Pool
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_input(image):
    """
    Converts image to grayscale using cv2, then computes binary matrix
    of the pixels that are above a certain threshold, then takes out
    the first row where a certain percentage of the pixels are above the
    threshold will be the first clip point. Same idea for col, max row, max col.
    """
    percentage = 0.02

    input = np.array(image)
    input_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    i = input_gray > 0.1 * np.mean(input_gray[input_gray != 0])
    row_sums = np.sum(i, axis=1)
    col_sums = np.sum(i, axis=0)
    rows = np.where(row_sums > input.shape[1] * percentage)[0]
    cols = np.where(col_sums > input.shape[0] * percentage)[0]
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    crop = input[min_row: max_row + 1, min_col: max_col + 1]
    return Image.fromarray(crop)
    # """
    # Stole this from some stackoverflow post but can't remember which,
    # this will add padding to maintain the aspect ratio.
    # """
    # crop_size = input_crop.size # size is in (width, height) format
    # ratio = float(desired_size) / max(crop_size)
    # resize_size = tuple([int(x * ratio) for x in crop_size])
    # input_resize = image.resize(resize_size, Image.ANTIALIAS)
    # input_new = Image.new("RGB", (desired_size, desired_size))
    # input_new.paste(input_resize, ((desired_size - resize_size[0]) // 2, (desired_size - resize_size[1]) // 2))
    # return input_new


def resize_maintain_aspect(image, desired_size):
    """
    Add padding to maintain the aspect ratio.
    """
    original_size = image.size  # original_size is in (width, height) format
    ratio = float(desired_size) / max(original_size)
    new_size = tuple([int(x * ratio) for x in original_size])
    im = image.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def save_image(args):
    input, input_path, output_path, output_size = args
    # Open image in RGB mode
    input_original = Image.open(os.path.join(input_path, input))
    input_resize = resize_input(input_original)
    input_resize = resize_maintain_aspect(input_resize, desired_size=output_size[0])
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

    inputs = [(file, input_path, output_path, output_size)
              for file in os.listdir(input_path)]

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_image, inputs), total=len(inputs)))


if __name__ == "__main__":
    root = "../Data/"
    input_path = os.path.join(root, "test1/")
    output_path = os.path.join(root, "test1_resized_150/")
    output_size = (150, 150)
    fast_image_resize(input_path, output_path, output_size=output_size)
