__author__ = 'fucus'
import numpy as np
import configuration as conf

def imresize_padding(img, shape, padding_val=0):
    height = img.shape[0]
    width = img.shape[1]
    to_height = shape[0]
    to_width = shape[1]

    mask = np.zeros(img.shape) + 1
    res_img = np.array(img)

    assert to_height >= height
    assert to_width >= width

    s_left = int((to_width - width) / 2)
    s_right = to_width - width - s_left
    s_up = int((to_height - height) / 2)
    s_down = to_height - height - s_up

    if s_left > 0:
        res_img = shift_left(res_img, s_left, padding_val)
        mask = shift_left(mask, s_left, padding_val)
    if s_right > 0:
        res_img = shift_right(res_img, s_right, padding_val)
        mask = shift_right(mask, s_right, padding_val)
    if s_up > 0:
        res_img = shift_up(res_img, s_up, padding_val)
        mask = shift_up(mask, s_up, padding_val)
    if s_down > 0:
        res_img = shift_down(res_img, s_down, padding_val)
        mask = shift_down(mask, s_down, padding_val)
    return res_img, mask



def shift_left(img, left=10.0, padding_val=0, padding=True):
    """
    :param numpy.array img: represented by numpy.array
    :param float left: how many pixels to shift to left, this value can be negative that means shift to
                    right {-left} pixels
    :return: numpy.array
    """
    if 0 < abs(left) < 1:
        left = int(left * img.shape[1])
    else:
        left = int(left)

    if len(img.shape) <= 2:
        is_grey = True
    else:
        is_grey = False

    if padding:
        width = img.shape[1] + abs(left)
        img_shift_left = np.zeros((img.shape[0], width)) + padding_val
    else:
        img_shift_left = np.zeros(img.shape) + padding_val

    if left >= 0:
        if is_grey:
            if padding:
                img_shift_left[:, left:] = img
            else:
                img_shift_left = img[:, left:]
        else:
            if padding:
                img_shift_left[:, left:, :] = img
            else:
                img_shift_left = img[:, left:, :]

    else:
        if is_grey:
            if padding:
                img_shift_left[:, :left] = img
            else:
                img_shift_left = img[:, :left]
        else:
            if padding:
                img_shift_left[:, :left, :] = img
            else:
                img_shift_left = img[:, :left, :]
    return img_shift_left


def shift_right(img, right=10.0, padding_val=0, padding=True):
    return shift_left(img, -right, padding_val, padding)


def shift_up(img, up=10.0, padding_val=0, padding=True):
    """
    :param numpy.array img: represented by numpy.array
    :param float up: how many pixels to shift to up, this value can be negative that means shift to
                    down {-up} pixels
    :return: numpy.array
    """


    if 0 < abs(up) < 1:
        up = int(up * img.shape[0])
    else:
        up = int(up)

    if len(img.shape) <= 2:
        is_grey = True
    else:
        is_grey = False

    if padding:
        height = abs(up) + img.shape[0]
        img_shift_up = np.zeros((height, img.shape[1])) + padding_val
    else:
        img_shift_up = np.zeros(img.shape) + padding_val
    if up >= 0:
        if is_grey:
            if padding:
                img_shift_up[up:, :] = img
            else:
                img_shift_up = img[up:, :]
        else:
            if padding:
                img_shift_up[up:, :, :] = img
            else:
                img_shift_up = img[up:, :, :]
    else:
        if is_grey:
            if padding:
                img_shift_up[:up, :] = img
            else:
                img_shift_up = img[:up, :]
        else:
            if padding:
                img_shift_up[:up, :, :] = img
            else:
                img_shift_up = img[:up, :, :]

    return img_shift_up


def shift_down(img, down=10.0, padding_val=0, padding=True):
    return shift_up(img, -down, padding_val, padding)


def extract_np(img, left_up_corner, box_size):
    """

    :param img: numpy array, test rbg numpy array
    :param left_up_corner: [{i}, {j}], i for height dim direction, j for width dim direction
    :param box_size:    [{height}, {width}]
    :return: img after crop
    """

    height = img.shape[0]
    width = img.shape[1]

    up_blank = left_up_corner[0] - (conf.data.crop_size_height - box_size[0]) / 2
    left_blank = left_up_corner[1] - (conf.data.crop_size_width - box_size[1]) / 2

    down_blank = height - up_blank - conf.data.crop_size_height
    right_blank = width - left_blank - conf.data.crop_size_width

    img = shift_left(img, left_blank, padding=False)
    img = shift_right(img, right_blank, padding=False)
    img = shift_up(img, up_blank, padding=False)
    img = shift_down(img, down_blank, padding=False)

    return img


