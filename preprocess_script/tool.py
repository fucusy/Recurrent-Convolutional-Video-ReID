__author__ = 'fucus'
import os
import re
import numpy as np
import configuration as conf

def get_video_path_by_video_id(video_id):
    return "%s/%s.avi" % (conf.data.video_path, video_id)


def extract_info_from_path(path):
    """
    :param path: xxxx/xxx/xxx/{hid}-{cond}-{seq}-{view}-xxx.xxx
        for example: /Users/fucus/Documents/irip/gait_recoginition/code/001-nm-01-090.avi
        with .avi, it also works
    :return: {hid}, {cond}, {seq}, {view}
    """
    basename = os.path.basename(path)
    split_base = basename.split('.')
    if len(split_base) > 1:
        img_id = ''.join(split_base[:-1])
    else:
        img_id = split_base[0]
    split_img_id = img_id.split('-')
    hid = split_img_id[0]
    cond = split_img_id[1]
    seq = split_img_id[2]
    view = split_img_id[3]
    return hid, cond, seq, view

def load_img_path_list(path, pattern=None):
    """

    :param path: the test img folder
    :return:
    """
    list_path = os.listdir(path)
    filtered = []
    # change to reg to match extension

    p = re.compile(r".*\.[jpg|png|bmp]")
    for x in list_path:
        if re.match(p, x):
            if pattern is not None and re.match(pattern, x):
                filtered.append(x)
            elif pattern is None:
                filtered.append(x)
    result = ["%s/%s" % (path, x) for x in filtered]
    result = sorted(result)
    return np.array(result)
