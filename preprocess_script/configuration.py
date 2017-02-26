__author__ = 'fucus'
import os
import time

time_now = time.localtime()

class project:
    base_folder = "/home/chenqiang/github/Recurrent-Convolutional-Video-ReID/preprocess_script"
    data_path = "/home/chenqiang/data/gait-rnn/"
    debug_data_path = "%s/debug_data/%s/" % (data_path, time.strftime('%y-%m-%d-%H-%M', time_now))
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(debug_data_path):
        os.makedirs(debug_data_path)

    debug_info_slower_speed = False

class data:
    video_path = "/home/chenqiang/data/CASIA_DatasetB"
    extract_max_height = 169
    extract_max_width = 93
    crop_size_height = extract_max_height + 8
    crop_size_width = extract_max_width + 8
