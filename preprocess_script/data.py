from multiprocessing import Pool
import numpy as np
import os
import subprocess
from scipy.misc import imsave, imread
from skimage.morphology import disk
from skimage import color as color, filter
import time

import logging
import sys
level = logging.NOTSET
log_filename = '%s.log' % __file__
logging.basicConfig(level=level,
            format='%(asctime)s [%(levelname)s] %(message)s ',
            filename=log_filename,
            datefmt='[%Y %b %d %H:%M:%S]')
import tool
import np_helper
import configuration as conf


def extract_cover(video_path, back_video_path):
    """
    using ffmpeg to split video to a sequence images at folder named with the video file name
    then get human box data by subtraction and save box data in txt file named image name

    and
    crop every image in img_folder by the human box data, save the crop image at the same
    folder, named specially

    :param video_path:
    :param back_video_path:
    :return: image folder name
    """
    video_name = os.path.basename(video_path)
    hid, cond, seq, view = tool.extract_info_from_path(video_name)

    video_id = '%s-%s-%s-%s' % (hid, cond, seq, view)
    img_folder = '%s/%s/%s/' % (conf.project.data_path, hid, video_id)
    origin_img_folder = "%s/origin/" % img_folder
    cover_path = '%s/cover/' % img_folder
    extract_path = '%s/extract/' % img_folder
    extract_cover_path = '%s/extract_cover/' % img_folder

    back_name = os.path.basename(back_video_path)
    back_video_id = ''.join(back_name.split('.')[:-1])
    back_folder = "%s/%s/%s/" % (conf.project.data_path, hid, back_video_id)
    origin_back_folder = '%s/origin/' % back_folder
    mean_filename = "%s/avg.jpg" % back_folder

    if not os.path.exists(origin_img_folder):
        os.makedirs(origin_img_folder)
        output = "%s/%%04d.jpg" % origin_img_folder
        subprocess.call(["ffmpeg", "-i", video_path, output])


    if not os.path.exists(origin_back_folder):
        os.makedirs(origin_back_folder)
        output = "%s/%%04d.jpg" % origin_back_folder
        subprocess.call(["ffmpeg", "-i", back_video_path, output])

    # get mean back img
    if not os.path.exists(mean_filename):
        back_img_path = tool.load_img_path_list(origin_back_folder)
        back_img = img_path_2_pic(back_img_path)
        mean_back_img = np.mean(back_img, axis=0)
        mean_back_img = mean_back_img.astype(np.int32).astype(np.uint8)
        imsave(mean_filename, mean_back_img)
    else:
        mean_back_img = imread(mean_filename)


    img_path = tool.load_img_path_list(origin_img_folder)
    # do subtraction
    img_data = img_path_2_pic(img_path)
    grey_img_data = np.array([color.rgb2gray(x) for x in img_data])
    back_grey_img = color.rgb2grey(mean_back_img)

    box_filename = "%s/%s-box_file.txt" % (img_folder, video_id)
    box_file = open(box_filename, 'w')
    box_file.write("\t".join(["img", "left-up-height-i", "left-up-width-i", "box-height", "box-width"]))
    box_file.write("\n")

    if not os.path.exists(extract_cover_path):
        os.makedirs(cover_path)
        os.makedirs(extract_path)
        os.makedirs(extract_cover_path)

        for i, img in enumerate(grey_img_data):
            base = os.path.basename(img_path[i])
            save_filename = "%s/%s.bmp" % (cover_path, os.path.splitext(base)[0])
            sub_img, res = subtract(img, back_grey_img)
            imsave(save_filename, sub_img)

            line = [base]
            for el in res:
                line.append(str(el))
            box_file.write("\t".join(line))
            box_file.write("\n")

            # do crop
            extract_img = np_helper.extract_np(img_data[i], res[0:2], res[2:4])
            extract_img_filename = "%s/%s.jpg" % (extract_path, base.rstrip(".jpg"))
            height = extract_img.shape[0]
            width = extract_img.shape[1]
            if height == conf.data.crop_size_height and width == conf.data.crop_size_width:
                imsave(extract_img_filename, extract_img)

            extract_cover_img = np_helper.extract_np(sub_img, res[0:2], res[2:4])
            if height == conf.data.crop_size_height and width == conf.data.crop_size_width:
                imsave(extract_img_filename, extract_img)
                extract_cover_img = grey2bmp(extract_cover_img)
                extract_cover_filename = '%s/%s.bmp' % (extract_cover_path, base.rstrip('.jpg'))
                imsave(extract_cover_filename, extract_cover_img)

    box_file.close()
    return extract_cover_path


def get_human_seqs(video_id):
    hid, cond, seq, view = tool.extract_info_from_path(video_id)
    back_path = "%s/%s-bkgrd-%s.avi" % (conf.data.video_path, hid, view)
    video_path = tool.get_video_path_by_video_id(video_id)
    target_path = extract_cover(video_path, back_path)
    img_path_list = tool.load_img_path_list(target_path)
    return img_path_list, img_path_2_pic(img_path_list)


def similar_between_img_seqs(seqs1, seqs2):
    """
    :param seqs1:
    :param seqs2:
    :return:
    """
    # swap seqs1 with seqs2 if len seqs1 larger than seqs2, make sure len of seqs1 less or equal than seqs2
    have_swap = False
    if len(seqs1) > len(seqs2):
        tmp = seqs1
        seqs1 = seqs2
        seqs2 = tmp
        have_swap = True

    start2_range = range(len(seqs2) - len(seqs1) + 1)
    max_simi = 0
    max_start2 = -1
    max_start1 = -1
    len1 = len(seqs1)
    len2 = len(seqs2)

    similar_cache = np.zeros((len1, len2), dtype=np.float32)
    for idx1 in range(len(seqs1)):
        for idx2 in range(len(seqs2)):
            similar_cache[idx1][idx2] = similar_between_img(seqs1[idx1], seqs2[idx2])

    for start2 in start2_range:
        for start1 in range(len(seqs1)):
            simi_list = []
            for i in range(len(seqs1)):
                idx1 = (start1+i) % len(seqs1)
                idx2 = (start2+i) % len(seqs2)
                simi_list.append(similar_cache[idx1][idx2])
            simi_list_sorted = sorted(simi_list)
            skip_bi_count = 5
            simi = sum(simi_list_sorted[skip_bi_count:-1*skip_bi_count])
            if simi > max_simi:
                max_simi = simi
                max_start1 = start1
                max_start2 = start2

    if conf.project.debug_info_slower_speed:
        seqs1_path = '%s/match_seqs/seqs1/' % conf.project.debug_data_path
        seqs2_path = '%s/match_seqs/seqs2/' % conf.project.debug_data_path
        for p in [seqs1_path, seqs2_path]:
            if not os.path.exists(p):
                os.makedirs(p)

        for i in range(len(seqs1)):
            idx = (i+max_start1) % len(seqs1)
            filename = '%s/%04d.jpg' % (seqs1_path, i)
            imsave(filename, seqs1[idx])

        for i in range(len(seqs2)):
            idx = (i+max_start2) % len(seqs2)
            filename = '%s/%04d.jpg' % (seqs2_path, i)
            imsave(filename, seqs2[idx])

    if have_swap:
        tmp = max_start1
        max_start1 = max_start2
        max_start2 = tmp

    return max_simi / len(seqs1), max_start1, max_start2


def similar_between_img(img1, img2):
    shape1 = img1.shape
    shape2 = img2.shape

    height1 = shape1[0]
    width1 = shape1[1]

    height2 = shape2[0]
    width2 = shape2[1]

    size1 = width1 * height1
    size2 = width2 * height2

    max_width = max(width1, width2)
    max_height = max(height1, height2)

    to_shape = (max_height, max_width)
    resize1, mask1 = np_helper.imresize_padding(img1, to_shape)
    resize2, mask2 = np_helper.imresize_padding(img2, to_shape)
    mask = mask1 * mask2
    if conf.project.debug_info_slower_speed:
        if not os.path.exists(conf.project.debug_data_path):
            os.makedirs(conf.project.debug_data_path)
        img1_res_filename = '%s/similar_between_img_img1_res.jpg' % conf.project.debug_data_path
        img2_res_filename = '%s/similar_between_img_img2_res.jpg' % conf.project.debug_data_path
        imsave(img1_res_filename, resize1)
        imsave(img2_res_filename, resize2)
    resize1 /= 255
    resize2 /= 255
    mul = resize1 * resize2 * mask
    both_1_count = np.sum(mul)

    # (resize1 - 1) * (resize2 - 1)
    both_0_count = np.sum((mul - resize1 - resize2 + 1) * mask)
    return both_1_count * 2.0 / (np.sum(resize1) + np.sum(resize2))


def grey2bmp(im):
    im[im > 0] = 1
    return im


def retrieve_similar(human_seqs, top=3):
    pass


def get_most_id(hids):
    pass


def img_path_2_pic(img_paths, func=None):
    img_pics = []
    for img_path in img_paths:
        im = imread(img_path)
        if func is not None:
            im = func(im)
        img_pics.append(im)
    return np.array(img_pics)

def subtract(img, back):
    """

    :param img: numpy array
    :param back: numpy array
    :param save_filename: save the sub result to this filename
    :return: a black and white img stored in numpy array
    """

    if len(img.shape) == 3 and img.shape[2] == 3:
        img = color.rgb2grey(img)

    if len(back.shape) == 3 and back.shape[2] == 3:
        back = color.rgb2grey(back)

    sub_img = np.subtract(img, back)
    sub_img = np.absolute(sub_img)
    clap = 0.08
    low_values_indices = sub_img < clap  # Where values are low
    sub_img[low_values_indices] = 0  # All low values set to 0


    sub_img_uint8 = sub_img * 255
    sub_img_uint8 = sub_img_uint8.astype(np.uint8)
    sub_img_uint8 = filter.median(sub_img_uint8, disk(5))

    sub_img = sub_img_uint8.astype(np.float32) / 255
    return sub_img, get_human_position(sub_img_uint8)


def get_human_position(img):
    """
    :param img: grey type numpy.array image
    :return: left up corner and width, height of the box,
        [{left-up-corner-i}, {left-up-corner-j}, {box-i}, {box-j}]

        you can get the left up pixel by img[{left-up-corner-i}][{left-up-corner-j}]
        right bottom pixel by img[{left-up-corner-i}+{box-i}][{left-up-corner-j}+{box-j}]
    """
    left_blank = 0
    right_blank = 0

    up_blank = 0
    down_blank = 0

    height = img.shape[0]
    width = img.shape[1]

    for i in range(height):
        if np.sum(img[i, :]) == 0:
            up_blank += 1
        else:
            break

    for i in range(height-1, -1, -1):
        if np.sum(img[i, :]) == 0:
            down_blank += 1
        else:
            break

    for i in range(width):
        if np.sum(img[:, i]) == 0:
            left_blank += 1
        else:
            break

    for i in range(width-1, -1, -1):
        if np.sum(img[:, i]) == 0:
            right_blank += 1
        else:
            break
    left_up_corner_i = up_blank
    left_up_corner_j = left_blank
    box_i = max(0, height - up_blank - down_blank)
    box_j = max(0, width - left_blank - right_blank)
    return [left_up_corner_i, left_up_corner_j, box_i, box_j]


def pre_compute():
    hids = ['%03d' % i for i in range(1, 125)]
    views = ['090']
    seqs = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'cl-01', 'cl-02', 'bg-01', 'bg-02']
    all_count = len(hids) * len(views) * len(seqs)
    count = 0
    for hid in hids:
        for seq in seqs:
            for view in views:
                count += 1
                video_id = '%s-%s-%s' % (hid, seq, view)
                logging.info('deal %04dth/%04d video:%s' % (count, all_count, video_id))
                get_human_seqs(video_id)


def pre_compute_similarity():
    hids = ['%03d' % i for i in range(1, 125)]
    views = ['090']
    seqs = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'cl-01', 'cl-02', 'bg-01', 'bg-02']
    all_count = len(hids) * len(views) * len(seqs)
    count = 0
    if not os.path.exists(conf.project.debug_data_path):
        os.makedirs(conf.project.debug_data_path)
    similar_filename = '%s/similar_file.txt' % conf.project.debug_data_path
    similar_file = open(similar_filename, 'w')
    video_ids = []
    for hid in hids:
        for seq in seqs:
            for view in views:
                video_id = '%s-%s-%s' % (hid, seq, view)
                video_ids.append(video_id)

    for i in range(len(video_ids)):
        video_id1 = video_ids[i]
        count += 1
        logging.info('deal %04dth/%04d video:%s' % (count, all_count, video_id1))
        img_path_list1, data1 = get_human_seqs(video_id1)
        for j in range(i+1, len(video_ids)):
            video_id2 = video_ids[j]
            img_path_list2, data2 = get_human_seqs(video_id2)
            for k in range(len(img_path_list1)):
                for l in range(len(img_path_list2)):
                    filename1 = img_path_list1[k]
                    filename2 = img_path_list2[l]
                    simi = similar_between_img(data1[k], data2[l])
                    logging.info('similar between %s,%s %.03f' % (filename1, filename2, simi))
                    similar_file.write('%s\t%s\t%.03f\n' % (filename1, filename2, simi))
    similar_file.close()


def pre_compute_similarity_single_seqs(video_id):
    hids = ['%03d' % i for i in range(1, 125)]
    name = video_id
    print 'Run task %s (%s)...' % (name, os.getpid())
    start = time.time()
    match_seqs = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
    views = ['090']
    similar_filename = '%s/single_match_file_%s.txt' % (conf.project.debug_data_path, video_id)
    similar_file = open(similar_filename, 'w')
    img_path_list1, data1 = get_human_seqs(video_id)
    hid, _, _, _ = tool.extract_info_from_path(video_id)
    logging.info('process video id:%s' % video_id)
    for hid in hids:
        for seq in match_seqs:
            for view in views:
                video_id2 = '%s-%s-%s' % (hid, seq, view)
                img_path_list2, data2 = get_human_seqs(video_id2)
                simi, start1, start2 = similar_between_img_seqs(data1, data2)
                logging.info('simi between %s, %s is %.03f, %03d, %03d' % (video_id, video_id2, simi, start1, start2))
                similar_file.write('%s\t%s\t%.03f\t%03d\t%03d\n' % (video_id, video_id2, simi, start1, start2))
    similar_file.close()
    end = time.time()
    print 'Task %s runs %0.2f seconds.' % (name, (end - start))


def pre_compute_similarity_seqs():
    print 'Parent process %s.' % os.getpid()
    hids = ['%03d' % i for i in range(4, 125)]
    views = ['090']
    seqs = ['nm-05', 'nm-06', 'cl-01', 'cl-02', 'bg-01', 'bg-02', ]
    seqs = ['bg-01', 'bg-02', ]
    if not os.path.exists(conf.project.debug_data_path):
        os.makedirs(conf.project.debug_data_path)
    video_ids = []
    for hid in hids:
        for seq in seqs:
            for view in views:
                video_id = '%s-%s-%s' % (hid, seq, view)
                video_ids.append(video_id)

    p = Pool()
    for video_id in video_ids:
        p.apply_async(pre_compute_similarity_single_seqs, args=(video_id,))
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    print 'All subprocesses done.'

    print 'cat to one file'

    #similar_filename = '%s/seq_match_result_nm_cl_bg.txt' % conf.project.debug_data_path
    #single_filename = '%s/single_match_file_*' % conf.project.debug_data_path
    #cmd = 'cat %s > %s' % (single_filename, similar_filename)
    #os.system(cmd)
    print 'done all'

def max_height_width(views=['180', ]):
    seqs = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'cl-01', 'cl-02', 'bg-01', 'bg-02', ]
    hids = ['%03d' % i for i in range(4, 125) if i != 77]
    video_ids = []
    for hid in hids:
        for seq in seqs:
            for view in views:
                video_id = '%s-%s-%s' % (hid, seq, view)
                video_ids.append(video_id)
    max_height = 0
    max_width = 0
    count = 0
    output_every = 1000
    for video_id in video_ids:
        img_path_list, imgs = get_human_seqs(video_id)
        for i, img in enumerate(imgs):
            count += 1
            if img.shape[0] == 240 or img.shape[1] == 320:
                logging.info('error shape %s' % img_path_list[i])
                continue

            if img.shape[0] > max_height:
                logging.info('change max_height from %s to %s cause %s' % (max_height, img.shape[0], img_path_list[i]))
                max_height = img.shape[0]
            if img.shape[1] > max_width:
                logging.info('change max_width from %s to %s cause %s' % (max_width, img.shape[1], img_path_list[i]))
                max_width = img.shape[1]

            if count % output_every == 0:
                logging.info('max height:%s' % max_height)
                logging.info('max width:%s' % max_width)

    logging.info('at last, max height %s' % max_height)
    logging.info('at last, max width %s' % max_width)

if __name__ == '__main__':
    #pre_compute_similarity_single_seqs('066-cl-01-090')
    #pre_compute_similarity_seqs()
    max_height_width()
