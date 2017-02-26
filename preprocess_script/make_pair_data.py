import os
import logging
from multiprocessing import Pool
import time, random
# generate same random result when run this program
random.seed(2017)

level = logging.NOTSET
log_filename = '%s.log' % __file__
logging.basicConfig(level=level,
            format='%(asctime)s [%(levelname)s] %(message)s ',
            filename=log_filename,
            datefmt='[%d/%b/%Y %H:%M:%S]')


import configuration as conf
from data import get_human_seqs, similar_between_img
import tool

def make_data_compute_simi_single(video_id):
    name = video_id
    print 'Run task %s (%s)...' % (name, os.getpid())
    start = time.time()

    match_seqs = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', ]
    views = ['090']
    similar_filename = '%s/single_similar_file_cl_bg_%s.txt' % (conf.project.debug_data_path, video_id)
    similar_file = open(similar_filename, 'w')
    img_path_list1, data1 = get_human_seqs(video_id)
    hid, _, _, _ = tool.extract_info_from_path(video_id)
    logging.info('process video id:%s' % video_id)
    for seq in match_seqs:
        for view in views:
            video_id2 = '%s-%s-%s' % (hid, seq, view)
            img_path_list2, data2 = get_human_seqs(video_id2)
            for k in range(len(img_path_list1)):
                for l in range(len(img_path_list2)):
                    filename1 = img_path_list1[k]
                    filename2 = img_path_list2[l]
                    simi = similar_between_img(data1[k], data2[l])
                    similar_file.write('%s\t%s\t%.03f\n' % (filename1, filename2, simi))
    similar_file.close()
    end = time.time()
    print 'Task %s runs %0.2f seconds.' % (name, (end - start))

def make_data_compute_simi():
    print 'Parent process %s.' % os.getpid()
    hids = ['%03d' % i for i in range(1, 125)]
    views = ['090']
    seqs = ['cl-01', 'cl-02', 'bg-01', 'bg-02', ]

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
        p.apply_async(make_data_compute_simi_single, args=(video_id,))
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    print 'All subprocesses done.'

    print 'cat to one file'

    similar_filename = '%s/similar_file_cl_bg.txt' % conf.project.debug_data_path
    single_filename = '%s/single_similar_file_cl_bg_*' % conf.project.debug_data_path
    cmd = 'cat %s > %s' % (single_filename, similar_filename)
    os.system(cmd)
    print 'done all'


def make_full_pair(hids):
    pos_video_id_pairs = []
    neg_video_id_pairs = []
    seqs = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'cl-01', 'cl-02', 'bg-01', 'bg-02', ]
    views = ['%03d' % i for i in range(0, 181, 18)]
    for hid in hids:
        video_ids = []

        for seq in seqs:
            for view in views:
                video_id = '%s-%s-%s' % (hid, seq, view)
                video_ids.append(video_id)
        for i, video_id_1 in enumerate(video_ids):
            for video_id_2 in video_ids[i+1:]:
                pos_video_id_pairs.append([video_id_1, video_id_2])

                random_hid = hids[random.randrange(0, len(hids))]
                while random_hid == hid:
                    random_hid = hids[random.randrange(0, len(hids))]
                random_seq = seqs[random.randrange(0, len(seqs))]
                random_view = views[random.randrange(0, len(views))]
                random_video_id = '%s-%s-%s' % (random_hid, random_seq, random_view)
                neg_video_id_pairs.append([video_id_1, random_video_id])
    return pos_video_id_pairs, neg_video_id_pairs

def make_data_for_rnn():
    train_hids = ['%03d' % i for i in range(1, 51)]
    val_hids = ['%03d' % i for i in range(51, 75)]
    test_hids = ['%03d' % i for i in range(51, 125)]
    train_pos, train_neg = make_full_pair(train_hids)
    val_pos, val_neg = make_full_pair(val_hids)
    test_pos, test_neg = make_full_pair(test_hids)
    train_filename = '%s/train_pairs.txt' % conf.project.data_path
    val_filename = '%s/val_pairs.txt' % conf.project.data_path
    test_filename = '%s/test_pairs.txt' % conf.project.data_path

    train_file = open(train_filename, 'w')
    for pos in train_pos:
        train_file.write('%s,%s,1\n' % (pos[0], pos[1]))
    for neg in train_neg:
        train_file.write('%s,%s,0\n' % (neg[0], neg[1]))
    train_file.close()

    val_file = open(val_filename, 'w')
    for pos in val_pos:
        val_file.write('%s,%s,1\n' % (pos[0], pos[1]))
    for neg in val_neg:
        val_file.write('%s,%s,0\n' % (neg[0], neg[1]))
    val_file.close()

    test_file = open(test_filename, 'w')
    for pos in test_pos:
        test_file.write('%s,%s,1\n' % (pos[0], pos[1]))
    for neg in test_neg:
        test_file.write('%s,%s,0\n' % (neg[0], neg[1]))
    test_file.close()

    seqs = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'cl-01', 'cl-02', 'bg-01', 'bg-02', ]
    views = ['%03d' % i for i in range(0, 181, 18)]
    hids = ['%03d' % i for i in range(1, 125)]
    video_id_path_filename = '%s/video_id_image_list' % conf.project.data_path
    video_id_path_file = open(video_id_path_filename, 'w')
    count = 0
    all = 0
    for hid in hids:
        for view in views:
            for seq in seqs:
                all += 1
    for hid in hids:
        for view in views:
            for seq in seqs:
                video_id = '%s-%s-%s' % (hid, seq, view)
                image_path = '%s/%s/%s/extract/' % (conf.project.data_path, hid, video_id)
                filenames = os.listdir(image_path)
                video_id_path_file.write('%s,%s\n' % (video_id, ','.join(filenames)))

                count += 1
                if count % 100 == 0:
                    logging.info('%05dth/%05d, %s' % (count, all, video_id))
    video_id_path_file.close()

if __name__ == '__main__':
    make_data_for_rnn()