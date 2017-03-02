__author__ = 'fucus'
import logging
level = logging.NOTSET
log_filename = '%s.log' % __file__
format = '%(asctime)-12s[%(levelname)s] %(message)s'
datefmt='%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=level,
            format=format,
            filename=log_filename,
            datefmt= datefmt)
import os
from multiprocessing import Pool
import time

import tool
import configuration as conf
from data import extract_cover


def prepare_same_size_seq_rbg(video_id):
    name = video_id
    print 'Run task %s (%s)...' % (name, os.getpid())
    start = time.time()
    hid, cond, seq, view = tool.extract_info_from_path(video_id)
    back_path = "%s/%s-bkgrd-%s.avi" % (conf.data.video_path, hid, view)
    video_path = tool.get_video_path_by_video_id(video_id)
    target_path = extract_cover(video_path, back_path)
    logging.info('cal %s' % video_id)
    end = time.time()
    print 'Task %s runs %0.2f seconds.' % (name, (end - start))

def prepare_same_size_seq_rbg_main(test=True):
    print 'Parent process %s.' % os.getpid()
    hids = ['%03d' % i for i in range(1, 125)]
    views = ['%03d' % i for i in range(0, 181, 18)]
    seqs = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05', 'nm-06', 'cl-01', 'cl-02', 'bg-01', 'bg-02', ]
    if not os.path.exists(conf.project.debug_data_path):
        os.makedirs(conf.project.debug_data_path)
    video_ids = []
    for hid in hids:
        for seq in seqs:
            for view in views:
                video_id = '%s-%s-%s' % (hid, seq, view)
                video_ids.append(video_id)

    if test:
        for video_id in video_ids:
            logging.info('I am going to deal with %s' % video_id)
    else:
        p = Pool(4)
        for video_id in video_ids:
            p.apply_async(prepare_same_size_seq_rbg, args=(video_id,))
        print 'Waiting for all subprocesses done...'
        p.close()
        p.join()
        print 'All subprocesses done.'

if __name__ == '__main__':
    #test = False
    # prepare_same_size_seq_rbg_main(test)
    video_id = '005-nm-06-000'
    prepare_same_size_seq_rbg(video_id)

