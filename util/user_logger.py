import logging
import time
import os

def logger_process(cur_path,log_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    riqi = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    log_path = cur_path + '/logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = log_path + riqi + '.log'
    fh = logging.FileHandler(log_name, mode='w')
    sh = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s-%(filename)s[line:%(lineno)d]-%(levelname)s:%(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
