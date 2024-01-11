import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool


def image_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)


parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default=r'C:\Users\Eon\PycharmProjects\GanZoo\dataset\B301MM\high')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default=r'C:\Users\Eon\PycharmProjects\GanZoo\dataset\B301MM\low')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default=r'C:\Users\Eon\PycharmProjects\GanZoo\dataset\B301MM\combine')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', help='If used, chooses single CPU execution instead of parallel execution', action='store_true',default=True)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_A)

if not args.no_multiprocessing:
    pool=Pool()

for sp in splits:
    img_list = os.listdir(args.fold_A)

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    if not os.path.isdir(args.fold_AB):
        os.makedirs(args.fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(args.fold_A, name_A)

        path_B = os.path.join(args.fold_B, 'U'+name_A)
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = 'combine'+name_A
            path_AB = os.path.join(args.fold_AB, name_AB)
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_A, path_B, path_AB))
            else:
                im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)

if not args.no_multiprocessing:
    pool.close()
    pool.join()
