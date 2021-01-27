import numpy as np
import glob
import os
import cv2

prob_images = glob.glob('original_USGS_txt_04/txt_pixel_result_*.jpg')
#mask_dir = 'original_more_OS_USGS_pixel_GT'
mask_dir = 'original_OS_USGS_pixel_GT'
idx = 0
all_boxes = []
all_confs = []
precisions = []
recalls = []
for map_path in prob_images:
    base_name = os.path.basename(map_path)[17:]
    this_mask_name = mask_dir + '/' + 'masked_' + base_name
    print(this_mask_name)
    prob_map_img = cv2.imread(map_path)
    #prob_map_img = cv2.resize(prob_map_img, (512, 512))
    #prob_map_img = (prob_map_img / 255. )[:,:,0]
    prob_map_img = (prob_map_img / 255.)[:, :, 2]
    prob_map = np.array(prob_map_img > 1. / 3).astype(np.int32)

    mask_img = cv2.imread(this_mask_name)
    #mask_img = cv2.resize(mask_img, (512, 512), interpolation=cv2.INTER_NEAREST) / 255.
    # mask_img = mask_img[:, :, 1]
    mask_img = (mask_img / 255.)[:, :, 1]
    mask_img = np.array(mask_img > 1. / 3).astype(np.int32)


    gt_num_pos_pixels = np.sum(mask_img == 1)
    pred_num_pos_pixels = np.sum(prob_map == 1)
    inter_num_pos_pixels = np.sum((mask_img == 1) & (prob_map == 1))
    if gt_num_pos_pixels == 0:
        precision = 1.0
        recall = 1.0
    else:
        if pred_num_pos_pixels == 0:
            precision = 0.
        else:
            precision = 1.0 * inter_num_pos_pixels / pred_num_pos_pixels
        recall = 1.0 * inter_num_pos_pixels / gt_num_pos_pixels
    print('precision', precision, 'recall', recall)
    precisions.append(precision)
    recalls.append(recall)

print('average precision', 1.0 * np.sum(precisions) / len(precisions))
print('average recall', 1.0 * np.sum(recalls) / len(recalls))