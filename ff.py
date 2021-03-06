import os
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import cv2
import argparse
sys.setrecursionlimit(40000)

def parse_arguments():
    """Parse arguments from command line"""
    description = "Train a model."
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--weight_path', '-w',
        default="model_frcnn.hdf5",
        help = 'The pretrained model'
        )

    parser.add_argument('--test_path', '-t',
                default="/mnt/disk1/dat/a2d_lele/id_test_expand.txt",
                help = 'the data path'
                )
    return parser.parse_args()
args = parse_arguments()
weight_path = args.weight_path
test_path = args.test_path
class Options:
    def __init__(self, test_path, num_rois=32, config_filename='config.pickle'):
        self.test_path = test_path
        self.num_rois = num_rois
        self.config_filename = config_filename


sys.setrecursionlimit(40000)

options = Options(test_path, num_rois=32)


config_output_filename = options.config_filename

with open(config_output_filename, 'r') as f_in:
    C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path


def format_img(img, C):
    # img_min_side = float(C.im_size)
    # (height,width,_) = img.shape
    #
    # if width <= height:
    #     f = img_min_side/width
    #     new_height = int(f * height)
    #     new_width = int(img_min_side)
    # else:
    #     f = img_min_side/height
    #     new_width = int(f * width)
    #     new_height = int(img_min_side)
    # img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    # img = img[:, :, (2, 1, 0)]
    # img = img.astype(np.float32)
    img = np.array(img)
    X = img.copy()
    X[:, :, 0] -= C.img_channel_mean[0]
    X[:, :, 1] -= C.img_channel_mean[1]
    X[:, :, 2] -= C.img_channel_mean[2]
    X /= C.img_scaling_factor
    # img = np.transpose(img, (2, 0, 1))
    # img = np.expand_dims(img, axis=0)
    return X, img


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.iteritems()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (1024, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)
# model_classifier_only.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

# model_classifier.load_weights()

# model_rpn.summary()
# model_classifier.summary()

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True

img_paths = []
prefix = '/media/lele/DATA/a2d_lele/labeled_frame_images/'
with open(img_path) as f:
    for line in f:
        img_paths.append(prefix + line[:-1] + '.png')
for img_name in img_paths:
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(img_name)
    st = time.time()
    filepath = img_name
    img_name = img_name.split('/')[-1]
    img = Image.open(filepath)

    # X, img_scaled = format_img(img, C)
    img_scaled = np.array(img, dtype=np.float32)
    X = img_scaled.copy()
    X[:, :, 0] -= C.img_channel_mean[0]
    X[:, :, 1] -= C.img_channel_mean[1]
    X[:, :, 2] -= C.img_channel_mean[2]
    X /= C.img_scaling_factor

    # img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
    # img_scaled[:, :, 0] += 123.68
    # img_scaled[:, :, 1] += 116.779
    # img_scaled[:, :, 2] += 103.939

    X = np.expand_dims(X, axis=0)

    img_scaled = img
    # img_scaled = img_scaled.astype(np.uint8)
    # img_scaled = Image.fromarray(img_scaled)

    # if K.image_dim_ordering() == 'tf':
    #     X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)


    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([16*x, 16*y, 16*(x+w), 16*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]

            draw = ImageDraw.Draw(img_scaled)
            # draw.rectangle((x1, y1), (x2, y2), outline=class_to_color[key])
            cv2.rectangle(img_scaled,(x1, y1), (x2, y2), class_to_color[key],2)

            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))

            # (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            # textOrg = (x1, y1-0)

            draw.text((x2 - x1, y2 - y1), key)
            # cv2.rectangle(img_scaled, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
            # cv2.rectangle(img_scaled, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            # cv2.putText(img_scaled, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    print('Elapsed time = {}'.format(time.time() - st))
    # cv2.imshow('img', img_scaled)
    # cv2.waitKey(0)
    #cv2.imwrite('./imgs/{}.png'.format(idx),img_scaled)
    img_scaled.save('/media/lele/DATA/a2d_lele/output/' + img_name, 'PNG')
    print(all_dets)
