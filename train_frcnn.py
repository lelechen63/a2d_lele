import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras_frcnn import resnet as nn
import keras_frcnn.roi_helpers as roi_helpers

# weight_path = '/home/lchen63/project/a2d_lele/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def parse_arguments():
    """Parse arguments from command line"""
    description = "Train a model."
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--weight_path', '-w',
        default="/home/lchen63/project/a2d_lele/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
        help = 'The pretrained resnet model'
        )

    parser.add_argument('--train_path', '-t',
                default="/mnt/disk1/dat/a2d_lele/",
                help = 'the data path'
                )
    return parser.parse_args()
args = parse_arguments()


class Options:
    def __init__(self, train_path, parser, num_rois=32, horizontal_flips=True, vertical_flips=True,
                 rot_90=False, num_epochs=15, config_filename='config.pickle', output_weight_path='model_frcnn.hdf5',
                 input_weight_path=weight_path):
        self.train_path = train_path
        self.parser = parser
        self.num_rois = num_rois
        self.horizontal_flips = horizontal_flips
        self.vertical_flips = vertical_flips
        self.rot_90 = rot_90
        self.num_epochs = num_epochs
        self.config_filename = config_filename
        self.output_weight_path = output_weight_path
        self.input_weight_path = input_weight_path


sys.setrecursionlimit(40000)
train_path =args.train_path
weight_path = args.weight_path
options = Options(train_path, parser='a2d', num_rois=32, num_epochs=15)

if not options.train_path:   # if filename is not given
    print 'Error: path to training data must be specified. Pass --path to command line'

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
elif options.parser == 'a2d':
    from keras_frcnn.a2d_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

C = config.Config()

C.num_rois = int(options.num_rois)
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path

if options.input_weight_path:
    C.base_net_weights = options.input_weight_path

all_imgs, classes_count, class_mapping = get_data(options.train_path)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.iteritems()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'w') as config_f:
    pickle.dump(C,config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))


data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    ))

optimizer_rpn = Adam(lr=1e-4)
optimizer_classifier = Adam(lr=1e-4)
model_rpn.compile(optimizer=optimizer_rpn, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier,
                         loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)],
                         smetrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')
model_all.load_weights(C.model_path)

epoch_length = len(train_imgs)
num_epochs = int(options.num_epochs)
iter_num = 0
epoch_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.iteritems()}
print('Starting training')
while True:
    # try:

    if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
        rpn_accuracy_rpn_monitor = []
        print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
        if mean_overlapping_bboxes == 0:
            print('RPN is not producing bounding boxes that overlap the ground truth boxes. Results will not be satisfactory. Keep training.')

    X, Y, img_data = data_gen_train.next()

    loss_rpn = model_rpn.train_on_batch(X, Y)

    P_rpn = model_rpn.predict_on_batch(X)

    R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
    X2, Y1, Y2 = roi_helpers.calc_iou(R, img_data, C, class_mapping)

    if X2 is None:
        rpn_accuracy_rpn_monitor.append(0)
        rpn_accuracy_for_epoch.append(0)
        continue

    neg_samples = np.where(Y1[0, :, -1] == 1)
    pos_samples = np.where(Y1[0, :, -1] == 0)

    if len(neg_samples) > 0:
        neg_samples = neg_samples[0]
    else:
        neg_samples = []

    if len(pos_samples) > 0:
        pos_samples = pos_samples[0]
    else:
        pos_samples = []

    rpn_accuracy_rpn_monitor.append(len(pos_samples))
    rpn_accuracy_for_epoch.append((len(pos_samples)))

    if C.num_rois > 1:
        if len(neg_samples) == 0:
            print img_data['filepath']
            continue
        if len(pos_samples) < C.num_rois/2:
            selected_pos_samples = pos_samples.tolist()
        else:
            selected_pos_samples = np.random.choice(pos_samples, C.num_rois/2, replace=False).tolist()

        try:
            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
        except:
            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

        sel_samples = selected_pos_samples + selected_neg_samples
    else:
        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
        selected_pos_samples = pos_samples.tolist()
        selected_neg_samples = neg_samples.tolist()
        if np.random.randint(0, 2):
            sel_samples = random.choice(neg_samples)
        else:
            sel_samples = random.choice(pos_samples)

    loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

    losses[iter_num, 0] = loss_rpn[1]
    losses[iter_num, 1] = loss_rpn[2]

    losses[iter_num, 2] = loss_class[1]
    losses[iter_num, 3] = loss_class[2]
    losses[iter_num, 4] = loss_class[3]

    iter_num += 1
    print 'epoch %d-%d, p/n: %d/%d, loss: %.8f' % (epoch_num, iter_num, len(selected_pos_samples), len(selected_neg_samples), np.sum(loss_rpn) + np.sum(loss_class))

    if iter_num == epoch_length:
        loss_rpn_cls = np.mean(losses[:, 0])
        loss_rpn_regr = np.mean(losses[:, 1])
        loss_class_cls = np.mean(losses[:, 2])
        loss_class_regr = np.mean(losses[:, 3])
        class_acc = np.mean(losses[:, 4])

        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
        rpn_accuracy_for_epoch = []

        if C.verbose:
            print('Epoch {}:'.format(epoch_num))
            print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
            print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
            print('Loss RPN classifier: {}'.format((loss_rpn_cls)))
            print('Loss RPN regression: {}'.format((loss_rpn_regr)))
            print('Loss Classifier classifier: {}'.format((loss_class_cls)))
            print('Loss Classifier regression: {}'.format((loss_class_regr)))
            print('Elapsed time: {}'.format(time.time() - start_time))
        else:
            print('loss_rpn_cls,{},loss_rpn_regr,{},loss_class_cls,{},loss_class_regr,{},class_acc,{},elapsed_time,{}'.format(loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, class_acc, time.time() - start_time))
        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
        iter_num = 0
        start_time = time.time()
        epoch_num += 1
        if epoch_num == 1 or curr_loss < best_loss:
            if C.verbose:
                print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
            best_loss = curr_loss
            model_all.save_weights(C.model_path)
    if epoch_num == num_epochs:
        print('Training complete, exiting.')
        sys.exit()
    # except Exception as e:
    #     print('Exception: {}'.format(e))
    #     continue

