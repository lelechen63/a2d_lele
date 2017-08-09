import os
import h5py
import numpy as np


CLASS_MAPPING = {0: 'adult', 1: 'baby', 2: 'ball', 3: 'bird', 4: 'car', 5: 'cat', 6: 'dog'}
CLASS_MAPPING_INV = {v: k for k, v in CLASS_MAPPING.iteritems()}
CLASSES_COUNT = {}

def get_labels(path):
    # return type: list
    f = h5py.File(path, 'r')
    v = f['id'].value
    shape_v = v.shape
    v = np.int16(np.squeeze(v)).tolist()
    if np.prod(shape_v) == 1:
        v = [v]
    return v


def get_actor_labels(path):
    # return type: list

    f = h5py.File(path, 'r')
    v = f['id'].value
    shape_v = v.shape
    v = np.int16(np.squeeze(v)) / 10 - 1
    v = v.tolist()
    if np.prod(shape_v) == 1:
        v = [v]
    return v


def get_hw(path):
    f = h5py.File(path, 'r')
    mask = f['reMask'].value
    if len(mask.shape) < 3:
        return mask.shape
    else:
        return mask.shape[1:]


def get_mask(path):
    f = h5py.File(path, 'r')
    mask = f['reMask'].value
    labels = f['id'].value
    labels = np.int16(np.squeeze(labels)).tolist()
    if len(mask.shape) == 3:
        mask = np.transpose(mask, [2, 1, 0])
        new_mask = np.zeros(mask.shape[:2], dtype=np.int16)
        for i in range(len(labels)):
            new_mask[np.where(mask[:, :, i] == 1)] = labels[i]
    else:
        mask = np.transpose(mask, [1, 0])
        new_mask = np.zeros(mask.shape[:2], dtype=np.int16)
        new_mask[np.where(mask == 1)] = labels
    return new_mask


def get_bboxes(path):
    # return: nparray, bboxes in columns
    f = h5py.File(path, 'r')
    v_bbox = f['reBBox'].value
    return v_bbox


def get_all(path, imageset, filepath):
    # return: {'bboxes': ..., 'imageset:'..., 'height':..., 'width':..., 'filepath':...}

    bboxes = get_bboxes(path)
    labels = get_actor_labels(path)
    hw = get_hw(path)
    ret_dic = {}
    bb_list = []
    for i in range(bboxes.shape[1]):
        bb = {}
        bb['x1'] = bboxes[0, i]
        bb['y1'] = bboxes[1, i]
        bb['x2'] = bboxes[2, i]
        bb['y2'] = bboxes[3, i]
        bb['class'] = CLASS_MAPPING[labels[i]]
        class_name = CLASS_MAPPING[labels[i]]
        if class_name not in CLASSES_COUNT:
            CLASSES_COUNT[class_name] = 1
        else:
            CLASSES_COUNT[class_name] += 1
        bb_list.append(bb)
    ret_dic['bboxes'] = bb_list
    ret_dic['imageset'] = imageset
    ret_dic['width'] = hw[0]
    ret_dic['height'] = hw[1]
    ret_dic['filepath'] = filepath
    return ret_dic


def get_data(input_path):
    all_imgs = []
    visualise = False

    # data_paths = [os.path.join(input_path,s) for s in ['VOC2012']]
    data_paths = [input_path]


    print('Parsing annotation files')

    for data_path in data_paths:

        annot_path = os.path.join(data_path, 'Annotations/mat')
        imgs_path = os.path.join(data_path, 'labeled_frame_images')
        imgsets_path_trainval = os.path.join(data_path, 'id_train_expand.txt')
        imgsets_path_test = os.path.join(data_path, 'id_val_expand.txt')

        trainval_files = []
        test_files = []
        try:
            with open(imgsets_path_trainval) as f:
                for line in f:
                    trainval_files.append(line.strip() + '.png')
            with open(imgsets_path_test) as f:
                for line in f:
                    test_files.append(line.strip() + '.png')
        except Exception as e:
            print(e)


        annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]

        for annot in annots:
            print annot
            print '+++'
            if annot[0]=='.':
                continue
            id = annot.split('/')[-1]
            mat_list = os.listdir(annot)
            for i in range(len(mat_list)):
                id_frame = id + str(i) + '.png'
                filepath = os.path.join(imgs_path, id_frame)
                path = os.path.join(annot, mat_list[i])
                if id_frame in trainval_files:
                    imageset = 'trainval'
                else:
                    imageset = 'test'
                imgs = get_all(path, imageset, filepath)
                all_imgs.append(imgs)


    return all_imgs, CLASSES_COUNT, CLASS_MAPPING_INV
