import numpy as np
import copy
import PIL.Image as Image


def augment(img_data, config, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)

    img = Image.open(img_data_aug['filepath'])

    if augment:
        # rows, cols = img.shape[:2]
        cols, rows = img.size
        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            # img = cv2.flip(img, 1)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            # img = cv2.flip(img, 0)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2


    img_data_aug['width'] = img.size[0]
    img_data_aug['height'] = img.size[1]
    return img_data_aug, img
