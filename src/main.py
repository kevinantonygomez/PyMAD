import img_processing
import model
import numpy as np


def driver(): 
    '''
    When a0 is high, even the deepest valleys (or areas of low intensity) remain far from the viewpoint.
    When setting a1 to high values, the transformation will result in significant valleys and ridges even when the pixels that created them had similar intensity values.
    '''


    TEST_OR_TRAIN = 'train' # 'train'
    GOOD_UNGOOD = 'good'
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    A0 = 1000
    A1 = 1
    SAVE_DIR = f'data/output.nosync/Chest-RSNA/{TEST_OR_TRAIN}_{IMG_HEIGHT}_{IMG_WIDTH}_{A0}_{A1}/{GOOD_UNGOOD}' # False
    SRC_DIR = f'data/images.nosync/Chest-RSNA/val/{GOOD_UNGOOD}'
    TRAIN_DIR = f'data/output.nosync/Chest-RSNA/train_512_512_1000_1/{GOOD_UNGOOD}' # will be same as SAVE_DIR if used after VisibilityDetection
    PKL_DUMP_PATH = f'data/pickles/'
    PKL_FILE_NAME = f'{TEST_OR_TRAIN}_{IMG_HEIGHT}_{IMG_WIDTH}_{A0}_{A1}_{GOOD_UNGOOD}_hundred'
    EIGEN_PKL_FILE_NAME = f'{TEST_OR_TRAIN}_{IMG_HEIGHT}_{IMG_WIDTH}_{A0}_{A1}_{GOOD_UNGOOD}_hundred_eigen'
    K = 3

    # chest_train_img_processor = img_processing.VisibilityDetection(SRC_DIR, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, a0=A0, a1=A1, save=SAVE_DIR)
    # chest_train_img_processor.run()

    chest_train_model = model.Model(IMG_HEIGHT, IMG_WIDTH, train_dir=TRAIN_DIR)
    chest_train_model.extract_features(PKL_DUMP_PATH, PKL_FILE_NAME, concurrent=True)
    chest_train_model.load_features(f'{PKL_DUMP_PATH}/{PKL_FILE_NAME}')
    chest_train_model.reorder_features(k=K)
    chest_train_model.knnn(PKL_DUMP_PATH, EIGEN_PKL_FILE_NAME, k=K)
    chest_train_model.load_eigen_mem(f'{PKL_DUMP_PATH}/{EIGEN_PKL_FILE_NAME}')

driver()