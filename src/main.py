import model
import img_processing
import os
import pandas as pd

def document_results(results_dict):
    results_file_path = 'data/results.csv'
    if os.path.exists(results_file_path):
        df = pd.read_csv(results_file_path)
        new_df = pd.DataFrame([results_dict])
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(results_dict, index=[0])
    df.to_csv(results_file_path, index=False)


def KNNN(): 
    for nn in [100]:
        for n in [3]:
            for set_size in [8]:
                KNNN_CONFIG = {
                    'embedding_model': 'dinov2_vits14',
                    'dataset_name': 'mrnetknee', #chest, brain, mrnetknee
                    'img_prep_type': 'Hist', #BW, OG, VD, Hist
                    'transform_height': 256, # 512, 240, 256
                    'transform_width': 256,
                    'train_data_path': 'data/images.nosync/mrnetkneemris/MRNet-v1.0/train/axial_imgs_hist',
                    'test_data_path': ('data/images.nosync/mrnetkneemris/MRNet-v1.0/valid/good/axial_imgs_hist', 'data/images.nosync/mrnetkneemris/MRNet-v1.0/valid/ungood/axial_imgs_hist'), # (good, ungood)
                    'output_root_path': 'data/output.nosync',
                    'set_size': set_size,
                    'nn': nn,
                    'n': n
                }

                knnn = model.Model(KNNN_CONFIG)
                knnn.train()
                roc_score = knnn.test()

                results_dict = {
                    'model': KNNN_CONFIG['embedding_model'],
                    'dataset_name': KNNN_CONFIG['dataset_name'],
                    'img_prep_type': KNNN_CONFIG['img_prep_type'],
                    '# Train': 'all',
                    '# Test': 'all',
                    'K_TRAIN': KNNN_CONFIG['nn'],
                    'K_TEST': KNNN_CONFIG['n'],
                    'SET_SIZE': KNNN_CONFIG['set_size'],
                    'SCORE': roc_score
                }

                document_results(results_dict)


def vid_det():
    A0 = 0.05
    A1 = 5
    chest_train_img_processor = img_processing.VisibilityDetection('data/images.nosync/BraTS2021_slice/mytest', img_height=240, img_width=240, hpr_param = (A0,A1), \
    tpo_param = (A0,A1), save='')
    chest_train_img_processor.run()

def hist_eq():
    x = img_processing.HistogramEqualization('data/images.nosync/mrnetkneemris/MRNet-v1.0/valid/ungood/axial_imgs', 256, 256, 'data/images.nosync/mrnetkneemris/MRNet-v1.0/valid/ungood/axial_imgs_hist')
    x.run_equalize()

def clahe_eq():
    x = img_processing.HistogramEqualization('data/images.nosync/mrnetkneemris/MRNet-v1.0/train/axial_imgs', 256, 256, 'data/images.nosync/mrnetkneemris/MRNet-v1.0/train/axial_imgs_clahe3', clip_limit=3)
    x.run_clahe_equalize()

def bw():
    x = img_processing.HistogramEqualization('data/images.nosync/BraTS2021_slice/train/good', 240, 240, 'data/images.nosync/BraTS2021_slice/train/good_bw')
    x.run_bw()

KNNN()
# vid_det()
# hist_eq()
# clahe_eq()
# bw()