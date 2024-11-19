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
    for nn in [30]:
        for n in [5]:
            for set_size in [4]:
                KNNN_CONFIG = {
                    'embedding_model': 'vit_h_14', # ViT-B/32 (always sets height and width to 224), vit_h_14 (always sets height and width to 518) dinov2_vits14
                    'dataset_name': 'retina', #chest, brain, mrnetknee, retina, wrist, pathology
                    'img_prep_type': 'OG', #BW, OG, VD, Hist, Clahe1
                    'transform_height': 518, # 512, 240, 256, 256
                    'transform_width': 518,
                    'train_data_path': 'data/images.nosync/RESC/Train/train/good',
                    'test_data_path': ('data/images.nosync/RESC/Test/test/good', 'data/images.nosync/RESC/Test/test/ungood'), # (good, ungood)
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
                    'transform_height': KNNN_CONFIG['transform_height'],
                    'transform_width': KNNN_CONFIG['transform_width'],
                    '# Train': 'all',
                    '# Test': 'all',
                    'K_TRAIN': KNNN_CONFIG['nn'],
                    'K_TEST': KNNN_CONFIG['n'],
                    'SET_SIZE': KNNN_CONFIG['set_size'],
                    'SCORE': roc_score
                }

                document_results(results_dict)


def vid_det():
    A0 = 0.5
    A1 = 5
    chest_train_img_processor = img_processing.VisibilityDetection('data/images.nosync/RESC/Test/my_test', img_height=512, img_width=512, hpr_param = (A0,A1), \
    tpo_param = (A0,A1), save='')
    chest_train_img_processor.run()

def hist_eq():
    x = img_processing.HistogramEqualization('data/images.nosync/RESC/Test/test/Ungood', 512, 512, 'data/images.nosync/RESC/Test/test/Ungood_hist')
    x.run_equalize()

def clahe_eq():
    x = img_processing.HistogramEqualization('data/images.nosync/RESC/Test/test/Ungood', 512, 512, 'data/images.nosync/RESC/Test/test/Ungood_clahe1_512', clip_limit=1)
    x.run_clahe_equalize()

def bw():
    x = img_processing.HistogramEqualization('data/images.nosync/BraTS2021_slice/train/good', 240, 240, 'data/images.nosync/BraTS2021_slice/train/good_bw')
    x.run_bw()

KNNN()
# vid_det()
# hist_eq()
# clahe_eq()
# bw()