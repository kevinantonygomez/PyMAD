'''
    This file contains an example driver code to run the K-NNN algorithm.
    Author: Kevin Antony Gomez
'''

import model
import os
import pandas as pd


def document_results(results_dict:dict, dataset_name:str) -> None:
    '''
    Writes results to a csv file. Creates one if it doesn't exist. Appends if it does exist.
    Args: 
        results_dict: dict containing results
        dataset_name: csv file name
    Returns:
        None
    '''
    results_file_path = f'data/results/results_{dataset_name}.csv'
    if os.path.exists(results_file_path):
        df = pd.read_csv(results_file_path)
        new_df = pd.DataFrame([results_dict])
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(results_dict, index=[0])
    df.to_csv(results_file_path, index=False)


def KNNN(KNNN_CONFIG:dict) -> None: 
    '''
    Runs the K-NNN algirithm and outputs results via document_results
    Args: 
        KNNN_CONFIG: dict containing the K-NNN configs. Must have the following keys:
            'embedding_model':str       Embedding model types supported: ViT-B/32, vit_h_14, dinov2_vits14, dinov2_vitb14
            'dataset_name': str         Dataset name. Does not affect functionality
            'img_prep_type': str        Type of preprocessing done on the images. Does not affect functionality
            'transform_height': int     target height to resize images prior to feature extraction
            'transform_width': int      target width to resize images prior to feature extraction
            'interpolationmode': str    'bilinear' or 'bicubic'. Type of interpolation to use during image resizing
            'train_data_path': str      directory of train images
            'test_data_path': tuple     directory of test images -> (path/to/good/imgs, path/to/ungood/imgs)
            'output_root_path': str     directory to store computed values during training for use during testing
            'set_size': int             desired feature set size
            'nn': int                   number of neighbors neighbors
            'n': int                    number of neighbors. High values increase computational time
    Returns:
        None
    '''
    knnn = model.Model(KNNN_CONFIG)
    knnn.train()
    roc_score = knnn.test()

    results_dict = {
        'model': KNNN_CONFIG['embedding_model'],
        'dataset_name': KNNN_CONFIG['dataset_name'],
        'img_prep_type': KNNN_CONFIG['img_prep_type'],
        'transform_height': KNNN_CONFIG['transform_height'],
        'transform_width': KNNN_CONFIG['transform_width'],
        'interpolationmode': KNNN_CONFIG['interpolationmode'],
        '# Train': 'all',
        '# Test': 'all',
        'K_TRAIN': KNNN_CONFIG['nn'],
        'K_TEST': KNNN_CONFIG['n'],
        'SET_SIZE': KNNN_CONFIG['set_size'],
        'SCORE': roc_score
    }

    document_results(results_dict, KNNN_CONFIG['dataset_name'])



if __name__ == "__main__":
    KNNN_CONFIG = {
        # 
        'embedding_model': 'dinov2_vits14', 
        'dataset_name': 'brain',
        'img_prep_type': 'OG',
        'transform_height': 518,
        'transform_width': 518, 
        'interpolationmode': 'bilinear',
        'train_data_path': 'path/to/train/images',
        'test_data_path': ('path/to/test/good_images', 'path/to/test/ungood_images'),
        'output_root_path': 'path/to/output/folder',
        'set_size': 64,
        'nn': 100,
        'n': 50
    }
    KNNN(KNNN_CONFIG)
