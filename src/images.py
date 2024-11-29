'''
    Contains the Image and Images classes
    Author: Kevin Antony Gomez
'''

import file_handler
import os
from PIL import Image as im

class Image:
    def __init__(self, path:str, save_path=False) -> None:
        '''
        Args: 
            path: path to folder containing images
            save_path: path to save image data. If False, does not save
        Returns:
            None
        '''
        self.path = path
        self.file_name = os.path.split(self.path)[-1]
        self.data = None
        self.save_path = save_path
    
    def save(self, path=None)-> None:
        '''
        Save image data
        Args: 
            path: path to save image data. If False, does not save
        Returns:
            None
        '''
        if self.save_path == False:
            print(f"Won't save image since self.save_path = {self.save_path}")
        if path is None:
            path = f'{self.save_path}/{self.file_name}'
            try:
                image = im.fromarray(self.data)
                image.save(path)
            except Exception as e:
                print(f'Failed to save image to {path}\nException:\n{e}')


class Images:
    '''
    Creates an object containing a list of Image objects
    '''
    def __init__(self, path:str, save_path=False) -> None:
        '''
        Args: 
            path: path to folder containing images
            save_path: path to save image data. If False, does not save
        Returns:
            None
        '''
        self.path = path
        self.save_path = save_path
        self.file_handler = file_handler.FileHandler()
        self.images = list()
        self._init_img_list()

    def _init_img_list(self) -> None:
        '''
        Intitalizes self.images with Image objects
        Args: 
            None
        Returns:
            None
        '''
        image_list = self.file_handler.get_image_files(self.path)
        if self.save_path:
            self.file_handler.create_folder(self.save_path)
        for img in image_list:
            self.images.append(Image(img, self.save_path))