import os
import pickle
import bz2file as bz2
from pathlib import Path

class FileHandler:
    '''
    Handles file i/o
    '''
    def __init__(self) -> None:
       pass

    def _type_check(self, obj_name:str, obj, type)->bool:
        '''
        Private method to type check an object
        :param obj_name: variable name
        :param obj: actual obj
        :param type: expected type of obj
        '''
        try:
            if not isinstance(obj, type):
                print(f'!!! {obj_name} should be of type {type} not {type(obj)}')
            else:
                return True
        except Exception as e:
            print(f"!!! Error in _type_check: \n {e}")
        return False


    def get_image_files(self, dir_path:str, include_sub_dirs=False) -> list:
        '''
        Return list of images in the passed directory
        :param dir_path: path to folder of images to be encoded
        :param include_sub_dirs: optionally recurse into nested folders
        '''
        if not self._type_check('dir_path', dir_path, str):
            raise TypeError 
        if os.path.exists(dir_path):
            if include_sub_dirs:
                if not dir_path.endswith('/'): dir_path = f'{dir_path}/'
                files_gen = (p.resolve() for p in Path(dir_path).glob("**/*") if p.suffix in {".jpeg", ".jpg", ".png", ".webp"} and not p.name.startswith('.'))
                files = [str(f) for f in list(files_gen)]
            else:
                files = [f'{dir_path}/{f}' for f in os.listdir(dir_path) if (f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") \
                    or f.endswith(".webp") or f.endswith(".bmp") and not f.startswith('.'))]
            return files
        else:
            print(f'!!! {dir_path} does not exist')
            return