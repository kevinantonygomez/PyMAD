'''
    This class handles file i/o
    Author: Kevin Antony Gomez
'''

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

    def create_folder(self, dir_path:str):
        if not self._type_check('dir_path', dir_path, str):
            raise TypeError
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                print(f'Output folder {dir_path} successfully created')
            except Exception as e:
                print(f"Error while making save directory {dir_path}")
                raise e
    
    def pkl_exists(self, dir_path):
        if not self._type_check('dir_path', dir_path, str):
            raise TypeError
        pkl_parent_path, pkl_name = os.path.split(dir_path)
        if '.pbz2' not in pkl_name:
            dir_path = f'{pkl_parent_path}/{pkl_name}.pbz2'
        if not os.path.exists(dir_path):
            return False
        return True

    def dump_data(self, data:object, dump_path:str, file_name:str, silent:bool=False) -> bool:
        '''
        Pickle dumps an object
        :param silent: print success/failure message if False.
        '''
        if not self._type_check('data', data, object):
            raise TypeError 
        if not self._type_check('dump_path', dump_path, str):
            raise TypeError 
        if not self._type_check('file_name', file_name, str):
            raise TypeError 
        if not self._type_check('silent', silent, bool):
            raise TypeError 
        if not os.path.exists(dump_path):
            self.create_folder(dump_path)
        try:
            # if not silent: print('Saving...')
            with bz2.BZ2File(f'{dump_path}/{file_name}.pbz2', 'w') as file:
                pickle.dump(data, file)
            if not silent:
                print(f'+++ Saved: {dump_path}/{file_name}.pbz2')
            return True
        except Exception as e:
            print(f'!!! Failed to save: {dump_path}/{file_name}.pbz2\n', e)
        return False

    def load_data(self, pkl_path:str, silent:bool=False) -> bool:
        '''
        Pickle dumps an object
        :param silent: print success/failure message if False.
        '''
        if not self._type_check('pkl_path', pkl_path, str):
            raise TypeError 
        if not self._type_check('silent', silent, bool):
            raise TypeError 
        try:
            head, tail = os.path.split(pkl_path)
            if '.pkl' not in tail:
                pkl_path = f'{pkl_path}.pbz2'
            # if not silent: print('Loading...')
            data = bz2.BZ2File(pkl_path, 'rb')
            result = pickle.load(data)
            if not silent:
                print(f'+++ Loaded: {pkl_path}')
                # if type(result) is dict:
                #     print(f'    Number of keys: {len(result.keys())}')
                # elif type(result) is list:
                #     print(f'    Number of objects: {len(result)}')
            return result
        except Exception as e:
            print(f'!!! Failed to load: {pkl_path}\n', e)

    def get_image_files(self, dir_path:str, include_sub_dirs=False) -> list:
        '''
        Return list of images in the passed directory
        :param dir_path: path to folder of images
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