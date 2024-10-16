import file_handler

class Image:
    def __init__(self, path) -> None:
        self.path = path
        self.data = None

class Images:
    def __init__(self, path:str) -> None:
        self.path = path
        self.file_handler = file_handler.FileHandler()
        self.images = list()
        self._init_img_list()

    def _init_img_list(self) -> None:
        image_list = self.file_handler.get_image_files(self.path)
        for img in image_list:
            self.images.append(Image(img))