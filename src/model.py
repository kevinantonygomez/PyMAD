'''
Main k-nnn + CNN model
'''
from copy import deepcopy
from sklearn.manifold import TSNE
import cv2
from torchvision.transforms.transforms import Resize
import images
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import file_handler
# from dinov2.models.vision_transformer import vit_small, vit_base
from torch import nn, optim
import math
import concurrent.futures as cf
from sklearn.neighbors import NearestNeighbors

class Model:
    def __init__(self, transform_height, transform_width, train_dir=None, test_dir=None, save=None) -> None:
        self.transform_height = transform_height
        self.transform_width = transform_width
        self.file_handler = file_handler.FileHandler()
        if train_dir is not None:
            self.train_dir = self.file_handler.check_folder(train_dir)
            self.images = images.Images(self.train_dir, save)
        elif test_dir is not None:
            self.test_dir = self.file_handler.check_folder(test_dir)
            self.images = images.Images(self.test_dir, save)
        else:
            print('train_dir and test_dir cannot both be None')
            raise ValueError
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.patch_size = self.dino.patch_size
        self.patch_based_height = math.ceil(self.transform_height/self.patch_size)*self.patch_size
        self.patch_based_width = math.ceil(self.transform_width/self.patch_size)*self.patch_size
        self.patch_h = self.patch_based_height//self.patch_size
        self.patch_w = self.patch_based_width//self.patch_size
        self.feat_dim = 384 # 384 vits14 | 768 vitb14 | 1024 vitl14 | 1536 vitg14
        self.extracted_features = list()
        self.img_transform = {
            "train": T.Compose([
                T.Resize(size=(self.transform_height, self.transform_width)),
                T.CenterCrop(518),
                T.RandomRotation(360),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "embedding": T.Compose([
                T.Resize(size=(self.patch_based_height, self.patch_based_width)),
                T.CenterCrop(size=(self.patch_based_height, self.patch_based_width)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        }

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
    
    def calculate_accuracy(self, outputs, labels):
        # Convert outputs to probabilities using sigmoid
        probabilities = torch.sigmoid(outputs)
        # Convert probabilities to predicted classes
        predicted_classes = probabilities > 0.5
        # Calculate accuracy
        correct_predictions = (predicted_classes == labels.byte()).sum().item()
        total_predictions = labels.size(0)
        return correct_predictions / total_predictions

    def train(self):
        if not self._type_check('train_dir', self.train_dir, str):
            raise TypeError
        try:
            image_datasets = {
                "train": datasets.ImageFolder(self.train_dir, self.img_transform["train"])
            }
            dataloaders = {
                "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=8, shuffle=True)
            }
            class_names = image_datasets["train"].classes
            print(class_names)
        except Exception as e:
            print('Error loading images in train')
            raise e
        
        # model = DinoVisionTransformerClassifier("base")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        model = model.to(self.device)
        model = model.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-6)
        num_epochs = 50
        epoch_losses = []
        epoch_accuracies = []

        print("Training...")
        for epoch in range(num_epochs):
            batch_losses = []
            batch_accuracies = []

            for data in dataloaders["train"]:
                # get the input batch and the labels
                batch_of_images, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # model prediction
                output = model(batch_of_images.to(self.device))
                output = output.squeeze(dim=1)
                # compute loss and do gradient descent
                loss = criterion(output, labels.unsqueeze(1).float().to(self.device))

                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                
                # Calculate and record batch accuracy
                accuracy = self.calculate_accuracy(output, labels.to(self.device))
                batch_accuracies.append(accuracy)

            epoch_losses.append(np.mean(batch_losses))
            epoch_accuracy = np.mean(batch_accuracies)
            epoch_accuracies.append(epoch_accuracy)
            print("  -> Epoch {}: Loss = {:.5f}, Accuracy = {:.3f}%".format(epoch, epoch_losses[-1], 100*epoch_accuracy))

    def _extract_features(self, image):
        image.data = Image.open(image.path).convert('RGB')
        img = self.img_transform['embedding'](image.data).unsqueeze(0)
        with torch.no_grad():
            features = self.dino.forward_features(img)["x_norm_patchtokens"]
        return image.path, features
        

    def extract_features(self, pkl_dump_path, pkl_file_name, concurrent=True):
        if concurrent:
            with cf.ThreadPoolExecutor() as executor:
                future_to_image = {executor.submit(self._extract_features, image): image for image in self.images.images[:10]}
                for future in cf.as_completed(future_to_image):
                    image_path, features = future.result()
                    print(image_path)
                    self.extracted_features.append((image_path, features))
        else:
            for image in self.images.images:
                image_path, features = self._extract_features(image)
                print(image_path, features.shape)
                self.extracted_features.append((image_path, features))
        self.file_handler.dump_data(self.extracted_features, pkl_dump_path, pkl_file_name)
        return self.extracted_features
    
    def load_features(self, pkl_path):
        self.extracted_features = self.file_handler.load_data(pkl_path)
        return self.extracted_features

    def calc_eigen_vect_var(self, all_features, k=3):
        N = all_features[-1].shape # dim of test point feature f
        S = N//k # divide into S sets
        L = k # dim of subfeature vectors. So more samples used to calculate the Eigen vectors = larger L
        pass

    def knnn(self, k=3):
        if len(self.extracted_features) == 0:
            print("Load features first!")
            raise Exception
        neigh = NearestNeighbors(n_neighbors=k)
        # v's original shape = (1, (self.patch_h)^2, feat_dim)
        # example: (1, 1369, 384) so 1369 = (518/14)^2
        samples = [v.view(-1).numpy() for (p,v) in self.extracted_features] 
        print(samples[0])
        neigh.fit(samples)
        for e_f_ind, (p,f) in enumerate(self.extracted_features):
            neighbors = neigh.kneighbors([f.view(-1).numpy()]) # example return: (array([[9.14509525e-04, 1.09526892e+03, 1.12253833e+03]]), array([[0, 1, 7]]))
            neighbor_indices = neighbors[1][0]
            try:
                all_features = [samples[i] for i in neighbor_indices] # add in all the neighbors
                all_features.append(samples[e_f_ind]) # add in the the test point feature
                self.calc_eigen_vect_var(all_features, k)
                # for i in neighbor_indices:
                #     print(i)
                #     self.calc_eignevalues(k)
                    # im = cv2.imread(self.extracted_features[i][0], cv2.IMREAD_GRAYSCALE)
                    # cv2.imshow(f"{self.extracted_features[i][0]}", im)
                # cv2.waitKey(0)
            except:
                pass
            quit()
        

            



# import torch
# from PIL import Image
# import torchvision.transforms as T
# dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
# patch_size = dinov2_vits14.patch_size
# img = Image.open('data/output.nosync/Chest-RSNA/train_512_512_1000_1/good/Normal-5a2f8c86-f21d-4815-98a3-1bb5fa6d9f44.png').convert('RGB')

# transform = T.Compose([
# T.Resize(518),
# T.CenterCrop(518),
# T.ToTensor(),
# T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])
# patch_h  = 518//patch_size
# patch_w  = 518//patch_size

# feat_dim = 384 # vits14

# img = transform(img).unsqueeze(0)

# with torch.no_grad():
#     features = dinov2_vits14.forward_features(img)["x_norm_patchtokens"]

# print(features.shape)
# features = features.squeeze(0)
# print(features.shape)
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.decomposition import PCA

# pca = PCA(n_components=3)
# pca.fit(features)

# pca_features = pca.transform(features)
# pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
# pca_features = pca_features * 255
# print(pca_features.shape)
# plt.imshow(pca_features.reshape(37, 37, 3).astype(np.uint8))

# plt.show()