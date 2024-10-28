'''
Main k-nnn + CNN model
'''
import images
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms as T
import file_handler
# from dinov2.models.vision_transformer import vit_small, vit_base
from torch import nn, optim
import math
import concurrent.futures as cf
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA

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
        self.reordered_feat_mat = None
        self.eigen_memory = dict()
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
            features = self.dino(img)
            # features = self.dino.forward_features(img)["x_norm_patchtokens"]
        return image.path, features
        

    def extract_features(self, pkl_dump_path, pkl_file_name, concurrent=True):
        self.dino.eval()
        if concurrent:
            with cf.ThreadPoolExecutor() as executor:
                future_to_image = {executor.submit(self._extract_features, image): image for image in self.images.images[:100]}
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
    
    def load_features(self, pkl_path, force=False):
        if len(self.extracted_features) == 0 or force==True:
            self.extracted_features = self.file_handler.load_data(pkl_path)
        return self.extracted_features
    
    def load_eigen_mem(self, pkl_path, force=False):
        if len(self.eigen_memory.keys()) == 0 or force==True:
            self.eigen_memory = self.file_handler.load_data(pkl_path)
        return self.eigen_memory

    
    def reorder_features(self, k=3):
        assert len(self.extracted_features)!=0 , 'Load features first!'

        ''' v's original shape = (1, (self.patch_h)^2, feat_dim) if using forward_features to extract features
            v's original shape = (1, feat_dim) if not
            below flattens to 1369*384 or (self.patch_h)^2*feat_dim if using forward_features to extract features
            below flattens to 1369*384 or (self.patch_h)^2*feat_dim if using forward_features to extract features'''
        feat_mat = np.array([v.view(-1).numpy() for (p,v) in self.extracted_features]) 
        N = self.feat_dim # dim of test point feature f (same for all points)
        S = N//k # divide into S sets
        L = k # dim of subfeature vectors. So more samples used to calculate the Eigen vectors = larger L
        corr_matrix = np.corrcoef(feat_mat, rowvar=False) # find correlation between all pairs in training set
        reordered_indices = list()
        curr_ind = 0
        set_num = 0
        m = np.zeros(N, dtype=bool)
        m[reordered_indices] = True
        while curr_ind < N:
            for j in range(L):
                if j == 0: # the first element selection differs
                    if set_num == 0:
                        reordered_indices.append(0)
                        m[curr_ind] = True # curr_ind is 0 here
                    else:
                        prev_index = reordered_indices[curr_ind-1]
                        prev_prev_index = reordered_indices[curr_ind-2]
                        arr1 = np.ma.array(corr_matrix[prev_prev_index], mask=m)
                        arr2 = np.ma.array(corr_matrix[prev_index], mask=m)
                        mean_arr = np.ma.mean(np.ma.array([arr1, arr2]), axis=0)
                        min_avg_cor_to_prev_two_ind = np.argmin(mean_arr)
                        reordered_indices.append(min_avg_cor_to_prev_two_ind)
                        m[min_avg_cor_to_prev_two_ind] = True
                elif j == 1:
                    max_cor_to_prev_ind = np.argmax(np.ma.array(corr_matrix[reordered_indices[curr_ind-1]], mask=m))
                    reordered_indices.append(max_cor_to_prev_ind)
                    m[max_cor_to_prev_ind] = True
                    # print(max_cor_to_prev_ind, corr_matrix[0, max_cor_to_prev_ind])
                else:
                    prev_index = reordered_indices[curr_ind-1]
                    prev_prev_index = reordered_indices[curr_ind-2]
                    arr1 = np.ma.array(corr_matrix[prev_prev_index], mask=m)
                    arr2 = np.ma.array(corr_matrix[prev_index], mask=m)
                    mean_arr = np.ma.mean(np.ma.array([arr1, arr2]), axis=0)
                    max_avg_cor_to_prev_two_ind = np.argmax(mean_arr)
                    reordered_indices.append(max_avg_cor_to_prev_two_ind)
                    m[max_avg_cor_to_prev_two_ind] = True
                curr_ind += 1
            set_num += 1
        assert len(set(reordered_indices)) == self.feat_dim, "Feature reorder error!"
        self.reordered_feat_mat = feat_mat[:, reordered_indices]
        return self.reordered_feat_mat

    def calc_eigen_in_sets(self, neighbor_features_mat, S, L):
        set_matrices = [neighbor_features_mat[:, i*L:(i+1)*L] for i in range(S)]
        all_eigenvals = list()
        all_eigenvects = list()
        for submatrix in set_matrices:
            eigenvalues, eigenvectors = LA.eig(submatrix) # normalized eigenvectors (each column)
            all_eigenvals.append(eigenvalues)
            all_eigenvects.append(eigenvectors)
        return (all_eigenvals, all_eigenvects)

    def knnn(self, pkl_dump_path, pkl_file_name, k=3, exclude_test_point=True):
        assert self.reordered_feat_mat is not None, 'Reorder features first!'
        N = self.feat_dim # dim of test point feature f (same for all points)
        S = N//k # divide into S sets
        L = k # dim of subfeature vectors. So more samples used to calculate the Eigen vectors = larger L
        if exclude_test_point:
            neigh = NearestNeighbors(n_neighbors=k+1)
        else:
            neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(self.reordered_feat_mat)
        for i, f in enumerate(self.reordered_feat_mat):
            neighbors = neigh.kneighbors([f]) # returns ([distances], [indices]). Eg: (array([[9.14509525e-04, 1.09526892e+03, 1.12253833e+03]]), array([[0, 1, 7]]))
            neighbor_indices = neighbors[1][0]
            if exclude_test_point: # the first point should NearestNeighbors be the test point but this is to be extra sure
                neighbor_indices = np.delete(neighbor_indices, np.argwhere(neighbor_indices==i)) 
            neighbor_features_mat = self.reordered_feat_mat[neighbor_indices]
            self.eigen_memory[i] = self.calc_eigen_in_sets(neighbor_features_mat, S, L)
        self.file_handler.dump_data(self.eigen_memory, pkl_dump_path, pkl_file_name)