'''
Main k-nnn + CNN model
'''
from sklearn.externals._packaging.version import InvalidVersion
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
from tqdm import tqdm
import cv2

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
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]
            )
        }

    def load_features(self, pkl_path, force=False):
        if len(self.extracted_features) == 0 or force==True:
            self.extracted_features = self.file_handler.load_data(pkl_path)
        return self.extracted_features

    def load_reordered_features(self, pkl_path, force=False):
        if self.reordered_feat_mat is None or force==True:
            self.reordered_feat_mat = self.file_handler.load_data(pkl_path)
        return self.reordered_feat_mat
    
    def load_eigen_mem(self, pkl_path, force=False):
        if len(self.eigen_memory.keys()) == 0 or force==True:
            self.eigen_memory = self.file_handler.load_data(pkl_path)
        return self.eigen_memory


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
    

    def _extract_features(self, image):
        if type(image) is str:
            image_data = Image.open(image).convert('RGB')
            img = self.img_transform['embedding'](image_data).unsqueeze(0)
        else:
            image.data = Image.open(image.path).convert('RGB')
            img = self.img_transform['embedding'](image.data).unsqueeze(0)
            # x = self.img_transform['embedding'](image.data)
            # x = x.numpy()
            # x = np.transpose(x, (1, 2, 0))
            # cv2.imshow('im', x)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # quit()
        with torch.no_grad():
            features = self.dino(img)
            # features = self.dino.forward_features(img)["x_norm_patchtokens"]
        if type(image) is str:
            return image, features
        return image.path, features
        

    def extract_features(self, pkl_dump_path, pkl_file_name, concurrent=True, select_images=None):
        assert len(self.images.images) != 0, 'No images!'
        num_images = len(self.images.images)
        assert select_images is None \
            or (type(select_images) is int and select_images > 0 and select_images <= num_images), 'Invalid select_images val'
        if select_images is not None:
            num_images = select_images
        self.dino.eval()
        if concurrent:
            with cf.ThreadPoolExecutor() as executor:
                future_to_image = {executor.submit(self._extract_features, image): image for image in self.images.images[:num_images]}
                for future in tqdm(cf.as_completed(future_to_image), total=len(future_to_image), desc='Extracting img features'):
                    image_path, features = future.result()
                    # print(image_path)
                    self.extracted_features.append(features.view(-1).numpy())
        else:
            for image in self.images.images:
                image_path, features = self._extract_features(image)
                print(image_path, features.shape)
                self.extracted_features.append((image_path, features))
        self.extracted_features = np.array(self.extracted_features)
        self.file_handler.dump_data(self.extracted_features, pkl_dump_path, pkl_file_name)
        return self.extracted_features
    
    def compute_correlation_matrix(self, data):
        """
        Computes the correlation matrix for a given 2D NumPy array.
        
        :param data: A 2D NumPy array of size [n, m] where n is the number of samples
                    and m is the number of features.
        :return: The correlation matrix of size [m, m].
        """
        # Standardize each feature (column) to have mean 0 and variance 1
        standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)

        # Compute the correlation matrix
        correlation_matrix = np.dot(standardized_data.T, standardized_data) / (data.shape[0] - 1)

        return correlation_matrix
    
    def reorder_features(self, pkl_dump_path, pkl_file_name, set_size=3):
        assert len(self.extracted_features)!=0 , 'Load features first!'

        ''' v's original shape = (1, (self.patch_h)^2, feat_dim) if using forward_features to extract features
            v's original shape = (1, feat_dim) if not
            below flattens to 1369*384 or (self.patch_h)^2*feat_dim if using forward_features to extract features
            below flattens to 384 or feat_dim if not'''
        feat_mat = self.extracted_features
        N = self.feat_dim # dim of test point feature f (same for all points)
        S = N//set_size # divide into S sets
        L = set_size # dim of subfeature vectors. So more samples used to calculate the Eigen vectors = larger L
        corr_matrix = np.corrcoef(feat_mat, rowvar=False) # find correlation between all pairs in training set
        # correlation_matrix = self.compute_correlation_matrix(feat_mat) # verified returns the same as above
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
        self.file_handler.dump_data(self.reordered_feat_mat, pkl_dump_path, pkl_file_name)
        return self.reordered_feat_mat

    def calc_eigen_in_sets(self, neighbor_features_mat, S, L):
        set_matrices = [neighbor_features_mat[:, i*L:(i+1)*L] for i in range(S)]
        all_eigenvals = list()
        all_eigenvects = list()
        for submatrix in set_matrices:
            cov_matrix = np.cov(submatrix, rowvar=False)
            try:
                eigenvalues, eigenvectors = LA.eig(cov_matrix)  # normalized eigenvectors (each column)
            except:
                print("Eigenvector calc exception")
                # If the covariance matrix is singular, use just an identity matrix
                eigenvalues = np.ones(cov_matrix.shape[0])
                eigenvectors = np.eye(cov_matrix.shape[0])

            all_eigenvals.append(eigenvalues)
            all_eigenvects.append(eigenvectors)
            # print(eigenvalues)
        return (all_eigenvals, all_eigenvects)

    def calc_anomaly_score(self, f, neighbor_features_mat, eigen_mem, S, L, k=3):
        neigh_sets = [neighbor_features_mat[:, i*L:(i+1)*L] for i in range(S)]
        f_sets = [f[i*L:(i+1)*L] for i in range(S)]
        score = 0
        for i in range(k):
            for j in range(L):
                for s in range(S):
                    # print(len(eigen_mem[i][0]))
                    # print(len(eigen_mem[i][1]))
                    # print(neigh_sets)
                    # print('\n')
                    # print(neigh_sets[s][i])
                    # print(f_sets[s] - neigh_sets[s][i])
                    a = np.abs(np.dot(f_sets[s] - neigh_sets[s][i], eigen_mem[i][1][s][j]))
                    assert eigen_mem[i][0][s][j] > 0, f'Negative or zero eigenvalue! {eigen_mem[i][0][s][j]}'
                    b = 1/np.emath.sqrt(eigen_mem[i][0][s][j])
                    score += np.dot(a,b)
                    # print(a)
                    # print(b)
                    # print(a,b)
                    # print(score)
                    # quit()
        # quit()
        return score


    def knnn(self, pkl_dump_path, pkl_file_name, mode, k, set_size, exclude_test_point=True, test_ds=set()):
        assert self.reordered_feat_mat is not None, 'Reorder features first!'
        N = self.feat_dim # dim of test point feature f (same for all points)
        S = N//set_size # divide into S sets
        L = set_size # dim of subfeature vectors. So more samples used to calculate the Eigen vectors = larger L
        print(f'k:{k}, N: {N}, S:{S}, L:{L}')
        if exclude_test_point and mode == 'train':
            neigh = NearestNeighbors(n_neighbors=k+1)
        else:
            neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(self.reordered_feat_mat)
        if mode == 'train':
            for i, f in enumerate(self.reordered_feat_mat):
                neighbors = neigh.kneighbors([f]) # returns ([distances], [indices]). Eg: (array([[9.14509525e-04, 1.09526892e+03, 1.12253833e+03]]), array([[0, 1, 7]]))
                neighbor_indices = neighbors[1][0]
                if exclude_test_point: # the first point should be the test point but this is to be extra sure
                    neighbor_indices = np.delete(neighbor_indices, np.argwhere(neighbor_indices==i)) 
                neighbor_features_mat = self.reordered_feat_mat[neighbor_indices]
                self.eigen_memory[i] = self.calc_eigen_in_sets(neighbor_features_mat, S, L)
            self.file_handler.dump_data(self.eigen_memory, pkl_dump_path, pkl_file_name)

        elif mode == 'test':
            assert len(self.eigen_memory.keys()) != 0, 'Load eigen memory first!'

            scores = list()
            # for image in tqdm(self.images.images, desc='Processing images'):
            for image in self.images.images[:25]:
                f = self._extract_features(image.path)[1].view(-1).numpy()
                neighbors = neigh.kneighbors([f])
                neighbor_indices = neighbors[1][0]
                neighbor_features_mat = self.reordered_feat_mat[neighbor_indices] 
                req_eigen_mem = {i: self.eigen_memory[key] for i, key in enumerate(neighbor_indices)}
                score = self.calc_anomaly_score(f, neighbor_features_mat, req_eigen_mem, S, L, k=k)
                print(score)
                scores.append(score)
            avg_score = np.mean(scores)
            print('AVG score:', avg_score)
            print('Min score:', np.min(scores))
            print('Max score:', np.max(scores))

        else:
            print(f'Invalid value for mode: {mode}')
            raise ValueError
