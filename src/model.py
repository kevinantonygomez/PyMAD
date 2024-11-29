'''
    This class represents the K-NNN model and contains all the functionality required to train and test the K-NNN algorithm 
    Author: Kevin Antony Gomez
'''

import numpy
from torchvision.transforms.functional import InterpolationMode
import images as images_class
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
import file_handler
import math
import concurrent.futures as cf
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
import clip
from torchvision.models import vit_h_14, ViT_H_14_Weights

class Model:
    def __init__(self, KNNN_CONFIG:dict)-> None:
        '''
            Init class with passed configs
            Args: 
                KNNN_CONFIG: dict containing the K-NNN configs. Must have the following keys:
                    'embedding_model':str -> dinov2_vits14, dinov2_vitb14, ViT-B/32, vit_h_14
                    'dataset_name': str
                    'img_prep_type': str
                    'transform_height': int
                    'transform_width': int
                    'interpolationmode': str -> bilinear, bicubic
                    'train_data_path': str
                    'test_data_path': str
                    'output_root_path': str
                    'set_size': int
                    'nn': int
                    'n': int
            Returns:
                None
            '''
        self.dataset_name = KNNN_CONFIG['dataset_name']
        self.transform_height = KNNN_CONFIG['transform_height']
        self.transform_width = KNNN_CONFIG['transform_width']
        self.train_data_path = KNNN_CONFIG['train_data_path']
        self.test_data_path_good = KNNN_CONFIG['test_data_path'][0]
        self.test_data_path_ungood = KNNN_CONFIG['test_data_path'][1]
        self.output_root_path = KNNN_CONFIG['output_root_path']
        self.set_size = KNNN_CONFIG['set_size']
        self.nn = KNNN_CONFIG['nn']
        self.n = KNNN_CONFIG['n']
        self.embedding_model = KNNN_CONFIG['embedding_model'] 
        self.img_prep_type = KNNN_CONFIG['img_prep_type']
        self.interpolationmode = KNNN_CONFIG['interpolationmode']
        if self.interpolationmode == 'bicubic':
            interpolationmode = InterpolationMode.BICUBIC
        else:
            interpolationmode = InterpolationMode.BILINEAR
        self.file_handler = file_handler.FileHandler()
        if torch.mps.is_available(): # support for GPU acceleration on MacOS
            self.device = torch.device("mps")
        if torch.cuda.is_available(): # support for GPU acceleration on cuda-compatible platforms
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self.embedding_model == 'dinov2_vits14':
            self.feat_dim = 384 # 384 vits14 | 768 vitb14 | 1024 vitl14 | 1536 vitg14
            self.model = torch.hub.load("facebookresearch/dinov2", self.embedding_model)
        elif self.embedding_model == 'dinov2_vitb14':
            self.feat_dim = 768
            self.model = torch.hub.load("facebookresearch/dinov2", self.embedding_model)
        elif self.embedding_model == 'ViT-B/32':
            self.feat_dim = 512
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.embedding_model = 'ViT-B_32'
        elif self.embedding_model == 'vit_h_14':
            self.feat_dim = 1280
            self.weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
            self.model = vit_h_14(self.weights)
        self.inst_resize = False
        self.model.to(self.device)
        if 'dinov2' in self.embedding_model:
            self.patch_size = self.model.patch_size
        if self.transform_height is None or self.transform_width is None:
            self.inst_resize = True
            self.img_transform = {
                "embedding": T.Compose([
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )
            }
        else:
            if 'dinov2' in self.embedding_model:
                self.patch_based_height = math.ceil(self.transform_height/self.patch_size)*self.patch_size # ensure dims work with selected model
                self.patch_based_width = math.ceil(self.transform_width/self.patch_size)*self.patch_size
                self.img_transform = {
                    "embedding": T.Compose([
                        T.Resize(size=(self.patch_based_height, self.patch_based_width),interpolation=interpolationmode),
                        T.CenterCrop(size=(self.patch_based_width, self.patch_based_width)),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ]
                    )
                }
            elif self.embedding_model == 'ViT-B_32':
                self.img_transform = {
                    "embedding": T.Compose([
                        T.Resize(size=(224, 224),interpolation=interpolationmode),
                        T.CenterCrop(size=(224, 224)),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ]
                    )
                }
            elif self.embedding_model == 'vit_h_14':
                self.img_transform = {
                    "embedding": T.Compose([
                        T.Resize(size=(518, 518),interpolation=interpolationmode),
                        T.CenterCrop(size=(518, 518)),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ]
                    )
                }
        self.features_clusters = None
        self.reordered_features = None
        self.eigen_memory = None
        

    def _extract_features(self, image:images_class.Image)-> torch.Tensor:
        '''
        Helper function that extracts and returns features from a passed image
        Args: 
            image: image object
        Returns:
            features: tensor of extracted features
        '''
        image.data = Image.open(image.path).convert('RGB')
        w, h = image.data.size
        if self.inst_resize:
            transform = T.Resize((math.ceil(w/self.patch_size)*self.patch_size, math.ceil(h/self.patch_size)*self.patch_size))
            img = transform(image.data)
            img = self.img_transform['embedding'](img).unsqueeze(0).to(self.device)
        else:
            img = self.img_transform['embedding'](image.data).unsqueeze(0).to(self.device)
        if 'dinov2' in self.embedding_model:
            with torch.no_grad():
                features = self.model(img)
        elif self.embedding_model == 'ViT-B_32':
            # img = self.preprocess(image.data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(img)
        elif self.embedding_model == 'vit_h_14':
            with torch.no_grad():
                feats = self.model._process_input(img)
                batch_class_token = self.model.class_token.expand(img.shape[0], -1, -1)
                feats = torch.cat([batch_class_token, feats], dim=1)
                feats = self.model.encoder(feats)
                features = feats[:, 0]
        features = features.view(-1).numpy() # flatten
        return features
        

    def extract_features(self, src_path:str, concurrent=True, select_images=None)-> numpy.ndarray:
        '''
        Extracts and returns features from all images given the path to the folder of images
        Args: 
            src_path: path to folder containing images
            concurrent: if True, extract features concurrently. Else, extract sequentially
            select_images: if None, extract all images in folder. Else, if int, extract specified number of images
        Returns:
            extracted_features: array of feature tensors of images in src_path
        '''
        try:
            images = images_class.Images(src_path)
        except Exception as e:
            print(e)
            return
        assert len(images.images) != 0, f'No images found in {src_path}'
        num_images = len(images.images)
        assert select_images is None or (type(select_images) is int and select_images > 0 and select_images <= num_images), 'Invalid value for select_images'
        if select_images is not None:
            num_images = select_images
        self.model.eval()
        extracted_features = list()
        if concurrent:
            with cf.ThreadPoolExecutor() as executor:
                future_to_image = {executor.submit(self._extract_features, image): image for image in images.images[:num_images]}
                for future in tqdm(cf.as_completed(future_to_image), total=len(future_to_image), desc=f'Extracting img features from {src_path}'):
                    features = future.result()
                    extracted_features.append(features)
        else:
            for image in images.images:
                features = self._extract_features(image)
                extracted_features.append(features)
        extracted_features = np.array(extracted_features)
        return extracted_features


    def hierarchical_clustering(self, correlation_matrix:numpy.ndarray, do_plot:bool=False)-> numpy.ndarray:
        '''
        Perform hierarchical clustering given a correlation matrix
        Args: 
            correlation_matrix: correlation matrix
            do_plot: if True, plot the dendrogram of linkage matrix
        Returns:
            Z: The hierarchical clustering encoded as a linkage matrix
        '''
        # Transform the correlation matrix to a distance matrix
        distance_matrix = 1 - np.abs(correlation_matrix)
        distance_matrix = distance_matrix.astype(np.float32) # original float64 causes symmetric matrix issues in squareform
        # Ensure the diagonal elements are zero
        np.fill_diagonal(distance_matrix, 0)
        # Perform hierarchical/agglomerative clustering
        Z = linkage(squareform(distance_matrix), method='average')
        if do_plot:
            dendrogram(Z)
            plt.title('Hierarchical Cluster Dendrogram')
            plt.xlabel('Data Point Indexes')
            plt.ylabel('Distance')
            plt.show()
        return Z


    def equal_size_clustering(self, Z:numpy.ndarray, m:int, target_size:int)-> numpy.ndarray:
        """
        Divides features into approximately equal-sized clusters.
        Args: 
            Z: Linkage matrix from hierarchical clustering.
            m: Total number of features.
            target_size: Desired number of features in each cluster.
        Returns:
            initial_clusters: Array indicating cluster membership for each feature.
        """
        
        if m % target_size != 0:
            raise ValueError("Number of features target_size must be a divisor of number of features m")
        k = m // target_size
        # Initial clustering
        initial_clusters = fcluster(Z, k, criterion='maxclust')
        # If the initial clustering does not result in k clusters, we need to adjust it
        for cluster_ind in range(1, k + 1):
            cluster_size = np.sum(initial_clusters == cluster_ind) 
            if cluster_size == 0:
                # find non empty cluster
                for i in range(1, k + 1):
                    if i != cluster_ind:
                        cluster_size_other = np.sum(initial_clusters == i) 
                        if cluster_size_other > target_size:
                            # change an elemnet to be of the other 
                            initial_clusters[np.where(initial_clusters == i)[0][0]] = cluster_ind
                            break
        
        # Target size for each cluster
        target_size = m // k

        # Create a list to hold the merge level for each feature
        merge_levels = np.zeros(2 * m - 1)

        # Fill in the merge levels from the linkage matrix
        for i in range(m - 1):
            cluster_formed = int(Z[i, 0]), int(Z[i, 1])
            for j in cluster_formed:
                merge_levels[j] = Z[i, 2]

        # Adjustment for equal size
        for cluster_id in range(1, k + 1):
            while np.sum(initial_clusters == cluster_id) > target_size:
                # Find the loosest feature in this cluster
                indices_in_cluster = np.where(initial_clusters == cluster_id)[0]
                loosest_feature = indices_in_cluster[np.argmax(merge_levels[indices_in_cluster])]

                # Find the closest cluster to move the loosest feature into
                closest_cluster = None
                min_distance = np.inf
                for i in range(1, k + 1):
                    if i != cluster_id and np.sum(initial_clusters == i) < target_size:
                        indices_in_other_cluster = np.where(initial_clusters == i)[0]
                        if indices_in_other_cluster.size > 0:
                            cluster_distances = merge_levels[indices_in_other_cluster + m - 1]  # Adjust indices for clusters
                            distance = np.abs(merge_levels[loosest_feature + m - 1] - cluster_distances.mean())  # Adjust index for current feature
                            if distance < min_distance:
                                min_distance = distance
                                closest_cluster = i

                # Move the feature to the closest cluster
                if closest_cluster is not None:
                    initial_clusters[loosest_feature] = closest_cluster

        return initial_clusters

        
    def get_features_clusters(self, data:numpy.ndarray, target_size:int)-> numpy.ndarray:
        """
        Performs hierarchical clustering on features and returns the clusters.
        Args:
            data: Data matrix with features in columns.
            target_size: Desired number of features in each cluster.
        Returns:
            Array indicating cluster membership for each feature.
        """
        if data.shape[1] < target_size:
            raise ValueError("Number of target_size must be less than or equal the number of features in the data")
        if data.shape[1] == target_size or target_size <= 1:
            return np.zeros(data.shape[1])
        correlation_matrix = np.corrcoef(data, rowvar=False)
        Z = self.hierarchical_clustering(np.abs(correlation_matrix))
        Z = np.where(Z < 0, 0, Z)
        m = data.shape[1]
        clusters = self.equal_size_clustering(Z, m, target_size)
        return clusters


    def get_features_sets(self, data:numpy.ndarray, features_clusters:numpy.ndarray) -> list:
        """
        Creates sets of features using clusters from hierarchical clustering
        Args:
            data: Data matrix with features in columns
            features_clusters: clusters from hierarchical clustering on features
        Returns:
            sets: list of feature sets
        """
        sets = []
        for cluster in np.unique(features_clusters):
            sets.append(np.ascontiguousarray(data[:, features_clusters == cluster]))
        return sets
    

    def reorder_features(self, extracted_features:numpy.ndarray, features_clusters:numpy.ndarray)-> numpy.ndarray:
        """
        Reorder features with features of highest correlation next to each other
        Args:
            extracted_features: array of extracted feature tensors
            features_clusters: clusters from hierarchical clustering on features
        Returns:
            reordered_feat_mat: reordered features matrix
        """
        feature_sets = self.get_features_sets(extracted_features, features_clusters)
        reordered_feat_mat = np.concatenate(feature_sets, axis=1)
        return reordered_feat_mat


    def calc_eigen_in_sets(self, neighbor_features_mat:numpy.ndarray, S:int, L:int) ->tuple:
        """
        Calculate the eigenvectors and values set-wise
        Args:
            neighbor_features_mat: matrix of neighbor features to calculate eigenvectors/values from
            S: # sets
            L: # features per set
        Returns:
            (all_eigenvals, all_eigenvects): tuple of lists of calculated eigenvalues and eigenvectors
        """
        set_matrices = [neighbor_features_mat[:, i*L:(i+1)*L] for i in range(S)] # create sets of features
        all_eigenvals = list()
        all_eigenvects = list()
        for submatrix in set_matrices:
            cov_matrix = np.cov(submatrix, rowvar=False)
            inverse_covariance_matrix = np.linalg.inv(cov_matrix)
            try:
                eigenvalues, eigenvectors = LA.eig(inverse_covariance_matrix)  # normalized eigenvectors (each column)
                eigenvectors = eigenvectors.T # make it each row
            except:
                print("Eigenvector calc exception")
                # If the covariance matrix is singular, use an identity matrix
                eigenvalues = np.ones(cov_matrix.shape[0])
                eigenvectors = np.eye(cov_matrix.shape[0])
            all_eigenvals.append(eigenvalues)
            all_eigenvects.append(eigenvectors)
        return (all_eigenvals, all_eigenvects)


    def train(self)-> None:
        """
        Trains the K-NNN model in the following steps:
            - extract image features
            - cluster features
            - reorder features
            - calculate eigenvalues/vectors
        All of the computed values are stored in memory. If any of the values were previously computed and stored, they
        are read from memory and not recomputed. 
        Args:
            None
        Returns:
            None
        """
        parent_path = f'{self.output_root_path}/{self.dataset_name}/{self.embedding_model}_{self.img_prep_type}_{self.transform_height}_{self.transform_width}_{self.interpolationmode}'
        features_clusters_name = f'feature_clusters_{self.set_size}'
        reordered_features_name = f'reordered_{self.set_size}'
        eigen_mem_name = f'eigen_{self.set_size}_{self.nn}'
        # only extract features if they haven't been extracted and stored before
        if not self.file_handler.pkl_exists(f'{parent_path}/train'):
            extracted_features = self.extract_features(self.train_data_path)
            self.file_handler.dump_data(extracted_features, parent_path, 'train')
        else: # read extracted features from mem
            extracted_features = self.file_handler.load_data(f'{parent_path}/train')

        # only get feature clusters if they haven't been computed and stored before
        if not self.file_handler.pkl_exists(f'{parent_path}/{features_clusters_name}'):
            self.features_clusters = self.get_features_clusters(extracted_features, self.set_size)
            self.file_handler.dump_data(self.features_clusters, parent_path, features_clusters_name)
        else: # read feature clusters from mem
            self.features_clusters = self.file_handler.load_data(f'{parent_path}/{features_clusters_name}')
        
        # only reorder feature if they haven't been reordered and stored before
        if not self.file_handler.pkl_exists(f'{parent_path}/{reordered_features_name}'):
            self.reordered_features = self.reorder_features(extracted_features, self.features_clusters)
            self.file_handler.dump_data(self.reordered_features, parent_path, reordered_features_name)
        else:
            self.reordered_features = self.file_handler.load_data(f'{parent_path}/{reordered_features_name}')

        # only fit if eigen mem not computed and stored before
        if not self.file_handler.pkl_exists(f'{parent_path}/{eigen_mem_name}'):
            neigh = NearestNeighbors(n_neighbors=self.nn+1, metric="cosine")
            neigh.fit(self.reordered_features)
            N = self.feat_dim # dim of test point feature f (same for all points)
            S = N//self.set_size # divide into S sets
            L = self.set_size # dim of subfeature vectors. More samples used to calculate the Eigen vectors = larger L
            self.eigen_memory = dict()

            for i, f in tqdm(enumerate(self.reordered_features), total=len(self.reordered_features), desc='Training'):
                neighbor_indices = neigh.kneighbors([f], return_distance=False)[0]
                neighbor_indices = np.delete(neighbor_indices, np.argwhere(neighbor_indices==i)) 
                neighbor_features_mat = self.reordered_features[neighbor_indices]
                self.eigen_memory[i] = self.calc_eigen_in_sets(neighbor_features_mat, S, L)

            self.file_handler.dump_data(self.eigen_memory, parent_path, eigen_mem_name)

        else:
            self.eigen_memory = self.file_handler.load_data(f'{parent_path}/{eigen_mem_name}')


    def make_ds(self, good_feats:numpy.ndarray, ungood_feats:numpy.ndarray) -> tuple:
        """
        Stacks the passed good and ungood features to create a testing dataset.
        Also creates labels for the dataset. 0 = good class. 1 = ungood class
        Args:
            good_feats: array of good features
            ungood_feats: array of ungood features
        Returns:
            (test_features, test_labels): tuple of created dataset and corresponding labels
        """
        test_features = np.vstack((good_feats, ungood_feats))
        zeros = np.zeros(good_feats.shape[0])
        ones = np.ones(ungood_feats.shape[0])
        test_labels = np.concatenate((zeros, ones))
        return (test_features, test_labels)


    def calc_anomaly_score(self, f:torch.Tensor, neighbor_features_mat:numpy.ndarray, eigen_mem:dict, S:int, L:int, k:int)-> float:
        """
        Calculates the anomaly score for each feature as described in:
        'k-NNN: Nearest Neighbors of Neighbors for Anomaly Detection'
        (https://doi.org/10.48550/arXiv.2305.17695)
        Args:
            f: feature tensor 
            neighbor_features_mat: matrix of neighboring features of f
            eigen_mem: precomputed eigenvalues/vectors associated with neighboring features
            S: # sets
            L: # features per set
            k: # neighbors
        Returns:
            score: calculated anomaly score
        """
        neigh_sets = neighbor_features_mat.reshape(neighbor_features_mat.shape[0], S, L)
        f_sets = [f[i*L:(i+1)*L] for i in range(S)] # create sets of features
        score = 0
        for i in range(k):
            for j in range(L):
                for s in range(S):
                    a = np.abs(np.dot(f_sets[s] - neigh_sets[i][s], eigen_mem[i][1][s][j]))
                    assert eigen_mem[i][0][s][j] > 0, f'Negative or zero eigenvalue! {eigen_mem[i][0][s][j]}'
                    b = 1/np.emath.sqrt(eigen_mem[i][0][s][j])
                    score += np.dot(a,b)
        return score


    def test(self)->float:
        """
        Tests the K-NNN model in the following steps:
            - extract features from test images
            - create test dataset from extracted features
            - create feature sets
            - calculate anomaly scores for each image in the dataset
            - compute and return roc_auc_score
        Args:
            None
        Returns:
            roc_auc_score: ROC-AUC score
        """
        if self.features_clusters is None or self.reordered_features is None or self.eigen_memory is None:
            print('Have to load required data from memory. Training first...')
            self.train()
        parent_path = f'{self.output_root_path}/{self.dataset_name}/{self.embedding_model}_{self.img_prep_type}_{self.transform_height}_{self.transform_width}_{self.interpolationmode}'
        N = self.feat_dim # dim of test point feature f (same for all points)
        S = N//self.set_size # divide into S sets
        L = self.set_size # dim of subfeature vectors. So more samples used to calculate the Eigen vectors = larger L

        # only extract features if they haven't been extracted and stored before
        if not self.file_handler.pkl_exists(f'{parent_path}/test_good'):
            test_good_features = self.extract_features(self.test_data_path_good)
            self.file_handler.dump_data(test_good_features, parent_path, 'test_good')
        else: # read extracted features from mem
            test_good_features = self.file_handler.load_data(f'{parent_path}/test_good')
        
        # only extract features if they haven't been extracted and stored before
        if not self.file_handler.pkl_exists(f'{parent_path}/test_ungood'):
            test_ungood_features = self.extract_features(self.test_data_path_ungood)
            self.file_handler.dump_data(test_ungood_features, parent_path, 'test_ungood')
        else: # read extracted features from mem
            test_ungood_features = self.file_handler.load_data(f'{parent_path}/test_ungood')
        (test_features, test_labels) = self.make_ds(test_good_features, test_ungood_features)
        feature_sets = self.get_features_sets(test_features, self.features_clusters)
        test_features = np.concatenate(feature_sets, axis=1)

        neigh = NearestNeighbors(n_neighbors=self.n, metric="cosine")
        neigh.fit(self.reordered_features)
        scores = list()

        for f in tqdm(test_features, total=len(test_features), desc='Testing'):
            neighbor_indices = neigh.kneighbors([f], return_distance=False)[0]
            neighbor_features_mat = self.reordered_features[neighbor_indices] 
            req_eigen_mem = {i: self.eigen_memory[key] for i, key in enumerate(neighbor_indices)}
            score = self.calc_anomaly_score(f, neighbor_features_mat, req_eigen_mem, S, L, self.n)
            scores.append(score)

        roc_score = roc_auc_score(test_labels, scores)
        fpr, tpr, thresholds = roc_curve(test_labels, scores, pos_label=0)
        print(f'k:{self.n}, N: {N}, S:{S}, L:{L}, roc_auc_score: {roc_score}')
        return roc_score
