import images
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

class VisibilityDetection:
    def __init__(self, path:str, img_height=1024, img_width=1024, a0=1, a1=1) -> None:
        self.images = images.Images(path)
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.scaled_coords_x = {i:(i-(self.IMG_HEIGHT/2))/self.IMG_HEIGHT for i in range(max(self.IMG_WIDTH, self.IMG_HEIGHT))}
        self.scaled_coords_y = {i:(i-(self.IMG_WIDTH/2))/self.IMG_WIDTH for i in range(max(self.IMG_WIDTH, self.IMG_HEIGHT))}
        self.a0 = a0 # When a0 is high, even the deepest valleys (or areas of low intensity) remain far from the viewpoint.
        self.a1 = a1 # When setting a1 to high values, the transformation will result in significant valleys and ridges even when the pixels that created them had similar intensity values.
        self.run()
        

    def vis_3d(self, r_list, theta_list, phi_list, intensity_list):
        # Sample data: Polar coordinates and grayscale values
        r = np.array(r_list)  # Radius (fixed for unit sphere)
        theta = np.array(theta_list)  # Polar angles (in radians)
        phi = np.array(phi_list)    # Azimuthal angles (in radians)
        gray_values = np.array(intensity_list)  # Grayscale values (0 to 255)

        # Convert spherical coordinates to Cartesian coordinates for plotting
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        # Normalize grayscale values (for coloring)
        gray_norm = gray_values / 255
        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot in 3D with grayscale coloring
        sc = ax.scatter(x, y, z, c=gray_norm, cmap='gray', s=100)

        # Add a colorbar to show the grayscale intensity
        fig.colorbar(sc, ax=ax, label="Grayscale Intensity")

        plt.show()


    def cartesian_to_spherical(self, x, y, z, i):
        r = math.sqrt((x**2) + (y**2) + z)
        theta = math.acos(z/r) # radians
        phi = math.atan2(y, x) # radians
        r = (self.a1 * i) + self.a0
        return (r, theta, phi)

    def spherical_to_cartesian(self, r, theta, phi):
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.sin(theta) * math.sin(phi)
        z = r * math.cos(theta)
        return (x, y, z)

    def HPR_transform_points(self, points, gamma):
        points = list(points)
        points_norm = np.linalg.norm(points)
        transformed_points = (points/points_norm) * (points_norm**gamma)
        return transformed_points

    def HPR_convex_hull(self, transformed_points):
        hull=ConvexHull(transformed_points)
        return hull.vertices

    def view_visible_img(self, img, vertices):
        row_indices, col_indices = np.unravel_index(vertices, img.data.shape)
        # highlighted_image = np.full_like(img.data, 0)
        # highlighted_image[row_indices, col_indices] = 255
        # highlighted_image = np.copy(img.data)
        # highlighted_image[row_indices, col_indices] = 255
        highlighted_image = np.full_like(img.data, 255)
        highlighted_image[row_indices, col_indices] = img.data[row_indices, col_indices]

        cv2.imshow("original_img", img.data)
        cv2.waitKey(0)
        cv2.imshow("visibile_img", highlighted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def scale_pixels(self, img:images.Image) -> images.Image:
        z_default = 1
        gamma = -0.001
        r_list = list()
        theta_list = list()
        phi_list = list()
        intensity_list = list()
        all_transformed_points = list()
        for i in range(self.IMG_HEIGHT): #x
            for j in range(self.IMG_WIDTH): #y
                # print(self.scaled_coords_x[i], self.scaled_coords_y[j], z, img.data[i][j])
                (r, theta, phi) = self.cartesian_to_spherical(self.scaled_coords_x[i], self.scaled_coords_y[j], z_default, img.data[i][j])
                (x, y, z) = self.spherical_to_cartesian(r, theta, phi)
                all_transformed_points.append(self.HPR_transform_points((x, y, z), gamma))
                r_list.append(r)
                theta_list.append(theta)
                phi_list.append(phi)
                intensity_list.append(img.data[i][j])
        # self.vis_3d(r_list[:100000], theta_list[:100000], phi_list[:100000], intensity_list[:100000]) # subsampled vis for large images
        visible_vertices = self.HPR_convex_hull(all_transformed_points)
        # 3d vis for visibile points
        self.vis_3d(np.take(r_list, visible_vertices), np.take(theta_list, visible_vertices), np.take(phi_list, visible_vertices), np.take(intensity_list, visible_vertices))
        # 3d vis for all points
        self.vis_3d(r_list, theta_list, phi_list, intensity_list)
        self.view_visible_img(img, visible_vertices)


    def run(self):
        for image in self.images.images:
            print(image.path, image.data)
            img = cv2.imread(image.path, cv2.IMREAD_GRAYSCALE)
            image.data = img
            self.scale_pixels(image)
            break