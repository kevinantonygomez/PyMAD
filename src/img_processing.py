import images
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.spatial as sp
from scipy.interpolate import griddata
import images
import sys

class VisibilityDetection:
    def __init__(self, path:str, img_height=1024, img_width=1024, a0=1, a1=1, save=None) -> None:
        self.hist_eq = HistogramEqualization()
        self.images = images.Images(path, save)
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.scaled_coords_x = {i:(i-(self.IMG_HEIGHT/2))/max(self.IMG_WIDTH, self.IMG_HEIGHT) for i in range(self.IMG_HEIGHT)}
        self.scaled_coords_y = {i:(i-(self.IMG_WIDTH/2))/max(self.IMG_WIDTH, self.IMG_HEIGHT) for i in range(self.IMG_WIDTH)}
        assert min(self.scaled_coords_x.values()) >= -0.5, f'Coordinate Scaling error! Got {min(self.scaled_coords_x.values())}'
        assert min(self.scaled_coords_y.values()) >= -0.5, f'Coordinate Scaling error! Got {min(self.scaled_coords_y.values())}'
        assert max(self.scaled_coords_x.values()) <= 0.5, f'Coordinate Scaling error! Got {max(self.scaled_coords_x.values())}'
        assert max(self.scaled_coords_y.values()) <= 0.5, f'Coordinate Scaling error! Got {max(self.scaled_coords_y.values())}'
        self.a0 = a0 # When a0 is high, even the deepest valleys (or areas of low intensity) remain far from the viewpoint.
        self.a1 = a1 # When setting a1 to high values, the transformation will result in significant valleys and ridges even when the pixels that created them had similar intensity values.
        

    def vis_3d(self, points, intensity_list, title='Figure'):
        def on_press(event):
            sys.stdout.flush()
            if event.key in ('enter', 'return'):
                plt.close()

        # print(points)
        points_array = np.array(points)
        x, y, z = points_array[:, 0], points_array[:, 1], points_array[:, 2]
        # Normalize grayscale values (for coloring)
        gray_norm = np.array(intensity_list) / 255
        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        # Scatter plot in 3D with grayscale coloring
        sc = ax.scatter(x, y, z, c=gray_norm, cmap='gray', s=1)

        # Add a colorbar to show the grayscale intensity
        fig.colorbar(sc, ax=ax, label="Grayscale Intensity")
        fig.canvas.mpl_connect('key_press_event', on_press)
        return plt
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()


    def cartesian_to_spherical(self, x, y, z, i):
        theta = math.atan2(y, x) # azimuth angle (radians)
        phi = math.atan2(math.sqrt(x**2 + y**2), z) # polar angle (radians)
        r = (self.a1 * i) + self.a0
        return (r, theta, phi)

    def spherical_to_cartesian(self, r, theta, phi):
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return (x, y, z) 

    def transform_points(self, points, gamma):
        points = list(points)
        points_norm = np.linalg.norm(points)
        transformed_points = np.dot((points/points_norm),(points_norm**gamma))
        return transformed_points

    def convex_hull(self, transformed_points):
        return sp.ConvexHull(transformed_points)


    def view_visible_img(self, img:images.Image, vertices):
        # indices_to_set = np.setdiff1d(np.arange(img.data.size), vertices) # get points NOT visible
        row_indices, col_indices = np.unravel_index(vertices, img.data.shape)
        highlighted_image = cv2.cvtColor(img.data, cv2.COLOR_GRAY2RGB)
        # highlighted_image[row_indices, col_indices] = img.data[row_indices, col_indices]
        highlighted_image[row_indices, col_indices] = [0, 0, 255]
        return highlighted_image

    def gen_smoothed_img(self, img:images.Image, vertices):
        # indices_to_set = np.setdiff1d(np.arange(img.data.size), vertices) # get points NOT visible
        row_indices, col_indices = np.unravel_index(vertices, img.data.shape)
        values = img.data[row_indices, col_indices]
        points = np.column_stack((row_indices, col_indices))
        x = np.arange(0, img.data.shape[1])
        y = np.arange(0, img.data.shape[0])
        grid_x, grid_y = np.meshgrid(x, y)
        # print(len(points))
        # print(len(values))
        # print(grid_x)
        # print(grid_y.shape)
        # print(len((grid_y, grid_x)))
        smoothed_image = griddata(points, values, (grid_y, grid_x), method='cubic')
        # print("smoothed_image")
        # print(smoothed_image)
        # smoothed_image = np.clip(smoothed_image, 0, 255)  # Ensure values are within valid range
        # print("smoothed_image")
        # print(smoothed_image)
        # smoothed_image = smoothed_image.astype(np.uint8)   # Convert to 8-bit grayscale format
        # print("smoothed_image")
        # print(smoothed_image)
        return smoothed_image

    def test(self, image):
        for i in range(0, 256):
            smooth = np.full_like(image, i)
            final_img = smooth - image
            # final_img = (final_img - np.min(final_img)) / (np.max(final_img) - np.min(final_img))
            # final_img = (final_img * 255).astype(np.uint8)
            cv2.imshow(f"final_img_{i}", final_img)
            cv2.waitKey(0)

    def final_img(self, img:images.Image, smoothed_image, visibile_img):
        # cv2.imshow("original_img_eq", self.hist_eq.equalize(img.data))
        # cv2.waitKey(0)
        # cv2.imshow("original_img", img.data)
        # cv2.waitKey(0)
        # cv2.imshow("visibile_img", visibile_img)
        # cv2.waitKey(0)
        # cv2.imshow("smoothed_image", smoothed_image)
        # cv2.waitKey(0)
        final_img = smoothed_image - img.data
        # print("smoothed_image")
        # print(smoothed_image[:10])

        # print("img.data")
        # print(img.data[:10])

        # print("final_img")
        # print(final_img[:10])
        # cv2.imshow("final_img_1", final_img)
        # cv2.waitKey(0)

        final_img = (final_img - np.min(final_img)) / (np.max(final_img) - np.min(final_img))
        # print("final_img")
        # print(final_img[:10])
        # cv2.imshow("final_img", final_img)
        # cv2.waitKey(0)
        final_img = (final_img * 255).astype(np.uint8)
        # cv2.imshow("final_img_255", final_img)
        # cv2.waitKey(0)
        # print("final_img")
        # print(final_img[:10])
        # print(np.array_equal(x, final_img))
        # cv2.imshow("final_img_1", final_img)
        # cv2.waitKey(0)

        # self.test(img.data)
        final_img = 255 - final_img
        imaxxxge = self.hist_eq.equalize(final_img)
        img.data = imaxxxge
        # cv2.imshow("final_img_eq", imaxxxge)
        # cv2.waitKey(0)

        # final_img1 = img.data - smoothed_image
        # final_img1 = (final_img1 - np.min(final_img1)) / (np.max(final_img1) - np.min(final_img1))
        # cv2.imshow("final_img1", final_img1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        

    def algorithm(self, img:images.Image) -> images.Image:
        z_default = 1
        # gamma = -0.008
        # gamma = -0.00004
        hpr_gamma = -0.0005 # < 0 higher sees more
        # tpo_gamma = 0.00005 # > 0 lower sees more
        tpo_gamma = 0.05 # > 0 lower sees more
        # tpo_gamma = 0.00001 # > 0 lower sees more writing
        print(hpr_gamma, tpo_gamma)
        r_list = list()
        theta_list = list()
        phi_list = list()
        intensity_list = list()
        hpr_transformed_points = list()
        tpo_transformed_points = list()
        hpr_tpo_transformed_points = list()
        for i in range(self.IMG_HEIGHT): #x
            for j in range(self.IMG_WIDTH): #y
                # print(self.scaled_coords_x[i], self.scaled_coords_y[j], z, img.data[i][j])
                (r, theta, phi) = self.cartesian_to_spherical(self.scaled_coords_x[i], self.scaled_coords_y[j], z_default, img.data[i][j])
                (x, y, z) = self.spherical_to_cartesian(r, theta, phi)
                hpr_transformed_points.append(self.transform_points((x, y, z), hpr_gamma))
                tpo_transformed_points.append(self.transform_points((x, y, z), tpo_gamma))
                r_list.append(r)
                theta_list.append(theta)
                phi_list.append(phi)
                intensity_list.append(img.data[i][j])
     
        # hpr_hull = self.convex_hull(hpr_transformed_points)
        tpo_hull = self.convex_hull(tpo_transformed_points)
        # hpr_tpo_transformed_points.extend(hpr_transformed_points)
        # hpr_tpo_transformed_points.extend(tpo_transformed_points)
        # hpr_tpo_hull = self.convex_hull(hpr_tpo_transformed_points)

        # '''3d vis for HPR'''
        # plt = self.vis_3d(hpr_transformed_points, intensity_list, title='HPR ALL')
        # plt.show(block=False)
        # self.vis_3d(np.take(hpr_transformed_points, hpr_hull.vertices, axis=0), np.take(intensity_list, hpr_hull.vertices, axis=0), title='HPR VIS')
        # plt.show(block=False)

        '''3d vis for TPO'''
        # plt = self.vis_3d(tpo_transformed_points, intensity_list, title='TPO ALL')
        # plt.show(block=False)
        # self.vis_3d(np.take(tpo_transformed_points, tpo_hull.vertices, axis=0), np.take(intensity_list, tpo_hull.vertices, axis=0), title='TPO VIS')
        # plt.show()

        # '''3d vis for HPR + TPO'''
        # plt = self.vis_3d(hpr_tpo_transformed_points, intensity_list, title='HPR + TPO ALL')
        # plt.show(block=False)
        # self.vis_3d(np.take(hpr_tpo_transformed_points, hpr_tpo_hull.vertices, axis=0), np.take(intensity_list, hpr_tpo_hull.vertices, axis=0), title='HPR + TPO VIS')
        # plt.show()

        # visible_img = self.view_visible_img(img, hpr_hull.vertices)
        # smoothed_image = self.gen_smoothed_img(img, hpr_hull.vertices)
        # self.final_img(img, smoothed_image, visible_img)

        visible_img = self.view_visible_img(img, tpo_hull.vertices)
        smoothed_image = self.gen_smoothed_img(img, tpo_hull.vertices)
        self.final_img(img, smoothed_image, visible_img)

        # visible_img = self.view_visible_img(img, hpr_tpo_hull.vertices)
        # smoothed_image = self.gen_smoothed_img(img, hpr_tpo_hull.vertices)
        # self.final_img(img, smoothed_image, visible_img)

    def resize_image(self, img):
        h,w = img.shape
        if self.IMG_HEIGHT != h and self.IMG_WIDTH != w:
            img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
        return img

    def run(self):
        for image in self.images.images:
            print(image.path)
            original_img = cv2.imread(image.path, cv2.IMREAD_GRAYSCALE)
            image.data = self.hist_eq.equalize(self.resize_image(original_img))
            # image.data = cv2.imread(image.path, cv2.IMREAD_GRAYSCALE)
            self.algorithm(image)
            image.save()
            # break


# HISTOGRAM EQUALIZATION
class HistogramEqualization:
    def __init__(self):
        pass

    def equalize(self, image):
        self.equalized_image = cv2.equalizeHist(image)
        return self.equalized_image
        
    # def display_images(self):
    #     """
    #     Display the original and equalized images side by side.
    #     """
    #     if self.image is None or self.equalized_image is None:
    #         raise ValueError("Either the original or equalized image is missing.")
        
    #     plt.figure(figsize=(10, 5))
        
    #     # Original Image
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(self.image, cmap='gray')
    #     plt.title('Original Image')
    #     #plt.axis('off')
        
    #     # Equalized Image
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(self.equalized_image, cmap='gray')
    #     plt.title('Histogram Equalized Image')
    #     #plt.axis('off')
        
    #     plt.show()