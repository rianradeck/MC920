import cv2 as cv
import sys
import numpy as np

def read_grayscale(s):
    img = cv.imread(s, cv.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit("Could not read the image.")
    return img

def read_colorscale(s):
    img = cv.imread(s, cv.IMREAD_COLOR)
    if img is None:
        sys.exit("Could not read the image.")
    return img

def get_edges(height, width, i, j, grid_height, grid_width):
    return i, min(height, i + grid_height), max(0, j - grid_width // 2), min(width , j + grid_width // 2 + 1)

def halftoning(img_, grid):
    img = img_.copy()
    img = np.full((img_.shape[0] + 10, img_.shape[1] + 10), 0, img.dtype)
    img[5:img_.shape[0] + 5, 5:img_.shape[1] + 5] = img_
    a = np.full(img.shape, 0, img.dtype)
    
    height, width = a.shape
    grid_height, grid_width = grid.shape
    fliped_grid = grid[::, ::-1]

    for i in range(5, height):
        for j_ in range(5, width):
            j = j_ if height % 2 == 1 else width - j_ - 1
            i_st, i_nd, j_st, j_nd = get_edges(height, width, i, j, grid_height, grid_width)

            a[i][j] = 0 if img[i][j] < 128 else 255
            error = float(img[i][j]) - float(a[i][j])

            try:
                img[i_st:i_nd, j_st:j_nd] = (img[i_st:i_nd, j_st:j_nd] + error * (grid if height % 2 == 1 else fliped_grid))
                # for k in range(grid.shape[0]):
                #     for l in range(grid.shape[1]):
                #         ll = l - ((grid.shape[1] + 1) // 2)
                #         if i + k < img.shape[0] and 0 <= j + ll and j + ll < img.shape[1]:
                #             if i % 2 == 0:
                #                 img[i + k, j + ll] += error * grid[k, l]
                #             else:
                #                 img[i + k, j + ll] += error * grid[::,::-1][k, l]
            except:
                None

    cv.imshow("b.png", img)
    return a.astype(img.dtype)

img = read_grayscale("images/baboon.pgm")
# img = np.full((512, 512), 255, img.dtype)
G = np.array([[0, 0, 7/16],
              [3/16, 5/16, 1/16],])
# G = np.array([[0.5]])
cv.imshow("half_shade.png", halftoning(img, G))
cv.imshow("original.png", img)

k = cv.waitKey()