import cv2 as cv
import sys
import numpy as np

def read_grayscale(s):
    img = cv.imread(s, cv.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit("Could not read the image.")
    return img

def negative_filter(img):
    return 255 - img

def interval_filter(img, lb, ub):
    ans = img.copy()
    conversion_map = np.rint(np.linspace(lb, ub, 256))
    ans = conversion_map[ans]
    return ans.astype(img.dtype)

def evenflip_filter(img):
    ans = img.copy()
    ans[::2] = ans[::2,::-1]
    return ans

def uppermirror_filter(img):
    ans = img.copy()
    ans[len(ans) // 2::] = ans[:len(ans) // 2:][::-1]
    return ans

def verticalflip_filter(img):
    return img[::-1]

def gamma_filter(img, x):
    ans = img.copy()
    conversion_map = np.linspace(0, 1, 256)
    
    ans = conversion_map[ans]
    ans = ans ** (1 / x)
    ans = np.rint(ans * 255)
    
    return ans.astype(img.dtype)

def bitmask_filter(img, bit):
    ans = img.copy()
    return np.where((ans & int(2 ** bit)), 255, 0).astype(img.dtype)

def mosaic_filter(img, x, y, grid):
    mosaic_parts = []
    part_heigth = len(img) / y
    part_width = len(img[0]) / x
    ans = img.copy()
    for i in range(y):
        for j in range(x):
            mosaic_parts.append(img[int(i * part_heigth):int((i + 1) * part_heigth):,int(j * part_width):int((j + 1) * part_width):])
    for i in range(y):
        for j in range(x):
            ans[int(i * part_heigth):int((i + 1) * part_heigth):,int(j * part_width):int((j + 1) * part_width):] = mosaic_parts[grid[i][j] - 1]
    return ans.astype(img.dtype)


def merge(img, img_, p):
    q = 1 - p
    a = img * p
    b = img_ * q

    return np.rint(a + b).astype(img.dtype)

def custom_filter(img, frac, grid):
    offset = len(grid) // 2
    img_ = img.copy()

    for i in range(offset, len(img) - offset):
        for j in range(offset, len(img) - offset):
            new_val = (img[i - offset:i + offset + 1, j - offset: j + offset + 1] * grid).sum() // frac
            img_[i][j] = (new_val if (new_val < 256 and new_val >= 0) else (0 if new_val < 0 else 255))

    return img_

def intesity_transforming():
    a = read_grayscale("city.png")
    b = negative_filter(a)
    c = interval_filter(a, 100, 200)
    d = evenflip_filter(a)
    e = uppermirror_filter(a)
    f = verticalflip_filter(a)

    cv.imshow("a.png", a)
    cv.imshow("b.png", b)
    cv.imshow("c.png", c)
    cv.imshow("d.png", d)
    cv.imshow("e.png", e)
    cv.imshow("f.png", f)

def brightness_adjustment():
    a = read_grayscale("baboon.png")
    b = gamma_filter(a, 1.5)
    c = gamma_filter(a, 2.5)
    d = gamma_filter(a, 3.5)

    cv.imshow("a.png", a)
    cv.imshow("b.png", b)
    cv.imshow("c.png", c)
    cv.imshow("d.png", d)

def bit_planes():
    a = read_grayscale("baboon.png")
    b = bitmask_filter(a, 0)
    c = bitmask_filter(a, 4)
    d = bitmask_filter(a, 7)

    cv.imshow("a.png", a)
    cv.imshow("b.png", b)
    cv.imshow("c.png", c)
    cv.imshow("d.png", d)

def mosaic():
    a = read_grayscale("baboon.png")
    d = mosaic_filter(a, 4, 4, [[6, 11, 13, 3], [8, 16, 1, 9], [12, 14, 2, 7], [4, 15, 10, 5]])
    cv.imshow("mosaic.png", d)

def image_merging():
    a = read_grayscale("baboon.png")
    b = read_grayscale("butterfly.png")
    c = merge(a, b, 0.2)
    d = merge(a, b, 0.5)
    e = merge(a, b, 0.8)

    cv.imshow("c.png", c)
    cv.imshow("d.png", d)
    cv.imshow("e.png", e)

def image_filtering():
    img = read_grayscale("baboon.png")
    h1 = custom_filter(img, 1, [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
    h2 = custom_filter(img, 256, [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
    h3 = custom_filter(img, 1, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    h4 = custom_filter(img, 1, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    h5 = custom_filter(img, 1, [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    h6 = custom_filter(img, 9, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    h7 = custom_filter(img, 1, [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
    h8 = custom_filter(img, 1, [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    h9 = custom_filter(img, 9, [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]])
    h10 = custom_filter(img, 8, [[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]])
    h11 = custom_filter(img, 1, [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
    h3h4 = np.rint(np.sqrt(h3.astype(float) ** 2 + h4.astype(float) ** 2)).astype(img.dtype)

    cv.imshow("h1.png", h1)
    cv.imshow("h2.png", h2)
    cv.imshow("h3.png", h3)
    cv.imshow("h4.png", h4)
    cv.imshow("h5.png", h5)
    cv.imshow("h6.png", h6)
    cv.imshow("h7.png", h7)
    cv.imshow("h8.png", h8)
    cv.imshow("h9.png", h9)
    cv.imshow("h10.png", h10)
    cv.imshow("h11.png", h11)
    cv.imshow("h3h4.png", h3h4)

print("Choose what part of the work do you want to see the images - 1.1, 1.2, 1.3, 1.4, 1.5 or 1.6")
n = input()

if n == "1.1":
    intesity_transforming()
if n == "1.2":
    brightness_adjustment()
if n == "1.3":
    bit_planes()
if n == "1.4":
    mosaic()
if n == "1.5":
    image_merging()
if n == "1.6":
    image_filtering()

print("Press any key to exit!")
k = cv.waitKey(0)