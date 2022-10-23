import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
import collections

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
    a = np.full(img.shape, 0, img.dtype)
    
    channels = 1
    try:
        height, width, channels = a.shape
    except:
        height, width = a.shape
        a = a.reshape(height, width, 1)
        img = img.reshape((height, width, 1))
    
    grid_height, grid_width = grid.shape
    fliped_grid = grid[::, ::-1]

    for k in range(channels):
        for i in range(height):
            for j_ in range(width):
                j = j_ if height % 2 == 0 else width - j_ - 1
                G = grid if height % 2 == 0 else fliped_grid
                
                i_st, i_nd, j_st, j_nd = get_edges(height, width, i, j, grid_height, grid_width)

                a[i][j][k] = 0 if img[i][j][k] < 128 else 255
                error = float(img[i][j][k]) - float(a[i][j][k])

                if i + grid_height - 1 < height and j + grid_width // 2 < width and j - grid_width // 2 >= 0:
                    img[i_st:i_nd, j_st:j_nd, k] = (img[i_st:i_nd, j_st:j_nd, k] + error * (grid if height % 2 == 0 else fliped_grid))

    return a.astype(img.dtype)

def distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

# CHANGE THIS TO APPLY TO ANOTHER IMAGE
# img = read_grayscale("images/baboon.pgm")
img = read_colorscale("images/baboon.png")

floyd_steinberg = np.array([[0, 0, 7/16],
                            [3/16, 5/16, 1/16]])
stevenson_arce = np.array([[0, 0, 0, 0, 0, 32/200, 0],
                           [12/200, 0, 26/200, 0, 30/200, 0, 16/200],
                           [0, 12/200, 0, 26/200, 0, 12/200, 0],
                           [5/200, 0, 12/200, 0, 12/200, 0, 5/200]])
burkes = np.array([[0, 0, 0, 8/32, 4/32],
                   [2/32, 4/32, 8/32, 4/32, 2/32]])
sierra = np.array([[0, 0, 0, 5/32, 3/32],
                   [2/32, 4/32, 5/32, 4/32, 2/32],
                   [0, 2/32, 3/32, 2/32, 0]])
stucki = np.array([[0, 0, 0, 8/42, 4/42],
                   [2/42, 4/42, 8/42, 4/42, 2/42],
                   [1/42, 2/42, 4/42, 2/42, 1/42]])
jarvis_judice_ninke = np.array([[0, 0, 0, 7/48, 5/48],
                                [3/48, 5/48, 7/48, 5/48, 3/48],
                                [1/48, 3/48, 5/48, 3/48, 1/48]])
# G = np.array([[0.5]])
cv.imshow("original.png", img)
cv.imshow("floyd_steinberg.png", halftoning(img, floyd_steinberg))
cv.imshow("burkes.png", halftoning(img, burkes))
cv.imshow("sierra.png", halftoning(img, sierra))
cv.imshow("stucki.png", halftoning(img, stucki))
cv.imshow("jarvis_judice_ninke.png", halftoning(img, jarvis_judice_ninke))

def get_image(complex_input):
    return np.log(1 + np.abs(complex_input))

def get_centered_spectrum(img):
    spectrum = np.fft.fft2(img)
    centered_spectrum = np.fft.fftshift(spectrum)
    return centered_spectrum

def get_processed_from_centralized(spectrum):
    decentralized_spectrum = np.fft.ifftshift(spectrum)
    final_img = np.fft.ifft2(decentralized_spectrum)
    return np.abs(final_img)

# not working properly
def get_compressed_image(img_):
    img = img_.copy()
    spec = get_centered_spectrum(img)
    rows, cols = spec.shape
    d = {}
    for i in range(rows):
        for j in range(cols):
            d[(i, j)] = spec[i][j]
    od = dict(sorted(d.items()))
    sz = len(od)
    cnt = 0
    for k in od:
        v = d[k]
        if cnt < 0.9 * sz:
            # print(k, spec[k[0]][k[1]])
            spec[k[0]][k[1]] = complex(spec[k[0]][k[1]].real)
        cnt += 1
    plt.imshow(get_image(np.abs(spec)), "gray"), plt.title("AA")
    plt.show()
    return get_processed_from_centralized(spec)

img = read_grayscale("images/baboon.pgm")

rows, cols = img.shape
crow,ccol = rows // 2, cols // 2

low_pass = np.full((rows, cols), 0, np.uint8)
high_pass = np.full((rows, cols), 1, np.uint8)
strip_pass = np.full((rows, cols), 0, np.uint8)
strip_reject = np.full((rows, cols), 1, np.uint8)

# inner and outer radius of the strip, if filter is not strip kind r1 will be used
r1 = 30
r2 = 60 

center = (rows / 2, cols / 2)
for x in range(cols):
    for y in range(rows):
        dist = distance((y, x), center)
        if dist < r1:
            low_pass[y, x] = 1
            high_pass[y, x] = 0
        elif dist < r2:
            strip_pass[y, x] = 1
            strip_reject[y, x] = 0


# apply filter
cspec = get_centered_spectrum(img)
low_pass_spec = cspec * low_pass
high_pass_spec = cspec * high_pass
strip_pass_spec = cspec * strip_pass
strip_reject_spec = cspec * strip_reject

plt.subplot(131), plt.imshow(img, "gray"), plt.title("Original Image")
plt.subplot(132), plt.imshow(get_image(cspec), "gray"), plt.title("Centered Spectrum")
plt.subplot(133), plt.imshow(get_processed_from_centralized(cspec), "gray"), plt.title("Processed Image")

plt.show()

plt.subplot(141), plt.imshow(get_image(low_pass_spec), "gray"), plt.title("Low Pass Spectrum")
plt.subplot(142), plt.imshow(get_image(high_pass_spec), "gray"), plt.title("High Pass Spectrum")
plt.subplot(143), plt.imshow(get_image(strip_pass_spec), "gray"), plt.title("Strip Pass Spectrum")
plt.subplot(144), plt.imshow(get_image(strip_reject_spec), "gray"), plt.title("Strip Reject Spectrum")

plt.show()

low_pass_processed = get_processed_from_centralized(low_pass_spec)
high_pass_processed = get_processed_from_centralized(high_pass_spec)
strip_pass_processed = get_processed_from_centralized(strip_pass_spec)
strip_reject_processed = get_processed_from_centralized(strip_reject_spec)

plt.subplot(141), plt.imshow(low_pass_processed, "gray"), plt.title("Processed Low Pass")
plt.subplot(142), plt.imshow(high_pass_processed, "gray"), plt.title("Processed High Pass")
plt.subplot(143), plt.imshow(strip_pass_processed, "gray"), plt.title("Processed Strip Pass")
plt.subplot(144), plt.imshow(strip_reject_processed, "gray"), plt.title("Processed Strip Reject")

plt.show()

low_pass_hist = low_pass_processed.reshape(low_pass_processed.shape[0] * low_pass_processed.shape[1])
plt.subplot(121), plt.hist(low_pass_hist, bins = 128), plt.title("Low Pass")

high_pass_hist = high_pass_processed.reshape(high_pass_processed.shape[0] * high_pass_processed.shape[1])
plt.subplot(122), plt.hist(high_pass_hist, bins = 128), plt.title("High Pass")

plt.show()

strip_pass_hist = strip_pass_processed.reshape(strip_pass_processed.shape[0] * strip_pass_processed.shape[1])
plt.subplot(121), plt.hist(strip_pass_hist, bins = 128), plt.title("Strip Pass")

strip_reject_hist = strip_reject_processed.reshape(strip_reject_processed.shape[0] * strip_reject_processed.shape[1])
plt.subplot(122), plt.hist(strip_reject_hist, bins = 128), plt.title("Strip Reject")

plt.show()

k = cv.waitKey()