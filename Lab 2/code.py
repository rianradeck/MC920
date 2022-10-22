import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def read_grayscale(s):
    img = cv.imread(s, cv.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit("Could not read the image.")
    return img

def get_edges(height, width, i, j, n):
    return max(0, i - n), min(height, i + n + 1), max(0, j - n), min(width , j + n + 1)

def global_method(img, T = 128):
    a = img.copy()

    a[img <= T] = 255
    a[img > T] = 0

    return a

def bernsen_method(img, n = 25):
    n = int(n / 2)
    a = img.copy()

    height = len(a)
    width = len(a[0])

    for i in range(height):
        for j in range(width):
            i_st, i_nd, j_st, j_nd = get_edges(height, width, i, j, n)

            z_min = img[i_st:i_nd, j_st:j_nd].min()
            z_max = img[i_st:i_nd, j_st:j_nd].max()

            threshold = (int(z_min) + int(z_max)) // 2
            a[i][j] = (255 if a[i][j] <= threshold else 0)

    return a.astype(img.dtype)

def niblack_method(img, n = 5, k = 0.2):
    n = int(n / 2)
    a = img.copy()

    height = len(a)
    width = len(a[0])

    for i in range(height):
        for j in range(width):
            i_st, i_nd, j_st, j_nd = get_edges(height, width, i, j, n)

            mu = img[i_st:i_nd, j_st:j_nd].mean()
            sigma = img[i_st:i_nd, j_st:j_nd].std()

            threshold = mu + k * sigma
            a[i][j] = (255 if a[i][j] <= threshold else 0)

    return a.astype(img.dtype)

def sauvola_pietaksinen_method(img, n = 5, k = 0.5, r = 128):
    n = int(n / 2)
    a = img.copy()

    height = len(a)
    width = len(a[0])

    for i in range(height):
        for j in range(width):
            i_st, i_nd, j_st, j_nd = get_edges(height, width, i, j, n)

            mu = img[i_st:i_nd, j_st:j_nd].mean()
            sigma = img[i_st:i_nd, j_st:j_nd].std()

            threshold = mu * (1 + k * (sigma / r - 1))
            a[i][j] = (255 if a[i][j] <= threshold else 0)

    return a.astype(img.dtype)

def phansalskar_more_sabale_method(img, n = 5, k = 0.25, r = 0.5, p = 2, q = 10):
    n = int(n / 2)
    a = img.copy()
    b = img.copy()
    a = a / 255
    b = b / 255

    height = len(a)
    width = len(a[0])

    for i in range(height):
        for j in range(width):
            i_st, i_nd, j_st, j_nd = get_edges(height, width, i, j, n)

            mu = b[i_st:i_nd, j_st:j_nd].mean()
            sigma = b[i_st:i_nd, j_st:j_nd].std()

            threshold = mu * (1 + p * np.exp(-q * mu) + k * (sigma / r - 1))
            a[i][j] = (255 if a[i][j] <= threshold else 0)

    return a.astype(img.dtype)

def contrast_method(img, n = 5):
    n = int(n / 2)
    a = img.copy()

    height = len(a)
    width = len(a[0])

    for i in range(height):
        for j in range(width):
            i_st, i_nd, j_st, j_nd = get_edges(height, width, i, j, n)

            z_min = img[i_st:i_nd, j_st:j_nd].min()
            z_max = img[i_st:i_nd, j_st:j_nd].max()

            a[i][j] = 0 if abs(int(a[i][j]) - int(z_min)) <= abs(int(a[i][j]) - int(z_max)) else 255

    return a.astype(img.dtype)

def mean_method(img, n = 5):
    n = int(n / 2)
    a = img.copy()

    height = len(a)
    width = len(a[0])

    for i in range(height):
        for j in range(width):
            i_st, i_nd, j_st, j_nd = get_edges(height, width, i, j, n)

            mu = img[i_st:i_nd, j_st:j_nd].mean()

            a[i][j] = 0 if int(a[i][j]) > mu else 255

    return a.astype(img.dtype)

def median_method(img, n = 5):
    n = int(n / 2)
    a = img.copy()

    height = len(a)
    width = len(a[0])

    for i in range(height):
        for j in range(width):
            i_st, i_nd, j_st, j_nd = get_edges(height, width, i, j, n)

            med = np.median(img[i_st:i_nd, j_st:j_nd])

            a[i][j] = 0 if int(a[i][j]) > med else 255

    return a.astype(img.dtype)

############################################################################################
# Plot the histogram with 16 bins (can be changed), and an continuous approximation.       #
#                                                                                          #
# About the arguments:                                                                     #
# img: The histogram image (dataset)                                                       #
# bins: Must be less than 256 (powers of two give a better result)                         #
# step: Must be less than 256, 256 // bins will just give the best resolution by bins      #
# y_ticks_count: Resolution of Y ticks                                                     #
############################################################################################
def histogram(img, bins = 16, step = 256 // 16, y_ticks_count = 16):
    a = img.copy()
    a = a.reshape(len(img) * len(img[0]))
    a = np.append(a, [0, 256])
    
    x, y = np.unique(img, return_counts=True)
    d = dict(zip(x, y))
    for i in range(256):
        if i not in d.keys():
            d[i] = 0

    d = sorted(d.items(), key=lambda x: x[0])
    x = np.array([x[0] for x in d])
    y = np.array([x[1] for x in d])

    x_ = np.array([])
    y_ = np.array([])
    for i in range(0, len(x), step):
        x_ = np.append(x_, np.rint(np.median(x[i:i + step])))
        y_ = np.append(y_, max(0, np.rint(y[i:i + step].sum())))

    X_Y_Spline = make_interp_spline(x_, y_)
    X_ = np.linspace(x_.min(), x_.max(), 500)
    Y_ = X_Y_Spline(X_)
    Y_[Y_ < 0] = 0

    plt.hist(a, bins)
    plt.plot(X_, Y_)
    plt.title("Histogram of original image - {}".format(image_name))
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency (10^3)")
    plt.xticks(np.arange(0, 256 + 1, step))
    plt.yticks(np.arange(0, y_.max(), y_.max() // y_ticks_count), labels = [str(round(x / 1e3, 2)) for x in np.arange(0, y_.max(), y_.max() // y_ticks_count)])
    plt.grid(which='both', linestyle='--', linewidth=.5)
    # plt.savefig("histogram.png")
    plt.show()

def bw_percentage(img):
    pixel, freq = np.unique(img, return_counts=True)

    freq = freq / freq.sum()

    b_color = [40 / 256] * 3
    w_color = [200 / 256] * 3
    fig, ax = plt.subplots()
    bar = plt.bar(x = ['Black', 'White'], height = freq, color = [b_color, w_color])
    plt.title("BW percentage - {} - {}".format(image_name, selected_method))
    plt.xlabel("Color")
    plt.ylabel("Percentage")
    ax.bar_label(bar)
    # plt.savefig("bw_percentage.png")
    plt.show()


image_name = "monarch"
selected_method = "Global Method"

babo = read_grayscale("original_images/baboon.pgm")
fidu = read_grayscale("original_images/fiducial.pgm")
mona = read_grayscale("original_images/monarch.pgm")
pepp = read_grayscale("original_images/peppers.pgm")
reti = read_grayscale("original_images/retina.pgm")
sonn = read_grayscale("original_images/sonnet.pgm")
wedg = read_grayscale("original_images/wedge.pgm")

cv.imshow("global_method1.png", median_method(babo, n = 35))
cv.imshow("global_method2.png", median_method(fidu, n = 35))
cv.imshow("global_method3.png", median_method(mona, n = 35))
cv.imshow("global_method4.png", median_method(pepp, n = 35))
cv.imshow("global_method5.png", median_method(reti, n = 35))
cv.imshow("global_method6.png", median_method(sonn, n = 35))
cv.imshow("global_method7.png", median_method(wedg, n = 35))
k = cv.waitKey(0)

# cv.imshow("global_method.png", global_method(img))
# cv.imshow("bernsen_method.png", bernsen_method(img, 5))
# cv.imshow("niblack_method.png", niblack_method(img, 5, -0.2))
# cv.imshow("sauvola_pietaksinen_method.png", sauvola_pietaksinen_method(img, 5, k = 0.5, r = 128))
# cv.imshow("phansalskar_more_sabale_method.png", phansalskar_more_sabale_method(img, 3, k = 0.25, r = 0.5, p = 2, q = 10))
# cv.imshow("contrast_method.png", contrast_method(img, 5))
# cv.imshow("mean_method.png", mean_method(img, 5))
# cv.imshow("median_method.png", median_method(img, 5))

n = int(input("Choose what image you want to use\n1 - Baboon\n2 - Fiducial\n3 - Monarch\n4 - Peppers\n5 - Retina\n6 - Sonnet\n7 - Wedge\n"))
if n == 1:
    image_name = "baboon"
if n == 2:
    image_name = "fiducial"
if n == 3:
    image_name = "monarch"
if n == 4:
    image_name = "peppers"
if n == 5:
    image_name = "retina"
if n == 6:
    image_name = "sonnet"
if n == 7:
    image_name = "wedge"

img = read_grayscale("original_images/{}.pgm".format(image_name))

n = int(input("Choose what method you want to use\n1 - Global\n2 - Bernsen's\n3 - Niblack's\n4 - Sauvola's and Pietaksinen's\n5 - Phansalskar's, More's and Sabale's\n6 - Contrast\n7 - Mean\n8 - Median\n"))
if n == 1:
    try:
        T = int(input("Choose your threshold: "))
        binary_image = global_method(img, T)
    except:
        print("Invalid Arguments, using standart ones.")
        binary_image = global_method(img)
    cv.imshow("global_method.png", binary_image)
    selected_method = "Global Method"
if n == 2:
    try:
        sz = int(input("Choose neighbourhood size (8-neighbourhood): "))
        binary_image = bernsen_method(img, sz)
    except:
        print("Invalid Arguments, using standart ones.")
        binary_image = bernsen_method(img)
    cv.imshow("bernsen_method.png", binary_image)
    selected_method = "Bernsen's Method"
if n == 3:
    try:
        sz, k = map(float, input("Choose neighbourhood size (8-neighbourhood) and k: ").split())
        binary_image = niblack_method(img, int(sz), k)
    except:
        print("Invalid Arguments, using standart ones.")
        binary_image = niblack_method(img)
    cv.imshow("niblack_method.png", binary_image)
    selected_method = "Niblack's Method"
if n == 4:
    try:
        sz, k, r = map(float, input("Choose neighbourhood size (8-neighbourhood), k and R: ").split())
        binary_image = sauvola_pietaksinen_method(img, int(sz), k, r)
    except:
        print("Invalid Arguments, using standart ones.")
        binary_image = sauvola_pietaksinen_method(img)    
    cv.imshow("sauvola_pietaksinen_method.png", binary_image)
    selected_method = "Sauvola's and Pietaksinen's Method"
if n == 5:
    try:
        sz, k, r, p, q = map(float, input("Choose neighbourhood size (8-neighbourhood), k, R, p and q: ").split())
        binary_image = phansalskar_more_sabale_method(img, int(sz), k, r, p, q)
    except:
        print("Invalid Arguments, using standart ones.")
        binary_image = phansalskar_more_sabale_method(img)    
    cv.imshow("phansalskar_more_sabale_method.png", binary_image)
    cv.imwrite("phansalskar_more_sabale_method.png", binary_image)
    selected_method = "Phansalskar's, More's and Sabale's Method"
if n == 6:
    try:
        sz = int(input("Choose neighbourhood size (8-neighbourhood): "))
        binary_image = contrast_method(img, sz)
    except:
        print("Invalid Arguments, using standart ones.")
        binary_image = contrast_method(img)    
    cv.imshow("contrast_method.png", binary_image)
    selected_method = "Contrast Method"
if n == 7:
    try:
        sz = int(input("Choose neighbourhood size (8-neighbourhood): "))
        binary_image = mean_method(img, sz)
    except:
        print("Invalid Arguments, using standart ones.")
        binary_image = mean_method(img)    
    cv.imshow("mean_method.png", binary_image)
    selected_method = "Mean Method"
if n == 8:
    try:
        sz = int(input("Choose neighbourhood size (8-neighbourhood): "))
        binary_image = median_method(img, sz)
    except:
        print("Invalid Arguments, using standart ones.")
        binary_image = median_method(img)    
    cv.imshow("median_method.png", binary_image)
    selected_method = "Median Method"

cv.imshow("{}.pgm".format(image_name), img)
histogram(img)
bw_percentage(binary_image)

print("Press any key to exit!")
k = cv.waitKey(0)