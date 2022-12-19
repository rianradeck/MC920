import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

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

def monochromatic(img):
    src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, src = cv.threshold(src_gray, 254, 255, 0)
    return src

def borders(img):
    ans = np.zeros(img.shape)
    ans[True] = 255

    d = [1, 0, -1, 0, 1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.min(img[i][j]) == 255:
                continue
            for k in range(4):
                i_ = i + d[k]
                j_ = j + d[k + 1]

                if 0 <= i_ and i_ < img.shape[0] and 0 <= j_ and j_ < img.shape[1] and np.min(img[i_][j_]) == 255:
                    ans[i][j] = [255, 0, 0]
    return ans

#considering 4-neighbourhood
def get_convex_components(img):
    ans = []

    comp = np.zeros(img.shape)
    comp_cnt = 0
    d = [1, 0, -1, 0, 1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if comp[i][j] == 0 and img[i][j] == 0:
                comp_cnt += 1
                
                q = [(i, j)]
                comp[i][j] = comp_cnt
                min_i = int(1e9)
                max_i = int(-1e9)
                min_j = int(1e9)
                max_j = int(-1e9)
                coordinates = [(i, j)]

                while len(q) > 0:

                    i_, j_ = q[0]
                    q = q[1:]

                    min_i = min(min_i, i_)
                    max_i = max(max_i, i_)
                    min_j = min(min_j, j_)
                    max_j = max(max_j, j_)

                    for k in range(4):
                        new_i = i_ + d[k]
                        new_j = j_ + d[k + 1]

                        if 0 <= new_i and new_i < img.shape[0] and 0 <= new_j and new_j < img.shape[1] and comp[new_i][new_j] == 0 and img[new_i][new_j] == 0:
                            comp[new_i][new_j] = comp_cnt
                            q.append((new_i, new_j))
                            coordinates.append((new_i, new_j))

                ans.append(((i, j), (min_i, max_i, min_j, max_j), coordinates, comp_cnt))

    return ans

original_img = read_colorscale("images/objetos3.png")
regions_img = original_img.copy()
border = borders(original_img)
cv.imshow("border.png", border)

img = monochromatic(original_img)

cv.imshow("monochromatic.png", img)
components = get_convex_components(img)
hist = []

print("número de regiões:", len(components))
print("")
for root, edges, coordinates, idx in components:
    src = img[edges[0]:edges[1] + 1, edges[2]:edges[3] + 1]
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if (i + edges[0], j + edges[2]) not in coordinates:
                src[i][j] = 0
            else:
                src[i][j] = 255
    
    print("região ", idx - 1, sep = '', end = ': ')
    text_sz = cv.getTextSize(str(idx - 1), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    mid_point = ((edges[1] + 1 + edges[0] + text_sz[0][1] - 1) // 2, (edges[3] + 1 + edges[2] - text_sz[0][0]) // 2)

    cv.putText(regions_img, str(idx - 1), mid_point[::-1], cv.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

    region = measure.regionprops(src)

    print("área:", region[0]['area'], end = ' ')
    print("perímetro:", region[0]['perimeter'], end = ' ')
    print("excentricidade:", region[0]['eccentricity'], end = ' ')
    print("solidez:", region[0]['solidity'])
    hist.append(region[0]['area'])
    
cv.imshow('regions.png', regions_img)

labels = [0, 0, 0]
for x in hist:
    labels[0 if x < 1500 else 1 if x < 3000 else 2] += 1

fig, ax = plt.subplots(1, 1)
ax.hist(hist, bins = [0, 1499, 2999, max(np.max(hist), 4500)], edgecolor = 'black', color = 'blue')
ax.set_title("Histograma de áreas dos objetos")
ax.set_xlabel('Área')
ax.set_ylabel('Quantidade de objetos')
plt.xticks([0, 1500, 3000, max(np.max(hist), 4500)])
  
rects = ax.patches

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, label, ha = 'center', va = 'bottom')
  
plt.show()