import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage

focal_stack_dir = './data/raw/'
lapla_stack_dir = './data/laplacian/'
sharpness_dir = './data/sharpness/'
align_dir = './data/align/'
edge_dir = './data/edge/'

# calculate sharpness by local variance of Laplacian
def local_variance(lapla, win_rows, win_cols):
    lapla_mean = ndimage.uniform_filter(lapla, (win_rows, win_cols))
    lapla_sqr_mean = ndimage.uniform_filter(lapla**2, (win_rows, win_cols))
    lapla_var = lapla_sqr_mean - lapla_mean**2
    lapla_var = cv2.blur(lapla_var, ksize=(win_rows, win_cols))
    return lapla_var

# read focal stack
focal_stack = []
for fname in sorted(os.listdir(focal_stack_dir)):
    img = cv2.imread(os.path.join(focal_stack_dir, fname))
    if img is None:
        print (f'Error opening image {fname}')
        exit(-1)
    focal_stack.append(img)
focal_stack = np.array(focal_stack)

N, H, W, C = focal_stack.shape


# naive optimization
max_scale = 1.2
steps = 50
scale = np.arange(start=1, stop=max_scale, step=(max_scale-1)/steps)

for i, img in enumerate(focal_stack):
    # NOTE: 0.jpg is the image that focus on the nearest object
    if i == 0:
        cv2.imwrite(os.path.join(align_dir, f'0.jpg'), focal_stack[0])
        continue
    # use template matching to find the scaling ratio and offset parameters
    found = None
    for s in tqdm(scale):
        resized = cv2.resize(focal_stack[i], (int(W * s), int(H * s)))
        result = cv2.matchTemplate(resized, focal_stack[0], cv2.TM_CCOEFF)
        (_, correlation, _, location) = cv2.minMaxLoc(result)

        if found is None or correlation > found[0]:
            found = (correlation, location, s)

    _, (x, y), s = found
    print(f'focal stack [{i}] is matched using (s, x, y) = ({s}, {x}, {y})')
    aligned = cv2.resize(focal_stack[i], (int(W * s), int(H * s)))
    aligned = aligned[y:y+H, x:x+W]
    cv2.imwrite(os.path.join(align_dir, f'{i}.jpg'), aligned)


# calculate sharpness
lapla_stack = np.zeros((N, H, W))
lapla_var_stack = np.zeros((N, H, W))
win_rows, win_cols = H//20, W//20
for i, img in enumerate(focal_stack):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lapla = cv2.Laplacian(gray, ksize=5, ddepth=cv2.CV_32F)
    lapla = cv2.blur(lapla, ksize=(H//20, W//20))
    lapla_stack[i, :, :] = lapla
    cv2.imwrite(os.path.join(lapla_stack_dir, f'{i}.jpg'), lapla/np.max(lapla)*255)
    # calculate sharpness by local variance of Laplacian
    lapla_var = local_variance(lapla, win_rows, win_cols)
    lapla_var_stack[i, :, :] = lapla_var
    cv2.imwrite(os.path.join(lapla_stack_dir, f'var_{i}.jpg'), lapla_var/np.max(lapla_var)*255)


plt.figure()
sharpness = np.argmax(lapla_stack, axis=0)
plt.imshow(sharpness)
plt.colorbar(shrink=0.5)
plt.savefig(os.path.join(sharpness_dir, 'sharpness.jpg'))

plt.figure()
sharpness = np.argmax(lapla_var_stack, axis=0)
# save sharpness for GUI
with open(os.path.join(sharpness_dir, 'sharpness.npy'), 'wb') as f:
    np.save(f, sharpness)
plt.imshow(sharpness)
plt.colorbar(shrink=0.5)
plt.savefig(os.path.join(sharpness_dir, 'sharpness_var.jpg'))