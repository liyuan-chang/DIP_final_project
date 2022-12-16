import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage

focal_stack_dir = './data/focal_stack_gen/bamboo_2/'
sharpness_dir = './data/sharpness_video/'
align_stack_dir = './data/align_stack_gen/'
# output directory
os.makedirs(sharpness_dir, exist_ok=True)
os.makedirs(align_stack_dir, exist_ok=True)

# only calculate sharpness of some images
frame_step = 10

# calculate sharpness by local variance of Laplacian
def local_variance(lapla, win_rows, win_cols):
    lapla_mean = ndimage.uniform_filter(lapla, (win_rows, win_cols))
    lapla_sqr_mean = ndimage.uniform_filter(lapla**2, (win_rows, win_cols))
    lapla_var = lapla_sqr_mean - lapla_mean**2
    lapla_var = cv2.blur(lapla_var, ksize=(win_rows, win_cols))
    return lapla_var

# read focal stack: [frame, 4, imgs]
focal_stack_depth = 4
focal_stack = [[] for i in range(focal_stack_depth)]
for i, folder in enumerate(sorted(os.listdir(focal_stack_dir))):
    for fname in sorted(os.listdir(os.path.join(focal_stack_dir, folder))):
        img = cv2.imread(os.path.join(focal_stack_dir, folder, fname))
        if img is None:
            print (f'Error opening image {fname}')
            exit(-1)
        focal_stack[i].append(img)
focal_stack = np.array(focal_stack)
focal_stack = np.transpose(focal_stack, (1, 0, 2, 3, 4))

num_frames, N, H, W, C = focal_stack.shape

#%% alignment
# naive optimization
max_scale = 1.1
steps = 20
scale = np.arange(start=1, stop=max_scale, step=(max_scale-1)/steps)

for frame_id in range(num_frames):
    for i, img in enumerate(focal_stack[frame_id]):
        folder_name = os.path.join(align_stack_dir, str(i))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        # NOTE: 0.jpg is the image that focus on the nearest object
        if i == 0:
            cv2.imwrite(os.path.join(folder_name, f'{frame_id:03d}.png'), focal_stack[frame_id][0])
            continue
        # use template matching to find the scaling ratio and offset parameters
        found = None
        for s in tqdm(scale):
            resized = cv2.resize(focal_stack[frame_id][i], (int(W * s), int(H * s)))
            result = cv2.matchTemplate(resized, focal_stack[frame_id][0], cv2.TM_CCOEFF)
            (_, correlation, _, location) = cv2.minMaxLoc(result)
    
            if found is None or correlation > found[0]:
                found = (correlation, location, s)
    
        _, (x, y), s = found
        print(f'focal stack [{frame_id}][{i}] is matched using (s, x, y) = ({s:.2f}, {x}, {y})')
        aligned = cv2.resize(focal_stack[frame_id][i], (int(W * s), int(H * s)))
        aligned = aligned[y:y+H, x:x+W]
        cv2.imwrite(os.path.join(folder_name, f'{frame_id:03d}.png'), aligned)

#%% sharpness
# calculate sharpness
frame_index = np.arange(0, num_frames, frame_step)
lapla_stack = np.zeros((len(frame_index), N, H, W))
lapla_var_stack = np.zeros((len(frame_index), N, H, W))
win_rows, win_cols = H//20, W//20
for k, frame_id in enumerate(frame_index):
    for i, img in enumerate(focal_stack[frame_id]):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lapla = cv2.Laplacian(gray, ksize=5, ddepth=cv2.CV_32F)
        lapla = cv2.blur(lapla, ksize=(H//20, W//20))
        lapla_stack[k, i, :, :] = lapla
        # calculate sharpness by local variance of Laplacian
        lapla_var = local_variance(lapla, win_rows, win_cols)
        lapla_var_stack[k, i, :, :] = lapla_var

# save sharpness for GUI
for i, lapla_var in enumerate(lapla_var_stack):
    sharpness = np.argmax(lapla_var, axis=0)
    with open(os.path.join(sharpness_dir, f'sharpness_{i}.npy'), 'wb') as f:
        np.save(f, sharpness)

plt.figure()
sharpness = np.argmax(lapla_stack[0], axis=0)
plt.imshow(sharpness)
plt.colorbar(shrink=0.5)

plt.figure()
sharpness = np.argmax(lapla_var_stack[0], axis=0)
plt.imshow(sharpness)
plt.colorbar(shrink=0.5)
plt.savefig(os.path.join(sharpness_dir, 'sharpness_var.jpg'))