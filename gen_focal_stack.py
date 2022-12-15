import os 
import cv2
import numpy as np

from light_field import LightField
from focal_stack import FocalStack


# simulate motion of focusing
ratio = [1.0, 1.0, 1.0, 1.0]
# ratio = [1.06, 1.04, 1.02, 1.0]

dataset = {
    'bamboo_2': {
        'light_field_dir': './data/light_field/bamboo_2',
        'shape': (436, 1024, 3),
        'view': (5, 5),
        'n_frame': 50,
        'focal_stack_dir': './data/focal_stack_gen/bamboo_2',
        'focus_depth': [3.4, 2.6, 1.8, 1.0]
    },
    'bamboo_3': {
        'light_field_dir': './data/light_field/bamboo_3',
        'shape': (436, 1024, 3),
        'view': (5, 5),
        'n_frame': 50,
        'focal_stack_dir': './data/focal_stack_gen/bamboo_3',
        'focus_depth': [6, 4.4, 2.4, 1.2]
    },
    'chickenrun_1': {
        'light_field_dir': './data/light_field/chickenrun_1',
        'shape': (436, 1024, 3),
        'view': (5, 5),
        'n_frame': 50,
        'focal_stack_dir': './data/focal_stack_gen/chickenrun_1',
        'focus_depth': [2, 1.2, 0.4, 0.0]
    },
    'shaman_1': {
        'light_field_dir': './data/light_field/shaman_1',
        'shape': (436, 1024, 3),
        'view': (5, 5),
        'n_frame': 50,
        'focal_stack_dir': './data/focal_stack_gen/shaman_1',
        'focus_depth': [15, 7, 3]
    }
}

# parameters
target = 'bamboo_2'
light_field_dir = dataset[target]['light_field_dir']
img_h, img_w, ch = dataset[target]['shape']
n_ver_view, n_hor_view = dataset[target]['view']
n_frame = dataset[target]['n_frame']
focal_stack_dir = dataset[target]['focal_stack_dir']
focus_depth = dataset[target]['focus_depth']

# load light field for every frame
tmp_dir = os.listdir(light_field_dir)[0]
for i, frame_fname in enumerate(sorted(os.listdir(os.path.join(light_field_dir, tmp_dir)))):
    print(f'Processing frame {i}', end='\r')
    
    # create light field objects
    LF = LightField(w=img_w, h=img_h, ch=ch, t=n_hor_view, p=n_ver_view)
    
    # create folder for storing focal stacks
    if not os.path.exists(focal_stack_dir):
        os.makedirs(focal_stack_dir)
    
    for j, view_folder in enumerate(sorted(os.listdir(light_field_dir))):
        phi = j // n_hor_view
        theta = j % n_hor_view
        tmp = [light_field_dir, view_folder, frame_fname]
        frame = cv2.imread(os.path.join(*tmp))
        LF.data[phi, theta, :, :, :] = frame / 255.0

    # generate focal stack 
    # focus_depth = [16 - i * 1 for i in range(21)]
    FS = FocalStack(w=img_w, h=img_h, ch=ch, t=n_hor_view, p=n_ver_view, focus_depth=focus_depth)
    FS.gen_FS_from_LF(LF)
    
    for k in range(len(focus_depth)):
        FS_folder_name = os.path.join(focal_stack_dir, str(k))
        if not os.path.exists(FS_folder_name):
            os.makedirs(FS_folder_name)
        
        # resize = FS.data[k]
        new_h, new_w = int(img_h * ratio[k]), int(img_w * ratio[k])
        offset_y, offset_x = (new_h - img_h) // 2, (new_w - img_w) // 2
        resize = cv2.resize(FS.data[k], (new_w, new_h))
        resize = resize[offset_y:offset_y+img_h, offset_x:offset_x+img_w, :]
        cv2.imwrite(os.path.join(FS_folder_name, '{:03d}.png'.format(i)), resize * 255.0)
        
    # break
print()