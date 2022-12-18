#!/usr/bin/env python
# encoding: utf-8

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from scatter import ModuleRenderScatter  # circular aperture

defocus_scale = 10.0
gamma_min   = 1.0
gamma_max   = 5.0
align_dir   = './data/align/'
bokeh_dir   = './data/bokeh/'
disp_path   = './data/sharpness/sharpness_var.png'
K           = 60                    #blur parameter
gamma       = 4                     #gamma value (1~5)
highlight  = True
highlight_RGB_threshold = 220/255
highlight_enhance_ratio = 0.4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classical_renderer = ModuleRenderScatter().to(device)

def pipeline(classical_renderer, image, defocus, gamma):
    bokeh_classical, defocus_dilate = classical_renderer(image**gamma, defocus*defocus_scale)
    bokeh_classical = bokeh_classical ** (1/gamma)
    defocus_dilate = defocus_dilate / defocus_scale
    gamma = (gamma - gamma_min) / (gamma_max - gamma_min)
    adapt_scale = max(defocus.abs().max().item(), 1)

    return bokeh_classical.clamp(0, 1)

disp_focus_list  = [0, 0.333, 0.667, 1]                  #refocused disparity (0~1)

for i, disp_focus in enumerate(disp_focus_list):
    print(f'Disparity focus [{disp_focus}] is executing...')
    image = cv2.imread(os.path.join(align_dir, f'{i}.jpg')).astype(np.float32) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image_ori = image.copy()

    disp = np.float32(cv2.imread(disp_path, cv2.IMREAD_GRAYSCALE))
    disp = (disp - disp.min()) / (disp.max() - disp.min())

    if highlight:
        mask1 = np.clip(np.tanh(200 * (np.abs(disp - disp_focus)**2 - 0.01)), 0, 1)[..., np.newaxis]  # out-of-focus areas
        mask2 = np.clip(np.tanh(10*(image - highlight_RGB_threshold)), 0, 1)    # highlight areas
        mask = mask1 * mask2
        image = image * (1 + mask * highlight_enhance_ratio)
    
    defocus = K * (disp - disp_focus) / defocus_scale

    with torch.no_grad():
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        defocus = torch.from_numpy(defocus).unsqueeze(0).unsqueeze(0)
        image = image.cuda()
        defocus = defocus.cuda()

        bokeh_classical = pipeline(
            classical_renderer, image, defocus, gamma
        )
    defocus = defocus[0][0].cpu().numpy()
    bokeh_classical = bokeh_classical[0].cpu().permute(1, 2, 0).numpy()
    
    cv2.imwrite(os.path.join(bokeh_dir, f'{i}.jpg'), bokeh_classical[..., ::-1] * 255)
