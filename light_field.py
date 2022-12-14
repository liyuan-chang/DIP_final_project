import os
from math import floor
import cv2
import numpy as np


class LightField:
    def __init__(self, w=512, h=512, ch=3, t=5, p=5):
        """Initialize light field class
        :param w: image with
        :param h: image height
        :param ch: number of channels
        :param t: number of horizontal views
        :param p: number of vertical views
        """
        assert t % 2 == 1, f'Number of horizontal view should be a odd number, got: {t}'
        assert p % 2 == 1, f'Number of vertical view should be a odd number, got: {p}'
        self.w = w
        self.h = h
        self.ch = ch
        self.t = t
        self.p = p
        self.data = np.zeros((p, t, h, w, ch))

    def load_LF(self, file_type='png', img_dir='./data/light_field/tower', rename=False) -> None:
        """Read light field images from the image directory
        :param img_dir: directory that contains light field images
        """
        i_fname = 0
        fname_list = sorted(os.listdir(img_dir))
        for phi in range(self.p):
            for theta in range(self.t):
                while True:
                    if file_type in fname_list[i_fname]:
                        break
                    else:
                        i_fname += 1
                
                print(f'Loading {fname_list[i_fname]}')
                img = cv2.imread(os.path.join(img_dir, fname_list[i_fname]))
                img = cv2.resize(img, (self.w, self.h))
                self.data[phi, theta, :, :, :] = img / 255.0
                if rename:
                    os.remove(os.path.join(img_dir, fname_list[i_fname]))
                i_fname += 1
        if rename:
            self.save_LF(save_dir=img_dir)

    def gen_focus_img(self, depth=0.0):
        """Generate the focused image at depth
        :param depth: focus depth
        :return img: focused image with value 0~1
        """
        img = np.zeros((self.h, self.w, self.ch))
        for phi in range(self.p):
            y_offset = -depth * (phi - (self.p - 1) / 2)
            for theta in range(self.t):
                x_offset = -depth * (theta - (self.t - 1) / 2)
                img_interp = self.get_interp(phi, theta, y_offset, x_offset)
                if y_offset < 0 and x_offset < 0:
                    img[-floor(y_offset):, -floor(x_offset):, :] += \
                        img_interp[:floor(y_offset), :floor(x_offset), :]
                elif y_offset < 0 and x_offset >= 0:
                    img[-floor(y_offset):, :self.w-floor(x_offset), :] += \
                        img_interp[:floor(y_offset), floor(x_offset):, :]
                elif y_offset >= 0 and x_offset < 0:
                    img[:self.h-floor(y_offset), -floor(x_offset):, :] += \
                        img_interp[floor(y_offset):, :floor(x_offset), :]
                else:
                    img[:self.h-floor(y_offset), :self.w-floor(x_offset), :] += \
                        img_interp[floor(y_offset):, floor(x_offset):, :]
        
        return img / self.p / self.t
    
    def get_interp(self, phi, theta, y_offset, x_offset):
        """Interpolate the view according to the offset
        :param phi: index of vertical view
        :param theta: index of horizontal view
        :param y_offset: the y offset with respect to y_coordinate
        :param x_offset: the x offset with respect to x_coordinate
        :return img_interp: bilinear interpolated image 
        """
        img_interp = np.zeros((self.h, self.w, self.ch))
        img_pad = cv2.copyMakeBorder(self.data[phi, theta, :, :, :], 0, 1, 0, 1, cv2.BORDER_REPLICATE)
        x_dist = x_offset - floor(x_offset)
        y_dist = y_offset - floor(y_offset)
        for y in range(2):
            for x in range(2):
                y_weight = y_dist if y == 1 else (1 - y_dist)
                x_weight = x_dist if x == 1 else (1 - x_dist)
                img_interp += img_pad[y:y+self.h, x:x+self.w] * x_weight * y_weight
        
        return img_interp

    
if __name__ == '__main__':
    LF = LightField(w=512, h=512, ch=3, t=5, p=5)
    LF.load_LF(file_type='png', img_dir='./data/light_field/tower')
