import os
import cv2
import numpy as np

from light_field import LightField

class FocalStack:
    def __init__(self, w=512, h=512, ch=3, t=5, p=5, focus_depth=[1.0, 0.0, -1.0]) -> None:
        """Initialize focal stack class
        :param w: image with
        :param h: image height
        :param ch: number of channels
        :param t: number of horizontal views
        :param p: number of vertical views
        :param focus_depth: array of the focus depth
        """
        self.w = w
        self.h = h
        self.ch = ch
        self.t = t
        self.p = p
        self.n_layer = len(focus_depth)
        self.focus_depth = focus_depth
        self.data = np.zeros((len(focus_depth), h, w, ch))

    def load_FS(self, file_type='png', img_dir='./data/focal_stack/town'):
        """Read focal stack images from the image directory
        :param img_dir: directory that contains focal stack images
        """
        i_fname = 0
        fname_list = sorted(os.listdir(img_dir))
        for i_layer in range(self.n_layer):
            while True:
                if file_type in fname_list[i_fname]:
                    break
                else:
                    i_fname += 1
            
            print(f'Loading {fname_list[i_fname]}')
            img = cv2.imread(os.path.join(img_dir, fname_list[i_fname]))
            img = cv2.resize(img, (self.w, self.h))
            self.data[i_layer, :, :, :] = img / 255.0
            i_fname += 1
    
    def gen_FS_from_LF(self, LF: LightField):
        """Generate focal stack from the light field images
        :param LF: light field class
        """
        for i_layer in range(self.n_layer):
            # print('Generating focus image at depth = {:.1f}'.format(self.focus_depth[i_layer]))
            self.data[i_layer, :, :, :] = LF.gen_focus_img(self.focus_depth[i_layer])
    
    def save_FS(self, save_dir='./result'):
        """Save the focal stack
        :param save_dir: directory for the generated focus images
        """
        for i_layer in range(self.n_layer):
            fname = os.path.join(save_dir, 'focus_{}_[{:.1f}].png'.format(str(i_layer).zfill(2), self.focus_depth[i_layer]))
            print(f'Saving focus image at {fname}')
            cv2.imwrite(fname, self.data[i_layer] * 255.0)


if __name__ == '__main__':
    LF = LightField(w=512, h=512, ch=3, t=5, p=5)
    LF.load_LF(file_type='png', img_dir='./data/light_field/tower')
    focus_depth = [2 - i * 0.2 for i in range(21)]
    # focus_depth = [1, 0.0, -1]
    FS = FocalStack(w=512, h=512, ch=3, t=5, p=5, focus_depth=focus_depth)
    FS.gen_FS_from_LF(LF)
    FS.save_FS(save_dir='./result')