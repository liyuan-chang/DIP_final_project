from cgi import test
import os
import numpy as np
from tkinter import *
from PIL import ImageTk, Image


original_dir = './data/raw/'
align_dir = './data/align/'
sharpness_fname = './data/sharpness/sharpness.npy'


def resize(img, window_w, window_h):
    w, h = img.size
    resize_flag = False

    if w > window_w:
        h = int(h * window_w / w)
        w = window_w
        resize_flag = True

    if h > window_h:
        w = int(w * window_h / h)
        h = window_h
        resize_flag = True

    if resize_flag:
        print(f"Resized image size: {w}x{h}")
        return img.resize((w,h))
    else:
        return img


class App():
    def __init__(self, root, original_dir=original_dir, align_dir=align_dir, sharpness_fname=sharpness_fname) -> None:
        # initialize parameters
        self.root = root
        self.original_dir = original_dir
        self.align_dir = align_dir
        self.sharpness_fname = sharpness_fname
        
        # set window size
        self.window_w = int(self.root.winfo_screenwidth() * 0.6)
        self.window_h = int(self.root.winfo_screenheight() * 0.6)
        print(f'Window size = {self.window_w}x{self.window_h}')
        self.root.geometry(f'{self.window_w}x{self.window_h}')
        
        # read all images 
        self.focal_stack = []
        self.align_stack = []
        self.sharpness = None
        self.read_data()
        
        # set the ratio of the original and the resized image
        img_w, img_h = self.focal_stack[0].width(), self.focal_stack[0].height()
        self.ratio_x, self.ratio_y = self.sharpness.shape[1] / img_w, self.sharpness.shape[0] / img_h
        
        # buttons
        self.flag_original = True
        self.flag_aligned = False
        original_btn = Button(self.root, text='Original', command=self.set_original)
        aligned_btn = Button(self.root, text='Aligned', command=self.set_aligned)
        # NOTE: not sure if this affects the window size
        original_btn.pack(side=BOTTOM)
        aligned_btn.pack(side=BOTTOM)
        
        # display the first image
        self.curr_idx = 0
        self.img_label = None
        self.display_image(self.focal_stack[self.curr_idx])
        
        # get mouse position when clicked
        self.mouse_x, self.mouse_y = None, None
        self.root.bind('<ButtonPress-1>', self.mouse_press)

    
    def read_data(self):
        """Read the original FS, aligned FS and the sharpness image"""
        for original, align in zip(sorted(os.listdir(self.original_dir)), sorted(os.listdir(self.align_dir))):
            img = Image.open(os.path.join(self.original_dir, original))
            if img is None:
                print (f'Error opening image {original}')
                exit(-1)
            self.focal_stack.append(ImageTk.PhotoImage(resize(img, window_w=self.window_w, window_h=self.window_h)))
            img = Image.open(os.path.join(self.align_dir, align))
            if img is None:
                print (f'Error opening image {align}')
                exit(-1)
            self.align_stack.append(ImageTk.PhotoImage(resize(img, window_w=self.window_w, window_h=self.window_h)))
        self.sharpness = np.load(self.sharpness_fname)
        
    
    def display_image(self, img):
        """Display the image"""
        if self.img_label:
            self.img_label.forget()
        self.img_label = Label(image=img)
        self.img_label.pack()
    
        
    def mouse_press(self, event):
        """Get mouse position and display the image at the specified index"""
        self.mouse_x, self.mouse_y = event.x, event.y
        sharpness_x, sharpness_y = int(self.mouse_x*self.ratio_x), int(self.mouse_y*self.ratio_y)
        self.curr_idx = self.sharpness[sharpness_y, sharpness_x]
        print(f'Sharpness at ({sharpness_x}, {sharpness_y}) index = {self.curr_idx}')
        if self.flag_original:
            self.display_image(self.focal_stack[self.curr_idx])
        else:
            self.display_image(self.align_stack[self.curr_idx])
        
        
    def set_original(self):
        """Set displaying original FS"""
        self.flag_original, self.flag_aligned = True, False
        self.display_image(self.focal_stack[self.curr_idx])
        
        
    def set_aligned(self):
        """Set displaying aligned FS"""
        self.flag_original, self.flag_aligned = False, True
        self.display_image(self.align_stack[self.curr_idx])
        
        
if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()