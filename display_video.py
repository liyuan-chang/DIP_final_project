from cgi import test
import os
import numpy as np
from tkinter import *
from PIL import ImageTk, Image
from gen_refocus_video import gen_refocus_video


original_dir = './data/focal_stack_gen/bamboo_2/'
align_dir = './data/align_stack_gen/'
sharpness_dir = './data/sharpness_video/'
refocus_video_dir = './data/refocus_video/bamboo_2'
sharpness_fnames = []
for fname in sorted(os.listdir(sharpness_dir)):
    if fname.split('.')[-1] == 'npy':
        sharpness_fnames.append(os.path.join(sharpness_dir, fname))
# output directory
os.makedirs(refocus_video_dir, exist_ok=True)

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
    def __init__(self, root, original_dir=original_dir, align_dir=align_dir, sharpness_fnames=sharpness_fnames) -> None:
        # initialize parameters
        self.root = root
        self.original_dir = original_dir
        self.align_dir = align_dir
        self.sharpness_fnames = sharpness_fnames
        
        # set window size
        self.window_w = int(self.root.winfo_screenwidth() * 0.6)
        self.window_h = int(self.root.winfo_screenheight() * 0.6)
        print(f'Window size = {self.window_w}x{self.window_h}')
        self.root.geometry(f'{self.window_w}x{self.window_h}')
        
        # read all images 
        # select some of images to display
        self.frame_step = 10
        self.num_frames = 50
        self.frame_index = np.arange(0, self.num_frames, self.frame_step)
        # focal stack shape: [frame, depth, imgs]
        self.focal_stack = [[] for i in range(self.num_frames)]
        self.align_stack = [[] for i in range(self.num_frames)]
        self.sharpness = None
        self.read_data()
        
        # set the ratio of the original and the resized image
        img_w, img_h = self.focal_stack[0][0].width(), self.focal_stack[0][0].height()
        self.ratio_x, self.ratio_y = self.sharpness[0].shape[1] / img_w, self.sharpness[0].shape[0] / img_h
        
        # buttons
        self.flag_original = True
        self.flag_aligned = False
        original_btn = Button(self.root, text='Original', command=self.set_original)
        aligned_btn = Button(self.root, text='Aligned', command=self.set_aligned)
        prev_btn = Button(self.root, text='Previous', command=self.set_prev)
        next_btn = Button(self.root, text='Next', command=self.set_next)
        export_btn = Button(self.root, text='Export', command=self.set_export)
        # NOTE: not sure if this affects the window size
        original_btn.pack(side=BOTTOM)
        aligned_btn.pack(side=BOTTOM)
        prev_btn.pack(side=BOTTOM)
        next_btn.pack(side=BOTTOM)
        export_btn.pack(side=BOTTOM)
        
        # display the first image
        self.curr_frame = 0
        self.curr_idx = 0
        self.img_label = None
        self.display_image(self.focal_stack[self.curr_frame][self.curr_idx])
        
        # get mouse position when clicked
        self.mouse_x, self.mouse_y = None, None
        self.root.bind('<ButtonPress-1>', self.mouse_press)
        
        # whether the mouse position is inside the image
        self.inside = False
        self.img_label.bind("<Enter>", self.set_inside)
        self.img_label.bind("<Leave>", self.set_outside)
        
        # record sequence to generate video
        # format: [(frame, depth), ...]
        self.video_sequence = {}
        for frame_id in self.frame_index:
            self.video_sequence[frame_id] = 0

    
    def read_data(self):
        """Read the original FS, aligned FS and the sharpness image"""
        for folder in sorted(os.listdir(self.original_dir)):
            for i, fname in enumerate(sorted(os.listdir(os.path.join(self.original_dir, folder)))):
                img = Image.open(os.path.join(self.original_dir, folder, fname))
                if img is None:
                    print (f'Error opening image {fname}')
                    exit(-1)
                self.focal_stack[i].append(ImageTk.PhotoImage(resize(img, window_w=self.window_w, window_h=self.window_h)))
        for folder in sorted(os.listdir(self.align_dir)):
            for i, fname in enumerate(sorted(os.listdir(os.path.join(self.align_dir, folder)))):
                img = Image.open(os.path.join(self.align_dir, folder, fname))
                if img is None:
                    print (f'Error opening image {fname}')
                    exit(-1)
                self.align_stack[i].append(ImageTk.PhotoImage(resize(img, window_w=self.window_w, window_h=self.window_h)))
        self.sharpness = [np.load(fname) for fname in self.sharpness_fnames]
        
    
    def display_image(self, img):
        """Display the image"""
        if self.img_label:
            self.img_label.forget()
        self.img_label = Label(image=img)
        self.img_label.pack()
        self.img_label.bind("<Enter>", self.set_inside)
        self.img_label.bind("<Leave>", self.set_outside)
    
        
    def mouse_press(self, event):
        """Get mouse position and display the image at the specified index"""
        if self.inside:
            self.mouse_x, self.mouse_y = event.x, event.y
            sharpness_x, sharpness_y = int(self.mouse_x*self.ratio_x), int(self.mouse_y*self.ratio_y)
            self.curr_idx = self.sharpness[self.curr_frame//self.frame_step][sharpness_y, sharpness_x]
            print(f'Sharpness at ({sharpness_x}, {sharpness_y}) index = {self.curr_idx}')
            if self.flag_original:
                self.display_image(self.focal_stack[self.curr_frame][self.curr_idx])
            else:
                self.display_image(self.align_stack[self.curr_frame][self.curr_idx])
            # record sequence to generate video
            # format: [(frame, depth), ...]
            self.video_sequence[self.curr_frame] = self.curr_idx
        
        
    def set_original(self):
        """Set displaying original FS"""
        self.flag_original, self.flag_aligned = True, False
        self.display_image(self.focal_stack[self.curr_frame][self.curr_idx])
        
        
    def set_aligned(self):
        """Set displaying aligned FS"""
        self.flag_original, self.flag_aligned = False, True
        self.display_image(self.align_stack[self.curr_frame][self.curr_idx])
    
    def set_prev(self):
        """Set displaying previous image"""
        if self.curr_frame >= self.frame_step:
            self.curr_frame = self.curr_frame - self.frame_step
        if self.flag_original:
            self.display_image(self.focal_stack[self.curr_frame][self.curr_idx])
        else:
            self.display_image(self.align_stack[self.curr_frame][self.curr_idx])
        
    def set_next(self):
        """Set displaying next image"""
        if self.curr_frame < (self.num_frames-self.frame_step-1):
            self.curr_frame = self.curr_frame + self.frame_step
        if self.flag_original:
            self.display_image(self.focal_stack[self.curr_frame][self.curr_idx])
        else:
            self.display_image(self.align_stack[self.curr_frame][self.curr_idx])
            
    def set_inside(self, event):
        """Mouse is inside the image"""
        self.inside = True

    def set_outside(self, event):
        """Mouse is outside the image"""
        self.inside = False
    
    def set_export(self):
        """Export video"""
        if self.flag_original:
            gen_refocus_video(sequence=list(self.video_sequence.items()),
                              n_frame=self.num_frames,
                              focal_stack_dir=self.original_dir,
                              refocus_video_dir=refocus_video_dir)
        else:
            gen_refocus_video(sequence=list(self.video_sequence.items()),
                              n_frame=self.num_frames,
                              focal_stack_dir=self.align_dir,
                              refocus_video_dir=refocus_video_dir)
        print('Finished exporting video.')

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.mainloop()
    print('------ Video sequence ------')
    print(list(app.video_sequence.items()))
    print('----------------------------')
    