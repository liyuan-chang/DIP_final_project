import os 
import cv2
import numpy as np


def interp_sequence(sequence, n_frame):
    new_seq = np.zeros(n_frame)
    i_seq = 0
    for i in range(n_frame):
        # the first element
        if i_seq == 0 and i <= sequence[i_seq][0]:
            new_seq[i] = sequence[i_seq][1]
        # the last element
        elif i_seq == len(sequence) - 1:
            new_seq[i] = sequence[i_seq][1]
        # other elements
        else:
            if i - sequence[i_seq][0] < sequence[i_seq+1][0] - i:
                new_seq[i] = sequence[i_seq][1]
            else:
                new_seq[i] = sequence[i_seq+1][1]
                i_seq += 1
    return new_seq
        

def gen_refocus_video(sequence, n_frame, focal_stack_dir, refocus_video_dir):
    new_seq = interp_sequence(sequence=sequence, n_frame=n_frame)
    if not os.path.exists(refocus_video_dir):
        os.makedirs(refocus_video_dir)
    
    for i, depth in enumerate(new_seq):
        frame_fname = '{:03d}.png'.format(i)
        tmp = [focal_stack_dir, str(int(depth)), frame_fname]
        # print(tmp)
        frame = cv2.imread(os.path.join(*tmp))
        tmp = [refocus_video_dir, frame_fname]
        cv2.imwrite(os.path.join(*tmp), frame)

    
    fps = 10
    # frame dimension: (width, height)
    frameSize = (1024, 436)
    out = cv2.VideoWriter('./data/refocus.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)
    for filename in sorted(os.listdir(refocus_video_dir)):
        img = cv2.imread(os.path.join(refocus_video_dir, filename)).astype(np.uint8)
        out.write(img)
    out.release()
    

if __name__ == '__main__':
    sequence = [(0, 3), (20, 2), (30, 1), (40, 0)] # (i_frame, i_depth)
    n_frame = 50
    focal_stack_dir = './data/focal_stack_gen/bamboo_2'
    refocus_video_dir = './data/refocus_video/bamboo_2'
    gen_refocus_video(sequence=sequence,
                      n_frame=n_frame,
                      focal_stack_dir=focal_stack_dir,
                      refocus_video_dir=refocus_video_dir)