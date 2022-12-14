import os

# NOTE: modify path of the executable
ffmpeg_path = "./ffmpeg/ffmpeg"

# default value of parameters
video_fname = './data/Inter4K/Inter4K/60fps/nHD/21.mp4'
yuv_fname = './data/raw/21_nHD.yuv'
framerate = 30
resolution = (144, 176)
frame_format = "./data/bgr/frame_%04d.png"

dec_yuv_fname = './data/raw/akiyo_qcif_dec.yuv'
dec_video_fname = './test.mp4'
predict_format = "./data/predict/frame_%04d.png"

def video2yuv(video_fname=video_fname, yuv_fname=yuv_fname):
    """Convert a video such as .y4m of .mp4 to .yuv"""
    os.system(f"{ffmpeg_path} -i {video_fname} {yuv_fname} -y")


def yuv2video(framerate=framerate, resolution=resolution, yuv_fname=dec_yuv_fname, video_fname=dec_video_fname):
    """Convert a .yuv file to a video that can be played"""
    os.system(f"{ffmpeg_path} -r {framerate} -s {resolution[1]}x{resolution[0]} -i {yuv_fname} -pix_fmt yuv420p {video_fname} -y")
    

def yuv2rgb(framerate=framerate, resolution=resolution, frame_format=frame_format, yuv_fname=dec_yuv_fname):
    """Convert a .yuv file into rgb frames"""
    os.system(f"{ffmpeg_path} -r {framerate} -s {resolution[1]}x{resolution[0]} -i {yuv_fname} -pix_fmt yuv420p {frame_format}")


def rgb2yuv(framerate=framerate, resolution=resolution, frame_format=frame_format, yuv_fname=yuv_fname):
    """Convert rgb frames into a .yuv file"""
    os.system(f"{ffmpeg_path} -r {framerate} -s {resolution[1]}x{resolution[0]} -i {frame_format} -pix_fmt yuv420p {yuv_fname}")


def rgb2video(framerate=framerate, resolution=resolution, frame_format=predict_format, video_fname=dec_video_fname):
    os.system(f"{ffmpeg_path} -r {framerate} -s {resolution[1]}x{resolution[0]} -i {frame_format} {video_fname} -y")

if __name__ == '__main__':
    # video2yuv()
    # yuv2rgb()
    # rgb2yuv()
    # yuv2video()
    rgb2video()