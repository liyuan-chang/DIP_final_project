# Seamless Focal-Stack Refocusing
2022 Fall Digital Image Processing Final Project

## Install Packages
Install the required libraries.
```bash
pip install -r requirements.txt
```
Install tkinter for display GUI.
```bash
sudo apt-get install python3-tk
```

## Prepare Data
Download required executables and images from [here](https://drive.google.com/drive/folders/1GSxM2Dyb1g_fLWesgHhDJqvm94lof-05?usp=sharing).
The data structure should be as follows.
```bash
data/
|---focal_stack_gen/
    |---bamboo_2/
        |---0/
            |---000.png
    ...
...
|---light_field/
    |---bamboo_2/
        |---00_00/
            |---000.png
    ...
```

## Execute
The output images are located in `data/`.

### Image Refocusing

Run the following command to align focal stack and calculate sharpness. The aligned focal stack will be in `data/align`. The results of Laplacian will be in `data/laplacian`. The sharpness will be in `data/sharpness`.
```bash
python main.py
```

Run the following command to launch the display app by executing the following command.
```bash
python display.py
```

### Bokeh Effect

Run the following command to generate focal stack with magnified bokeh effect. The focal stack will be in `data/bokeh`.
```bash
python bokeh.py
```

### Video Refocusing

Run the following command to generate focal stacks from light field videos. The focal stacks will be in `data/focal_stack_gen`.
```bash
python gen_focal_stack.py
```

Run the following command to align focal stacks and calculate sharpness. The aligned focal stacks will be in `data/align_stack_gen`. The sharpness will be in `data/sharpness_video`.
```bash
python main_video.py
```

Run the following command to launch the display app and choose the focal point of the selected frame. Press `Export` button to generate the refocused video. The refocused video will be in `data/refocus.mp4`.
```bash
python display_video.py
```
