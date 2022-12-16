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

Align focal stack and calculate sharpness.
```bash
python main.py
```

Launch the display app by executing the following command.
```bash
python display.py
```

### Video Refocusing

Generate focal stacks from light field videos.
```bash
python gen_focal_stack.py
```

Align focal stacks and calculate sharpness.
```bash
python main_video.py
```

Launch the display app and choose the focal point of the selected frame. Press `Export` button to generate refocused video.
```bash
python display_video.py
```
