# carrino haloween eyes

add this to /boot/firmware/config.txt
```bash
dtoverlay=pwm-2chan
```

debug
```bash
sudo cat /sys/kernel/debug/pwm  # pwm-2   (sysfs): requested enabled period: 1000000 ns duty: 500000 ns polarity: normal
pinctrl  | grep pwm -i # we use pin 12 which is gpio18 18: a3    pd | hi // GPIO18 = PWM0_CHAN2
```

runs at boot as a systemd service (track.service in this repo). to (re)install it:
```bash
sudo ln -sf /home/pi/hailo-rpi5-examples/track.service /etc/systemd/system/track.service
sudo systemctl daemon-reload && sudo systemctl enable --now track
sudo systemctl restart track                 # after changing code
sudo journalctl -u track -n 30 --no-pager    # its logs
```

see what the camera sees: the live view is on by default, open http://raspberrypi.local:8080/ on a phone/laptop
on the same wifi (boxes, cx grid, magenta line = where the eyes aim). it costs nothing while nobody is watching.
```bash
./track.sh                                         # same as at boot; view on :8080
./track.sh --debug-port 0                          # turn the view off
./track.sh --save-dir ~/eyes_debug --save-every 1  # record annotated frames to look at later
./track.sh --camera-size 640x480                   # capture size; default is 1280x720, which is a wider view on this camera
./track.sh --model yolov8m                         # bigger detector: sees small/far people better, ~half the fps
./track.sh --idle-after 30                         # look around after this long with nobody in view (default 15, 0 = never)
./track.sh --tiles 0                               # tiling (off by default): 0 = as many tiles as fit, or give a number
./track.sh --tiles 0 --tile-top 180 --tile-height 360   # the band of rows the tiles cover (default: middle half at 720p)
```
tiling: the model's input is 640x640, so squashing a 1280x720 frame into it makes people half as wide.
with --tiles the band of rows between --tile-top and --tile-top + --tile-height is cut out, scaled to 640
tall (so far-away people get bigger to the model), and 640x640 tiles across it are fed to the model one
per frame, round robin. the Hailo load is one inference per frame either way, but each tile is only seen
every Nth frame so the eyes update slower. someone standing in the overlap is seen by two tiles; the clipped
box is merged into the full one. off by default: the squashed frame is fine in daylight. the page
draws the tiles when they're on.
with nobody in view for `IDLE_AFTER` seconds (15) the eyes look around on their own: an eased look from one
end of `IDLE_RANGE` (cx 0.1 to 0.9, through the calibration) to the other taking `IDLE_MOVE` seconds (16), then
`IDLE_REST` seconds (15) still, then a look back. it starts from wherever the eyes are, stops the moment someone
is seen, and the header/page say LOOKING AROUND / LOOK. holding the eyes from the page also stops it.
when the person being followed vanishes mid-frame while walking at a steady pace (behind the trellises or the
tree) the eyes keep going at that pace: for `COAST_MAX` seconds (3), or if they vanished at one of the `BLOCKED`
spans (cx ranges, tinted red on the page; set them for your view) until they should be out the other side.
the header says COASTING. someone reappearing takes over at once; a person who stops in view is still seen.
people look small in the wide view. yolo_person.json's `detection_threshold` (0.2) is the hard floor for what
reaches the code; `MIN_CONFIDENCE` in track_x.py (0.4, or `--min-confidence`) is what gets followed. red boxes on
the page are the band in between: if real people show up red, lower it; if bushes show up red, don't. to try a
value without editing anything, put it on the ExecStart line of track.service (`track.sh --min-confidence 0.5`),
`sudo systemctl daemon-reload`, restart.
a person has to be seen in `PRESENT_FRAMES` of the last `PRESENT_WINDOW` frames (3 of 5, 0.1 s) to be followed or
to reset the idle timer: a one-frame flicker on a bush used to move the eyes and park them for 15 s.
boxes touching the left/right edge narrower than `EDGE_MIN_WIDTH` (3% of the frame) are ignored entirely:
a pole or car corner half out of shot kept getting called a person at 0.2-0.38.
the model always gets 640x640 whatever the capture size is, so a bigger capture costs the Hailo nothing
(only some CPU for jpeg decode). `v4l2-ctl -d /dev/video0 --list-formats-ext` lists what the camera offers.
green box = person being tracked, yellow = other person, red = below MIN_CONFIDENCE

tune where the eyes point, from the phone page (http://ai.local:8080/):
1. tap **Hold eyes** so tracking stops fighting you
2. stand somewhere, wait for the green box, use the -5/-1/+1/+5 buttons until the eyes look at you
3. tap **Mark (cx, duty)**. repeat at a few spots across the view (left, middle, right)
4. tap **Apply marks**: the marks become the calibration, saved to calibration.json (loaded at boot,
   overrides CALIBRATION in track_x.py). **Reset calibration** goes back to duty = cx*100.
   **Flip direction** mirrors left/right (camera mounted the other way up).

the header shows the cx it sees and the duty it sent; in hold mode it also shows what the
current calibration would send, so you can see how far off it is. each Mark also saves a
snapshot to ~/eyes_marks/. the old stdin way still exists: `./track.sh --calibrate`.

ssh
```bash
ssh pi@raspberrypi.local        # or ssh pi@<ip>, find the ip with `hostname -I` on the pi
ssh-copy-id pi@raspberrypi.local # once, so you don't need a password
```



![Banner](doc/images/hailo_rpi_examples_banner.png)

# Hailo Raspberry Pi 5 Examples
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hailo-ai/hailo-rpi5-examples)

Welcome to the Hailo Raspberry Pi 5 Examples repository. This project showcases various community projects and examples demonstrating the capabilities of the Hailo AI processor. These examples will help you get started with AI on embedded devices.
The examples in this repository are designed to work with the Raspberry Pi AI Kit and AI HAT, and x86_64 Ubuntu machine supporting both the Hailo8 (26 TOPS) and Hailo8L (13 TOPS) AI processors.
Visit the [Hailo Official Website](https://hailo.ai/) and [Hailo Community Forum](https://community.hailo.ai/) for more information.

## Hailo Apps Infra Repository
Hailo's official examples and pipelines are available in the [Hailo Apps Infra repo](https://github.com/hailo-ai/hailo-apps-infra) repository.
See the Hailo Apps Infra repo for more information and documentation on how to use the pipelines and development guide.

## Install Hailo Hardware and Software Setup on Raspberry Pi

For instructions on how to set up Hailo's hardware and software on the Raspberry Pi 5, see the [Hailo Raspberry Pi 5 installation guide](doc/install-raspberry-pi5.md#how-to-set-up-raspberry-pi-5-and-hailo).

# Hailo RPi5 Basic Pipelines
The basic pipelines examples demonstrate object detection, human pose estimation, and instance segmentation, providing a solid foundation for your own projects.
This repo is using our new [Hailo Apps Infra](https://github.com/hailo-ai/hailo-apps-infra) repo as a dependency.
See our Developement Guide for more information on how to use the pipelines to create your own custom pipelines.

## Installation

### Clone the Repository
```bash
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git
```
Navigate to the repository directory:
```bash
cd hailo-rpi5-examples
```

### Installation
Run the following script to automate the installation process:
```bash
./install.sh
```

### Documentation
For additional information and documentation on how to use the pipelines to create your own custom pipelines, see the [Basic Pipelines Documentation](doc/basic-pipelines.md).

### Running The Examples
When opening a new terminal session, ensure you have sourced the environment setup script:
```bash
source setup_env.sh
```
### Detection Example
For more information see [Detection Example Documentation.](doc/basic-pipelines.md#detection-example)

![Detection Example](doc/images/detection.gif)

#### Run the simple detection example:
```bash
python basic_pipelines/detection_simple.py
```
To close the application, press `Ctrl+C`.

This is lightweight version of the detection example, mainly focusing on demonstrating Hailo performance while minimizing CPU load. The internal GStreamer video processing pipeline is simplified by minimizing video processing tasks, and the YOLOv6 Nano model is used.

#### Run the full detection example:
This is the full detection example, including object tracker and multiple video resolution support - see more information [Detection Example Documentation](doc/basic-pipelines.md#detection-example):

```bash
python basic_pipelines/detection.py
```
To close the application, press `Ctrl+C`.

#### Running with Raspberry Pi Camera input:
```bash
python basic_pipelines/detection.py --input rpi
```

#### Running with USB camera input (webcam):
There are 2 ways:

Specify the argument `--input` to `usb`:
```bash
python basic_pipelines/detection.py --input usb
```

This will automatically detect the available USB camera (if multiple are connected, it will use the first detected).

Second way:

Detect the available camera using this script:
```bash
get-usb-camera
```
Run example using USB camera input - Use the device found by the previous script:
```bash
python basic_pipelines/detection.py --input /dev/video<X>
```

For additional options, execute:
```bash
python basic_pipelines/detection.py --help
```

#### Retrained Networks Support
The retrain guide is available in the [Hailo Apps Infra repo: Retraining Example](https://github.com/hailo-ai/hailo-apps-infra/blob/main/doc/developer_guide/retraining_example.md).

### Pose Estimation Example
For more information see [Pose Estimation Example Documentation.](doc/basic-pipelines.md#pose-estimation-example)

![Pose Estimation Example](doc/images/pose_estimation.gif)

#### Run the pose estimation example:
```bash
python basic_pipelines/pose_estimation.py
```
To close the application, press `Ctrl+C`.
See Detection Example above for additional input options examples.

### Instance Segmentation Example
For more information see [Instance Segmentation Example Documentation.](doc/basic-pipelines.md#instance-segmentation-example)

![Instance Segmentation Example](doc/images/instance_segmentation.gif)

#### Run the instance segmentation example:
```bash
python basic_pipelines/instance_segmentation.py
```
To close the application, press `Ctrl+C`.
See Detection Example above for additional input options examples.

### Depth Estimation Example
For more information see [Depth Estimation Example Documentation.](doc/basic-pipelines.md#depth-estimation-example)

![Depth Estimation Example](doc/images/depth.gif)

#### Run the depth estimation example:
```bash
python basic_pipelines/depth.py
```
To close the application, press `Ctrl+C`.
See Detection Example above for additional input options examples.

### Community Projects

Get involved and make your mark! Explore our Community Projects and start contributing today, because together, we build better things! 🚀
Check out our [Community Projects](community_projects/community_projects.md) for more information.

# Additional Examples and Resources

## Hailo Apps Infra
Hailo RPi5 Examples are using the [Hailo Apps Infra Repository](https://github.com/hailo-ai/hailo-apps-infra) as a dependency. The Hailo Apps Infra repository contains the infrastructure of Hailo applications and pipelines.
It is aimed for to provide tools for developers who want to create their own custom pipelines and applications. It features a simple and easy-to-use API for creating custom pipelines and applications.
It it installed as a pip package and can be used as a dependency in your own projects. See more information in its documentation and Development Guide.

### CLIP Application

CLIP (Contrastive Language-Image Pre-training) predicts the most relevant text prompt on real-time video frames using Hailo8/8l AI processor.
See the [hailo-CLIP Repository](https://github.com/hailo-ai/hailo-CLIP) for more information.
Click the image below to watch the demo on YouTube.

[![Watch the demo on YouTube](https://img.youtube.com/vi/XXizBHtCLew/0.jpg)](https://youtu.be/XXizBHtCLew)


#### Frigate Integration

Frigate is an open-source video surveillance software that runs on a Raspberry Pi.
Hailo is officially integrated into Frigate framework starting from version 0.16.0.
See [Hailo Official Integration with Frigate](https://community.hailo.ai/t/hailo-official-integration-with-frigate/13679) for more information.


### Raspberry Pi Official Examples

#### rpicam-apps

Raspberry Pi [rpicam-apps](https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-apps) Hailo post-processing examples.
This is Raspberry Pi's official example for AI post-processing using the Hailo AI processor integrated into their CPP camera framework.
The documentation on how to use rpicam-apps can be found [here](https://www.raspberrypi.com/documentation/computers/ai.html).

#### picamera2

Raspberry Pi [picamera2](https://github.com/raspberrypi/picamera2) is the libcamera-based replacement for Picamera, which was a Python interface to the Raspberry Pi's legacy camera stack. Picamera2 also presents an easy-to-use Python API.

## Additional Resources

### Hailo Python API
The Hailo Python API is now available on the Raspberry Pi 5. This API allows you to run inference on the Hailo-8L AI processor using Python.
For examples, see our [Python code examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/python).
Additional examples can be found in RPi [picamera2](#picamera2) code.
Visit our [HailoRT Python API documentation](https://hailo.ai/developer-zone/documentation/hailort-v4-18-0/?page=api%2Fpython_api.html#module-hailo_platform.drivers) for more information.

### Hailo Dataflow Compiler (DFC)

The Hailo Dataflow Compiler (DFC) is a software tool that enables developers to compile their neural networks to run on the Hailo-8/8L AI processors.
The DFC is available for download from the [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/) (Registration required).
For examples, tutorials, and retrain instructions, see the [Hailo Model Zoo Repo](https://github.com/hailo-ai/hailo_model_zoo).
Additional documentation and [tutorials](https://hailo.ai/developer-zone/documentation/dataflow-compiler/latest/?sp_referrer=tutorials/tutorials.html) can be found in the [Hailo Developer Zone Documentation](https://hailo.ai/developer-zone/documentation/).
For a full end-to-end training and deployment example, see the [Hailo Apps Infra repo: Retraining Example](https://github.com/hailo-ai/hailo-apps-infra/blob/main/doc/developer_guide/retraining_example.md).

## Contributing

We welcome contributions from the community. You can contribute by:
1. Contribute to our [Community Projects](community_projects/community_projects.md).
2. Reporting issues and bugs.
3. Suggesting new features or improvements.
4. Joining the discussion on the [Hailo Community Forum](https://community.hailo.ai/).


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This code example is provided by Hailo solely on an “AS IS” basis and “with all faults.” No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness, or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.
