# 💡 Power Saver using OpenCV and Raspberry Pi

**Power Saver using OpenCV and Raspberry Pi** is a smart automation system that uses YOLOv4-powered person detection to automatically control appliances based on room occupancy. When people are present, devices turn ON, and they shut OFF when the room is empty—helping save energy efficiently.

---

## 🔍 Features

- **Real-time People Detection**: Utilizes YOLOv4 to detect and count people in the camera frame.
- **Automated Appliance Control**: Turns appliances ON/OFF using GPIO signals based on occupancy.
- **Seamless Integration with Raspberry Pi**: Lightweight, runs reliably on Pi hardware.
- **Single Python Script**: `powersaver_switch.py` handles detection and control with minimal configuration.

---

## ⚙️ How It Works

1. **Object Detection (YOLOv4)**  
   - The script loads YOLOv4 weights, config, and class labels.
   - Every video frame is analyzed to detect people.

2. **Occupancy Decision**  
   - If any person is detected (`count > 0`), a GPIO pin is set HIGH to turn appliances ON.
   - If no one is detected, the GPIO pin is set LOW to switch appliances OFF.

3. **GPIO Control**  
   - Utilizes RPi.GPIO to send control signals to a relay or switch circuitry.

---

## 🎯 Prerequisites

- Raspberry Pi with camera support
- Python packages:
  ```bash
  pip install opencv-python numpy RPi.GPIO


## 🚀 Getting Started
Clone the repository:
   
    ```bash
    Copy
    Edit
    git clone https://github.com/Rishikarpe/Power-saver-using-Open-CV-and-R-pi.git
    cd Power-saver-using-Open-CV-and-R-pi

-Download YOLOv4 assets:

    bash
    # Example download commands
    wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O coco.names

Execute the script:

    ```bash
     python3 powersaver_switch.py

-Monitor Output:
 -The script will display video feed with detected persons.
 -It logs actions like GPIO HIGH → power ON or GPIO LOW → power OFF.

## 🔧 Hardware Setup
Raspberry Pi with compatible camera
Relay Module connected via GPIO pin (e.g., GPIO 17)
Appliances connected through the relay for automated control
Power Supply properly grounded/shared between Pi and appliances

