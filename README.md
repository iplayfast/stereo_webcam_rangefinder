# Webcam Stereo Vision with Object Detection

A real-time stereo vision system that uses two webcams to detect objects and measure their distances using YOLO-based triangulation.

# There are a lot of test files, but webcam_stereo_vision.py seems to work the best so far

## Features

- 🎯 **YOLO Object Detection** - Uses YOLOv8 to detect objects in both cameras
- 📏 **Distance Measurement** - Calculates real-world distances using triangulation
- 🔧 **Interactive Calibration** - Easy one-time calibration using real measurements
- 🎮 **Multiple Interfaces** - Choose between Pygame GUI or terminal interface
- 📊 **Object Tracking** - Persistent tracking of detected objects
- 💾 **Flexible Model Support** - Use different YOLO models for speed vs accuracy

## Visual Indicators

- 🟢 **Green boxes**: Objects detected in both cameras (with distance measurement)
- 🔵 **Blue boxes**: Objects only detected in one camera
- ✓ **Visible status**: Object currently being tracked
- ○ **Lost status**: Object not seen recently but still tracked

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Test your setup
python setup.py
```

### 2. Basic Usage

```bash
# Use GUI interface (default)
python webcam_stereo_vision.py

# Use terminal interface (more compatible)
python webcam_stereo_vision.py --terminal

# Use different YOLO model
python webcam_stereo_vision.py --yolo-model yolov8s.pt

# Adjust detection sensitivity
python webcam_stereo_vision.py --confidence 0.5
```

### 3. Calibration

1. Position two webcams 6-10cm apart, facing the same direction
2. Run the system - it will detect objects but show "NOT CALIBRATED"
3. When an object appears in BOTH cameras, calibration mode activates
4. Measure the actual distance to that object and enter it
5. System is now calibrated for accurate distance measurements!

## Camera Setup

```
[Camera 1]  <-- 6-10cm -->  [Camera 2]
     |                           |
     v                           v
    Objects to detect
```

- **Distance**: 6-10cm apart works best
- **Alignment**: Point both cameras at the same area
- **Height**: Same height for both cameras
- **USB**: Each camera needs its own USB port

## Controls

### GUI Mode (Pygame)
- **C** - Calibrate system
- **L/H** - Lower/Higher confidence threshold  
- **M** - Change YOLO model
- **R** - Reset object tracking
- **Q** - Quit

### Terminal Mode
- **'c' + Enter** - Calibrate system
- **'l' + Enter** - Lower confidence (more detections)
- **'h' + Enter** - Higher confidence (fewer detections)
- **'m' + Enter** - Change YOLO model
- **'s' + Enter** - Show current settings
- **'d' + Enter** - Debug information
- **'q' + Enter** - Quit

## YOLO Models

Choose based on your needs:

| Model | Speed | Accuracy | Size | Best For |
|-------|-------|----------|------|----------|
| yolov8n.pt | ⭐⭐⭐⭐⭐ | ⭐⭐ | ~6MB | Real-time, weak hardware |
| yolov8s.pt | ⭐⭐⭐⭐ | ⭐⭐⭐ | ~22MB | Balanced (recommended) |
| yolov8m.pt | ⭐⭐⭐ | ⭐⭐⭐⭐ | ~52MB | Better accuracy |
| yolov8l.pt | ⭐⭐ | ⭐⭐⭐⭐⭐ | ~87MB | High accuracy |
| yolov8x.pt | ⭐ | ⭐⭐⭐⭐⭐ | ~136MB | Maximum accuracy |

## Troubleshooting

### GUI Won't Start
```bash
# Try terminal mode instead
python webcam_stereo_vision.py --terminal

# Or run diagnostics
python pygame_diagnostics.py
```

### No Objects Detected
- Lower confidence: press 'l' (terminal) or L (GUI)
- Better lighting helps YOLO detection
- Try different YOLO model: press 'm'

### Calibration Issues
- Ensure exactly ONE object is visible in BOTH cameras
- Object should be clearly detected (green boxes in GUI)
- Measure distance accurately (use ruler/tape measure)

### Camera Issues
```bash
# Test camera setup
python setup.py
```

## File Structure

```
webcam/
├── webcam_stereo_vision.py      # Main application (FIXED)
├── webcam_stereo_advanced.py    # Advanced version (complex features)
├── stereo_gui.py               # Pygame GUI component
├── setup.py                    # Setup and testing
├── requirements.txt            # Dependencies
├── pygame_diagnostics.py       # Troubleshooting tool
└── calibration/               # Calibration data (auto-created)
    └── stereo_calibration_yolo.txt
```

## Advanced Usage

### Command Line Options
```bash
python webcam_stereo_vision.py --help

# Examples:
python webcam_stereo_vision.py --confidence 0.7 --yolo-model yolov8m.pt
python webcam_stereo_vision.py --terminal --calibration-file my_calib.txt
```

### Configuration
- Calibration data is automatically saved
- Use different models during runtime (press 'm')
- Adjust confidence on the fly (press 'l'/'h')

## Requirements

- **Python 3.7+**
- **2 USB webcams**
- **OpenCV** (`opencv-python`)
- **YOLO** (`ultralytics`)
- **NumPy** (`numpy`)
- **Pygame** (optional, for GUI)

## Performance Tips

1. **Use yolov8n.pt** for maximum speed
2. **Lower confidence** for more detections  
3. **Good lighting** improves detection accuracy
4. **Terminal mode** is more compatible than GUI
5. **USB 3.0 ports** for better camera performance

## Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| "No cameras found" | Check USB connections, try different ports |
| "GUI failed to start" | Use `--terminal` flag |
| "No objects detected" | Lower confidence, improve lighting |
| "YOLO not available" | Run `pip install ultralytics` |
| Inaccurate distances | Recalibrate with known object |

## Getting Help

1. **Run setup script**: `python setup.py`
2. **Check debug info**: Press 'd' in terminal mode
3. **Test different models**: Press 'm' to change YOLO model
4. **Try terminal mode**: `python webcam_stereo_vision.py --terminal`
