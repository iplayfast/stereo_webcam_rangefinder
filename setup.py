#!/usr/bin/env python3
"""
Setup script for Webcam Stereo Vision project
=============================================

This script helps set up the environment and test the system.
"""

import subprocess
import sys
import os
import cv2

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("   Try running manually: pip install -r requirements.txt")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def test_opencv():
    """Test OpenCV installation."""
    print("\n🔍 Testing OpenCV...")
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        return True
    except ImportError:
        print("❌ OpenCV not found")
        print("   Try: pip install opencv-python")
        return False

def test_yolo():
    """Test YOLO installation."""
    print("\n🎯 Testing YOLO...")
    try:
        from ultralytics import YOLO
        print("✅ YOLO (ultralytics) is available")
        return True
    except ImportError:
        print("❌ YOLO not found")
        print("   Try: pip install ultralytics")
        return False

def find_cameras():
    """Find available cameras."""
    print("\n📷 Scanning for cameras...")
    cameras = []
    
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cameras.append(i)
                print(f"   ✅ Camera {i}: Available ({frame.shape[1]}x{frame.shape[0]})")
            cap.release()
    
    if len(cameras) >= 2:
        print(f"✅ Found {len(cameras)} cameras - stereo vision possible!")
        return True
    elif len(cameras) == 1:
        print("⚠️  Only 1 camera found - stereo vision requires 2 cameras")
        return False
    else:
        print("❌ No cameras found")
        return False

def test_gui():
    """Test OpenCV GUI capabilities."""
    print("\n🖥️  Testing GUI capabilities...")
    try:
        import cv2
        import numpy as np
        
        # Create test image
        test_img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(test_img, "GUI Test - Press any key", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('GUI Test', test_img)
        print("   A test window should appear - press any key to continue...")
        key = cv2.waitKey(5000)  # Wait 5 seconds
        cv2.destroyAllWindows()
        
        if key != -1:
            print("✅ GUI is working")
            return True
        else:
            print("⚠️  GUI test timed out - may be running headless")
            return False
            
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    dirs = ['output', 'calibration']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"   ✅ Created {dir_name}/")
        else:
            print(f"   ✅ {dir_name}/ already exists")

def download_yolo_model():
    """Download YOLO model if needed."""
    print("\n⬇️  Checking YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("✅ YOLO model ready")
        return True
    except Exception as e:
        print(f"❌ Failed to download YOLO model: {e}")
        return False

def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Position two webcams 6-10cm apart, facing the same direction")
    print("2. Run the system:")
    print("   python webcam_stereo_vision.py")
    print("3. When prompted, measure distance to first detected object")
    print("4. Enjoy real-time stereo depth detection!")
    print()
    print("Helpful commands:")
    print("  python webcam_stereo_vision.py --help")
    print("  python webcam_stereo_vision.py --confidence 0.5")
    print("  python webcam_stereo_vision.py --debug")
    print()
    print("For help, see README.md")
    print("="*60)

def main():
    """Main setup function."""
    print("WEBCAM STEREO VISION SETUP")
    print("="*30)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install dependencies
    if success and not install_dependencies():
        success = False
    
    # Test components
    if success:
        test_opencv()
        test_yolo()
        camera_ok = find_cameras()
        gui_ok = test_gui()
        
        if not camera_ok:
            print("\n⚠️  WARNING: Insufficient cameras for stereo vision")
            print("   Connect at least 2 USB webcams to continue")
        
        # Create directories
        create_directories()
        
        # Download YOLO model
        download_yolo_model()
    
    if success:
        print_next_steps()
    else:
        print("\n❌ Setup failed - please resolve the issues above")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())