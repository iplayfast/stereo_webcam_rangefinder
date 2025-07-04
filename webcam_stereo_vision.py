#!/usr/bin/env python3
"""
Webcam Stereo Vision with Object Detection
==========================================

A real-time stereo vision system that:
- Uses two webcams for depth estimation
- Detects objects using YOLO
- Calculates distance by triangulating the center of YOLO bounding boxes.
- Can use either terminal or pygame GUI interface

USAGE:
    python webcam_stereo_vision.py                    # Use GUI (default)
    python webcam_stereo_vision.py --terminal         # Use terminal mode
    python webcam_stereo_vision.py --yolo-model yolov8s.pt  # Use different model
"""

import cv2
import numpy as np
import time
import threading
import signal
import sys
import os
import argparse
from collections import defaultdict

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install with: pip install ultralytics")

# Try to import the GUI module
try:
    from stereo_gui import StereoVisionGUI
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

def check_for_input():
    """Check if user has typed something without blocking."""
    if sys.platform != 'win32':
        # Unix/Linux/Mac
        import select
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return True
    else:
        # Windows
        import msvcrt
        if msvcrt.kbhit():
            return True
    return False

class StereoVisionSystem:
    def __init__(self, config=None, use_gui=True, dual_terminal=False):
        self.running = True
        self.use_gui = use_gui and GUI_AVAILABLE
        self.dual_terminal = dual_terminal
        self.gui_working = False
        
        # Configuration
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)

        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        print("Initializing YOLO-based Triangulation System...")
        
        # Dual terminal communication files
        if self.dual_terminal:
            self.command_file = '/tmp/stereo_commands.json'
            self.status_file = '/tmp/stereo_status.json'
            self.last_command_time = 0
            print("📡 Dual-terminal mode enabled")
            print("   Start controller: python stereo_controller.py")
        
        # Initialize GUI or test OpenCV GUI
        if self.use_gui and not self.dual_terminal:
            try:
                self.gui = StereoVisionGUI()
                self.gui.set_system_reference(self)
                print("✓ Pygame GUI initialized")
                self.gui_working = True
            except Exception as e:
                print(f"✗ Failed to initialize Pygame GUI: {e}")
                print("  💡 Recommendation: Use terminal mode for better compatibility")
                print("     Command: python webcam_stereo_vision.py --terminal")
                print("  🔄 Automatically falling back to terminal mode...")
                self.use_gui = False
                self.test_opencv_gui()
        else:
            self.test_opencv_gui()

        # Threading and Frame Management
        self.frame_lock = threading.Lock()
        self.latest_frame_left = None
        self.latest_frame_right = None
        self.display_frame_left = None
        self.display_frame_right = None
        
        # Trigger for forcing recalibration
        self.force_calibration_trigger = False
        
        # Camera orientation (auto-detected during calibration)
        self.cameras_swapped = False
        
        # Object tracking for persistent display
        self.tracked_objects = {}  # {class_name: {'distance': float, 'last_seen': time, 'count': int}}
        self.detection_count = 0

        # Load YOLO model
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                print("Loading YOLO model...")
                yolo_model_path = self.config.get('yolo_model', 'yolov8n.pt')
                success, message = self.load_yolo_model(yolo_model_path)
                if success:
                    print("✓ YOLO loaded successfully")
                else:
                    print(f"✗ {message}")
            except Exception as e:
                print(f"✗ YOLO loading failed: {e}")

        # Calibration system
        self.is_calibrated = False
        self.calibration_factor = None
        self.calibration_file = self.config.get('calibration_file', 'calibration/stereo_calibration_yolo.txt')
        self.load_calibration()
        
        # Initialize status for dual terminal mode
        if self.dual_terminal:
            self.update_status_file()
        
        # Show initial startup message
        if not self.is_calibrated and not self.use_gui and not self.dual_terminal:
            print("\n⚠️  SYSTEM NOT CALIBRATED")
            print("   Type any key then 'c' when ready to calibrate with a known object")
            print("   Calibration requires exactly ONE object visible in BOTH cameras\n")
    
    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}. Shutting down...")
        self.running = False

    def test_opencv_gui(self):
        print("Testing OpenCV GUI capabilities...")
        try:
            cv2.imshow('GUI_TEST', np.zeros((50, 50), dtype=np.uint8))
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            self.gui_working = True
            print("✓ OpenCV GUI is working")
        except Exception:
            self.gui_working = False
            print("✗ OpenCV GUI not working. Will run in text-only mode.")

    def load_yolo_model(self, model_path):
        """Load a new YOLO model."""
        if not YOLO_AVAILABLE:
            return False, "YOLO not available. Install with: pip install ultralytics"
        
        try:
            new_model = YOLO(model_path)
            self.yolo_model = new_model
            self.config['yolo_model'] = model_path
            
            # Update status for dual terminal
            if self.dual_terminal:
                self.send_status_message(f"Model changed to {model_path}", model_changed=True)
            
            return True, f"Successfully loaded {model_path}"
        except Exception as e:
            return False, f"Failed to load {model_path}: {str(e)}"

    def update_status_file(self):
        """Update status file for dual terminal mode."""
        if not self.dual_terminal:
            return
            
        try:
            status = {
                'calibrated': self.is_calibrated,
                'model': self.config.get('yolo_model', 'unknown'),
                'confidence': self.confidence_threshold,
                'tracked_objects': len(self.tracked_objects),
                'timestamp': time.time()
            }
            
            if self.is_calibrated:
                status['calibration_factor'] = self.calibration_factor
                status['cameras_swapped'] = self.cameras_swapped
            
            with open(self.status_file, 'w') as f:
                import json
                json.dump(status, f)
        except:
            pass  # Ignore file errors

    def send_status_message(self, message, **kwargs):
        """Send a status message to the controller."""
        if not self.dual_terminal:
            return
            
        try:
            status = {
                'message': message,
                'timestamp': time.time(),
                **kwargs
            }
            
            with open(self.status_file, 'w') as f:
                import json
                json.dump(status, f)
        except:
            pass

    def check_commands(self):
        """Check for commands from controller in dual terminal mode."""
        if not self.dual_terminal:
            return
            
        try:
            if not os.path.exists(self.command_file):
                return
                
            # Get file modification time
            file_time = os.path.getmtime(self.command_file)
            if file_time <= self.last_command_time:
                return  # No new command
                
            self.last_command_time = file_time
            
            # Read command
            with open(self.command_file, 'r') as f:
                import json
                cmd_data = json.load(f)
            
            command = cmd_data.get('command')
            params = cmd_data.get('params', {})
            
            # Process command
            if command == 'quit':
                self.running = False
                
            elif command == 'calibrate':
                self.force_calibration_trigger = True
                self.send_status_message("Calibration mode activated")
                
            elif command == 'confidence_change':
                delta = params.get('delta', 0)
                old_conf = self.confidence_threshold
                self.confidence_threshold = max(0.1, min(0.9, old_conf + delta))
                self.send_status_message(f"Confidence: {old_conf:.2f} → {self.confidence_threshold:.2f}", 
                                       confidence_changed=True, confidence=self.confidence_threshold)
                
            elif command == 'set_confidence':
                old_conf = self.confidence_threshold
                self.confidence_threshold = params.get('value', old_conf)
                self.send_status_message(f"Confidence set to: {self.confidence_threshold:.2f}",
                                       confidence_changed=True, confidence=self.confidence_threshold)
                
            elif command == 'reset_tracking':
                self.tracked_objects.clear()
                self.detection_count = 0
                self.send_status_message("Object tracking reset")
                
            elif command == 'change_model':
                model = params.get('model')
                if model:
                    success, message = self.load_yolo_model(model)
                    if success:
                        self.tracked_objects.clear()
                        self.detection_count = 0
                        self.send_status_message(f"Model changed to {model}", 
                                               model_changed=True, current_model=model)
                    else:
                        self.send_status_message(f"Model change failed: {message}")
            
            # Delete command file after processing
            try:
                os.remove(self.command_file)
            except:
                pass
                
        except Exception as e:
            pass  # Ignore command processing errors

    def find_cameras(self):
        print("Scanning for cameras...")
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cameras.append(i)
                    print(f"  ✓ Camera {i}: Available")
                cap.release()
        print(f"Found cameras: {cameras}")
        return cameras

    def load_calibration(self):
        if not os.path.exists(self.calibration_file):
             print("✗ No calibration file found. System must be calibrated on first run.")
             return
        try:
            with open(self.calibration_file, 'r') as f:
                lines = f.readlines()
                self.calibration_factor = float(lines[0].strip())
                # Load camera orientation if available (backward compatibility)
                if len(lines) > 1:
                    self.cameras_swapped = lines[1].strip().lower() == 'true'
                else:
                    self.cameras_swapped = False
                    
                self.is_calibrated = True
                orientation_msg = " (cameras swapped)" if self.cameras_swapped else ""
                print(f"✓ Loaded calibration factor: {self.calibration_factor:.2f}{orientation_msg}")
        except Exception as e:
            print(f"✗ Error loading calibration: {e}")

    def save_calibration(self):
        try:
            os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
            with open(self.calibration_file, 'w') as f:
                f.write(f"{self.calibration_factor}\n")
                f.write(f"{self.cameras_swapped}\n")
            print(f"✓ Calibration saved to {self.calibration_file}")
        except Exception as e:
            print(f"✗ Error saving calibration: {e}")

    def calculate_disparity(self, left_det, right_det):
        """Calculate disparity - simple and straightforward."""
        return left_det['center'][0] - right_det['center'][0]

    def disparity_to_distance(self, disparity):
        if self.calibration_factor is None or disparity <= 0:
            return -1.0
        return self.calibration_factor / disparity

    def clear_screen(self):
        """Clear the terminal screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

    def update_tracked_objects(self, matched_detections):
        """Update the persistent object tracking."""
        import time
        current_time = time.time()
        self.detection_count += 1
        
        # Update objects that were detected this cycle
        detected_classes = set()
        for detection in matched_detections:
            class_name = detection['class_name']
            detected_classes.add(class_name)
            
            if class_name not in self.tracked_objects:
                self.tracked_objects[class_name] = {
                    'distance': detection['distance_mm'],
                    'last_seen': current_time,
                    'count': 1,
                    'first_seen': self.detection_count
                }
            else:
                self.tracked_objects[class_name]['distance'] = detection['distance_mm']
                self.tracked_objects[class_name]['last_seen'] = current_time
                self.tracked_objects[class_name]['count'] += 1
        
        # Mark objects not detected this cycle (but keep them for display)
        for class_name in list(self.tracked_objects.keys()):
            if class_name not in detected_classes:
                # Keep the object but mark distance as unknown if not seen recently
                time_since_seen = current_time - self.tracked_objects[class_name]['last_seen']
                if time_since_seen > 2.0:  # If not seen for 2 seconds, show '?'
                    self.tracked_objects[class_name]['distance'] = None

    def get_sorted_objects(self):
        """Get objects sorted by most recently/frequently detected."""
        import time
        current_time = time.time()
        
        # Sort by: 1) Recently seen (within 5 seconds), 2) Detection count, 3) First seen order
        def sort_key(item):
            class_name, data = item
            time_since_seen = current_time - data['last_seen']
            recently_seen = time_since_seen < 5.0
            return (
                not recently_seen,  # Recently seen objects first
                -data['count'],     # Then by detection count (descending)
                data['first_seen']  # Then by order first detected
            )
        
        return sorted(self.tracked_objects.items(), key=sort_key)

    def draw_static_display(self, status_message=""):
        """Draw the static display with current object tracking."""
        self.clear_screen()
        
        print("=" * 80)
        print("🎯 STEREO VISION OBJECT TRACKING")
        print("=" * 80)
        
        # System status
        if self.is_calibrated:
            orientation = " (cameras swapped)" if self.cameras_swapped else ""
            print(f"✓ CALIBRATED - Factor: {self.calibration_factor:.1f}{orientation}")
        else:
            print("⚠️  NOT CALIBRATED - Type any key then 'c' to calibrate")
        
        # YOLO model info
        current_model = self.config.get('yolo_model', 'yolov8n.pt')
        model_status = "✓ Loaded" if self.yolo_model else "✗ Failed"
        print(f"🤖 Model: {current_model} ({model_status})")
        print(f"📊 Confidence: {self.confidence_threshold:.1f}")
        print("-" * 80)
        
        # Object tracking table
        sorted_objects = self.get_sorted_objects()
        if sorted_objects:
            print("📦 DETECTED OBJECTS:")
            print(f"{'Object':<15} {'Distance':<12} {'Count':<8} {'Status':<15}")
            print("-" * 60)
            
            for class_name, data in sorted_objects:
                if data['distance'] is not None:
                    distance_str = f"{data['distance']/1000:.2f} m"
                    status = "✓ Visible"
                else:
                    distance_str = "?"
                    status = "○ Lost"
                
                print(f"{class_name:<15} {distance_str:<12} {data['count']:<8} {status:<15}")
        else:
            print("📦 No objects detected yet...")
            if self.is_calibrated:
                if self.yolo_model:
                    print("   Try typing any key then 'l' to lower confidence threshold")
                else:
                    print("   YOLO model failed to load - try typing any key then 'm' to change model")
        
        print("-" * 80)
        
        # Status message
        if status_message:
            print(f"📢 {status_message}")
            print("-" * 80)
        
        # Controls
        print("🎮 CONTROLS:")
        print("  Type any key to pause, then enter commands:")
        print("  'c' = Calibrate    'l' = Lower confidence    'h' = Higher confidence")
        print("  'm' = Change model 'd' = Debug info         's' = Show settings      'q' = Quit")
        print("=" * 80)

    def change_yolo_model(self):
        """Interactive YOLO model changing interface."""
        self.clear_screen()
        print("=" * 80)
        print("🤖 CHANGE YOLO MODEL")
        print("=" * 80)
        
        # Show available models
        available_models = [
            ('yolov8n.pt', 'Nano - Fastest, least accurate (~6MB)'),
            ('yolov8s.pt', 'Small - Good balance (~22MB)'),
            ('yolov8m.pt', 'Medium - More accurate (~52MB)'),
            ('yolov8l.pt', 'Large - Very accurate (~87MB)'),
            ('yolov8x.pt', 'Extra Large - Most accurate (~136MB)'),
        ]
        
        current_model = self.config.get('yolo_model', 'yolov8n.pt')
        print(f"Current model: {current_model}")
        print("-" * 80)
        print("Available models:")
        
        for i, (model, description) in enumerate(available_models, 1):
            marker = "→" if model == current_model else " "
            print(f"{marker} {i}. {model:<12} - {description}")
        
        print("-" * 80)
        print("Options:")
        print("  Enter number (1-5) to select a model")
        print("  Enter custom path (e.g., '/path/to/model.pt')")
        print("  Press Enter to cancel")
        print("-" * 80)
        
        try:
            choice = input("Your choice: ").strip()
            
            if not choice:
                return "Model change cancelled"
            
            # Check if it's a number (predefined model)
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_models):
                    model_path = available_models[choice_num - 1][0]
                else:
                    return "Invalid choice number"
            else:
                # Custom path
                model_path = choice
            
            print(f"\nAttempting to load: {model_path}")
            print("This may take a moment for first-time downloads...")
            
            success, message = self.load_yolo_model(model_path)
            
            if success:
                # Clear tracking data since different models might detect different objects
                self.tracked_objects.clear()
                self.detection_count = 0
            
            input(f"\n{message}\nPress Enter to continue...")
            return ""
            
        except Exception as e:
            input(f"\nError: {e}\nPress Enter to continue...")
            return ""

    def detect_objects(self, frame, debug=False):
        detections = []
        if self.yolo_model is None: 
            return detections
        
        try:
            results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_name = self.yolo_model.names[int(box.cls[0])]
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'bbox': (x1, y1, x2, y2)
                    })
        except Exception as e:
            pass  # Silently ignore YOLO errors to keep display clean
        
        return detections

    def perform_calibration(self, left_det, right_det, cameras_swapped=False):
        """Handle calibration process with user input."""
        disparity = self.calculate_disparity(left_det, right_det)
        
        # Clear screen and show calibration interface
        self.clear_screen()
        
        print("=" * 80)
        print("🎯 CALIBRATION MODE")
        print("=" * 80)
        print(f"Object detected: {left_det['class_name']}")
        print(f"Left center: {left_det['center']}")
        print(f"Right center: {right_det['center']}")
        print(f"Disparity: {disparity:.2f} pixels")
        if cameras_swapped:
            print("🔄 Cameras auto-swapped due to negative disparity")
        print("-" * 80)
        
        if disparity <= 0:
            print("⚠️  Invalid disparity (<=0). Check camera positioning.")
            input("Press Enter to continue...")
            self.force_calibration_trigger = False
            return ""
            
        try:
            real_distance_mm_str = input("Enter REAL distance to object in mm (or press Enter to skip): ")
            
            if real_distance_mm_str.strip():
                real_distance_mm = float(real_distance_mm_str)
                if real_distance_mm <= 0:
                    print("⚠️  Distance must be positive")
                    input("Press Enter to continue...")
                    self.force_calibration_trigger = False
                    return ""
                    
                self.calibration_factor = real_distance_mm * disparity
                self.cameras_swapped = cameras_swapped
                self.is_calibrated = True
                self.save_calibration()
                
                print(f"✓ Calibration successful! Factor: {self.calibration_factor:.2f}")
                print("System is now ready for distance measurement.")
                input("Press Enter to continue...")
                
                return ""
            else:
                print("Calibration skipped.")
                input("Press Enter to continue...")
                return ""
                
        except (ValueError, TypeError) as e:
            print(f"⚠️  Input error: {e}")
            input("Press Enter to continue...")
            return ""
        except Exception as e:
            print(f"⚠️  Calibration error: {e}")
            input("Press Enter to continue...")
            return ""
        finally:
            self.force_calibration_trigger = False

    def keyboard_input_thread(self):
        """Keyboard input handler for terminal controls."""
        while self.running:
            try:
                user_input = input().strip().lower()
                if user_input == 'q':
                    self.running = False
                    break
                elif user_input == 'c':
                    self.force_calibration_trigger = True
                elif user_input == 'l':
                    old_conf = self.confidence_threshold
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
                elif user_input == 'h':
                    old_conf = self.confidence_threshold
                    self.confidence_threshold = min(0.9, self.confidence_threshold + 0.1)
                elif user_input == 'm':
                    # Change YOLO model
                    self.change_yolo_model()
                elif user_input == 'd':
                    # Show debug info in a separate screen
                    self.clear_screen()
                    print("=" * 80)
                    print("🔍 DEBUG INFORMATION")
                    print("=" * 80)
                    print(f"YOLO model: {self.config.get('yolo_model', 'yolov8n.pt')}")
                    print(f"YOLO loaded: {self.yolo_model is not None}")
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                    print(f"Calibrated: {self.is_calibrated}")
                    if self.is_calibrated:
                        print(f"Calibration factor: {self.calibration_factor:.2f}")
                        print(f"Cameras swapped: {self.cameras_swapped}")
                    print(f"GUI working: {self.gui_working}")
                    print(f"Total tracked objects: {len(self.tracked_objects)}")
                    
                    # Show object classes that YOLO can detect
                    if self.yolo_model:
                        print(f"YOLO classes available: {len(self.yolo_model.names)}")
                        print("Sample classes: " + ", ".join(list(self.yolo_model.names.values())[:10]) + "...")
                    
                    print("-" * 80)
                    input("Press Enter to return...")
                elif user_input == 's':
                    # Show settings in a separate screen
                    self.clear_screen()
                    print("=" * 80)
                    print("⚙️  CURRENT SETTINGS")
                    print("=" * 80)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                    print(f"YOLO model: {self.config.get('yolo_model', 'yolov8n.pt')}")
                    print(f"YOLO status: {'Loaded' if self.yolo_model else 'Failed'}")
                    print(f"Calibration file: {self.calibration_file}")
                    print(f"System calibrated: {self.is_calibrated}")
                    if self.is_calibrated:
                        print(f"Calibration factor: {self.calibration_factor:.2f}")
                        orientation = "Swapped" if self.cameras_swapped else "Normal"
                        print(f"Camera orientation: {orientation}")
                    print("-" * 80)
                    input("Press Enter to return...")
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break

    def match_objects(self, detections_left, detections_right):
        """Match objects between left and right cameras using Y-coordinate proximity."""
        grouped_right = defaultdict(list)
        for det in detections_right: 
            grouped_right[det['class_name']].append(det)
        
        matched_detections = []
        
        for det_l in detections_left:
            best_match = None
            smallest_y_diff = float('inf')
            class_name = det_l['class_name']

            if class_name in grouped_right:
                # Find the corresponding right-camera object with the closest Y-coordinate
                for det_r in grouped_right[class_name]:
                    y_diff = abs(det_l['center'][1] - det_r['center'][1])
                    # Assume objects on the same horizontal plane are matches
                    if y_diff < smallest_y_diff and y_diff < 50: # 50-pixel vertical tolerance
                        smallest_y_diff = y_diff
                        best_match = det_r
            
            if best_match:
                # Calculate disparity based on camera orientation
                if self.cameras_swapped:
                    # If cameras were swapped during calibration, swap them for distance calculation too
                    disparity = best_match['center'][0] - det_l['center'][0]
                else:
                    # Normal orientation
                    disparity = det_l['center'][0] - best_match['center'][0]
                    
                distance_mm = self.disparity_to_distance(disparity)

                if distance_mm > 0:
                    matched_detections.append({
                        "class_name": class_name,
                        "distance_mm": distance_mm,
                        "left_bbox": det_l['bbox'],
                        "cam1_xy": det_l['center'],
                        "cam2_xy": best_match['center']
                    })
        
        return matched_detections

    def _processing_loop(self):
        """Main processing loop using YOLO-based triangulation with non-blocking input."""
        import time
        status_message = ""
        last_display_update = time.time()
        last_status_update = time.time()
        
        while self.running:
            # Check for commands in dual terminal mode
            if self.dual_terminal:
                self.check_commands()
            
            # Check for user input first - PAUSE UPDATES WHEN USER IS TYPING
            if check_for_input():
                # User is typing - pause updates and handle input
                print(f"\n{'='*60}")
                print("🎮 PAUSED FOR INPUT")
                print("Commands: c=calibrate, l=lower conf, h=higher conf, d=debug, s=settings, q=quit")
                
                try:
                    command = input("Enter command: ").strip().lower()
                    
                    if command == 'q':
                        self.running = False
                    elif command == 'c':
                        self.force_calibration_trigger = True
                        print("✓ Calibration mode activated")
                    elif command == 'l':
                        old_conf = self.confidence_threshold
                        self.confidence_threshold = max(0.1, old_conf - 0.1)
                        print(f"✓ Confidence: {old_conf:.2f} → {self.confidence_threshold:.2f}")
                    elif command == 'h':
                        old_conf = self.confidence_threshold
                        self.confidence_threshold = min(0.9, old_conf + 0.1)
                        print(f"✓ Confidence: {old_conf:.2f} → {self.confidence_threshold:.2f}")
                    elif command == 'd':
                        # DEBUG: Show current detections
                        frame_left, frame_right = None, None
                        with self.frame_lock:
                            if self.latest_frame_left is not None:
                                frame_left = self.latest_frame_left.copy()
                                frame_right = self.latest_frame_right.copy()
                        
                        if frame_left is not None:
                            detections_left = self.detect_objects(frame_left)
                            detections_right = self.detect_objects(frame_right)
                            
                            print(f"\n📊 CURRENT DETECTIONS:")
                            print(f"LEFT CAMERA ({len(detections_left)} objects):")
                            for i, det in enumerate(detections_left):
                                print(f"  {i+1}. {det['class_name']} (conf: {det['confidence']:.2f}) at {det['center']}")
                            
                            print(f"\nRIGHT CAMERA ({len(detections_right)} objects):")
                            for i, det in enumerate(detections_right):
                                print(f"  {i+1}. {det['class_name']} (conf: {det['confidence']:.2f}) at {det['center']}")
                            
                            # Show why no matches
                            if detections_left and detections_right:
                                print(f"\n🔍 MATCHING ANALYSIS:")
                                for det_l in detections_left:
                                    for det_r in detections_right:
                                        if det_l['class_name'] == det_r['class_name']:
                                            y_diff = abs(det_l['center'][1] - det_r['center'][1])
                                            x_disparity = det_l['center'][0] - det_r['center'][0]
                                            
                                            print(f"  {det_l['class_name']}: Y-diff={y_diff:.1f} (max=50), X-disp={x_disparity:.1f}")
                                            if y_diff < 50 and x_disparity > 0:
                                                print(f"    ✓ This should match!")
                                            elif y_diff >= 50:
                                                print(f"    ✗ Y-difference too large")
                                            elif x_disparity <= 0:
                                                print(f"    ✗ Negative/zero disparity (check camera order)")
                            elif detections_left and not detections_right:
                                print("🔍 ISSUE: Objects only in LEFT camera")
                            elif detections_right and not detections_left:
                                print("🔍 ISSUE: Objects only in RIGHT camera")
                            else:
                                print("🔍 ISSUE: No objects detected in either camera")
                    
                    elif command == 's':
                        print(f"\n⚙️  CURRENT SETTINGS:")
                        print(f"  Confidence: {self.confidence_threshold:.2f}")
                        print(f"  YOLO Model: {self.config.get('yolo_model', 'unknown')}")
                        print(f"  Calibrated: {self.is_calibrated}")
                        print(f"  Tracked Objects: {len(self.tracked_objects)}")
                    elif command == 'r':
                        self.tracked_objects.clear()
                        self.detection_count = 0
                        print("✓ Object tracking reset")
                    elif command == 'm':
                        print("Model change - use 'm' during normal display mode")
                    elif command == '':
                        pass  # Empty command
                    else:
                        print(f"Unknown command: '{command}' - Available: c, l, h, d, s, r, q")
                        
                    input("\nPress Enter to resume...")
                    
                except KeyboardInterrupt:
                    print("\nResuming...")
                
                print("▶️  RESUMING DISPLAY...\n")
                # Force immediate display update
                last_display_update = 0

            # Acquire frames safely
            frame_left, frame_right = None, None
            with self.frame_lock:
                if self.latest_frame_left is not None:
                    frame_left = self.latest_frame_left.copy()
                    frame_right = self.latest_frame_right.copy()

            if frame_left is None:
                time.sleep(0.01)
                continue

            # Determine if we're in calibration mode
            in_calibration_mode = not self.is_calibrated or self.force_calibration_trigger
            
            # Perform object detection
            detections_left = self.detect_objects(frame_left)
            detections_right = self.detect_objects(frame_right)

            matched_detections = []
            
            # Handle calibration mode
            if in_calibration_mode:
                # Find a single, unambiguous object for calibration
                calibration_candidate = None
                grouped_right = defaultdict(list)
                for det in detections_right: 
                    grouped_right[det['class_name']].append(det)
                
                # Look for objects that appear exactly once in each camera
                for det_l in detections_left:
                    class_name = det_l['class_name']
                    left_count = len([d for d in detections_left if d['class_name'] == class_name])
                    right_count = len(grouped_right[class_name])
                    
                    if left_count == 1 and right_count == 1:
                        calibration_candidate = (det_l, grouped_right[class_name][0])
                        break

                if calibration_candidate:
                    left_det, right_det = calibration_candidate
                    raw_disparity = left_det['center'][0] - right_det['center'][0]
                    
                    # Check if cameras are swapped (negative disparity)
                    if raw_disparity < 0:
                        status_message = f"🔧 Ready to calibrate {left_det['class_name']} (cameras will be auto-swapped)"
                        # Only auto-calibrate in single terminal mode
                        if not self.use_gui and not self.dual_terminal:
                            result = self.perform_calibration(right_det, left_det, cameras_swapped=True)
                            status_message = result if result else ""
                    else:
                        status_message = f"🔧 Ready to calibrate {left_det['class_name']} (normal orientation)"
                        # Only auto-calibrate in single terminal mode
                        if not self.use_gui and not self.dual_terminal:
                            result = self.perform_calibration(left_det, right_det, cameras_swapped=False)
                            status_message = result if result else ""
                        
                elif self.force_calibration_trigger:
                    if not detections_left and not detections_right:
                        status_message = "🔧 Calibration mode: No objects detected"
                    elif not detections_left:
                        status_message = "🔧 Calibration mode: No objects in left camera"
                    elif not detections_right:
                        status_message = "🔧 Calibration mode: No objects in right camera"  
                    else:
                        status_message = "🔧 Calibration mode: Need exactly ONE object type in BOTH cameras"
                else:
                    if self.dual_terminal:
                        status_message = "🔧 Calibration mode: Use controller to start calibration"
                    else:
                        status_message = "🔧 Calibration mode: Type any key then 'c' to start"

            # Perform object matching and distance calculation
            if self.is_calibrated:
                matched_detections = self.match_objects(detections_left, detections_right)
                self.update_tracked_objects(matched_detections)
                
                if not in_calibration_mode:
                    status_message = ""  # Clear status when not calibrating

            # Update display and status
            current_time = time.time()
            
            # Update display (only for terminal mode)
            if current_time - last_display_update > 0.2:
                # Check if we've switched from GUI to terminal mode
                if not self.use_gui and hasattr(self, 'gui') and getattr(self.gui, 'gui_failed', False):
                    # We've switched to terminal mode, start showing terminal display
                    self.draw_static_display(status_message)
                elif not self.use_gui:
                    # We're in terminal mode from the start
                    self.draw_static_display(status_message)
                last_display_update = current_time
            
            # Update status file for dual terminal mode
            if self.dual_terminal and current_time - last_status_update > 1.0:
                self.update_status_file()
                last_status_update = current_time

            # Update GUI display frames (for both GUI and OpenCV windows)
            if self.gui_working:
                display_l, display_r = frame_left.copy(), frame_right.copy()
                
                # Draw all detections in blue
                for det in detections_left: 
                    cv2.rectangle(display_l, det['bbox'][:2], det['bbox'][2:], (255, 100, 0), 2)
                for det in detections_right: 
                    cv2.rectangle(display_r, det['bbox'][:2], det['bbox'][2:], (255, 100, 0), 2)
                
                # Draw matched detections in green with distance
                for res in matched_detections:
                    x1, y1, x2, y2 = res['left_bbox']
                    cv2.rectangle(display_l, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    dist_text = f"{res['distance_mm']/1000:.2f} m"
                    cv2.putText(display_l, dist_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                with self.frame_lock:
                    self.display_frame_left = display_l
                    self.display_frame_right = display_r
            
            time.sleep(0.05)  # Faster processing for smoother updates

    def run(self):
        """Main execution loop."""
        cameras = self.find_cameras()
        if len(cameras) < 2: 
            print("✗ Need at least 2 cameras for stereo vision")
            return

        cam_left, cam_right = cv2.VideoCapture(cameras[0]), cv2.VideoCapture(cameras[1])
        if not (cam_left.isOpened() and cam_right.isOpened()): 
            print("✗ Failed to open cameras")
            return

        # Camera warm-up
        print("Warming up cameras...")
        for i in range(60):
            ret_left, _ = cam_left.read()
            ret_right, _ = cam_right.read()
            if ret_left and ret_right and i > 15: 
                break
            time.sleep(0.02)
        print("✓ Cameras are ready.")

        # Start processing thread
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()
        
        # Start appropriate input handler
        gui_thread = None
        input_thread = None
        
        if self.use_gui:
            # Start GUI in separate thread
            gui_thread = threading.Thread(target=self.gui.run, daemon=True)
            gui_thread.start()
            print("🎮 GUI thread started...")
        else:
            # Start terminal input thread
            input_thread = threading.Thread(target=self.keyboard_input_thread, daemon=True)
            input_thread.start()
            # Initialize the static display
            self.draw_static_display("System starting up...")
            time.sleep(1)

        # Main frame capture loop
        try:
            frame_count = 0
            last_interface_check = time.time()
            
            while self.running:
                ret_left, frame_left = cam_left.read()
                ret_right, frame_right = cam_right.read()
                if not (ret_left and ret_right): 
                    continue

                with self.frame_lock:
                    self.latest_frame_left = frame_left
                    self.latest_frame_right = frame_right
                
                # Check if we need to switch from GUI to terminal mode
                current_time = time.time()
                if self.use_gui and current_time - last_interface_check > 2.0:  # Check every 2 seconds
                    if hasattr(self.gui, 'gui_failed') and self.gui.gui_failed:
                        print("🔄 GUI has failed, switching to terminal mode...")
                        self.use_gui = False
                        # Start terminal interface
                        input_thread = threading.Thread(target=self.keyboard_input_thread, daemon=True)
                        input_thread.start()
                        print("✓ Terminal interface activated")
                    last_interface_check = current_time
                
                # Handle display based on current mode
                if self.use_gui:
                    # GUI mode - frames are handled by GUI thread
                    pass
                elif self.gui_working:
                    # Terminal mode with OpenCV windows
                    if self.display_frame_left is not None and self.display_frame_right is not None:
                        cv2.imshow('Left Camera', self.display_frame_left)
                        cv2.imshow('Right Camera', self.display_frame_right)
                    else:
                        cv2.imshow('Left Camera', frame_left)
                        cv2.imshow('Right Camera', frame_right)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        self.running = False
                
                frame_count += 1

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
            self.running = False
        
        # Graceful shutdown
        print("Cleaning up...")
        self.running = False
        
        # Wait for threads to finish
        processing_thread.join(timeout=1.0)
        if input_thread:
            input_thread.join(timeout=0.5)
        if gui_thread:
            gui_thread.join(timeout=1.0)

        # Release resources
        cam_left.release()
        cam_right.release()
        if self.gui_working and not self.use_gui: 
            cv2.destroyAllWindows()
        if self.use_gui and hasattr(self, 'gui'):
            self.gui.cleanup()
        print("✓ Cleanup complete")

def main():
    parser = argparse.ArgumentParser(description='Webcam Stereo Vision with YOLO-based Triangulation')
    parser.add_argument('--confidence', type=float, default=0.3, 
                       help='YOLO confidence threshold (default: 0.3, lower = more detections)')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', 
                       help='YOLO model path (yolov8n.pt=fastest, yolov8s.pt=balanced, yolov8m.pt=accurate)')
    parser.add_argument('--calibration-file', type=str, default='calibration/stereo_calibration_yolo.txt', 
                       help='Calibration file path')
    parser.add_argument('--gui', action='store_true', default=True,
                       help='Use pygame GUI interface (default)')
    parser.add_argument('--terminal', action='store_true',
                       help='Use terminal interface instead of GUI')
    
    args = parser.parse_args()
    
    # Determine interface mode
    use_gui = args.gui and not args.terminal
    
    # Set environment variables for better pygame compatibility
    if use_gui:
        import os
        # Try to force software rendering if needed
        if not os.environ.get('SDL_VIDEODRIVER'):
            # Let pygame try hardware first, fall back automatically
            pass
    
    if use_gui and not GUI_AVAILABLE:
        print("⚠️  Pygame GUI not available.")
        print("   Install with: pip install pygame")
        print("   Falling back to terminal mode...")
        use_gui = False
    
    # Show interface mode
    interface_mode = "Pygame GUI" if use_gui else "Terminal"
    print(f"Interface mode: {interface_mode}")
    
    # Show YOLO model recommendations
    model_info = {
        'yolov8n.pt': 'Nano - Fastest, least accurate (~6MB)',
        'yolov8s.pt': 'Small - Good balance of speed/accuracy (~22MB)', 
        'yolov8m.pt': 'Medium - More accurate, slower (~52MB)',
        'yolov8l.pt': 'Large - Very accurate, much slower (~87MB)',
        'yolov8x.pt': 'Extra Large - Most accurate, slowest (~136MB)'
    }
    
    model_name = args.yolo_model
    if model_name in model_info:
        print(f"Using YOLO model: {model_name} ({model_info[model_name]})")
    else:
        print(f"Using custom YOLO model: {model_name}")
    
    print(f"Detection confidence threshold: {args.confidence}")
    
    if use_gui:
        print("💡 GUI Controls:")
        print("   - C = Calibrate    L/H = Confidence    M = Change model")
        print("   - R = Reset tracking    Q = Quit")
        print("   📝 Note: If GUI fails to start, it will automatically fall back to terminal mode")
    else:
        print("💡 Terminal controls available:")
        print("   - Type any key to pause, then enter commands")
        print("   - 'd' for debug mode to see detection details")
        print("   - 'l'/'h' to adjust confidence threshold")
    print("")
    
    config = {
        'confidence_threshold': args.confidence,
        'yolo_model': args.yolo_model,
        'calibration_file': args.calibration_file
    }
    
    try:
        system = StereoVisionSystem(config, use_gui=use_gui)
        system.run()
    except KeyboardInterrupt:
        print("\n✓ Application stopped by user")
    except Exception as e:
        print(f"\n✗ Application error: {e}")
        if use_gui:
            print("   💡 Try running with --terminal flag for better compatibility")

if __name__ == "__main__":
    main()