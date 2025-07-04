#!/usr/bin/env python3
"""
Webcam Stereo Vision with Object Detection
==========================================

A real-time stereo vision system that:
- Uses two webcams for depth estimation
- Detects objects using YOLO
- Measures distances to detected objects
- Interactive calibration using real-world measurements
- Temporal buffering for robust stereo matching

Author: AI Assistant
License: MIT
"""

import cv2
import numpy as np
import pickle
import time
import threading
import signal
import sys
import os
import argparse

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install with: pip install ultralytics")

class StereoVisionSystem:
    def __init__(self, config=None):
        self.running = True
        self.gui_working = False
        self.save_counter = 0
        
        # Configuration
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
        self.buffer_time_window = self.config.get('buffer_time_window', 0.5)
        self.max_buffer_size = self.config.get('max_buffer_size', 10)
        
        # Set up signal handlers for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("Initializing Stereo Vision System...")
        
        # Test GUI capabilities
        self.test_gui()
        
        # Initialize stereo matcher with improved parameters
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=-64,           # Allow negative disparity 
            numDisparities=128,         # More disparity levels (must be divisible by 16)
            blockSize=3,                # Smaller block for more detail
            P1=8 * 3 * 3**2,           # Smoothness parameter 1
            P2=32 * 3 * 3**2,          # Smoothness parameter 2 
            disp12MaxDiff=5,           # Max allowed difference in left-right check
            uniquenessRatio=15,        # Margin by which best match must win
            speckleWindowSize=50,      # Maximum speckle size
            speckleRange=16,           # Maximum disparity variation in speckle
            preFilterCap=31,           # Truncate prefiltered pixels
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Load YOLO if available
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                print("Loading YOLO model...")
                yolo_model_path = self.config.get('yolo_model', 'yolov8n.pt')
                self.yolo_model = YOLO(yolo_model_path)
                print("✓ YOLO loaded successfully")
            except Exception as e:
                print(f"✗ YOLO loading failed: {e}")
        
        # Calibration system
        self.is_calibrated = False
        self.calibration_factor = None  # focal_length * baseline product
        self.calibration_unit = None  # User's preferred unit
        self.calibration_unit_factor = 1.0  # Conversion factor from mm to user's unit
        self.calibration_file = self.config.get('calibration_file', 'stereo_calibration_interactive.txt')
        self.pending_calibration = None  # Stores data for calibration
        
        # Debug mode for matching
        self.debug_matching = False
        
        # Adaptive stereo tuning system
        self.adaptive_tuning = True
        self.stereo_parameter_history = []
        self.best_parameters = {
            'numDisparities': 128,
            'blockSize': 3,
            'uniquenessRatio': 15,
            'P1_factor': 8,
            'P2_factor': 32,
            'disp12MaxDiff': 5
        }
        self.tuning_candidates = [
            {'numDisparities': 96, 'blockSize': 3, 'uniquenessRatio': 10, 'P1_factor': 8, 'P2_factor': 32, 'disp12MaxDiff': 2},
            {'numDisparities': 128, 'blockSize': 3, 'uniquenessRatio': 15, 'P1_factor': 8, 'P2_factor': 32, 'disp12MaxDiff': 5},
            {'numDisparities': 160, 'blockSize': 5, 'uniquenessRatio': 20, 'P1_factor': 10, 'P2_factor': 40, 'disp12MaxDiff': 3},
            {'numDisparities': 96, 'blockSize': 7, 'uniquenessRatio': 12, 'P1_factor': 6, 'P2_factor': 24, 'disp12MaxDiff': 8}
        ]
        
        # Temporal detection buffers for stereo matching
        self.left_detection_buffer = []   # Stores recent left camera detections
        self.right_detection_buffer = []  # Stores recent right camera detections
        
        # Load existing calibration if available
        self.load_calibration()
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print(f"\nReceived signal {signum}. Shutting down...")
        self.running = False
    
    def test_gui(self):
        """Test if OpenCV GUI is working."""
        print("Testing OpenCV GUI capabilities...")
        
        try:
            # Create a test window
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.putText(test_img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('GUI_TEST', test_img)
            
            # Test if window shows up
            key = cv2.waitKey(1000)  # Wait 1 second
            cv2.destroyWindow('GUI_TEST')
            
            self.gui_working = True
            print("✓ OpenCV GUI is working")
            
        except Exception as e:
            print(f"✗ OpenCV GUI not working: {e}")
            print("  Running in headless mode - will save images instead")
            self.gui_working = False
    
    def find_cameras(self):
        """Find available cameras."""
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
        """Load calibration from file."""
        try:
            with open(self.calibration_file, 'r') as f:
                lines = f.readlines()
                self.calibration_factor = float(lines[0].strip())
                if len(lines) > 1 and not lines[1].startswith('#'):
                    parts = lines[1].strip().split()
                    if len(parts) >= 2:
                        self.calibration_unit = parts[0]
                        self.calibration_unit_factor = float(parts[1])
                else:
                    # Old format - assume millimeters
                    self.calibration_unit = "mm"
                    self.calibration_unit_factor = 1.0
                
                self.is_calibrated = True
                print(f"✓ Loaded calibration factor: {self.calibration_factor:.1f}")
                print(f"  Unit: {self.calibration_unit} (factor: {self.calibration_unit_factor})")
                print("  System is calibrated and ready for accurate distance measurements")
        except FileNotFoundError:
            print("✗ No calibration file found")
            print("  System will prompt for calibration on first detection")
        except Exception as e:
            print(f"✗ Error loading calibration: {e}")
    
    def save_calibration(self):
        """Save calibration to file."""
        try:
            with open(self.calibration_file, 'w') as f:
                f.write(f"{self.calibration_factor}\n")
                f.write(f"{self.calibration_unit} {self.calibration_unit_factor}\n")
                f.write(f"# Stereo vision calibration factor\n")
                f.write(f"# Unit: {self.calibration_unit}, Factor: {self.calibration_unit_factor}\n")
                f.write(f"# Calibrated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"✓ Calibration saved to {self.calibration_file}")
        except Exception as e:
            print(f"✗ Error saving calibration: {e}")
    
    def request_calibration(self, detection, disparity_value):
        """Request user input for calibration."""
        print("\n" + "="*60)
        print("🎯 CALIBRATION NEEDED")
        print("="*60)
        print(f"Detected: {detection['class_name']} (confidence: {detection['confidence']:.2f})")
        print(f"Measured disparity: {disparity_value:.1f} pixels")
        print()
        print("To calibrate the distance measurements, please measure the actual")
        print("distance to this object and enter it below.")
        print()
        print("You can measure in any unit (mm, cm, inches, feet, etc.)")
        print("Just be consistent - all future measurements will use the same unit.")
        print()
        
        while True:
            try:
                print("How far away is this object? (enter number + unit)")
                print("Examples: '120cm', '1.2m', '48inches', '4feet', '1200mm'")
                user_input = input("Distance: ").strip().lower()
                
                if not user_input:
                    continue
                
                # Parse the input to extract number and unit
                distance_mm, unit = self.parse_distance_input(user_input)
                
                if distance_mm > 0:
                    # Calculate calibration factor (in millimeters internally)
                    # Use absolute disparity since cameras might be swapped
                    abs_disparity = abs(disparity_value)
                    self.calibration_factor = distance_mm * abs_disparity
                    
                    # Store the user's preferred unit for display
                    self.calibration_unit = unit
                    self.calibration_unit_factor = self.get_unit_conversion_factor(unit)
                    
                    self.is_calibrated = True
                    
                    self.save_calibration()
                    
                    print(f"✓ Calibration complete!")
                    print(f"  Distance: {user_input}")
                    print(f"  Calibration factor: {self.calibration_factor:.1f}")
                    print(f"  All future measurements will be in {unit}")
                    print("="*60)
                    break
                else:
                    print("Invalid input. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nCalibration cancelled. Using uncalibrated measurements.")
                return False
            except Exception as e:
                print(f"Error: {e}. Please try again.")
        
        return True
    
    def parse_distance_input(self, user_input):
        """Parse user distance input into millimeters and unit name."""
        import re
        
        # Extract number and unit
        match = re.match(r'([0-9.]+)\s*([a-zA-Z]+)', user_input)
        if not match:
            return -1, ""
        
        number = float(match.group(1))
        unit = match.group(2).lower()
        
        # Convert to millimeters for internal calculations
        conversions = {
            'mm': 1.0,
            'millimeter': 1.0, 'millimeters': 1.0,
            'cm': 10.0,
            'centimeter': 10.0, 'centimeters': 10.0,
            'm': 1000.0, 'meter': 1000.0, 'meters': 1000.0,
            'inch': 25.4, 'inches': 25.4, 'in': 25.4,
            'foot': 304.8, 'feet': 304.8, 'ft': 304.8,
            'yard': 914.4, 'yards': 914.4, 'yd': 914.4
        }
        
        if unit in conversions:
            distance_mm = number * conversions[unit]
            return distance_mm, unit
        else:
            print(f"Unknown unit: {unit}")
            return -1, ""
    
    def get_unit_conversion_factor(self, unit):
        """Get conversion factor from mm to user's unit."""
        conversions = {
            'mm': 1.0, 'millimeter': 1.0, 'millimeters': 1.0,
            'cm': 10.0, 'centimeter': 10.0, 'centimeters': 10.0,
            'm': 1000.0, 'meter': 1000.0, 'meters': 1000.0,
            'inch': 25.4, 'inches': 25.4, 'in': 25.4,
            'foot': 304.8, 'feet': 304.8, 'ft': 304.8,
            'yard': 914.4, 'yards': 914.4, 'yd': 914.4
        }
        return conversions.get(unit.lower(), 1.0)
    
    def compute_disparity(self, img_left, img_right):
        """Compute disparity map from stereo pair."""
        # Convert to grayscale
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        # Preprocessing
        gray_left = cv2.equalizeHist(gray_left)
        gray_right = cv2.equalizeHist(gray_right)
        gray_left = cv2.GaussianBlur(gray_left, (3, 3), 0)
        gray_right = cv2.GaussianBlur(gray_right, (3, 3), 0)
            
        # Compute disparity
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # Basic filtering
        disparity[disparity <= 0] = 0
        disparity = cv2.medianBlur(disparity.astype(np.uint8), 5).astype(np.float32)
        
        return disparity
    
    def disparity_to_distance(self, disparity_value):
        """Convert disparity to distance using calibration."""
        # Use absolute value since cameras might be swapped (negative disparity)
        abs_disparity = abs(disparity_value)
        
        if abs_disparity <= 0:
            return -1
        
        if self.is_calibrated:
            # Use calibrated factor for accurate measurements (result in mm)
            distance_mm = self.calibration_factor / abs_disparity
            # Convert to user's preferred unit
            distance_user_unit = distance_mm / self.calibration_unit_factor
            return distance_user_unit
        else:
            # Use rough estimate for uncalibrated measurements (in mm)
            estimated_factor = 500 * 65  # rough focal_length * baseline estimate
            distance_mm = estimated_factor / abs_disparity
            return distance_mm
    
    def detect_objects(self, frame):
        """Detect objects using YOLO."""
        detections = []
        
        if self.yolo_model is None:
            return detections
        
        try:
            results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_name': class_name,
                            'class_id': class_id
                        })
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return detections
    
    def add_detections_to_buffer(self, detections_left, detections_right, timestamp):
        """Add current detections to temporal buffers with timestamps."""
        current_time = timestamp
        
        # Add left detections
        for det in detections_left:
            buffered_det = dict(det)  # Copy detection
            buffered_det['timestamp'] = current_time
            self.left_detection_buffer.append(buffered_det)
        
        # Add right detections
        for det in detections_right:
            buffered_det = dict(det)  # Copy detection
            buffered_det['timestamp'] = current_time
            self.right_detection_buffer.append(buffered_det)
        
        # Clean old detections from buffers
        self.clean_detection_buffers(current_time)
    
    def clean_detection_buffers(self, current_time):
        """Remove old detections from buffers."""
        cutoff_time = current_time - self.buffer_time_window
        
        # Remove old left detections
        self.left_detection_buffer = [det for det in self.left_detection_buffer 
                                     if det['timestamp'] > cutoff_time]
        
        # Remove old right detections
        self.right_detection_buffer = [det for det in self.right_detection_buffer 
                                      if det['timestamp'] > cutoff_time]
        
        # Limit buffer sizes
        if len(self.left_detection_buffer) > self.max_buffer_size:
            self.left_detection_buffer = self.left_detection_buffer[-self.max_buffer_size:]
        
        if len(self.right_detection_buffer) > self.max_buffer_size:
            self.right_detection_buffer = self.right_detection_buffer[-self.max_buffer_size:]
    
    def match_temporal_detections(self, debug=False):
        """
        Match detections using temporal buffering - matches detections that occurred
        within the time window, even if not on the same exact frame.
        """
        matched_pairs = []
        used_right_indices = set()
        
        if debug:
            print(f"\n🕐 TEMPORAL STEREO MATCHING DEBUG:")
            print(f"Left buffer: {len(self.left_detection_buffer)} detections")
            print(f"Right buffer: {len(self.right_detection_buffer)} detections")
            print(f"Time window: {self.buffer_time_window:.1f} seconds")
        
        for left_idx, left_det in enumerate(self.left_detection_buffer):
            left_x1, left_y1, left_x2, left_y2 = left_det['bbox']
            left_center_x = (left_x1 + left_x2) / 2
            left_center_y = (left_y1 + left_y2) / 2
            left_width = left_x2 - left_x1
            left_height = left_y2 - left_y1
            left_area = left_width * left_height
            left_time = left_det['timestamp']
            
            if debug:
                print(f"\nLeft {left_idx}: {left_det['class_name']} at ({left_center_x:.1f}, {left_center_y:.1f}), "
                      f"time: {left_time:.2f}, size: {left_width:.1f}x{left_height:.1f}")
            
            best_match = None
            best_score = 0
            best_right_idx = -1
            best_time_diff = float('inf')
            
            for right_idx, right_det in enumerate(self.right_detection_buffer):
                if right_idx in used_right_indices:
                    continue
                    
                # Only match same class objects
                if left_det['class_id'] != right_det['class_id']:
                    if debug:
                        print(f"  Right {right_idx}: {right_det['class_name']} - DIFFERENT CLASS")
                    continue
                
                right_x1, right_y1, right_x2, right_y2 = right_det['bbox']
                right_center_x = (right_x1 + right_x2) / 2
                right_center_y = (right_y1 + right_y2) / 2
                right_width = right_x2 - right_x1
                right_height = right_y2 - right_y1
                right_area = right_width * right_height
                right_time = right_det['timestamp']
                
                # Calculate temporal and spatial differences
                time_diff = abs(left_time - right_time)
                y_diff = abs(left_center_y - right_center_y)
                size_ratio = min(left_area, right_area) / max(left_area, right_area)
                x_disparity = left_center_x - right_center_x  # Should be positive
                
                # Temporal and spatial matching criteria
                max_time_diff = self.buffer_time_window  # Must be within time window
                max_y_diff = 100  # pixels
                min_size_ratio = 0.3
                min_disparity = -300  # Allow negative disparity (cameras might be swapped)
                max_disparity = 300
                
                if debug:
                    print(f"  Right {right_idx}: {right_det['class_name']} at ({right_center_x:.1f}, {right_center_y:.1f}), "
                          f"time: {right_time:.2f}")
                    print(f"    Time diff: {time_diff:.3f}s (max: {max_time_diff:.1f}s)")
                    print(f"    Y-diff: {y_diff:.1f} (max: {max_y_diff})")
                    print(f"    Size ratio: {size_ratio:.2f} (min: {min_size_ratio})")
                    print(f"    X-disparity: {x_disparity:.1f} (range: {min_disparity}-{max_disparity})")
                
                if (time_diff <= max_time_diff and
                    y_diff < max_y_diff and 
                    size_ratio > min_size_ratio and 
                    min_disparity < x_disparity < max_disparity):
                    
                    # Calculate matching score (higher is better)
                    time_score = max(0, 1 - time_diff / max_time_diff)  # Newer is better
                    y_score = max(0, 1 - y_diff / max_y_diff)
                    size_score = size_ratio
                    # Use absolute disparity for scoring
                    abs_disparity = abs(x_disparity)
                    disparity_score = min(1, (abs_disparity - 1) / (299))  # Adjusted for abs value
                    
                    # Weight temporal score highly for more recent matches
                    total_score = (time_score * 0.3 + y_score * 0.3 + size_score * 0.2 + disparity_score * 0.2)
                    
                    if debug:
                        print(f"    Scores: Time={time_score:.2f}, Y={y_score:.2f}, Size={size_score:.2f}, "
                              f"Disp={disparity_score:.2f}, Total={total_score:.2f}")
                    
                    # Prefer better scores, and among equal scores, prefer more recent matches
                    if (total_score > best_score and total_score > 0.3) or \
                       (total_score == best_score and time_diff < best_time_diff):
                        best_match = right_det
                        best_score = total_score
                        best_right_idx = right_idx
                        best_time_diff = time_diff
                        if debug:
                            print(f"    NEW BEST MATCH!")
                else:
                    if debug:
                        reasons = []
                        if time_diff > max_time_diff:
                            reasons.append(f"Time diff too large ({time_diff:.3f}s > {max_time_diff:.1f}s)")
                        if y_diff >= max_y_diff:
                            reasons.append(f"Y-diff too large ({y_diff:.1f} >= {max_y_diff})")
                        if size_ratio <= min_size_ratio:
                            reasons.append(f"Size ratio too small ({size_ratio:.2f} <= {min_size_ratio})")
                        if not (min_disparity < x_disparity < max_disparity):
                            reasons.append(f"X-disparity out of range ({x_disparity:.1f} not in {min_disparity}-{max_disparity})")
                        print(f"    REJECTED: {', '.join(reasons)}")
            
            if best_match is not None:
                used_right_indices.add(best_right_idx)
                matched_pairs.append({
                    'left': left_det,
                    'right': best_match,
                    'match_score': best_score,
                    'time_diff': best_time_diff
                })
                if debug:
                    print(f"  ✓ TEMPORAL MATCH with Right {best_right_idx} "
                          f"(score: {best_score:.2f}, time_diff: {best_time_diff:.3f}s)")
            else:
                if debug:
                    print(f"  ✗ NO TEMPORAL MATCH FOUND")
        
        if debug:
            print(f"\nTemporal matched pairs: {len(matched_pairs)}")
            if matched_pairs:
                for i, pair in enumerate(matched_pairs):
                    print(f"  Match {i+1}: {pair['left']['class_name']} "
                          f"(time_diff: {pair['time_diff']:.3f}s, score: {pair['match_score']:.2f})")
        
        return matched_pairs
    
    def create_stereo_matcher(self, params):
        """Create stereo matcher with given parameters."""
        return cv2.StereoSGBM_create(
            minDisparity=-64,
            numDisparities=params['numDisparities'],
            blockSize=params['blockSize'],
            P1=params['P1_factor'] * 3 * params['blockSize']**2,
            P2=params['P2_factor'] * 3 * params['blockSize']**2,
            disp12MaxDiff=params['disp12MaxDiff'],
            uniquenessRatio=params['uniquenessRatio'],
            speckleWindowSize=50,
            speckleRange=16,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    def evaluate_disparity_quality(self, disparity, bbox):
        """
        Evaluate disparity quality in a bounding box region.
        Returns a score from 0-1 (higher = better).
        """
        x1, y1, x2, y2 = bbox
        h, w = disparity.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        roi = disparity[y1:y2, x1:x2]
        total_pixels = roi.size
        
        if total_pixels == 0:
            return 0
        
        # Get valid disparities
        valid_disparities = roi[roi > 1.0]
        valid_count = len(valid_disparities)
        
        if valid_count < 10:  # Need minimum valid pixels
            return 0
        
        # Calculate quality metrics
        coverage = valid_count / total_pixels
        
        # Consistency - how uniform is the disparity in this region?
        if valid_count > 1:
            std_dev = np.std(valid_disparities)
            mean_disp = np.mean(valid_disparities)
            consistency = 1.0 / (1.0 + std_dev / max(mean_disp, 1.0))
        else:
            consistency = 0
        
        # Range check - reasonable disparity values
        median_disp = np.median(valid_disparities)
        range_score = 1.0 if 2 <= abs(median_disp) <= 100 else 0.5
        
        # Combined score
        quality_score = (coverage * 0.4 + consistency * 0.4 + range_score * 0.2)
        
        return min(1.0, quality_score)
    
    def adaptive_stereo_tuning(self, frame_left, frame_right, detections):
        """
        Adaptively tune stereo parameters based on detected objects.
        """
        if not self.adaptive_tuning or len(detections) == 0:
            return self.compute_disparity(frame_left, frame_right)
        
        best_disparity = None
        best_score = -1
        best_params = None
        
        # Try different parameter sets
        for params in self.tuning_candidates:
            try:
                # Create temporary stereo matcher
                temp_stereo = self.create_stereo_matcher(params)
                
                # Compute disparity with these parameters
                disparity = self.compute_disparity_with_matcher(frame_left, frame_right, temp_stereo)
                
                # Evaluate quality across all detected objects
                total_score = 0
                valid_objects = 0
                
                for det in detections:
                    quality = self.evaluate_disparity_quality(disparity, det['bbox'])
                    if quality > 0.1:  # Only count decent quality regions
                        total_score += quality
                        valid_objects += 1
                
                avg_score = total_score / max(valid_objects, 1)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_disparity = disparity
                    best_params = params
                    
            except Exception as e:
                continue  # Skip failed parameter sets
        
        # Update best parameters if we found better ones
        if best_params and best_score > 0.3:  # Minimum quality threshold
            self.update_best_parameters(best_params, best_score)
        
        # Return best disparity, or fallback to current parameters
        if best_disparity is not None:
            return best_disparity
        else:
            return self.compute_disparity(frame_left, frame_right)
    
    def update_best_parameters(self, new_params, score):
        """Update the running average of best parameters."""
        # Add to history with score weighting
        self.stereo_parameter_history.append({
            'params': new_params.copy(),
            'score': score,
            'timestamp': time.time()
        })
        
        # Keep only recent history (last 20 measurements)
        if len(self.stereo_parameter_history) > 20:
            self.stereo_parameter_history = self.stereo_parameter_history[-20:]
        
        # Update best parameters using weighted average
        if len(self.stereo_parameter_history) >= 3:  # Need some history
            weighted_params = {}
            total_weight = 0
            
            for param_name in self.best_parameters.keys():
                weighted_sum = 0
                for entry in self.stereo_parameter_history[-10:]:  # Last 10 entries
                    weight = entry['score']
                    weighted_sum += entry['params'][param_name] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    weighted_params[param_name] = int(weighted_sum / total_weight)
            
            # Update best parameters
            if weighted_params:
                self.best_parameters.update(weighted_params)
                # Update the main stereo matcher
                self.stereo = self.create_stereo_matcher(self.best_parameters)
                
                if len(self.stereo_parameter_history) % 5 == 0:  # Print update every 5 tunings
                    print(f"🔧 Adaptive tuning: Updated stereo parameters (score: {score:.2f})")
    
    def compute_disparity_with_matcher(self, img_left, img_right, stereo_matcher):
        """Compute disparity with a specific stereo matcher."""
        # Convert to grayscale
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        # Preprocessing
        gray_left = cv2.equalizeHist(gray_left)
        gray_right = cv2.equalizeHist(gray_right)
        gray_left = cv2.GaussianBlur(gray_left, (3, 3), 0)
        gray_right = cv2.GaussianBlur(gray_right, (3, 3), 0)
            
        # Compute disparity
        disparity = stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # Basic filtering
        disparity[disparity <= 0] = 0
        disparity = cv2.medianBlur(disparity.astype(np.uint8), 5).astype(np.float32)
        
        return disparity
    
    def validate_disparity_region(self, disparity_map, x1, y1, x2, y2):
        """
        Validate that a region has sufficient disparity data for reliable distance measurement.
        Returns (is_valid, median_disparity, coverage_percentage)
        """
        h, w = disparity_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False, 0, 0
        
        roi = disparity_map[y1:y2, x1:x2]
        total_pixels = roi.size
        
        # Get valid disparities (remove zeros and very small values)
        valid_disparities = roi[roi > 1.0]
        valid_count = len(valid_disparities)
        
        if valid_count == 0:
            return False, 0, 0
        
        coverage_percentage = (valid_count / total_pixels) * 100
        median_disparity = np.median(valid_disparities)
        
        # Require at least 20% coverage and reasonable disparity values
        min_coverage = 20  # percent
        min_disparity = 2.0  # Use absolute minimum disparity
        max_disparity = 100.0
        
        # Check absolute disparity values since cameras might be swapped
        abs_median_disparity = abs(median_disparity)
        
        is_valid = (coverage_percentage >= min_coverage and 
                   min_disparity <= abs_median_disparity <= max_disparity)
        
        return is_valid, median_disparity, coverage_percentage
    
    def get_object_distance(self, disparity_map, x1, y1, x2, y2):
        """Get distance to object in bounding box using validated disparity data."""
        is_valid, median_disparity, coverage = self.validate_disparity_region(
            disparity_map, x1, y1, x2, y2)
        
        if not is_valid:
            return -1
            
        # Convert to distance
        distance = self.disparity_to_distance(median_disparity)
        return distance
    
    def detect_objects_stereo(self, frame_left, frame_right):
        """
        Detect objects using stereo vision with temporal buffering and adaptive tuning.
        """
        # Detect objects in both cameras first
        detections_left = self.detect_objects(frame_left)
        detections_right = self.detect_objects(frame_right)
        
        # Use adaptive stereo tuning based on detected objects
        all_detections = detections_left + detections_right
        disparity = self.adaptive_stereo_tuning(frame_left, frame_right, all_detections)
        
        # Add current detections to temporal buffers
        current_time = time.time()
        self.add_detections_to_buffer(detections_left, detections_right, current_time)
        
        # Match detections using temporal buffering
        matched_pairs = self.match_temporal_detections(debug=self.debug_matching)
        
        # Calculate distances for matched pairs
        stereo_detections = []
        for pair in matched_pairs:
            left_det = pair['left']
            right_det = pair['right']
            match_score = pair['match_score']
            time_diff = pair['time_diff']
            
            # Use left camera detection for position (primary view)
            x1, y1, x2, y2 = left_det['bbox']
            
            # Validate and calculate distance
            is_valid, median_disparity, coverage = self.validate_disparity_region(
                disparity, x1, y1, x2, y2)
            
            if is_valid:
                distance = self.disparity_to_distance(median_disparity)
                
                stereo_detections.append({
                    'bbox': left_det['bbox'],
                    'confidence': left_det['confidence'],
                    'class_name': left_det['class_name'],
                    'class_id': left_det['class_id'],
                    'distance': distance,
                    'match_score': match_score,
                    'time_diff': time_diff,
                    'disparity_coverage': coverage,
                    'median_disparity': median_disparity,
                    'timestamp': left_det['timestamp']
                })
        
        # For "only" detections, use recent detections from buffers
        # Find recent unmatched detections
        matched_left_times = {pair['left']['timestamp'] for pair in matched_pairs}
        matched_right_times = {pair['right']['timestamp'] for pair in matched_pairs}
        
        # Only show recent detections that weren't matched
        recent_time_threshold = current_time - 0.1  # Last 0.1 seconds
        
        left_only_detections = [det for det in self.left_detection_buffer 
                               if det['timestamp'] >= recent_time_threshold and 
                               det['timestamp'] not in matched_left_times]
        
        right_only_detections = [det for det in self.right_detection_buffer 
                                if det['timestamp'] >= recent_time_threshold and 
                                det['timestamp'] not in matched_right_times]
        
        return stereo_detections, left_only_detections, right_only_detections, disparity
    
    def draw_stereo_detections(self, frame_left, frame_right, stereo_detections, 
                              left_only_detections, right_only_detections):
        """Draw detections on both frames with color coding."""
        result_left = frame_left.copy()
        result_right = frame_right.copy()
        calibration_triggered = False
        
        # Draw stereo detections (objects visible in both cameras) - GREEN
        for i, det in enumerate(stereo_detections):
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            distance = det['distance']
            
            # Check for calibration trigger
            if (not self.is_calibrated and not calibration_triggered and 
                distance > 0 and confidence > 0.7 and 
                (x2-x1) * (y2-y1) > 5000):  # Reasonably sized detection
                
                self.pending_calibration = {
                    'detection': det,
                    'disparity': det['median_disparity'],
                    'frame': result_left.copy()
                }
                calibration_triggered = True
                print(f"\n🎯 Triggering calibration for: {class_name} (temporal match)")
            
            # Choose color based on distance (green family for stereo detections)
            if distance < 500:
                color = (0, 255, 0)      # Bright green - close
            elif distance < 1500:
                color = (0, 200, 100)    # Green-cyan - medium
            else:
                color = (0, 150, 50)     # Dark green - far
            
            # Create distance text with temporal info
            if self.is_calibrated:
                if distance < 10:
                    dist_text = f"{distance:.1f}{self.calibration_unit}"
                else:
                    dist_text = f"{distance:.0f}{self.calibration_unit}"
            else:
                dist_text = f"~{distance:.0f}mm (uncal)"
            
            # Add temporal information if significant time difference
            time_diff = det.get('time_diff', 0)
            if time_diff > 0.05:  # Show if > 50ms difference
                temporal_info = f" (Δt:{time_diff:.2f}s)"
            else:
                temporal_info = ""
            
            label = f"{class_name} {confidence:.2f} - {dist_text}{temporal_info}"
            
            # Draw on left frame
            cv2.rectangle(result_left, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_left, (x1, y1-text_h-5), (x1+text_w, y1), color, -1)
            cv2.putText(result_left, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw left-only detections - BLUE
        for det in left_only_detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            color = (255, 100, 0)  # Blue
            label = f"{class_name} {confidence:.2f} - Left only"
            
            cv2.rectangle(result_left, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_left, (x1, y1-text_h-5), (x1+text_w, y1), color, -1)
            cv2.putText(result_left, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw right-only detections - ORANGE
        for det in right_only_detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            color = (0, 165, 255)  # Orange
            label = f"{class_name} {confidence:.2f} - Right only"
            
            cv2.rectangle(result_right, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_right, (x1, y1-text_h-5), (x1+text_w, y1), color, -1)
            cv2.putText(result_right, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_left, result_right, calibration_triggered
    
    def colorize_disparity(self, disparity):
        """Create colorized disparity map."""
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_JET)
        disp_color[disparity <= 0] = [0, 0, 0]
        return disp_color
    
    def save_stereo_results(self, frame_left, frame_right, detection_frame_left, 
                           detection_frame_right, disparity_color, stereo_detections, 
                           left_only_detections, right_only_detections):
        """Save stereo results to files."""
        timestamp = int(time.time())
        
        cv2.imwrite(f'left_{timestamp}.jpg', frame_left)
        cv2.imwrite(f'right_{timestamp}.jpg', frame_right)
        cv2.imwrite(f'detections_left_{timestamp}.jpg', detection_frame_left)
        cv2.imwrite(f'detections_right_{timestamp}.jpg', detection_frame_right)
        cv2.imwrite(f'disparity_{timestamp}.jpg', disparity_color)
        
        # Save detection info to text file
        with open(f'detections_{timestamp}.txt', 'w') as f:
            f.write(f"Stereo Detection Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n")
            if self.is_calibrated:
                f.write(f"✓ System is calibrated - distances are in {self.calibration_unit}\n")
            else:
                f.write("✗ System not calibrated - distances are estimates in mm\n")
            f.write("="*60 + "\n")
            
            f.write("STEREO DETECTIONS (Objects visible in both cameras):\n")
            f.write("-" * 50 + "\n")
            for i, det in enumerate(stereo_detections):
                distance = det['distance']
                coverage = det['disparity_coverage']
                match_score = det['match_score']
                time_diff = det.get('time_diff', 0)
                
                if self.is_calibrated:
                    dist_str = f"{distance:.1f}{self.calibration_unit}"
                else:
                    dist_str = f"~{distance:.0f}mm (uncalibrated)"
                    
                f.write(f"{i+1}. {det['class_name']}: {dist_str} "
                       f"(conf: {det['confidence']:.2f}, match: {match_score:.2f}, "
                       f"time_diff: {time_diff:.3f}s, coverage: {coverage:.1f}%)\n")
            
            f.write(f"\nLEFT-ONLY DETECTIONS ({len(left_only_detections)} objects):\n")
            f.write("-" * 50 + "\n")
            for i, det in enumerate(left_only_detections):
                f.write(f"{i+1}. {det['class_name']} (conf: {det['confidence']:.2f}) - No temporal match\n")
            
            f.write(f"\nRIGHT-ONLY DETECTIONS ({len(right_only_detections)} objects):\n")
            f.write("-" * 50 + "\n")
            for i, det in enumerate(right_only_detections):
                f.write(f"{i+1}. {det['class_name']} (conf: {det['confidence']:.2f}) - No temporal match\n")
        
        print(f"✓ Saved stereo results with timestamp {timestamp}")
    
    def keyboard_input_thread(self):
        """Handle keyboard input in a separate thread."""
        print("\nTerminal keyboard controls (press Enter after each command):")
        print("  'q' + Enter: Quit")
        print("  's' + Enter: Save current frame")
        print("  'c' + Enter: Force calibration on next stereo detection")
        print("  'r' + Enter: Reset calibration")
        print("  'd' + Enter: Toggle debug mode for stereo matching")
        print("  'debug' + Enter: Show debug info for next detection")
        print("  'status' + Enter: Show system status")
        print("  'adaptive' + Enter: Toggle adaptive stereo tuning")
        print("  'tune' + Enter: Tune stereo parameters for better disparity")
        print("  'h' + Enter: Show this help")
        if self.gui_working:
            print("\nOR use OpenCV window controls (click on window first):")
            print("  'q' or ESC: Quit")
            print("  's': Save current frame")
            print("  'c': Force calibration")
            print("  'd': Toggle debug mode")
        
        while self.running:
            try:
                user_input = input().strip().lower()
                
                if user_input == 'q':
                    print("Terminal quit command received")
                    self.running = False
                    break
                elif user_input == 's':
                    print("Terminal save command received")
                    self.save_counter += 1
                elif user_input == 'c':
                    print("Terminal calibration reset command received")
                    self.is_calibrated = False
                    self.calibration_factor = None
                    self.calibration_unit = None
                    self.calibration_unit_factor = 1.0
                elif user_input == 'r':
                    print("Terminal reset calibration command received")
                    self.is_calibrated = False
                    self.calibration_factor = None
                    self.calibration_unit = None
                    self.calibration_unit_factor = 1.0
                    try:
                        os.remove(self.calibration_file)
                        print("✓ Calibration file removed")
                    except FileNotFoundError:
                        pass
                elif user_input == 'd':
                    self.debug_matching = not self.debug_matching
                    print(f"Debug mode: {'ON' if self.debug_matching else 'OFF'}")
                elif user_input == 'debug':
                    print("Enabling debug for next detection cycle...")
                    self.debug_matching = True
                elif user_input == 'status':
                    print(f"\nSystem Status:")
                    print(f"  Calibrated: {'YES' if self.is_calibrated else 'NO'}")
                    print(f"  Left buffer: {len(self.left_detection_buffer)} detections")
                    print(f"  Right buffer: {len(self.right_detection_buffer)} detections")
                    print(f"  Debug mode: {'ON' if self.debug_matching else 'OFF'}")
                    print(f"  Buffer time window: {self.buffer_time_window}s")
                    print(f"  Adaptive tuning: {'ON' if self.adaptive_tuning else 'OFF'}")
                    if self.adaptive_tuning:
                        print(f"  Tuning history: {len(self.stereo_parameter_history)} measurements")
                        print(f"  Current best parameters:")
                        for k, v in self.best_parameters.items():
                            print(f"    {k}: {v}")
                elif user_input == 'adaptive':
                    self.adaptive_tuning = not self.adaptive_tuning
                    print(f"Adaptive tuning: {'ON' if self.adaptive_tuning else 'OFF'}")
                elif user_input == 'tune':
                    self.tune_stereo_parameters()
                elif user_input == 'h':
                    self.show_help()
                elif user_input:
                    print(f"Unknown command: {user_input}")
                    
            except EOFError:
                break
            except Exception as e:
                print(f"Input error: {e}")
                break
    
    def tune_stereo_parameters(self):
        """Interactive stereo parameter tuning."""
        print("\n🔧 STEREO PARAMETER TUNING")
        print("="*40)
        print("Current parameters:")
        print(f"  numDisparities: {self.stereo.getNumDisparities()}")
        print(f"  blockSize: {self.stereo.getBlockSize()}")
        print(f"  uniquenessRatio: {self.stereo.getUniquenessRatio()}")
        print()
        print("Quick presets:")
        print("  1 - High detail (good lighting, close objects)")
        print("  2 - Balanced (default)")  
        print("  3 - Smooth (poor lighting, far objects)")
        print("  4 - Custom")
        
        try:
            choice = input("Choose preset (1-4): ").strip()
            
            if choice == '1':
                # High detail preset
                self.stereo = cv2.StereoSGBM_create(
                    minDisparity=-96, numDisparities=192, blockSize=3,
                    P1=8*3*3**2, P2=32*3*3**2, disp12MaxDiff=2,
                    uniquenessRatio=20, speckleWindowSize=30, speckleRange=8
                )
                print("✓ Applied high detail preset")
            elif choice == '2':
                # Balanced preset (current)
                print("✓ Using current balanced preset")
            elif choice == '3':
                # Smooth preset
                self.stereo = cv2.StereoSGBM_create(
                    minDisparity=-32, numDisparities=96, blockSize=7,
                    P1=8*3*7**2, P2=32*3*7**2, disp12MaxDiff=10,
                    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32
                )
                print("✓ Applied smooth preset")
            elif choice == '4':
                print("Custom tuning not implemented yet - using balanced preset")
            else:
                print("Invalid choice - keeping current settings")
                
        except KeyboardInterrupt:
            print("\nTuning cancelled")
    
    def show_help(self):
        """Show help information."""
        print("\nTerminal keyboard controls:")
        print("  'q' + Enter: Quit")
        print("  's' + Enter: Save current frame")
        print("  'c' + Enter: Force calibration on next stereo detection")
        print("  'r' + Enter: Reset calibration")
        print("  'd' + Enter: Toggle debug mode for stereo matching")
        print("  'debug' + Enter: Show debug info for next detection")
        print("  'status' + Enter: Show system status")
        print("  'adaptive' + Enter: Toggle adaptive stereo tuning")
        print("  'tune' + Enter: Tune stereo parameters for better disparity")
        print("  'h' + Enter: Show this help")
        print("\nColor coding:")
        print("  🟢 Green boxes: Objects visible in both cameras (with distance)")
        print("  🔵 Blue boxes: Objects only in left camera")
        print("  🟠 Orange boxes: Objects only in right camera")
        print("\nAdaptive Tuning:")
        print("  🔧 Automatically adjusts stereo parameters based on detected objects")
        print("  📊 Focuses on object regions rather than entire image")
        print("  📈 Keeps running average of best-performing parameters")
        if self.gui_working:
            print("\nGUI controls (click on OpenCV window first):")
            print("  'q' or ESC: Quit")
            print("  's': Save frame")
            print("  'c': Force calibration")
            print("  'd': Toggle debug mode")
    
    def run(self):
        """Main execution loop."""
        # Find cameras
        cameras = self.find_cameras()
        if len(cameras) < 2:
            print("Error: Need at least 2 cameras for stereo vision")
            return
        
        # Open cameras
        cam_left = cv2.VideoCapture(cameras[0])
        cam_right = cv2.VideoCapture(cameras[1])
        
        if not cam_left.isOpened() or not cam_right.isOpened():
            print("Error: Could not open cameras")
            return
        
        # Set resolution
        width, height = 640, 480
        cam_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cam_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        print(f"✓ Cameras opened: {cameras[0]} (left), {cameras[1]} (right)")
        
        # Start keyboard input thread regardless of GUI mode
        # This allows terminal input to work even when GUI is present
        input_thread = threading.Thread(target=self.keyboard_input_thread, daemon=True)
        input_thread.start()
        
        print("\n" + "="*60)
        if self.gui_working:
            print("RUNNING WITH GUI")
            print("  Option 1: Click on OpenCV windows then press 'q' to quit")
            print("  Option 2: Type 'q' + Enter in this terminal")
        else:
            print("RUNNING IN HEADLESS MODE - Type 'q' + Enter in terminal to quit")
        
        if not self.is_calibrated:
            print("\n🎯 CALIBRATION NOTICE:")
            print("  The system is not calibrated yet.")
            print("  When the first object is detected in BOTH cameras (even across")
            print("  different frames within 0.5s), you'll be asked to measure its")
            print("  actual distance to calibrate the system.")
            print("  This only needs to be done once!")
            print("\n📐 TEMPORAL STEREO DETECTION:")
            print("  The system now uses temporal buffering - it remembers detections")
            print("  for 0.5 seconds and matches them even if they don't occur on")
            print("  the exact same frame.")
            print("  🤖 ADAPTIVE TUNING: Automatically optimizes stereo parameters")
            print("     based on detected object regions for better disparity maps!")
            print("  🟢 Green boxes: Objects matched between cameras (distance measured)")
            print("  🔵 Blue boxes: Objects only in left camera")  
            print("  🟠 Orange boxes: Objects only in right camera")
        else:
            print(f"\n✓ SYSTEM IS CALIBRATED (factor: {self.calibration_factor:.1f})")
            print(f"  Distance measurements are in {self.calibration_unit}.")
            print("\n📐 TEMPORAL STEREO DETECTION:")
            print("  🤖 ADAPTIVE TUNING: Automatically optimizes stereo parameters")
            print("     based on detected object regions for better disparity maps!")
            print("  🟢 Green boxes: Objects matched between cameras (distance measured)")
            print("  🔵 Blue boxes: Objects only in left camera")  
            print("  🟠 Orange boxes: Objects only in right camera")
            print("  Time differences (Δt) shown when > 50ms")
        
        print("="*60)
        
        frame_count = 0
        last_save_request = 0
        
        # Initialize disparity for first frame
        disparity = np.zeros((height, width), dtype=np.float32)
        disparity_color = self.colorize_disparity(disparity)
        
        try:
            while self.running:
                # Capture frames
                ret_left, frame_left = cam_left.read()
                ret_right, frame_right = cam_right.read()
                
                if not ret_left or not ret_right:
                    print("Error: Failed to capture frames")
                    break
                
                frame_count += 1
                
                # Process every frame (you can skip frames for better performance)
                if frame_count % 1 == 0:  # Process every frame
                    try:
                        # Detect objects every 5th frame for performance
                        stereo_detections = []
                        left_only_detections = []
                        right_only_detections = []
                        calibration_triggered = False
                        
                        if frame_count % 5 == 0:
                            stereo_detections, left_only_detections, right_only_detections, new_disparity = \
                                self.detect_objects_stereo(frame_left, frame_right)
                            # Update disparity only when we compute new data
                            disparity = new_disparity
                            disparity_color = self.colorize_disparity(disparity)
                        
                        # Draw detections on both frames
                        detection_frame_left, detection_frame_right, calibration_triggered = \
                            self.draw_stereo_detections(frame_left, frame_right, stereo_detections, 
                                                      left_only_detections, right_only_detections)
                        
                        # Handle calibration if triggered
                        if calibration_triggered and self.pending_calibration:
                            print("\n⏸️  PAUSING for calibration...")
                            if self.gui_working:
                                # Save calibration frame for GUI mode
                                cv2.imwrite('calibration_frame.jpg', self.pending_calibration['frame'])
                                print("📸 Calibration frame saved as 'calibration_frame.jpg'")
                            
                            # Request calibration
                            success = self.request_calibration(
                                self.pending_calibration['detection'],
                                self.pending_calibration['disparity']
                            )
                            
                            if success:
                                # Recalculate detection frames with new calibration
                                detection_frame_left, detection_frame_right, _ = \
                                    self.draw_stereo_detections(frame_left, frame_right, stereo_detections, 
                                                              left_only_detections, right_only_detections)
                            
                            self.pending_calibration = None
                            print("▶️  Resuming detection...\n")
                        
                        # Display if GUI is working
                        if self.gui_working:
                            cv2.imshow('Left Camera + Detections', detection_frame_left)
                            cv2.imshow('Right Camera + Detections', detection_frame_right)
                            cv2.imshow('Disparity Map', disparity_color)
                            
                            # More responsive keyboard checking
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q') or key == 27:  # 'q' or ESC
                                print("GUI quit key pressed")
                                break
                            elif key == ord('s'):
                                print("GUI save key pressed")
                                self.save_counter += 1
                            elif key == ord('c'):
                                print("GUI calibration reset key pressed")
                                self.is_calibrated = False
                                self.calibration_factor = None
                                self.calibration_unit = None
                                self.calibration_unit_factor = 1.0
                            elif key == ord('d'):
                                self.debug_matching = not self.debug_matching
                                print(f"GUI debug mode: {'ON' if self.debug_matching else 'OFF'}")
                            elif key != 255 and key != 0:  # Some other key was pressed
                                print(f"GUI key pressed: {chr(key) if 32 <= key <= 126 else f'code-{key}'}")
                        
                        # Save if requested
                        if self.save_counter > last_save_request:
                            self.save_stereo_results(frame_left, frame_right, detection_frame_left, 
                                                   detection_frame_right, disparity_color, 
                                                   stereo_detections, left_only_detections, 
                                                   right_only_detections)
                            last_save_request = self.save_counter
                        
                        # Print status every 30 frames
                        if frame_count % 30 == 0:
                            cal_status = "✓ Calibrated" if self.is_calibrated else "✗ Uncalibrated"
                            total_detections = len(stereo_detections) + len(left_only_detections) + len(right_only_detections)
                            buffer_info = f"Buf: L={len(self.left_detection_buffer)} R={len(self.right_detection_buffer)}"
                            
                            print(f"Frame {frame_count:4d} | Total: {total_detections:2d} | "
                                  f"Stereo: {len(stereo_detections):2d} | "
                                  f"L-only: {len(left_only_detections):1d} | "
                                  f"R-only: {len(right_only_detections):1d} | "
                                  f"{buffer_info} | "
                                  f"Status: {'GUI' if self.gui_working else 'Headless'} | {cal_status}")
                            
                            if stereo_detections:
                                print("  📐 TEMPORAL STEREO MEASUREMENTS:")
                                for i, det in enumerate(stereo_detections):
                                    distance = det['distance']
                                    time_diff = det.get('time_diff', 0)
                                    if self.is_calibrated:
                                        dist_str = f"{distance:.1f}{self.calibration_unit}"
                                    else:
                                        dist_str = f"~{distance:.0f}mm (est)"
                                    temporal_str = f" (Δt:{time_diff:.3f}s)" if time_diff > 0.01 else ""
                                    print(f"    ✓ {det['class_name']}: {dist_str}{temporal_str}")
                            
                            if left_only_detections or right_only_detections:
                                print("  📷 RECENT SINGLE-CAMERA DETECTIONS:")
                                for det in left_only_detections:
                                    print(f"    📘 L: {det['class_name']} (no temporal match)")
                                for det in right_only_detections:
                                    print(f"    🟠 R: {det['class_name']} (no temporal match)")
                        
                        # Small delay to prevent overwhelming the system
                        time.sleep(0.01)
                        
                    except Exception as e:
                        print(f"Processing error: {e}")
                        # Show original frames if processing fails
                        if self.gui_working:
                            cv2.imshow('Left Camera + Detections', frame_left)
                            cv2.imshow('Right Camera + Detections', frame_right)
                            cv2.imshow('Disparity Map', np.zeros_like(frame_left))
                
        except Exception as e:
            print(f"Runtime error: {e}")
        finally:
            print("Cleaning up...")
            cam_left.release()
            cam_right.release()
            if self.gui_working:
                cv2.destroyAllWindows()
            print("✓ Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Webcam Stereo Vision with Object Detection')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--confidence', type=float, default=0.3, help='YOLO confidence threshold')
    parser.add_argument('--buffer-time', type=float, default=0.5, help='Temporal buffer time window in seconds')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--calibration-file', type=str, default='stereo_calibration_interactive.txt', 
                       help='Calibration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'confidence_threshold': args.confidence,
        'buffer_time_window': args.buffer_time,
        'yolo_model': args.yolo_model,
        'calibration_file': args.calibration_file
    }
    
    print("STEREO VISION + OBJECT DETECTION WITH TEMPORAL MATCHING")
    print("========================================================")
    print("🎯 This system uses BOTH cameras for object detection")
    print("📏 Objects are matched across cameras using temporal buffering")
    print("⏰ Detections within 0.5 seconds are considered for matching")
    print("🔧 Interactive calibration using real measurements")
    print("📐 Color coding shows which camera(s) detect each object")
    print()
    
    system = StereoVisionSystem(config)
    system.run()


if __name__ == "__main__":
    main()