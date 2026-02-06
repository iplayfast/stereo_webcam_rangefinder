#!/usr/bin/env python3
"""
Headless Webcam Stereo Vision with Object Detection
====================================================

A headless (no display) version of the stereo vision system that:
- Uses two webcams for depth estimation
- Detects objects using YOLO
- Measures distances to detected objects
- Outputs detection data as JSON
- Requires pre-loaded calibration (no interactive prompts)

Usage:
    python webcam_stereo_headless.py --config config.json
    python webcam_stereo_headless.py --calibration-file stereo_calibration_interactive.txt

Author: AI Assistant
License: MIT
"""

import cv2
import numpy as np
import time
import json
import logging
import signal
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HeadlessStereoVision:
    """Headless stereo vision system for object detection and distance measurement."""

    def __init__(self, config=None):
        self.running = True
        self.config = config or {}

        # Configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
        self.buffer_time_window = self.config.get('buffer_time_window', 0.5)
        self.max_buffer_size = self.config.get('max_buffer_size', 10)
        self.process_every_n_frames = self.config.get('process_every_n_frames', 5)
        self.status_interval = self.config.get('status_interval', 30)

        # Camera settings
        self.left_camera_index = self.config.get('left_camera_index', 0)
        self.right_camera_index = self.config.get('right_camera_index', 2)
        self.resolution_width = self.config.get('resolution_width', 640)
        self.resolution_height = self.config.get('resolution_height', 480)

        # Output settings
        self.output_dir = Path(self.config.get('output_directory', './output'))
        self.save_frames = self.config.get('save_frames', False)
        self.json_output_file = self.config.get('json_output_file', None)

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Initializing Headless Stereo Vision System...")

        # Initialize stereo matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=-64,
            numDisparities=128,
            blockSize=3,
            P1=8 * 3 * 3**2,
            P2=32 * 3 * 3**2,
            disp12MaxDiff=5,
            uniquenessRatio=15,
            speckleWindowSize=50,
            speckleRange=16,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Load YOLO
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                yolo_model_path = self.config.get('yolo_model', 'yolov8n.pt')
                logger.info(f"Loading YOLO model: {yolo_model_path}")
                self.yolo_model = YOLO(yolo_model_path)
                logger.info("YOLO loaded successfully")
            except Exception as e:
                logger.error(f"YOLO loading failed: {e}")
        else:
            logger.warning("YOLO not available. Install with: pip install ultralytics")

        # Calibration
        self.is_calibrated = False
        self.calibration_factor = None
        self.calibration_unit = "mm"
        self.calibration_unit_factor = 1.0
        self.calibration_file = self.config.get('calibration_file', 'stereo_calibration_interactive.txt')

        # Temporal detection buffers
        self.left_detection_buffer = []
        self.right_detection_buffer = []

        # Adaptive tuning
        self.adaptive_tuning = self.config.get('adaptive_tuning', True)
        self.stereo_parameter_history = []
        self.best_parameters = {
            'numDisparities': 128, 'blockSize': 3, 'uniquenessRatio': 15,
            'P1_factor': 8, 'P2_factor': 32, 'disp12MaxDiff': 5
        }
        self.tuning_candidates = [
            {'numDisparities': 96, 'blockSize': 3, 'uniquenessRatio': 10, 'P1_factor': 8, 'P2_factor': 32, 'disp12MaxDiff': 2},
            {'numDisparities': 128, 'blockSize': 3, 'uniquenessRatio': 15, 'P1_factor': 8, 'P2_factor': 32, 'disp12MaxDiff': 5},
            {'numDisparities': 160, 'blockSize': 5, 'uniquenessRatio': 20, 'P1_factor': 10, 'P2_factor': 40, 'disp12MaxDiff': 3},
            {'numDisparities': 96, 'blockSize': 7, 'uniquenessRatio': 12, 'P1_factor': 6, 'P2_factor': 24, 'disp12MaxDiff': 8}
        ]

        # Load calibration
        self._load_calibration()

        # Create output directory
        if self.save_frames:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _load_calibration(self):
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
                    self.calibration_unit = "mm"
                    self.calibration_unit_factor = 1.0

                self.is_calibrated = True
                logger.info(f"Loaded calibration: factor={self.calibration_factor:.1f}, unit={self.calibration_unit}")
        except FileNotFoundError:
            logger.warning(f"No calibration file found at {self.calibration_file}")
            logger.warning("Running in uncalibrated mode - distances will be estimates")
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")

    def find_cameras(self):
        """Find available cameras."""
        logger.info("Scanning for cameras...")
        cameras = []

        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cameras.append(i)
                    logger.info(f"Camera {i}: Available")
                cap.release()

        logger.info(f"Found cameras: {cameras}")
        return cameras

    def compute_disparity(self, img_left, img_right):
        """Compute disparity map from stereo pair."""
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right

        gray_left = cv2.equalizeHist(gray_left)
        gray_right = cv2.equalizeHist(gray_right)
        gray_left = cv2.GaussianBlur(gray_left, (3, 3), 0)
        gray_right = cv2.GaussianBlur(gray_right, (3, 3), 0)

        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0
        disparity = cv2.medianBlur(disparity.astype(np.uint8), 5).astype(np.float32)

        return disparity

    def disparity_to_distance(self, disparity_value):
        """Convert disparity to distance using calibration."""
        abs_disparity = abs(disparity_value)

        if abs_disparity <= 0:
            return -1

        if self.is_calibrated:
            distance_mm = self.calibration_factor / abs_disparity
            distance_user_unit = distance_mm / self.calibration_unit_factor
            return distance_user_unit
        else:
            estimated_factor = 500 * 65
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
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class_name': class_name,
                            'class_id': class_id
                        })
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")

        return detections

    def add_detections_to_buffer(self, detections_left, detections_right, timestamp):
        """Add current detections to temporal buffers."""
        for det in detections_left:
            buffered_det = dict(det)
            buffered_det['timestamp'] = timestamp
            self.left_detection_buffer.append(buffered_det)

        for det in detections_right:
            buffered_det = dict(det)
            buffered_det['timestamp'] = timestamp
            self.right_detection_buffer.append(buffered_det)

        self._clean_detection_buffers(timestamp)

    def _clean_detection_buffers(self, current_time):
        """Remove old detections from buffers."""
        cutoff_time = current_time - self.buffer_time_window

        self.left_detection_buffer = [det for det in self.left_detection_buffer
                                      if det['timestamp'] > cutoff_time]
        self.right_detection_buffer = [det for det in self.right_detection_buffer
                                       if det['timestamp'] > cutoff_time]

        if len(self.left_detection_buffer) > self.max_buffer_size:
            self.left_detection_buffer = self.left_detection_buffer[-self.max_buffer_size:]
        if len(self.right_detection_buffer) > self.max_buffer_size:
            self.right_detection_buffer = self.right_detection_buffer[-self.max_buffer_size:]

    def match_temporal_detections(self):
        """Match detections using temporal buffering."""
        matched_pairs = []
        used_right_indices = set()

        for left_det in self.left_detection_buffer:
            left_x1, left_y1, left_x2, left_y2 = left_det['bbox']
            left_center_x = (left_x1 + left_x2) / 2
            left_center_y = (left_y1 + left_y2) / 2
            left_area = (left_x2 - left_x1) * (left_y2 - left_y1)
            left_time = left_det['timestamp']

            best_match = None
            best_score = 0
            best_right_idx = -1
            best_time_diff = float('inf')

            for right_idx, right_det in enumerate(self.right_detection_buffer):
                if right_idx in used_right_indices:
                    continue

                if left_det['class_id'] != right_det['class_id']:
                    continue

                right_x1, right_y1, right_x2, right_y2 = right_det['bbox']
                right_center_x = (right_x1 + right_x2) / 2
                right_center_y = (right_y1 + right_y2) / 2
                right_area = (right_x2 - right_x1) * (right_y2 - right_y1)
                right_time = right_det['timestamp']

                time_diff = abs(left_time - right_time)
                y_diff = abs(left_center_y - right_center_y)
                size_ratio = min(left_area, right_area) / max(left_area, right_area)
                x_disparity = left_center_x - right_center_x

                max_time_diff = self.buffer_time_window
                max_y_diff = 100
                min_size_ratio = 0.3
                min_disparity = -300
                max_disparity = 300

                if (time_diff <= max_time_diff and
                    y_diff < max_y_diff and
                    size_ratio > min_size_ratio and
                    min_disparity < x_disparity < max_disparity):

                    time_score = max(0, 1 - time_diff / max_time_diff)
                    y_score = max(0, 1 - y_diff / max_y_diff)
                    size_score = size_ratio
                    abs_disparity = abs(x_disparity)
                    disparity_score = min(1, (abs_disparity - 1) / 299)

                    total_score = (time_score * 0.3 + y_score * 0.3 + size_score * 0.2 + disparity_score * 0.2)

                    if (total_score > best_score and total_score > 0.3) or \
                       (total_score == best_score and time_diff < best_time_diff):
                        best_match = right_det
                        best_score = total_score
                        best_right_idx = right_idx
                        best_time_diff = time_diff

            if best_match is not None:
                used_right_indices.add(best_right_idx)
                matched_pairs.append({
                    'left': left_det,
                    'right': best_match,
                    'match_score': best_score,
                    'time_diff': best_time_diff
                })

        return matched_pairs

    def validate_disparity_region(self, disparity_map, x1, y1, x2, y2):
        """Validate disparity data in a region."""
        h, w = disparity_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False, 0, 0

        roi = disparity_map[y1:y2, x1:x2]
        total_pixels = roi.size

        valid_disparities = roi[roi > 1.0]
        valid_count = len(valid_disparities)

        if valid_count == 0:
            return False, 0, 0

        coverage_percentage = (valid_count / total_pixels) * 100
        median_disparity = np.median(valid_disparities)

        abs_median_disparity = abs(median_disparity)
        is_valid = (coverage_percentage >= 20 and 2.0 <= abs_median_disparity <= 100.0)

        return is_valid, median_disparity, coverage_percentage

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
        """Evaluate disparity quality in a bounding box region."""
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

        valid_disparities = roi[roi > 1.0]
        valid_count = len(valid_disparities)

        if valid_count < 10:
            return 0

        coverage = valid_count / total_pixels

        if valid_count > 1:
            std_dev = np.std(valid_disparities)
            mean_disp = np.mean(valid_disparities)
            consistency = 1.0 / (1.0 + std_dev / max(mean_disp, 1.0))
        else:
            consistency = 0

        median_disp = np.median(valid_disparities)
        range_score = 1.0 if 2 <= abs(median_disp) <= 100 else 0.5

        quality_score = (coverage * 0.4 + consistency * 0.4 + range_score * 0.2)
        return min(1.0, quality_score)

    def compute_disparity_with_matcher(self, img_left, img_right, stereo_matcher):
        """Compute disparity with a specific stereo matcher."""
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right

        gray_left = cv2.equalizeHist(gray_left)
        gray_right = cv2.equalizeHist(gray_right)
        gray_left = cv2.GaussianBlur(gray_left, (3, 3), 0)
        gray_right = cv2.GaussianBlur(gray_right, (3, 3), 0)

        disparity = stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0
        disparity = cv2.medianBlur(disparity.astype(np.uint8), 5).astype(np.float32)

        return disparity

    def adaptive_stereo_tuning(self, frame_left, frame_right, detections):
        """Adaptively tune stereo parameters based on detected objects."""
        if not self.adaptive_tuning or len(detections) == 0:
            return self.compute_disparity(frame_left, frame_right)

        best_disparity = None
        best_score = -1
        best_params = None

        for params in self.tuning_candidates:
            try:
                temp_stereo = self.create_stereo_matcher(params)
                disparity = self.compute_disparity_with_matcher(frame_left, frame_right, temp_stereo)

                total_score = 0
                valid_objects = 0

                for det in detections:
                    quality = self.evaluate_disparity_quality(disparity, det['bbox'])
                    if quality > 0.1:
                        total_score += quality
                        valid_objects += 1

                avg_score = total_score / max(valid_objects, 1)

                if avg_score > best_score:
                    best_score = avg_score
                    best_disparity = disparity
                    best_params = params

            except Exception:
                continue

        if best_params and best_score > 0.3:
            self._update_best_parameters(best_params, best_score)

        if best_disparity is not None:
            return best_disparity
        else:
            return self.compute_disparity(frame_left, frame_right)

    def _update_best_parameters(self, new_params, score):
        """Update the running average of best parameters."""
        self.stereo_parameter_history.append({
            'params': new_params.copy(),
            'score': score,
            'timestamp': time.time()
        })

        if len(self.stereo_parameter_history) > 20:
            self.stereo_parameter_history = self.stereo_parameter_history[-20:]

        if len(self.stereo_parameter_history) >= 3:
            weighted_params = {}
            total_weight = 0

            for param_name in self.best_parameters.keys():
                weighted_sum = 0
                for entry in self.stereo_parameter_history[-10:]:
                    weight = entry['score']
                    weighted_sum += entry['params'][param_name] * weight
                    total_weight += weight

                if total_weight > 0:
                    weighted_params[param_name] = int(weighted_sum / total_weight)

            if weighted_params:
                self.best_parameters.update(weighted_params)
                self.stereo = self.create_stereo_matcher(self.best_parameters)

    def detect_objects_stereo(self, frame_left, frame_right):
        """Detect objects using stereo vision with temporal buffering."""
        detections_left = self.detect_objects(frame_left)
        detections_right = self.detect_objects(frame_right)

        all_detections = detections_left + detections_right
        disparity = self.adaptive_stereo_tuning(frame_left, frame_right, all_detections)

        current_time = time.time()
        self.add_detections_to_buffer(detections_left, detections_right, current_time)

        matched_pairs = self.match_temporal_detections()

        stereo_detections = []
        for pair in matched_pairs:
            left_det = pair['left']
            x1, y1, x2, y2 = left_det['bbox']

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
                    'match_score': pair['match_score'],
                    'time_diff': pair['time_diff'],
                    'disparity_coverage': coverage,
                    'median_disparity': float(median_disparity),
                    'timestamp': left_det['timestamp']
                })

        matched_left_times = {pair['left']['timestamp'] for pair in matched_pairs}
        matched_right_times = {pair['right']['timestamp'] for pair in matched_pairs}

        recent_time_threshold = current_time - 0.1

        left_only = [det for det in self.left_detection_buffer
                     if det['timestamp'] >= recent_time_threshold and
                     det['timestamp'] not in matched_left_times]

        right_only = [det for det in self.right_detection_buffer
                      if det['timestamp'] >= recent_time_threshold and
                      det['timestamp'] not in matched_right_times]

        return stereo_detections, left_only, right_only, disparity

    def format_detection_output(self, stereo_detections, left_only, right_only, frame_count):
        """Format detection data as a dictionary for JSON output."""
        timestamp = datetime.now().isoformat()

        output = {
            'timestamp': timestamp,
            'frame_count': frame_count,
            'calibrated': self.is_calibrated,
            'unit': self.calibration_unit if self.is_calibrated else 'mm (estimate)',
            'stereo_detections': [],
            'left_only_detections': [],
            'right_only_detections': []
        }

        for det in stereo_detections:
            output['stereo_detections'].append({
                'class_name': det['class_name'],
                'class_id': det['class_id'],
                'confidence': round(det['confidence'], 3),
                'distance': round(det['distance'], 2),
                'bbox': list(det['bbox']),
                'match_score': round(det['match_score'], 3),
                'time_diff_seconds': round(det['time_diff'], 4),
                'disparity_coverage_percent': round(det['disparity_coverage'], 1)
            })

        for det in left_only:
            output['left_only_detections'].append({
                'class_name': det['class_name'],
                'class_id': det['class_id'],
                'confidence': round(det['confidence'], 3),
                'bbox': list(det['bbox'])
            })

        for det in right_only:
            output['right_only_detections'].append({
                'class_name': det['class_name'],
                'class_id': det['class_id'],
                'confidence': round(det['confidence'], 3),
                'bbox': list(det['bbox'])
            })

        return output

    def save_frame_data(self, frame_left, frame_right, disparity, stereo_detections, timestamp):
        """Save frame data to files."""
        ts = int(timestamp)

        cv2.imwrite(str(self.output_dir / f'left_{ts}.jpg'), frame_left)
        cv2.imwrite(str(self.output_dir / f'right_{ts}.jpg'), frame_right)

        # Save disparity as normalized image
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_JET)
        disp_color[disparity <= 0] = [0, 0, 0]
        cv2.imwrite(str(self.output_dir / f'disparity_{ts}.jpg'), disp_color)

        logger.info(f"Saved frames with timestamp {ts}")

    def run(self):
        """Main execution loop."""
        # Find cameras
        cameras = self.find_cameras()

        # Use configured cameras if available, otherwise auto-detect
        if self.left_camera_index in cameras and self.right_camera_index in cameras:
            left_idx = self.left_camera_index
            right_idx = self.right_camera_index
        elif len(cameras) >= 2:
            left_idx = cameras[0]
            right_idx = cameras[1]
            logger.warning(f"Using auto-detected cameras: {left_idx} (left), {right_idx} (right)")
        else:
            logger.error("Need at least 2 cameras for stereo vision")
            return

        # Open cameras
        cam_left = cv2.VideoCapture(left_idx)
        cam_right = cv2.VideoCapture(right_idx)

        if not cam_left.isOpened() or not cam_right.isOpened():
            logger.error("Could not open cameras")
            return

        # Set resolution
        cam_left.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution_width)
        cam_left.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution_height)
        cam_right.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution_width)
        cam_right.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution_height)

        logger.info(f"Cameras opened: {left_idx} (left), {right_idx} (right)")
        logger.info(f"Resolution: {self.resolution_width}x{self.resolution_height}")
        logger.info(f"Calibrated: {self.is_calibrated}")
        if self.is_calibrated:
            logger.info(f"Distance unit: {self.calibration_unit}")

        frame_count = 0
        disparity = np.zeros((self.resolution_height, self.resolution_width), dtype=np.float32)

        # Open JSON output file if specified
        json_file = None
        if self.json_output_file:
            json_file = open(self.json_output_file, 'w')
            json_file.write('[\n')
            first_entry = True

        try:
            logger.info("Starting detection loop... (Ctrl+C to stop)")

            while self.running:
                ret_left, frame_left = cam_left.read()
                ret_right, frame_right = cam_right.read()

                if not ret_left or not ret_right:
                    logger.error("Failed to capture frames")
                    break

                frame_count += 1

                stereo_detections = []
                left_only = []
                right_only = []

                # Process every N frames
                if frame_count % self.process_every_n_frames == 0:
                    try:
                        stereo_detections, left_only, right_only, disparity = \
                            self.detect_objects_stereo(frame_left, frame_right)

                        # Output detection data
                        if stereo_detections or left_only or right_only:
                            output_data = self.format_detection_output(
                                stereo_detections, left_only, right_only, frame_count)

                            # Write to JSON file if configured
                            if json_file:
                                if not first_entry:
                                    json_file.write(',\n')
                                json_file.write(json.dumps(output_data, indent=2))
                                json_file.flush()
                                first_entry = False

                            # Log detections
                            for det in stereo_detections:
                                if self.is_calibrated:
                                    dist_str = f"{det['distance']:.1f}{self.calibration_unit}"
                                else:
                                    dist_str = f"~{det['distance']:.0f}mm"
                                logger.info(f"DETECTED: {det['class_name']} at {dist_str} "
                                          f"(conf: {det['confidence']:.2f})")

                        # Save frames if configured
                        if self.save_frames and stereo_detections:
                            self.save_frame_data(frame_left, frame_right, disparity,
                                               stereo_detections, time.time())

                    except Exception as e:
                        logger.error(f"Processing error: {e}")

                # Status update
                if frame_count % self.status_interval == 0:
                    total = len(stereo_detections) + len(left_only) + len(right_only)
                    logger.info(f"Frame {frame_count} | Stereo: {len(stereo_detections)} | "
                              f"Left-only: {len(left_only)} | Right-only: {len(right_only)} | "
                              f"Buffer: L={len(self.left_detection_buffer)} R={len(self.right_detection_buffer)}")

                # Small delay
                time.sleep(0.01)

        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            logger.info("Shutting down...")
            cam_left.release()
            cam_right.release()

            if json_file:
                json_file.write('\n]\n')
                json_file.close()
                logger.info(f"JSON output saved to {self.json_output_file}")

            logger.info("Cleanup complete")


def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Flatten nested config
        flat_config = {}

        if 'camera_settings' in config:
            flat_config.update({
                'left_camera_index': config['camera_settings'].get('left_camera_index', 0),
                'right_camera_index': config['camera_settings'].get('right_camera_index', 2),
                'resolution_width': config['camera_settings'].get('resolution_width', 640),
                'resolution_height': config['camera_settings'].get('resolution_height', 480)
            })

        if 'detection_settings' in config:
            flat_config.update({
                'confidence_threshold': config['detection_settings'].get('confidence_threshold', 0.3),
                'yolo_model': config['detection_settings'].get('yolo_model', 'yolov8n.pt'),
                'process_every_n_frames': config['detection_settings'].get('process_every_n_frames', 5)
            })

        if 'stereo_settings' in config:
            flat_config.update({
                'buffer_time_window': config['stereo_settings'].get('buffer_time_window', 0.5),
                'max_buffer_size': config['stereo_settings'].get('max_buffer_size', 10)
            })

        if 'calibration_settings' in config:
            flat_config['calibration_file'] = config['calibration_settings'].get(
                'calibration_file', 'stereo_calibration_interactive.txt')

        if 'output_settings' in config:
            flat_config.update({
                'output_directory': config['output_settings'].get('output_directory', './output'),
                'save_frames': config['output_settings'].get('save_detection_frames', False)
            })

        if 'display_settings' in config:
            flat_config['status_interval'] = config['display_settings'].get('status_update_interval', 30)

        return flat_config

    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Headless Stereo Vision with Object Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with config file
    python webcam_stereo_headless.py --config config.json

    # Run with command line options
    python webcam_stereo_headless.py --calibration-file stereo_calibration_interactive.txt

    # Output detections to JSON file
    python webcam_stereo_headless.py --json-output detections.json

    # Save detection frames
    python webcam_stereo_headless.py --save-frames --output-dir ./captures
        """
    )

    parser.add_argument('--config', type=str, help='Configuration file path (JSON)')
    parser.add_argument('--confidence', type=float, default=0.3, help='YOLO confidence threshold')
    parser.add_argument('--buffer-time', type=float, default=0.5, help='Temporal buffer time window (seconds)')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--calibration-file', type=str, default='stereo_calibration_interactive.txt',
                        help='Calibration file path')
    parser.add_argument('--left-camera', type=int, default=0, help='Left camera index')
    parser.add_argument('--right-camera', type=int, default=2, help='Right camera index')
    parser.add_argument('--json-output', type=str, help='Output detections to JSON file')
    parser.add_argument('--save-frames', action='store_true', help='Save detection frames')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory for frames')
    parser.add_argument('--status-interval', type=int, default=30, help='Status log interval (frames)')
    parser.add_argument('--no-adaptive', action='store_true', help='Disable adaptive stereo tuning')

    args = parser.parse_args()

    # Load config from file or build from args
    if args.config:
        config = load_config(args.config)
    else:
        config = {}

    # Override with command line arguments
    config.update({
        'confidence_threshold': args.confidence,
        'buffer_time_window': args.buffer_time,
        'yolo_model': args.yolo_model,
        'calibration_file': args.calibration_file,
        'left_camera_index': args.left_camera,
        'right_camera_index': args.right_camera,
        'json_output_file': args.json_output,
        'save_frames': args.save_frames,
        'output_directory': args.output_dir,
        'status_interval': args.status_interval,
        'adaptive_tuning': not args.no_adaptive
    })

    logger.info("=" * 60)
    logger.info("HEADLESS STEREO VISION + OBJECT DETECTION")
    logger.info("=" * 60)

    system = HeadlessStereoVision(config)
    system.run()


if __name__ == "__main__":
    main()
