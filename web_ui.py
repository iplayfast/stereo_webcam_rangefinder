#!/usr/bin/env python3
"""
Web UI for Stereo Vision System
================================

A web-based interface for the stereo vision rangefinder that:
- Streams camera feeds to the browser
- Shows real-time object detections with distances
- Displays system status and controls
- Uses WebSocket for low-latency updates

Usage:
    python web_ui.py
    python web_ui.py --port 8080
    python web_ui.py --config config.json

Then open http://localhost:5000 in your browser.

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
import base64
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce Flask/Werkzeug logging noise
logging.getLogger('werkzeug').setLevel(logging.WARNING)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stereo-vision-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


class WebStereoVision:
    """Stereo vision system with web streaming capabilities."""

    def __init__(self, config=None):
        self.running = False
        self.config = config or {}
        self.lock = threading.Lock()

        # Configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
        self.buffer_time_window = self.config.get('buffer_time_window', 0.5)
        self.max_buffer_size = self.config.get('max_buffer_size', 10)
        self.process_every_n_frames = self.config.get('process_every_n_frames', 5)

        # Camera settings
        self.left_camera_index = self.config.get('left_camera_index', 0)
        self.right_camera_index = self.config.get('right_camera_index', 2)
        self.resolution_width = self.config.get('resolution_width', 640)
        self.resolution_height = self.config.get('resolution_height', 480)
        self.jpeg_quality = self.config.get('jpeg_quality', 70)

        # Current frame data (thread-safe access)
        self.current_frame_left = None
        self.current_frame_right = None
        self.current_disparity = None
        self.current_detections = []
        self.left_only_detections = []
        self.right_only_detections = []

        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0

        # Cameras
        self.cam_left = None
        self.cam_right = None

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

        # Calibration
        self.is_calibrated = False
        self.calibration_factor = None
        self.calibration_unit = "mm"
        self.calibration_unit_factor = 1.0
        self.calibration_file = self.config.get('calibration_file', 'stereo_calibration_interactive.txt')

        # Temporal detection buffers
        self.left_detection_buffer = []
        self.right_detection_buffer = []

        # Load calibration
        self._load_calibration()

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
                self.is_calibrated = True
                logger.info(f"Loaded calibration: factor={self.calibration_factor:.1f}, unit={self.calibration_unit}")
        except FileNotFoundError:
            logger.warning(f"No calibration file found")
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")

    def find_cameras(self):
        """Find available cameras."""
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cameras.append(i)
                cap.release()
        return cameras

    def start_cameras(self):
        """Initialize and start cameras."""
        cameras = self.find_cameras()
        logger.info(f"Found cameras: {cameras}")

        if self.left_camera_index in cameras and self.right_camera_index in cameras:
            left_idx = self.left_camera_index
            right_idx = self.right_camera_index
        elif len(cameras) >= 2:
            left_idx = cameras[0]
            right_idx = cameras[1]
        else:
            logger.error("Need at least 2 cameras")
            return False

        self.cam_left = cv2.VideoCapture(left_idx)
        self.cam_right = cv2.VideoCapture(right_idx)

        if not self.cam_left.isOpened() or not self.cam_right.isOpened():
            logger.error("Could not open cameras")
            return False

        # Set resolution
        for cam in [self.cam_left, self.cam_right]:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution_width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution_height)

        logger.info(f"Cameras started: {left_idx} (left), {right_idx} (right)")
        return True

    def stop_cameras(self):
        """Stop and release cameras."""
        if self.cam_left:
            self.cam_left.release()
        if self.cam_right:
            self.cam_right.release()
        logger.info("Cameras stopped")

    def compute_disparity(self, img_left, img_right):
        """Compute disparity map."""
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        gray_left = cv2.equalizeHist(gray_left)
        gray_right = cv2.equalizeHist(gray_right)
        gray_left = cv2.GaussianBlur(gray_left, (3, 3), 0)
        gray_right = cv2.GaussianBlur(gray_right, (3, 3), 0)

        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        disparity[disparity <= 0] = 0
        disparity = cv2.medianBlur(disparity.astype(np.uint8), 5).astype(np.float32)

        return disparity

    def disparity_to_distance(self, disparity_value):
        """Convert disparity to distance."""
        abs_disparity = abs(disparity_value)
        if abs_disparity <= 0:
            return -1

        if self.is_calibrated:
            distance_mm = self.calibration_factor / abs_disparity
            return distance_mm / self.calibration_unit_factor
        else:
            return (500 * 65) / abs_disparity

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
            logger.error(f"Detection error: {e}")

        return detections

    def add_detections_to_buffer(self, detections_left, detections_right, timestamp):
        """Add detections to temporal buffers."""
        for det in detections_left:
            buffered_det = dict(det)
            buffered_det['timestamp'] = timestamp
            self.left_detection_buffer.append(buffered_det)

        for det in detections_right:
            buffered_det = dict(det)
            buffered_det['timestamp'] = timestamp
            self.right_detection_buffer.append(buffered_det)

        # Clean old detections
        cutoff_time = timestamp - self.buffer_time_window
        self.left_detection_buffer = [d for d in self.left_detection_buffer if d['timestamp'] > cutoff_time][-self.max_buffer_size:]
        self.right_detection_buffer = [d for d in self.right_detection_buffer if d['timestamp'] > cutoff_time][-self.max_buffer_size:]

    def match_temporal_detections(self):
        """Match detections between cameras."""
        matched_pairs = []
        used_right_indices = set()

        for left_det in self.left_detection_buffer:
            left_bbox = left_det['bbox']
            left_center_x = (left_bbox[0] + left_bbox[2]) / 2
            left_center_y = (left_bbox[1] + left_bbox[3]) / 2
            left_area = (left_bbox[2] - left_bbox[0]) * (left_bbox[3] - left_bbox[1])

            best_match = None
            best_score = 0
            best_idx = -1

            for right_idx, right_det in enumerate(self.right_detection_buffer):
                if right_idx in used_right_indices or left_det['class_id'] != right_det['class_id']:
                    continue

                right_bbox = right_det['bbox']
                right_center_x = (right_bbox[0] + right_bbox[2]) / 2
                right_center_y = (right_bbox[1] + right_bbox[3]) / 2
                right_area = (right_bbox[2] - right_bbox[0]) * (right_bbox[3] - right_bbox[1])

                time_diff = abs(left_det['timestamp'] - right_det['timestamp'])
                y_diff = abs(left_center_y - right_center_y)
                size_ratio = min(left_area, right_area) / max(left_area, right_area)
                x_disparity = left_center_x - right_center_x

                if (time_diff <= self.buffer_time_window and y_diff < 100 and
                    size_ratio > 0.3 and -300 < x_disparity < 300):

                    score = (1 - time_diff/self.buffer_time_window) * 0.3 + (1 - y_diff/100) * 0.3 + size_ratio * 0.4

                    if score > best_score and score > 0.3:
                        best_match = right_det
                        best_score = score
                        best_idx = right_idx

            if best_match:
                used_right_indices.add(best_idx)
                matched_pairs.append({'left': left_det, 'right': best_match, 'score': best_score})

        return matched_pairs

    def validate_disparity_region(self, disparity_map, x1, y1, x2, y2):
        """Validate disparity in a region."""
        h, w = disparity_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False, 0, 0

        roi = disparity_map[y1:y2, x1:x2]
        valid_disparities = roi[roi > 1.0]

        if len(valid_disparities) == 0:
            return False, 0, 0

        coverage = (len(valid_disparities) / roi.size) * 100
        median_disparity = np.median(valid_disparities)

        is_valid = coverage >= 20 and 2.0 <= abs(median_disparity) <= 100.0
        return is_valid, median_disparity, coverage

    def process_stereo(self, frame_left, frame_right):
        """Process stereo frames and detect objects."""
        detections_left = self.detect_objects(frame_left)
        detections_right = self.detect_objects(frame_right)
        disparity = self.compute_disparity(frame_left, frame_right)

        current_time = time.time()
        self.add_detections_to_buffer(detections_left, detections_right, current_time)

        matched_pairs = self.match_temporal_detections()

        stereo_detections = []
        for pair in matched_pairs:
            left_det = pair['left']
            x1, y1, x2, y2 = left_det['bbox']

            is_valid, median_disparity, coverage = self.validate_disparity_region(disparity, x1, y1, x2, y2)

            if is_valid:
                distance = self.disparity_to_distance(median_disparity)
                stereo_detections.append({
                    'bbox': left_det['bbox'],
                    'confidence': left_det['confidence'],
                    'class_name': left_det['class_name'],
                    'class_id': left_det['class_id'],
                    'distance': distance,
                    'coverage': coverage
                })

        # Get unmatched detections
        matched_left = {id(p['left']) for p in matched_pairs}
        matched_right = {id(p['right']) for p in matched_pairs}

        left_only = [d for d in detections_left if id(d) not in matched_left]
        right_only = [d for d in detections_right if id(d) not in matched_right]

        return stereo_detections, left_only, right_only, disparity

    def draw_detections(self, frame, detections, stereo_detections):
        """Draw detection boxes on frame."""
        result = frame.copy()

        # Draw stereo detections (green)
        for det in stereo_detections:
            x1, y1, x2, y2 = det['bbox']
            distance = det['distance']

            if self.is_calibrated:
                label = f"{det['class_name']} {distance:.1f}{self.calibration_unit}"
            else:
                label = f"{det['class_name']} ~{distance:.0f}mm"

            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (x1, y1-th-10), (x1+tw+10, y1), (0, 255, 0), -1)
            cv2.putText(result, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Draw single-camera detections (blue)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(result, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        return result

    def colorize_disparity(self, disparity):
        """Create colorized disparity map."""
        disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_color = cv2.applyColorMap(disp_norm.astype(np.uint8), cv2.COLORMAP_JET)
        disp_color[disparity <= 0] = [0, 0, 0]
        return disp_color

    def frame_to_base64(self, frame):
        """Convert frame to base64 JPEG."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return base64.b64encode(buffer).decode('utf-8')

    def capture_loop(self):
        """Main capture and processing loop."""
        if not self.start_cameras():
            return

        self.running = True
        logger.info("Starting capture loop")

        try:
            while self.running:
                ret_left, frame_left = self.cam_left.read()
                ret_right, frame_right = self.cam_right.read()

                if not ret_left or not ret_right:
                    logger.error("Failed to capture")
                    time.sleep(0.1)
                    continue

                self.frame_count += 1
                self.fps_frame_count += 1

                # Calculate FPS
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.fps = self.fps_frame_count / (now - self.last_fps_time)
                    self.fps_frame_count = 0
                    self.last_fps_time = now

                # Process every N frames
                if self.frame_count % self.process_every_n_frames == 0:
                    stereo_dets, left_only, right_only, disparity = self.process_stereo(frame_left, frame_right)

                    # Draw detections
                    frame_left_drawn = self.draw_detections(frame_left, left_only, stereo_dets)
                    frame_right_drawn = self.draw_detections(frame_right, right_only, [])
                    disparity_color = self.colorize_disparity(disparity)

                    with self.lock:
                        self.current_frame_left = frame_left_drawn
                        self.current_frame_right = frame_right_drawn
                        self.current_disparity = disparity_color
                        self.current_detections = stereo_dets
                        self.left_only_detections = left_only
                        self.right_only_detections = right_only

                    # Emit to websocket
                    detection_data = {
                        'timestamp': datetime.now().isoformat(),
                        'frame': self.frame_count,
                        'fps': round(self.fps, 1),
                        'calibrated': self.is_calibrated,
                        'unit': self.calibration_unit,
                        'detections': [
                            {
                                'class': d['class_name'],
                                'distance': round(d['distance'], 1),
                                'confidence': round(d['confidence'], 2),
                                'bbox': d['bbox']
                            }
                            for d in stereo_dets
                        ],
                        'left_only': len(left_only),
                        'right_only': len(right_only)
                    }
                    socketio.emit('detections', detection_data)

                time.sleep(0.01)

        except Exception as e:
            logger.error(f"Capture loop error: {e}")
        finally:
            self.stop_cameras()
            self.running = False

    def get_frame_jpeg(self, camera='left'):
        """Get current frame as JPEG bytes."""
        with self.lock:
            if camera == 'left' and self.current_frame_left is not None:
                frame = self.current_frame_left
            elif camera == 'right' and self.current_frame_right is not None:
                frame = self.current_frame_right
            elif camera == 'disparity' and self.current_disparity is not None:
                frame = self.current_disparity
            else:
                # Return placeholder
                frame = np.zeros((self.resolution_height, self.resolution_width, 3), dtype=np.uint8)
                cv2.putText(frame, "No Signal", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buffer.tobytes()

    def get_status(self):
        """Get current system status."""
        with self.lock:
            return {
                'running': self.running,
                'frame_count': self.frame_count,
                'fps': round(self.fps, 1),
                'calibrated': self.is_calibrated,
                'calibration_unit': self.calibration_unit,
                'stereo_detections': len(self.current_detections),
                'left_buffer': len(self.left_detection_buffer),
                'right_buffer': len(self.right_detection_buffer),
                'yolo_available': self.yolo_model is not None
            }


# Global vision system instance
vision_system = None


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stereo Vision Rangefinder</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #0f3460;
        }
        .header h1 { font-size: 1.5rem; color: #00ff88; }
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
        }
        .status-badge.running { background: #00ff88; color: #000; }
        .status-badge.stopped { background: #ff4757; color: #fff; }

        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            padding: 1rem;
            max-width: 1600px;
            margin: 0 auto;
        }

        .video-card {
            background: #16213e;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #0f3460;
        }
        .video-card h3 {
            padding: 0.75rem 1rem;
            background: #0f3460;
            font-size: 0.9rem;
            display: flex;
            justify-content: space-between;
        }
        .video-card img {
            width: 100%;
            height: auto;
            display: block;
        }

        .full-width { grid-column: 1 / -1; }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            padding: 1rem;
        }
        .stat-card {
            background: #0f3460;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        .stat-card .value {
            font-size: 2rem;
            font-weight: bold;
            color: #00ff88;
        }
        .stat-card .label {
            font-size: 0.8rem;
            color: #888;
            margin-top: 0.25rem;
        }

        .detections-panel {
            background: #16213e;
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid #0f3460;
        }
        .detections-panel h3 {
            margin-bottom: 1rem;
            color: #00ff88;
        }
        .detection-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .detection-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: #0f3460;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            border-left: 4px solid #00ff88;
        }
        .detection-item .class-name {
            font-weight: bold;
            font-size: 1.1rem;
        }
        .detection-item .distance {
            font-size: 1.5rem;
            color: #00ff88;
            font-weight: bold;
        }
        .detection-item .confidence {
            font-size: 0.8rem;
            color: #888;
        }

        .controls {
            display: flex;
            gap: 1rem;
            padding: 1rem;
            background: #16213e;
            border-radius: 10px;
            margin: 1rem;
            border: 1px solid #0f3460;
        }
        .controls button {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.1s;
        }
        .controls button:hover { transform: scale(1.05); }
        .controls button:active { transform: scale(0.95); }
        .btn-start { background: #00ff88; color: #000; }
        .btn-stop { background: #ff4757; color: #fff; }
        .btn-snapshot { background: #3498db; color: #fff; }

        .no-detections {
            color: #666;
            text-align: center;
            padding: 2rem;
        }

        @media (max-width: 768px) {
            .container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Stereo Vision Rangefinder</h1>
        <span id="status-badge" class="status-badge stopped">Stopped</span>
    </div>

    <div class="controls">
        <button class="btn-start" onclick="startSystem()">Start</button>
        <button class="btn-stop" onclick="stopSystem()">Stop</button>
        <button class="btn-snapshot" onclick="takeSnapshot()">Snapshot</button>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="value" id="fps">0</div>
            <div class="label">FPS</div>
        </div>
        <div class="stat-card">
            <div class="value" id="frame-count">0</div>
            <div class="label">Frames</div>
        </div>
        <div class="stat-card">
            <div class="value" id="detection-count">0</div>
            <div class="label">Detections</div>
        </div>
        <div class="stat-card">
            <div class="value" id="calibration-status">No</div>
            <div class="label">Calibrated</div>
        </div>
    </div>

    <div class="container">
        <div class="video-card">
            <h3>Left Camera <span id="left-label"></span></h3>
            <img id="left-feed" src="/video_feed/left" alt="Left Camera">
        </div>
        <div class="video-card">
            <h3>Right Camera <span id="right-label"></span></h3>
            <img id="right-feed" src="/video_feed/right" alt="Right Camera">
        </div>
        <div class="video-card">
            <h3>Disparity Map</h3>
            <img id="disparity-feed" src="/video_feed/disparity" alt="Disparity Map">
        </div>
        <div class="detections-panel">
            <h3>Live Detections</h3>
            <div id="detection-list" class="detection-list">
                <div class="no-detections">No objects detected</div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let refreshInterval;

        socket.on('connect', () => {
            console.log('Connected to server');
            updateStatus();
        });

        socket.on('detections', (data) => {
            document.getElementById('fps').textContent = data.fps;
            document.getElementById('frame-count').textContent = data.frame;
            document.getElementById('detection-count').textContent = data.detections.length;
            document.getElementById('calibration-status').textContent = data.calibrated ? 'Yes' : 'No';

            const list = document.getElementById('detection-list');
            if (data.detections.length === 0) {
                list.innerHTML = '<div class="no-detections">No objects detected</div>';
            } else {
                list.innerHTML = data.detections.map(d => `
                    <div class="detection-item">
                        <div>
                            <div class="class-name">${d.class}</div>
                            <div class="confidence">${(d.confidence * 100).toFixed(0)}% confidence</div>
                        </div>
                        <div class="distance">${d.distance}${data.unit}</div>
                    </div>
                `).join('');
            }
        });

        socket.on('status_update', (data) => {
            const badge = document.getElementById('status-badge');
            if (data.running) {
                badge.textContent = 'Running';
                badge.className = 'status-badge running';
            } else {
                badge.textContent = 'Stopped';
                badge.className = 'status-badge stopped';
            }
        });

        function updateStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    const badge = document.getElementById('status-badge');
                    if (data.running) {
                        badge.textContent = 'Running';
                        badge.className = 'status-badge running';
                        startRefresh();
                    } else {
                        badge.textContent = 'Stopped';
                        badge.className = 'status-badge stopped';
                    }
                });
        }

        function startRefresh() {
            if (refreshInterval) return;
            refreshInterval = setInterval(() => {
                const timestamp = Date.now();
                document.getElementById('left-feed').src = '/video_feed/left?' + timestamp;
                document.getElementById('right-feed').src = '/video_feed/right?' + timestamp;
                document.getElementById('disparity-feed').src = '/video_feed/disparity?' + timestamp;
            }, 100);
        }

        function stopRefresh() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
                refreshInterval = null;
            }
        }

        function startSystem() {
            fetch('/api/start', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'started') {
                        startRefresh();
                        updateStatus();
                    }
                });
        }

        function stopSystem() {
            fetch('/api/stop', { method: 'POST' })
                .then(r => r.json())
                .then(() => {
                    stopRefresh();
                    updateStatus();
                });
        }

        function takeSnapshot() {
            fetch('/api/snapshot', { method: 'POST' })
                .then(r => r.json())
                .then(data => alert('Snapshot saved: ' + data.filename));
        }

        // Initial status check
        updateStatus();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve main page."""
    return HTML_TEMPLATE


@app.route('/video_feed/<camera>')
def video_feed(camera):
    """Stream video frames."""
    if vision_system is None:
        return Response(status=503)

    jpeg = vision_system.get_frame_jpeg(camera)
    return Response(jpeg, mimetype='image/jpeg')


@app.route('/api/status')
def api_status():
    """Get system status."""
    if vision_system is None:
        return jsonify({'running': False, 'error': 'System not initialized'})
    return jsonify(vision_system.get_status())


@app.route('/api/start', methods=['POST'])
def api_start():
    """Start the vision system."""
    global vision_system
    if vision_system and not vision_system.running:
        thread = threading.Thread(target=vision_system.capture_loop, daemon=True)
        thread.start()
        time.sleep(0.5)  # Give it time to start
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running' if vision_system.running else 'error'})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop the vision system."""
    if vision_system:
        vision_system.running = False
        time.sleep(0.5)
        socketio.emit('status_update', {'running': False})
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not_running'})


@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    """Save current frames."""
    if vision_system and vision_system.running:
        timestamp = int(time.time())
        output_dir = Path('./snapshots')
        output_dir.mkdir(exist_ok=True)

        with vision_system.lock:
            if vision_system.current_frame_left is not None:
                cv2.imwrite(str(output_dir / f'left_{timestamp}.jpg'), vision_system.current_frame_left)
            if vision_system.current_frame_right is not None:
                cv2.imwrite(str(output_dir / f'right_{timestamp}.jpg'), vision_system.current_frame_right)
            if vision_system.current_disparity is not None:
                cv2.imwrite(str(output_dir / f'disparity_{timestamp}.jpg'), vision_system.current_disparity)

        return jsonify({'status': 'saved', 'filename': f'snapshots/*_{timestamp}.jpg'})
    return jsonify({'status': 'error', 'message': 'System not running'})


def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

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
            flat_config['calibration_file'] = config['calibration_settings'].get('calibration_file')

        return flat_config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def main():
    """Main entry point."""
    global vision_system

    parser = argparse.ArgumentParser(description='Web UI for Stereo Vision Rangefinder')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    parser.add_argument('--confidence', type=float, default=0.3, help='YOLO confidence threshold')
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--calibration-file', type=str, default='stereo_calibration_interactive.txt')
    parser.add_argument('--left-camera', type=int, default=0, help='Left camera index')
    parser.add_argument('--right-camera', type=int, default=2, help='Right camera index')
    parser.add_argument('--auto-start', action='store_true', help='Start capture automatically')

    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = {}

    # Override with command line args
    config.update({
        'confidence_threshold': args.confidence,
        'yolo_model': args.yolo_model,
        'calibration_file': args.calibration_file,
        'left_camera_index': args.left_camera,
        'right_camera_index': args.right_camera
    })

    # Initialize vision system
    vision_system = WebStereoVision(config)

    # Auto-start if requested
    if args.auto_start:
        thread = threading.Thread(target=vision_system.capture_loop, daemon=True)
        thread.start()

    logger.info(f"Starting web server on http://{args.host}:{args.port}")
    logger.info("Open this URL in your browser to view the stereo vision interface")

    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
