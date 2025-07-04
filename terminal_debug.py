#!/usr/bin/env python3
"""
Terminal-Based Stereo Debug Tool
================================

More reliable than OpenCV key handling - uses simple terminal input.

USAGE: python terminal_debug.py

Commands:
- Enter = Analyze current frame
- l = Lower confidence
- h = Higher confidence  
- q = Quit
"""

import cv2
import numpy as np
import time
import threading
from collections import defaultdict

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install with: pip install ultralytics")
    exit(1)

class TerminalStereoDebugger:
    def __init__(self):
        self.confidence_threshold = 0.3
        self.yolo_model = None
        self.running = True
        self.latest_frame_left = None
        self.latest_frame_right = None
        
        print("🔍 TERMINAL STEREO VISION DEBUGGER")
        print("=" * 40)
        print("Loading YOLO model...")
        
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            print("✓ YOLO model loaded")
        except Exception as e:
            print(f"✗ YOLO loading failed: {e}")
            exit(1)
    
    def detect_objects(self, frame):
        """Detect objects using YOLO."""
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
            print(f"YOLO detection error: {e}")
        
        return detections
    
    def analyze_matching(self, detections_left, detections_right):
        """Analyze why objects aren't matching between cameras."""
        print("\n🔍 MATCHING ANALYSIS:")
        print("-" * 50)
        
        if not detections_left and not detections_right:
            print("❌ NO OBJECTS detected in either camera")
            print("💡 SOLUTIONS:")
            print("   - Lower confidence threshold (type 'l')")
            print("   - Improve lighting")
            print("   - Move closer to cameras")
            print("   - Try different objects (person, bottle, chair, etc.)")
            return
        
        if detections_left and not detections_right:
            print("❌ Objects ONLY in LEFT camera")
            print("💡 SOLUTIONS:")
            print("   - Check if right camera is working")
            print("   - Ensure both cameras point at same area")
            print("   - Lower confidence threshold (type 'l')")
            return
        
        if detections_right and not detections_left:
            print("❌ Objects ONLY in RIGHT camera")
            print("💡 SOLUTIONS:")
            print("   - Check if left camera is working") 
            print("   - Ensure both cameras point at same area")
            print("   - Lower confidence threshold (type 'l')")
            return
        
        # Both cameras have detections - analyze matching
        print("✓ Objects detected in BOTH cameras - analyzing matches...")
        
        grouped_right = defaultdict(list)
        for det in detections_right:
            grouped_right[det['class_name']].append(det)
        
        matches_found = 0
        
        for det_l in detections_left:
            class_name = det_l['class_name']
            
            print(f"\n🔎 Analyzing {class_name} from left camera:")
            print(f"   Left position: {det_l['center']}")
            
            if class_name not in grouped_right:
                print(f"   ❌ No {class_name} found in right camera")
                continue
            
            best_match = None
            best_y_diff = float('inf')
            
            for det_r in grouped_right[class_name]:
                y_diff = abs(det_l['center'][1] - det_r['center'][1])
                x_disparity = det_l['center'][0] - det_r['center'][0]
                
                print(f"   Right candidate: {det_r['center']}")
                print(f"     Y-difference: {y_diff:.1f} pixels (max allowed: 50)")
                print(f"     X-disparity: {x_disparity:.1f} pixels")
                
                if y_diff < 50:
                    if x_disparity > 0:
                        print(f"     ✅ SHOULD MATCH! (Y-diff OK, positive disparity)")
                        matches_found += 1
                        if y_diff < best_y_diff:
                            best_y_diff = y_diff
                            best_match = det_r
                    else:
                        print(f"     ❌ Negative disparity - cameras might be swapped")
                        print(f"        💡 Try swapping camera indices 0⟷2")
                else:
                    print(f"     ❌ Y-difference too large (vertical misalignment)")
                    print(f"        💡 Cameras need better vertical alignment")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Expected matches: {matches_found}")
        print(f"   Left detections: {len(detections_left)}")
        print(f"   Right detections: {len(detections_right)}")
        
        if matches_found == 0:
            print("\n❌ NO MATCHES FOUND")
            print("💡 TOP SOLUTIONS:")
            print("   1. Check camera vertical alignment (Y-difference)")
            print("   2. Try swapping cameras: change config from 0,2 to 2,0")
            print("   3. Lower confidence threshold (type 'l')")
            print("   4. Ensure both cameras see the same area")
        else:
            print(f"\n✅ Found {matches_found} potential matches!")
            print("💡 If you're still seeing blue boxes instead of green:")
            print("   - Check your stereo matching code")
            print("   - Verify calibration is working")
            print("   - Make sure match_objects() method is being called")
    
    def capture_frames(self):
        """Capture frames in background thread."""
        cam_left = cv2.VideoCapture(0)
        cam_right = cv2.VideoCapture(2)
        
        if not (cam_left.isOpened() and cam_right.isOpened()):
            print("❌ Failed to open cameras 0 and 2")
            self.running = False
            return
        
        print("✓ Camera windows will appear...")
        
        try:
            while self.running:
                ret_left, frame_left = cam_left.read()
                ret_right, frame_right = cam_right.read()
                
                if ret_left and ret_right:
                    self.latest_frame_left = frame_left.copy()
                    self.latest_frame_right = frame_right.copy()
                    
                    # Show camera feeds
                    cv2.imshow('Left Camera (0)', frame_left)
                    cv2.imshow('Right Camera (2)', frame_right)
                    
                    # Check for window close button
                    if cv2.waitKey(1) & 0xFF == ord('x'):
                        self.running = False
                
                time.sleep(0.03)  # ~30 FPS
        finally:
            cam_left.release()
            cam_right.release()
            cv2.destroyAllWindows()
    
    def analyze_current_frame(self):
        """Analyze the current frame."""
        if self.latest_frame_left is None or self.latest_frame_right is None:
            print("❌ No frames available - check camera connection")
            return
        
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING FRAME (confidence: {self.confidence_threshold:.1f})")
        print(f"{'='*60}")
        
        detections_left = self.detect_objects(self.latest_frame_left)
        detections_right = self.detect_objects(self.latest_frame_right)
        
        print(f"\n📷 LEFT CAMERA ({len(detections_left)} objects):")
        for i, det in enumerate(detections_left):
            print(f"   {i+1}. {det['class_name']} (conf: {det['confidence']:.2f}) at {det['center']}")
        
        print(f"\n📷 RIGHT CAMERA ({len(detections_right)} objects):")
        for i, det in enumerate(detections_right):
            print(f"   {i+1}. {det['class_name']} (conf: {det['confidence']:.2f}) at {det['center']}")
        
        self.analyze_matching(detections_left, detections_right)
        print(f"\n{'='*60}")
    
    def run_debug_session(self):
        """Run interactive debug session."""
        print("\n🎥 Starting camera capture...")
        
        # Start camera capture in background
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        capture_thread.start()
        
        time.sleep(2)  # Give cameras time to start
        
        if not self.running:
            print("❌ Camera startup failed")
            return
        
        print("✓ Cameras connected")
        print("\n🎮 COMMANDS:")
        print("   Enter = Analyze current frame")
        print("   l = Lower confidence threshold")
        print("   h = Higher confidence threshold") 
        print("   q = Quit")
        print("-" * 50)
        
        try:
            while self.running:
                try:
                    command = input("Debug> ").strip().lower()
                    
                    if command == '' or command == 'a':  # Enter or 'a' for analyze
                        self.analyze_current_frame()
                        
                    elif command == 'l':  # Lower confidence
                        old_conf = self.confidence_threshold
                        self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
                        print(f"📊 Confidence: {old_conf:.1f} → {self.confidence_threshold:.1f}")
                        
                    elif command == 'h':  # Higher confidence
                        old_conf = self.confidence_threshold
                        self.confidence_threshold = min(0.9, self.confidence_threshold + 0.1)
                        print(f"📊 Confidence: {old_conf:.1f} → {self.confidence_threshold:.1f}")
                        
                    elif command == 'q':  # Quit
                        break
                        
                    elif command == 'help':
                        print("\nCommands:")
                        print("  Enter/a - Analyze frame")
                        print("  l - Lower confidence")  
                        print("  h - Higher confidence")
                        print("  q - Quit")
                        
                    else:
                        print(f"Unknown command: '{command}' (try 'help')")
                        
                except (EOFError, KeyboardInterrupt):
                    break
                    
        finally:
            self.running = False
            print("\n✓ Debug session ended")

def main():
    """Main entry point."""
    try:
        debugger = TerminalStereoDebugger()
        debugger.run_debug_session()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()