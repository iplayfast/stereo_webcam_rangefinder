#!/usr/bin/env python3
"""
Stereo Vision Controller - Separate Terminal Interface
====================================================

Run this in a SEPARATE terminal window to control the main stereo vision app
without interference from the constantly updating display.

USAGE:
1. Terminal 1: python webcam_stereo_vision.py --dual-terminal
2. Terminal 2: python stereo_controller.py

This gives you clean input/output separation.
"""

import os
import json
import time
import sys

class StereoController:
    def __init__(self):
        self.command_file = '/tmp/stereo_commands.json'
        self.status_file = '/tmp/stereo_status.json'
        self.running = True
        
        print("=" * 60)
        print("🎮 STEREO VISION CONTROLLER")
        print("=" * 60)
        print("This terminal controls the main stereo vision application.")
        print("Make sure the main app is running with --dual-terminal flag.")
        print("=" * 60)
        
    def send_command(self, command, params=None):
        """Send command to main application."""
        try:
            cmd_data = {
                'command': command,
                'params': params or {},
                'timestamp': time.time()
            }
            
            with open(self.command_file, 'w') as f:
                json.dump(cmd_data, f)
                
            print(f"✓ Sent command: {command}")
            
        except Exception as e:
            print(f"✗ Failed to send command: {e}")
    
    def read_status(self):
        """Read status from main application."""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                return status
        except:
            pass
        return None
    
    def show_status(self):
        """Display current system status."""
        status = self.read_status()
        if status:
            print("\n📊 SYSTEM STATUS:")
            print(f"  Calibrated: {'✓' if status.get('calibrated', False) else '✗'}")
            print(f"  Model: {status.get('model', 'unknown')}")
            print(f"  Confidence: {status.get('confidence', 'unknown')}")
            print(f"  Tracked Objects: {status.get('tracked_objects', 0)}")
            
            if 'message' in status:
                print(f"  Message: {status['message']}")
        else:
            print("⚠️  No status available. Is main app running with --dual-terminal?")
    
    def show_help(self):
        """Show available commands."""
        print("\n🎮 AVAILABLE COMMANDS:")
        print("  'c' - Trigger calibration")
        print("  'l' - Lower confidence threshold")
        print("  'h' - Higher confidence threshold") 
        print("  's' - Show system status")
        print("  'm' - Change YOLO model")
        print("  'r' - Reset object tracking")
        print("  'debug' - Request debug information")
        print("  'help' - Show this help")
        print("  'q' - Quit controller")
        print("  'quit' - Quit both controller and main app")
    
    def change_model(self):
        """Interactive model selection."""
        models = [
            'yolov8n.pt',
            'yolov8s.pt', 
            'yolov8m.pt',
            'yolov8l.pt',
            'yolov8x.pt'
        ]
        
        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        try:
            choice = input("Enter model number (1-5) or custom path: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= 5:
                model = models[int(choice) - 1]
            else:
                model = choice
                
            self.send_command('change_model', {'model': model})
            
        except (ValueError, KeyboardInterrupt):
            print("Model change cancelled")
    
    def run(self):
        """Main controller loop."""
        self.show_help()
        self.show_status()
        
        while self.running:
            try:
                print("\n" + "="*60)
                cmd = input("Controller> ").strip().lower()
                
                if cmd in ['q', 'exit']:
                    print("Controller shutting down...")
                    self.running = False
                    
                elif cmd == 'quit':
                    print("Shutting down both controller and main app...")
                    self.send_command('quit')
                    self.running = False
                    
                elif cmd == 'c':
                    self.send_command('calibrate')
                    
                elif cmd == 'l':
                    self.send_command('confidence_change', {'delta': -0.1})
                    
                elif cmd == 'h':
                    self.send_command('confidence_change', {'delta': 0.1})
                    
                elif cmd == 's':
                    self.show_status()
                    
                elif cmd == 'm':
                    self.change_model()
                    
                elif cmd == 'r':
                    self.send_command('reset_tracking')
                    
                elif cmd == 'debug':
                    print("Debug info will be shown in main app terminal")
                    self.send_command('debug')
                    
                elif cmd in ['help', '?']:
                    self.show_help()
                    
                elif cmd == '':
                    continue  # Empty input
                    
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for available commands")
                    
                # Small delay to let command process
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print("\nController interrupted")
                self.running = False
            except EOFError:
                print("\nInput stream closed")
                self.running = False
        
        # Cleanup
        try:
            if os.path.exists(self.command_file):
                os.remove(self.command_file)
        except:
            pass

def main():
    """Main entry point."""
    print("Starting Stereo Vision Controller...")
    
    # Check if main app is likely running
    controller = StereoController()
    
    # Quick status check
    initial_status = controller.read_status()
    if not initial_status:
        print("\n⚠️  WARNING: Main stereo vision app doesn't appear to be running")
        print("   Start it in another terminal with:")
        print("   python webcam_stereo_vision.py --dual-terminal")
        print("\n   This controller will still work once the main app starts.")
    
    try:
        controller.run()
    except Exception as e:
        print(f"Controller error: {e}")
    
    print("Controller shutdown complete")

if __name__ == "__main__":
    main()