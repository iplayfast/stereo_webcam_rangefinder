#!/usr/bin/env python3
"""
FILE: stereo_gui.py
PATH: Save this in the SAME DIRECTORY as webcam_stereo_vision.py

Pygame GUI for Stereo Vision System
==================================

A clean graphical interface that displays:
- Stereo camera feeds with object detection overlays
- Real-time object tracking table
- System status and controls
- Non-blocking keyboard input

REQUIREMENTS:
    pip install pygame

This module is imported by the main webcam_stereo_vision.py script.
Do not run this file directly - it's a supporting module.
"""

import pygame
import cv2
import numpy as np
import threading
import time
from collections import defaultdict

class StereoVisionGUI:
    def __init__(self, width=1200, height=800):
        # Initialize Pygame with comprehensive fallback options
        self.initialization_successful = False
        
        try:
            # Set environment for better compatibility BEFORE pygame init
            import os
            
            # Force software rendering to avoid OpenGL context issues
            print("🔧 Configuring pygame for maximum compatibility...")
            os.environ['SDL_VIDEODRIVER'] = 'x11'
            os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'  # Force software OpenGL
            os.environ['SDL_RENDER_DRIVER'] = 'software'  # Force software rendering
            
            pygame.init()
            pygame.font.init()
            
            # Create display with specific flags to avoid OpenGL
            try:
                # Try without any OpenGL flags first
                self.screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
                pygame.display.set_caption("Stereo Vision Object Tracking")
                
                # Test basic operations without OpenGL calls
                self.screen.fill((0, 0, 0))
                pygame.display.update()  # Use update() instead of flip()
                print("✓ Software rendering successful")
                
            except pygame.error as e:
                print(f"⚠️  Standard mode failed: {e}")
                print("   Trying compatibility mode...")
                
                # Try absolute minimum compatibility mode
                try:
                    pygame.display.quit()
                    self.screen = pygame.display.set_mode((width, height), 0)  # No flags at all
                    pygame.display.set_caption("Stereo Vision Object Tracking")
                    self.screen.fill((0, 0, 0))
                    pygame.display.update()
                    print("✓ Compatibility mode successful")
                    
                except pygame.error as e2:
                    print(f"⚠️  Compatibility mode failed: {e2}")
                    raise pygame.error(f"All display modes failed: {e2}")
                
        except Exception as e:
            print(f"✗ Pygame initialization completely failed: {e}")
            print("   Likely causes:")
            print("   • OpenGL drivers corrupted")
            print("   • Multiple graphics contexts conflicting")
            print("   • Hardware acceleration completely broken")
            raise Exception(f"Cannot initialize pygame: {e}") from e
        
        # Display settings
        self.width = width
        self.height = height
        self.use_flip = False  # Use update() instead of flip() to avoid OpenGL
        
        # Colors
        self.colors = {
            'bg': (20, 25, 35),           # Dark blue background
            'panel': (35, 40, 50),        # Panel background
            'text': (255, 255, 255),      # White text
            'accent': (64, 150, 255),     # Blue accent
            'success': (64, 255, 150),    # Green success
            'warning': (255, 200, 64),    # Yellow warning
            'error': (255, 100, 100),     # Red error
            'border': (80, 90, 110)       # Border color
        }
        
        # Fonts with better fallback handling
        self.fonts = {}
        try:
            self.fonts = {
                'title': pygame.font.Font(None, 24),
                'normal': pygame.font.Font(None, 20),
                'small': pygame.font.Font(None, 16),
                'large': pygame.font.Font(None, 28)
            }
        except:
            try:
                # Try system fonts
                self.fonts = {
                    'title': pygame.font.SysFont('Arial', 24),
                    'normal': pygame.font.SysFont('Arial', 20),
                    'small': pygame.font.SysFont('Arial', 16),
                    'large': pygame.font.SysFont('Arial', 28)
                }
            except:
                # Last resort - use default font
                default_font = pygame.font.get_default_font()
                self.fonts = {
                    'title': pygame.font.Font(default_font, 24),
                    'normal': pygame.font.Font(default_font, 20),
                    'small': pygame.font.Font(default_font, 16),
                    'large': pygame.font.Font(default_font, 28)
                }
        
        # Layout
        self.camera_width = 320
        self.camera_height = 240
        self.panel_width = 350
        
        # State
        self.running = True
        self.system_ref = None
        self.last_key_time = {}
        self.display_error_count = 0  # Track display errors
        self.gui_failed = False  # Track if GUI has failed
        
        # Input handling
        self.key_commands = {
            pygame.K_q: 'quit',
            pygame.K_c: 'calibrate',
            pygame.K_l: 'lower_confidence',
            pygame.K_h: 'higher_confidence',
            pygame.K_m: 'change_model',
            pygame.K_d: 'debug',
            pygame.K_s: 'settings',
            pygame.K_r: 'reset_tracking'
        }
        
        # Model selection state
        self.show_model_selector = False
        self.model_options = [
            ('yolov8n.pt', 'Nano - Fastest (~6MB)'),
            ('yolov8s.pt', 'Small - Balanced (~22MB)'),
            ('yolov8m.pt', 'Medium - Accurate (~52MB)'),
            ('yolov8l.pt', 'Large - Very Accurate (~87MB)'),
            ('yolov8x.pt', 'XLarge - Most Accurate (~136MB)')
        ]
        self.selected_model_index = 0
        
        self.initialization_successful = True
        print("✓ Pygame GUI initialized with software rendering")

    def set_system_reference(self, system):
        """Set reference to the main stereo vision system."""
        self.system_ref = system

    def handle_events(self):
        """Handle pygame events and keyboard input."""
        current_time = time.time()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                if self.system_ref:
                    self.system_ref.running = False
                return
            
            elif event.type == pygame.KEYDOWN:
                # Handle model selector
                if self.show_model_selector:
                    if event.key == pygame.K_ESCAPE:
                        self.show_model_selector = False
                    elif event.key == pygame.K_UP:
                        self.selected_model_index = (self.selected_model_index - 1) % len(self.model_options)
                    elif event.key == pygame.K_DOWN:
                        self.selected_model_index = (self.selected_model_index + 1) % len(self.model_options)
                    elif event.key == pygame.K_RETURN:
                        self.apply_model_selection()
                    continue
                
                # Normal command handling with rate limiting
                if event.key in self.key_commands:
                    command = self.key_commands[event.key]
                    
                    # Rate limiting for confidence changes
                    if command in ['lower_confidence', 'higher_confidence']:
                        if command in self.last_key_time:
                            if current_time - self.last_key_time[command] < 0.1:  # 100ms limit
                                continue
                        self.last_key_time[command] = current_time
                    
                    self.execute_command(command)

    def execute_command(self, command):
        """Execute a keyboard command."""
        if not self.system_ref:
            return
            
        if command == 'quit':
            self.running = False
            self.system_ref.running = False
            
        elif command == 'calibrate':
            self.system_ref.force_calibration_trigger = True
            
        elif command == 'lower_confidence':
            old_conf = self.system_ref.confidence_threshold
            self.system_ref.confidence_threshold = max(0.1, old_conf - 0.05)
            
        elif command == 'higher_confidence':
            old_conf = self.system_ref.confidence_threshold
            self.system_ref.confidence_threshold = min(0.9, old_conf + 0.05)
            
        elif command == 'change_model':
            self.show_model_selector = True
            
        elif command == 'reset_tracking':
            self.system_ref.tracked_objects.clear()
            self.system_ref.detection_count = 0

    def apply_model_selection(self):
        """Apply the selected model."""
        if not self.system_ref:
            return
            
        model_path = self.model_options[self.selected_model_index][0]
        success, message = self.system_ref.load_yolo_model(model_path)
        
        if success:
            # Clear tracking since different models detect different objects
            self.system_ref.tracked_objects.clear()
            self.system_ref.detection_count = 0
            
        self.show_model_selector = False

    def cv2_to_pygame(self, cv_image):
        """Convert OpenCV image to Pygame surface."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Rotate 90 degrees and flip to correct orientation
        rgb_image = np.rot90(rgb_image)
        rgb_image = np.flipud(rgb_image)
        # Create pygame surface
        return pygame.surfarray.make_surface(rgb_image)

    def draw_camera_feeds(self):
        """Draw the stereo camera feeds."""
        if not self.system_ref:
            return
            
        y_offset = 10
        
        # Left camera
        with self.system_ref.frame_lock:
            if self.system_ref.display_frame_left is not None:
                frame = cv2.resize(self.system_ref.display_frame_left, (self.camera_width, self.camera_height))
                surface = self.cv2_to_pygame(frame)
                self.screen.blit(surface, (10, y_offset))
                
        # Camera label
        label = self.fonts['small'].render("Left Camera", True, self.colors['text'])
        self.screen.blit(label, (10, y_offset + self.camera_height + 5))
        
        # Right camera
        with self.system_ref.frame_lock:
            if self.system_ref.display_frame_right is not None:
                frame = cv2.resize(self.system_ref.display_frame_right, (self.camera_width, self.camera_height))
                surface = self.cv2_to_pygame(frame)
                self.screen.blit(surface, (self.camera_width + 20, y_offset))
                
        # Camera label
        label = self.fonts['small'].render("Right Camera", True, self.colors['text'])
        self.screen.blit(label, (self.camera_width + 20, y_offset + self.camera_height + 5))

    def draw_system_status(self):
        """Draw system status panel."""
        if not self.system_ref:
            return
            
        panel_x = self.width - self.panel_width
        panel_y = 10
        panel_height = 200
        
        # Draw panel background
        pygame.draw.rect(self.screen, self.colors['panel'], 
                        (panel_x, panel_y, self.panel_width - 10, panel_height))
        pygame.draw.rect(self.screen, self.colors['border'], 
                        (panel_x, panel_y, self.panel_width - 10, panel_height), 2)
        
        y = panel_y + 10
        x = panel_x + 10
        line_height = 25
        
        # Title
        title = self.fonts['title'].render("System Status", True, self.colors['accent'])
        self.screen.blit(title, (x, y))
        y += 35
        
        # Calibration status
        if self.system_ref.is_calibrated:
            status_text = "✓ CALIBRATED"
            color = self.colors['success']
            if self.system_ref.cameras_swapped:
                status_text += " (swapped)"
        else:
            status_text = "⚠ NOT CALIBRATED"
            color = self.colors['warning']
            
        status = self.fonts['normal'].render(status_text, True, color)
        self.screen.blit(status, (x, y))
        y += line_height
        
        # YOLO model
        model_name = self.system_ref.config.get('yolo_model', 'yolov8n.pt')
        model_status = "✓" if self.system_ref.yolo_model else "✗"
        model_color = self.colors['success'] if self.system_ref.yolo_model else self.colors['error']
        
        model_text = f"Model: {model_name} {model_status}"
        model = self.fonts['normal'].render(model_text, True, model_color)
        self.screen.blit(model, (x, y))
        y += line_height
        
        # Confidence
        conf_text = f"Confidence: {self.system_ref.confidence_threshold:.2f}"
        conf = self.fonts['normal'].render(conf_text, True, self.colors['text'])
        self.screen.blit(conf, (x, y))
        y += line_height
        
        # Object count
        obj_count = len(self.system_ref.tracked_objects)
        obj_text = f"Tracked Objects: {obj_count}"
        obj = self.fonts['normal'].render(obj_text, True, self.colors['text'])
        self.screen.blit(obj, (x, y))

    def draw_object_tracking(self):
        """Draw object tracking table."""
        if not self.system_ref:
            return
            
        panel_x = self.width - self.panel_width
        panel_y = 220
        panel_height = 400
        
        # Draw panel background
        pygame.draw.rect(self.screen, self.colors['panel'], 
                        (panel_x, panel_y, self.panel_width - 10, panel_height))
        pygame.draw.rect(self.screen, self.colors['border'], 
                        (panel_x, panel_y, self.panel_width - 10, panel_height), 2)
        
        y = panel_y + 10
        x = panel_x + 10
        line_height = 20
        
        # Title
        title = self.fonts['title'].render("Detected Objects", True, self.colors['accent'])
        self.screen.blit(title, (x, y))
        y += 30
        
        # Headers
        headers = ["Object", "Distance", "Count", "Status"]
        header_x = [x, x + 80, x + 160, x + 220]
        for i, header in enumerate(headers):
            text = self.fonts['small'].render(header, True, self.colors['text'])
            self.screen.blit(text, (header_x[i], y))
        y += 25
        
        # Draw line under headers
        pygame.draw.line(self.screen, self.colors['border'], 
                        (x, y), (x + self.panel_width - 30, y), 1)
        y += 10
        
        # Object data
        sorted_objects = self.system_ref.get_sorted_objects()
        max_visible = 15  # Maximum objects to display
        
        for i, (class_name, data) in enumerate(sorted_objects[:max_visible]):
            if data['distance'] is not None:
                distance_str = f"{data['distance']/1000:.2f}m"
                status_str = "✓ Visible"
                status_color = self.colors['success']
            else:
                distance_str = "?"
                status_str = "○ Lost"
                status_color = self.colors['warning']
            
            # Object name
            name_text = self.fonts['small'].render(class_name[:10], True, self.colors['text'])
            self.screen.blit(name_text, (header_x[0], y))
            
            # Distance
            dist_text = self.fonts['small'].render(distance_str, True, self.colors['text'])
            self.screen.blit(dist_text, (header_x[1], y))
            
            # Count
            count_text = self.fonts['small'].render(str(data['count']), True, self.colors['text'])
            self.screen.blit(count_text, (header_x[2], y))
            
            # Status
            status_text = self.fonts['small'].render(status_str, True, status_color)
            self.screen.blit(status_text, (header_x[3], y))
            
            y += line_height

    def draw_controls(self):
        """Draw control instructions."""
        panel_x = self.width - self.panel_width
        panel_y = 630
        panel_height = 160
        
        # Draw panel background
        pygame.draw.rect(self.screen, self.colors['panel'], 
                        (panel_x, panel_y, self.panel_width - 10, panel_height))
        pygame.draw.rect(self.screen, self.colors['border'], 
                        (panel_x, panel_y, self.panel_width - 10, panel_height), 2)
        
        y = panel_y + 10
        x = panel_x + 10
        line_height = 18
        
        # Title
        title = self.fonts['title'].render("Controls", True, self.colors['accent'])
        self.screen.blit(title, (x, y))
        y += 25
        
        # Control list
        controls = [
            "C - Calibrate system",
            "L/H - Lower/Higher confidence",
            "M - Change YOLO model",
            "R - Reset object tracking",
            "D - Debug information",
            "S - Show settings",
            "Q - Quit application"
        ]
        
        for control in controls:
            text = self.fonts['small'].render(control, True, self.colors['text'])
            self.screen.blit(text, (x, y))
            y += line_height

    def draw_model_selector(self):
        """Draw model selection overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Selection panel
        panel_width = 500
        panel_height = 300
        panel_x = (self.width - panel_width) // 2
        panel_y = (self.height - panel_height) // 2
        
        pygame.draw.rect(self.screen, self.colors['panel'], 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['accent'], 
                        (panel_x, panel_y, panel_width, panel_height), 3)
        
        y = panel_y + 20
        x = panel_x + 20
        line_height = 35
        
        # Title
        title = self.fonts['large'].render("Select YOLO Model", True, self.colors['accent'])
        self.screen.blit(title, (x, y))
        y += 50
        
        # Model options
        for i, (model, description) in enumerate(self.model_options):
            if i == self.selected_model_index:
                # Highlight selected option
                pygame.draw.rect(self.screen, self.colors['accent'], 
                               (x - 5, y - 5, panel_width - 30, line_height - 5))
                text_color = self.colors['bg']
            else:
                text_color = self.colors['text']
            
            model_text = f"{model} - {description}"
            text = self.fonts['normal'].render(model_text, True, text_color)
            self.screen.blit(text, (x, y))
            y += line_height
        
        # Instructions
        y += 20
        instructions = [
            "↑↓ - Navigate",
            "Enter - Select",
            "Esc - Cancel"
        ]
        
        for instruction in instructions:
            text = self.fonts['small'].render(instruction, True, self.colors['text'])
            self.screen.blit(text, (x, y))
            y += 20

    def draw_calibration_status(self):
        """Draw calibration status message if in calibration mode."""
        if not self.system_ref or not self.system_ref.force_calibration_trigger:
            return
            
        # Calibration banner
        banner_height = 40
        pygame.draw.rect(self.screen, self.colors['warning'], 
                        (0, 0, self.width, banner_height))
        
        message = "CALIBRATION MODE - Place exactly ONE object in view of BOTH cameras"
        text = self.fonts['normal'].render(message, True, self.colors['bg'])
        text_rect = text.get_rect(center=(self.width // 2, banner_height // 2))
        self.screen.blit(text, text_rect)

    def update_display(self):
        """Main display update method with comprehensive error handling."""
        if self.gui_failed:
            return False
            
        try:
            # Clear screen
            self.screen.fill(self.colors['bg'])
            
            # Draw main components
            self.draw_camera_feeds()
            self.draw_system_status()
            self.draw_object_tracking()
            self.draw_controls()
            self.draw_calibration_status()
            
            # Draw overlays
            if self.show_model_selector:
                self.draw_model_selector()
            
            # Update display - use update() instead of flip() to avoid OpenGL context issues
            try:
                pygame.display.update()  # Safer than flip() - doesn't require OpenGL context
                self.display_error_count = 0  # Reset error count on success
                return True
            except pygame.error as e:
                self.display_error_count += 1
                if self.display_error_count <= 2:  # Only print first couple errors
                    print(f"⚠️  Display update error #{self.display_error_count}: {e}")
                
                if self.display_error_count >= 2:  # Quick failure for broken displays
                    print("✗ Display completely broken even with software rendering.")
                    print("   This indicates a fundamental graphics system issue.")
                    self.gui_failed = True
                    self.running = False
                    if self.system_ref:
                        print("🔄 Automatically switching to terminal mode...")
                        self.system_ref.use_gui = False
                        self.system_ref.gui_working = True  # Enable OpenCV windows
                    return False
                    
        except Exception as e:
            print(f"✗ Critical display error: {e}")
            self.gui_failed = True
            self.running = False
            if self.system_ref:
                self.system_ref.use_gui = False
                self.system_ref.gui_working = True
            return False
        
        return True

    def run(self):
        """Main GUI loop with better error handling and communication."""
        clock = pygame.time.Clock()
        
        print("🎮 GUI starting - if you see this message but blank window, it will auto-fallback...")
        
        # Test initial display capability
        try:
            if not self.update_display():
                print("✗ Initial display test failed - GUI not viable")
                return
        except Exception as e:
            print(f"✗ GUI completely non-functional: {e}")
            if self.system_ref:
                self.system_ref.use_gui = False
                self.system_ref.gui_working = True
            return
        
        print("✓ GUI display test passed - starting main loop")
        
        try:
            while self.running and not self.gui_failed:
                try:
                    self.handle_events()
                    if not self.update_display():
                        break  # Display failed, exit gracefully
                    clock.tick(30)  # 30 FPS
                except pygame.error as e:
                    print(f"⚠️  Pygame error in main loop: {e}")
                    self.display_error_count += 1
                    if self.display_error_count >= 5:
                        print("✗ Too many pygame errors, giving up on GUI")
                        break
                    continue
                except Exception as e:
                    print(f"✗ Unexpected error in GUI main loop: {e}")
                    break
                    
        except KeyboardInterrupt:
            print("\n⚠️  GUI interrupted by user")
        finally:
            print("🔄 GUI thread ending - cleaning up pygame...")
            try:
                pygame.quit()
            except:
                pass  # Ignore cleanup errors
            
            # Notify main system that GUI has ended
            if self.system_ref and self.gui_failed:
                print("✓ Switching to terminal interface...")
                self.system_ref.use_gui = False
                # Don't stop the whole system, just switch interface mode

    def cleanup(self):
        """Clean up resources."""
        pygame.quit()