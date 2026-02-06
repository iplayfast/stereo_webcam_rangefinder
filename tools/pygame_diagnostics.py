#!/usr/bin/env python3
"""
FILE: pygame_diagnostics.py
PATH: Save this as a diagnostic tool to identify pygame issues

Pygame Diagnostics Tool
=======================

This tool helps identify why pygame might be failing when it worked before.
Run this to get detailed information about your graphics setup.
"""

import os
import sys
import subprocess
import platform

def run_command(cmd):
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip(), result.stderr.strip()
    except:
        return "Command failed", ""

def check_display_environment():
    """Check display-related environment variables."""
    print("=" * 60)
    print("🖥️  DISPLAY ENVIRONMENT")
    print("=" * 60)
    
    env_vars = ['DISPLAY', 'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE', 'DESKTOP_SESSION']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var:<20}: {value}")
    
    # Check if we're in SSH
    if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
        print("🔍 SSH Session detected")
        if 'DISPLAY' not in os.environ:
            print("⚠️  No DISPLAY variable in SSH - X11 forwarding may not be enabled")
    
    print()

def check_graphics_info():
    """Check graphics hardware and drivers."""
    print("=" * 60)
    print("🎮 GRAPHICS HARDWARE & DRIVERS")
    print("=" * 60)
    
    # GPU info
    gpu_info, _ = run_command("lspci | grep -i vga")
    if gpu_info:
        print(f"GPU: {gpu_info}")
    
    # Graphics driver info
    nvidia_info, _ = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null")
    if nvidia_info and "NVIDIA" in gpu_info:
        print(f"NVIDIA Driver: {nvidia_info}")
    
    # OpenGL info
    glx_info, _ = run_command("glxinfo | grep 'OpenGL version' 2>/dev/null")
    if glx_info:
        print(f"OpenGL: {glx_info}")
    else:
        print("⚠️  glxinfo not available (install mesa-utils)")
    
    # Check if OpenGL is working
    glx_test, glx_err = run_command("glxgears -info 2>&1 | head -1")
    if "GL_RENDERER" in glx_test:
        print("✓ OpenGL appears to be working")
    else:
        print(f"⚠️  OpenGL test failed: {glx_err}")
    
    print()

def check_x11_wayland():
    """Check X11/Wayland status."""
    print("=" * 60)
    print("🪟 X11 / WAYLAND STATUS")
    print("=" * 60)
    
    session_type = os.environ.get('XDG_SESSION_TYPE', 'unknown')
    print(f"Session Type: {session_type}")
    
    if session_type == 'wayland':
        print("📝 Note: Wayland can sometimes cause pygame OpenGL issues")
        print("   Try setting XDG_SESSION_TYPE=x11 or use X11 session")
    
    # Check X11 accessibility
    xauth_info, _ = run_command("xauth list 2>/dev/null | wc -l")
    if xauth_info and int(xauth_info) > 0:
        print(f"✓ X11 authentication available ({xauth_info} entries)")
    else:
        print("⚠️  No X11 authentication found")
    
    # Test X11 connection
    xdpyinfo, xdpy_err = run_command("xdpyinfo 2>/dev/null | head -1")
    if xdpyinfo:
        print("✓ X11 connection working")
    else:
        print(f"⚠️  X11 connection failed: {xdpy_err}")
    
    print()

def check_pygame_specific():
    """Check pygame-specific issues."""
    print("=" * 60)
    print("🐍 PYGAME SPECIFIC TESTS")
    print("=" * 60)
    
    try:
        import pygame
        print(f"✓ Pygame version: {pygame.version.ver}")
        print(f"✓ SDL version: {pygame.version.SDL}")
        
        # Test pygame init
        try:
            pygame.init()
            print("✓ pygame.init() successful")
            
            # Test display modes
            modes = pygame.display.list_modes()
            print(f"✓ Available display modes: {len(modes) if modes != -1 else 'All'}")
            
            # Test video drivers
            drivers = pygame.display.get_driver()
            print(f"✓ Current video driver: {drivers}")
            
            # List all available drivers
            pygame.quit()
            os.environ['SDL_VIDEODRIVER'] = ''  # Reset
            
            print("\n📋 Testing video drivers:")
            test_drivers = ['x11', 'wayland', 'software', 'fbcon']
            
            for driver in test_drivers:
                try:
                    os.environ['SDL_VIDEODRIVER'] = driver
                    pygame.init()
                    test_screen = pygame.display.set_mode((100, 100))
                    test_screen.fill((0, 0, 0))
                    pygame.display.flip()
                    print(f"   ✓ {driver}: Working")
                    pygame.quit()
                except Exception as e:
                    print(f"   ✗ {driver}: Failed ({str(e)[:50]})")
                    pygame.quit()
            
            # Reset environment
            if 'SDL_VIDEODRIVER' in os.environ:
                del os.environ['SDL_VIDEODRIVER']
                
        except Exception as e:
            print(f"✗ pygame.init() failed: {e}")
            
    except ImportError:
        print("✗ Pygame not installed")
    
    print()

def check_system_changes():
    """Check for recent system changes that might affect graphics."""
    print("=" * 60)
    print("🔄 RECENT SYSTEM CHANGES")
    print("=" * 60)
    
    # Check recent package updates (Ubuntu/Debian)
    recent_updates, _ = run_command("grep 'install\\|upgrade' /var/log/apt/history.log 2>/dev/null | tail -5")
    if recent_updates:
        print("Recent package changes:")
        for line in recent_updates.split('\n'):
            print(f"   {line}")
    
    # Check kernel version
    kernel, _ = run_command("uname -r")
    print(f"Kernel: {kernel}")
    
    # Check if graphics drivers were recently updated
    driver_updates, _ = run_command("grep -i 'nvidia\\|mesa\\|graphics' /var/log/apt/history.log 2>/dev/null | tail -3")
    if driver_updates:
        print("Recent graphics-related updates:")
        for line in driver_updates.split('\n'):
            if line.strip():
                print(f"   {line}")
    
    print()

def suggest_solutions():
    """Suggest solutions based on common issues."""
    print("=" * 60)
    print("💡 POTENTIAL SOLUTIONS")
    print("=" * 60)
    
    solutions = [
        "1. Try forcing software rendering:",
        "   export SDL_VIDEODRIVER=software",
        "   python webcam_stereo_vision.py",
        "",
        "2. If using Wayland, try X11 session:",
        "   Log out → select X11 session → log back in",
        "",
        "3. Reset graphics drivers (NVIDIA):",
        "   sudo apt update && sudo apt install --reinstall nvidia-driver-*",
        "",
        "4. Update pygame:",
        "   pip install --upgrade pygame",
        "",
        "5. Use terminal mode (always works):",
        "   python webcam_stereo_vision.py --terminal",
        "",
        "6. Check for conflicting processes:",
        "   killall -9 python",
        "   # Then restart your application"
    ]
    
    for solution in solutions:
        print(solution)

def main():
    """Run complete diagnostics."""
    print("🔍 PYGAME GRAPHICS DIAGNOSTICS")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print()
    
    check_display_environment()
    check_graphics_info()
    check_x11_wayland()
    check_pygame_specific()
    check_system_changes()
    suggest_solutions()
    
    print("\n📋 SUMMARY:")
    print("Save this output and look for:")
    print("• ⚠️  Warnings that indicate potential issues")
    print("• ✗ Failed tests that show what's broken")
    print("• Recent changes that might have caused the problem")

if __name__ == "__main__":
    main()