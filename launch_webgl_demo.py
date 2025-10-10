#!/usr/bin/env python3
"""
WebGL Black Hole Visualization Launcher
======================================

Launches the interactive LTQG black hole visualization in a web browser.
This provides a 3D rotatable view with real-time σ-time animation,
geodesic tracking, and interactive controls.

Features:
- Real-time σ-time animation showing temporal evolution
- Interactive 3D rotation and zoom with mouse/touch
- Adjustable Schwarzschild radius and geometry parameters
- Toggleable event horizon, geodesics, and wireframe modes
- Smooth color transitions representing log-time coordinate

Usage:
    python launch_webgl_demo.py

Author: Denzil James Greenwood
"""

import os
import sys
import webbrowser
import http.server
import socketserver
import threading
import time
from pathlib import Path


def find_available_port(start_port=8000, max_attempts=50):
    """Find an available port starting from start_port."""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")


def start_local_server(port=8000, directory=None):
    """Start a local HTTP server to serve the WebGL demo."""
    if directory:
        os.chdir(directory)
    
    handler = http.server.SimpleHTTPRequestHandler
    
    class QuietHTTPServer(socketserver.TCPServer):
        def log_message(self, format, *args):
            # Suppress default HTTP server logging
            pass
    
    handler.log_message = lambda self, format, *args: None
    
    try:
        with QuietHTTPServer(("localhost", port), handler) as httpd:
            print(f"🌐 Local server running at http://localhost:{port}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")


def launch_webgl_demo():
    """Launch the WebGL black hole demonstration."""
    print("LTQG WebGL Black Hole Visualization Launcher")
    print("=" * 50)
    
    # Find the HTML file
    current_dir = Path(__file__).parent
    html_file = current_dir / "ltqg_black_hole_webgl.html"
    
    if not html_file.exists():
        print(f"❌ Error: {html_file} not found!")
        print("Please make sure the WebGL demo file exists.")
        return False
    
    # Check if we can use a simple file:// URL or need a server
    try:
        # Try to find an available port
        port = find_available_port()
        
        print(f"🚀 Starting local server on port {port}...")
        print(f"📁 Serving from: {current_dir}")
        
        # Start server in a separate thread
        server_thread = threading.Thread(
            target=start_local_server, 
            args=(port, str(current_dir)),
            daemon=True
        )
        server_thread.start()
        
        # Give server time to start
        time.sleep(1)
        
        # Open in browser
        url = f"http://localhost:{port}/ltqg_black_hole_webgl.html"
        print(f"🌍 Opening {url} in browser...")
        
        webbrowser.open(url)
        
        print("\n✨ WebGL Demo Features:")
        print("• 🔄 Interactive 3D rotation with mouse/touch")
        print("• ⏯️  Real-time σ-time animation (toggleable)")
        print("• 🎛️  Live parameter controls (top-right panel)")
        print("• 🕳️  Event horizon visualization")
        print("• 🛸 Animated geodesic with particle tracking")
        print("• 🎨 Dynamic color mapping of log-time coordinate")
        print("• 📏 Adjustable Schwarzschild radius and geometry")
        
        print(f"\n🎯 Key LTQG Insights Demonstrated:")
        print("• Smooth geometry across event horizon in σ-coordinates")
        print("• Singularity regularization via asymptotic silence")
        print("• Continuous funnel depth scaling ~ exp(-σ/κ)")
        print("• Geodesic regularization (no finite-time singularity)")
        
        print(f"\n📖 Controls:")
        print("• Mouse: Rotate view, scroll to zoom")
        print("• Top-right panel: Adjust all parameters")
        print("• Toggle wireframe, horizon, geodesic visibility")
        print("• Play/pause σ-time animation")
        
        print(f"\n🔧 Server running... Press Ctrl+C to stop")
        
        try:
            # Keep the main thread alive while server runs
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            return True
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        
        # Fallback: try to open directly
        print("🔄 Trying direct file access...")
        try:
            file_url = html_file.as_uri()
            webbrowser.open(file_url)
            print(f"✅ Opened {file_url}")
            print("⚠️  Note: Some browsers may restrict WebGL from file:// URLs")
            print("   If the demo doesn't work, try running with a local server.")
            return True
        except Exception as e2:
            print(f"❌ Failed to open file directly: {e2}")
            return False


def create_desktop_shortcut():
    """Create a desktop shortcut for easy access (Windows)."""
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, "LTQG Black Hole WebGL.lnk")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{__file__}"'
        shortcut.WorkingDirectory = str(Path(__file__).parent)
        shortcut.IconLocation = sys.executable
        shortcut.Description = "LTQG Black Hole WebGL Visualization"
        shortcut.save()
        
        print(f"✅ Desktop shortcut created: {shortcut_path}")
        return True
        
    except ImportError:
        print("⚠️  Desktop shortcut creation requires: pip install winshell pywin32")
        return False
    except Exception as e:
        print(f"⚠️  Could not create desktop shortcut: {e}")
        return False


def show_system_info():
    """Show system information for debugging."""
    print("\n🔍 System Information:")
    print(f"• Python: {sys.version}")
    print(f"• Platform: {sys.platform}")
    print(f"• Working directory: {os.getcwd()}")
    
    # Check for required files
    current_dir = Path(__file__).parent
    html_file = current_dir / "ltqg_black_hole_webgl.html"
    print(f"• HTML file exists: {html_file.exists()}")
    
    if html_file.exists():
        file_size = html_file.stat().st_size
        print(f"• HTML file size: {file_size:,} bytes")


def main():
    """Main launcher function with options."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--info":
            show_system_info()
            return
        elif sys.argv[1] == "--shortcut":
            create_desktop_shortcut()
            return
        elif sys.argv[1] == "--help":
            print(__doc__)
            print("\nOptions:")
            print("  --info      Show system information")
            print("  --shortcut  Create desktop shortcut (Windows)")
            print("  --help      Show this help message")
            return
    
    success = launch_webgl_demo()
    
    if not success:
        print("\n❌ Failed to launch WebGL demo")
        print("🔧 Troubleshooting:")
        print("• Make sure ltqg_black_hole_webgl.html exists")
        print("• Try running with --info for system details")
        print("• Check that your browser supports WebGL")
        print("• Some antivirus software may block local servers")
        sys.exit(1)


if __name__ == "__main__":
    main()