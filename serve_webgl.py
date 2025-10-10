#!/usr/bin/env python3
"""
Simple HTTP Server for LTQG WebGL Demo
=====================================

Serves the WebGL visualization on localhost to avoid CORS issues.
"""

import http.server
import socketserver
import webbrowser
import threading
import time
import os
import sys
from pathlib import Path


def find_port(start=8080):
    """Find an available port."""
    import socket
    for port in range(start, start + 50):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None


def serve_demo(filename="ltqg_black_hole_webgl.html"):
    """Serve the WebGL demo."""
    
    # Check if file exists
    if not Path(filename).exists():
        print(f"❌ Error: {filename} not found!")
        return False
    
    # Find available port
    port = find_port()
    if not port:
        print("❌ Error: No available ports found!")
        return False
    
    print(f"🌐 Starting server on http://localhost:{port}")
    print(f"📁 Serving: {filename}")
    
    # Create custom handler to suppress logs
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress default logging
    
    try:
        with socketserver.TCPServer(("", port), QuietHandler) as httpd:
            # Open browser after short delay
            def open_browser():
                time.sleep(1)
                url = f"http://localhost:{port}/{filename}"
                print(f"🚀 Opening {url}")
                webbrowser.open(url)
            
            browser_thread = threading.Thread(target=open_browser, daemon=True)
            browser_thread.start()
            
            print("✅ Server running... Press Ctrl+C to stop")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        return True
    except Exception as e:
        print(f"❌ Server error: {e}")
        return False


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "ltqg_black_hole_webgl.html"
    serve_demo(filename)