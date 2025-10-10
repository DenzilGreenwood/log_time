#!/usr/bin/env python3
"""
Manual WebGL Server Launcher
============================

Simple manual server to test the enhanced WebGL visualization.
"""

import http.server
import socketserver
import webbrowser
import threading
import time


def launch_server():
    """Launch server on port 8081."""
    PORT = 8081
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Quiet logging
    
    print("🌐 LTQG WebGL Server")
    print("=" * 30)
    print(f"📍 Server: http://localhost:{PORT}")
    print(f"📁 File: ltqg_black_hole_webgl.html")
    print("🎨 Enhanced: Multiple UI fixes + Interactive legend")
    print()
    print("✨ Recent Fixes:")
    print("  • Readable color picker labels (bright white)")
    print("  • Interactive legend with live color bar")
    print("  • Color-coded bullet points")
    print("  • Fixed geodesic animation (runs in all states)")
    print("  • Fixed dropdown visibility (dark theme)")
    print("  • Custom dropdown styling with blue accents")
    print("  • Fixed shadow system (ground plane + better quality)")
    print()
    
    # Auto-open browser
    def open_browser():
        time.sleep(1.5)
        url = f"http://localhost:{PORT}/ltqg_black_hole_webgl.html"
        print(f"🚀 Opening: {url}")
        webbrowser.open(url)
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("✅ Server running... Press Ctrl+C to stop")
            print("-" * 50)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    launch_server()