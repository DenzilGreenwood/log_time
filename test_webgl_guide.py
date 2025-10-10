#!/usr/bin/env python3
"""
LTQG WebGL Testing Guide
========================

This script helps test and troubleshoot the WebGL black hole visualization.
"""

import webbrowser
import time
import subprocess
import sys
from pathlib import Path
import socket


def check_port_available(port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0


def find_browser():
    """Find available browser."""
    browsers = [
        'firefox',
        'chrome', 
        'msedge',
        'iexplore',
        'opera'
    ]
    
    for browser in browsers:
        try:
            subprocess.run([browser, '--version'], 
                         capture_output=True, check=True)
            return browser
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    return None


def test_webgl_features():
    """Test WebGL features in browser."""
    print("🔧 Testing WebGL Features")
    print("=" * 30)
    
    webgl_test_html = """
<!DOCTYPE html>
<html>
<head>
    <title>WebGL Test</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .test { margin: 10px 0; }
        .pass { color: green; }
        .fail { color: red; }
    </style>
</head>
<body>
    <h1>WebGL Capability Test</h1>
    <div id="results"></div>
    
    <script>
        function testWebGL() {
            const results = document.getElementById('results');
            
            // Test WebGL support
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            
            if (gl) {
                results.innerHTML += '<div class="test pass">✅ WebGL is supported</div>';
                
                // Test WebGL features
                const vendor = gl.getParameter(gl.VENDOR);
                const renderer = gl.getParameter(gl.RENDERER);
                const version = gl.getParameter(gl.VERSION);
                
                results.innerHTML += `<div class="test">📊 Vendor: ${vendor}</div>`;
                results.innerHTML += `<div class="test">🎮 Renderer: ${renderer}</div>`;
                results.innerHTML += `<div class="test">📋 Version: ${version}</div>`;
                
                // Test extensions
                const extensions = gl.getSupportedExtensions();
                results.innerHTML += `<div class="test">🧩 Extensions: ${extensions.length} available</div>`;
                
                // Test max texture size
                const maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE);
                results.innerHTML += `<div class="test">🖼️ Max Texture Size: ${maxTexture}x${maxTexture}</div>`;
                
                // Test max viewport
                const maxViewport = gl.getParameter(gl.MAX_VIEWPORT_DIMS);
                results.innerHTML += `<div class="test">📐 Max Viewport: ${maxViewport[0]}x${maxViewport[1]}</div>`;
                
            } else {
                results.innerHTML += '<div class="test fail">❌ WebGL is not supported</div>';
                results.innerHTML += '<div class="test">💡 Try updating your browser or graphics drivers</div>';
            }
            
            // Test Three.js loading
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
            script.onload = function() {
                results.innerHTML += '<div class="test pass">✅ Three.js loaded successfully</div>';
                if (typeof THREE !== 'undefined') {
                    results.innerHTML += `<div class="test">📦 Three.js version: ${THREE.REVISION}</div>`;
                }
            };
            script.onerror = function() {
                results.innerHTML += '<div class="test fail">❌ Failed to load Three.js</div>';
            };
            document.head.appendChild(script);
        }
        
        testWebGL();
    </script>
</body>
</html>
    """
    
    # Save test file
    test_file = Path("webgl_test.html")
    test_file.write_text(webgl_test_html, encoding='utf-8')
    
    print(f"✅ Created WebGL test file: {test_file.absolute()}")
    return test_file


def main():
    """Main testing routine."""
    print("🧪 LTQG WebGL Testing Guide")
    print("=" * 40)
    
    # Check files exist
    files_to_check = [
        "ltqg_black_hole_webgl.html",
        "ltqg_black_hole_webgl_fixed.html",
        "serve_webgl.py"
    ]
    
    print("\n📁 Checking files:")
    for file in files_to_check:
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} - NOT FOUND")
    
    # Test WebGL capabilities
    print("\n🔧 Creating WebGL capability test...")
    test_file = test_webgl_features()
    
    # Check available ports
    print(f"\n🌐 Checking ports:")
    ports = [8080, 8081, 8000, 3000, 5000]
    available_port = None
    
    for port in ports:
        if check_port_available(port):
            print(f"  ✅ Port {port} available")
            if available_port is None:
                available_port = port
        else:
            print(f"  ⚠️ Port {port} in use")
    
    if available_port:
        print(f"\n🚀 Recommended port: {available_port}")
    else:
        print("\n⚠️ All common ports are in use")
    
    # Check browser
    print(f"\n🌍 Checking browser:")
    browser = find_browser()
    if browser:
        print(f"  ✅ Found browser: {browser}")
    else:
        print("  ⚠️ No browser found in PATH")
    
    # Instructions
    print("\n📋 Testing Instructions:")
    print("=" * 25)
    print("1. Run the server:")
    print("   python serve_webgl.py")
    print()
    print("2. Open the visualization:")
    print("   http://localhost:8080/ltqg_black_hole_webgl_fixed.html")
    print()
    print("3. Test WebGL capabilities:")
    print(f"   Open: {test_file.absolute()}")
    print()
    print("4. Expected features in LTQG visualization:")
    print("   ✅ 3D funnel shape (black hole embedding)")
    print("   ✅ Event horizon circle")
    print("   ✅ Smooth color gradients")
    print("   ✅ Interactive camera controls (mouse drag/wheel)")
    print("   ✅ Play/Pause button")
    print("   ✅ Sigma-time slider")
    print("   ✅ Color scheme selector")
    print("   ✅ Wireframe toggle")
    print("   ✅ Geodesic paths")
    print()
    print("5. Common issues and fixes:")
    print("   🔧 Black screen → Check browser console (F12)")
    print("   🔧 No controls → Try refreshing page")
    print("   🔧 Slow performance → Lower grid resolution")
    print("   🔧 No colors → Check WebGL support")
    print("   🔧 Server errors → Try different port")
    print()
    print("6. Browser console check:")
    print("   • Open Developer Tools (F12)")
    print("   • Check Console tab for errors")
    print("   • Look for Three.js loading messages")
    print("   • Check for WebGL context creation")


if __name__ == "__main__":
    main()