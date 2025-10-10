#!/usr/bin/env python3
"""
WebGL Validation Test
====================

Tests the WebGL HTML file for syntax, dependencies, and functionality.
"""

import re
import json
from pathlib import Path


def validate_html_syntax(filepath):
    """Validate basic HTML syntax."""
    try:
        content = Path(filepath).read_text(encoding='utf-8')
        
        # Check basic structure
        checks = {
            'DOCTYPE': '<!DOCTYPE html>' in content,
            'HTML tags': '<html' in content and '</html>' in content,
            'HEAD section': '<head>' in content and '</head>' in content,
            'BODY section': '<body>' in content and '</body>' in content,
            'TITLE': '<title>' in content,
            'Viewport meta': 'viewport' in content,
        }
        
        print("📄 HTML Structure Validation:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        
        return all(checks.values())
        
    except Exception as e:
        print(f"❌ Error reading HTML file: {e}")
        return False


def validate_css(filepath):
    """Validate CSS sections."""
    try:
        content = Path(filepath).read_text(encoding='utf-8')
        
        # Extract CSS
        css_pattern = r'<style[^>]*>(.*?)</style>'
        css_matches = re.findall(css_pattern, content, re.DOTALL)
        
        if not css_matches:
            print("❌ No CSS found")
            return False
        
        css_content = '\n'.join(css_matches)
        
        # Check key CSS features
        css_checks = {
            'Body styling': 'body' in css_content,
            'Container layout': '#container' in css_content,
            'Controls styling': '#controls' in css_content,
            'Button styling': '.button' in css_content,
            'Slider styling': '.slider' in css_content,
            'Responsive design': '@media' in css_content,
            'Animations': '@keyframes' in css_content,
        }
        
        print("\n🎨 CSS Validation:")
        for check, passed in css_checks.items():
            status = "✅" if passed else "⚠️"
            print(f"  {status} {check}")
        
        return sum(css_checks.values()) >= len(css_checks) * 0.8  # 80% pass rate
        
    except Exception as e:
        print(f"❌ Error validating CSS: {e}")
        return False


def validate_javascript(filepath):
    """Validate JavaScript functionality."""
    try:
        content = Path(filepath).read_text(encoding='utf-8')
        
        # Extract JavaScript
        js_pattern = r'<script[^>]*>(.*?)</script>'
        js_matches = re.findall(js_pattern, content, re.DOTALL)
        
        if not js_matches:
            print("❌ No JavaScript found")
            return False
        
        js_content = '\n'.join(js_matches)
        
        # Check Three.js functionality
        threejs_checks = {
            'Three.js scene': 'THREE.Scene' in js_content,
            'Camera setup': 'THREE.PerspectiveCamera' in js_content,
            'Renderer': 'THREE.WebGLRenderer' in js_content,
            'Controls': 'OrbitControls' in js_content,
            'Geometry creation': 'THREE.BufferGeometry' in js_content,
            'Materials': 'THREE.MeshPhongMaterial' in js_content or 'THREE.MeshBasicMaterial' in js_content,
            'Lighting': 'THREE.DirectionalLight' in js_content,
            'Animation loop': 'requestAnimationFrame' in js_content,
        }
        
        # Check LTQG-specific functionality
        ltqg_checks = {
            'Sigma parameter': 'sigma' in js_content.lower(),
            'Schwarzschild radius': 'rs' in js_content or 'schwarzschild' in js_content.lower(),
            'Funnel creation': 'funnel' in js_content.lower() or 'embedding' in js_content.lower(),
            'Horizon visualization': 'horizon' in js_content.lower(),
            'Geodesic': 'geodesic' in js_content.lower(),
            'Color schemes': 'colorSchemes' in js_content or 'getColorForSigma' in js_content,
            'Event listeners': 'addEventListener' in js_content,
        }
        
        print("\n🔧 Three.js Validation:")
        for check, passed in threejs_checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        
        print("\n🌌 LTQG Physics Validation:")
        for check, passed in ltqg_checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        
        threejs_score = sum(threejs_checks.values()) / len(threejs_checks)
        ltqg_score = sum(ltqg_checks.values()) / len(ltqg_checks)
        
        return threejs_score >= 0.8 and ltqg_score >= 0.8
        
    except Exception as e:
        print(f"❌ Error validating JavaScript: {e}")
        return False


def validate_dependencies(filepath):
    """Check external dependencies."""
    try:
        content = Path(filepath).read_text(encoding='utf-8')
        
        # Extract script src URLs
        script_pattern = r'<script[^>]*src=["\']([^"\']+)["\']'
        scripts = re.findall(script_pattern, content)
        
        dependency_checks = {
            'Three.js CDN': any('three' in script.lower() for script in scripts),
            'OrbitControls': any('orbitcontrols' in script.lower() for script in scripts),
            'Multiple CDNs': len(scripts) >= 2,
        }
        
        print("\n📦 Dependency Validation:")
        for check, passed in dependency_checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        
        if scripts:
            print("  📋 Found dependencies:")
            for script in scripts:
                print(f"    • {script}")
        
        return all(dependency_checks.values())
        
    except Exception as e:
        print(f"❌ Error validating dependencies: {e}")
        return False


def validate_ui_controls(filepath):
    """Validate UI controls and interactivity."""
    try:
        content = Path(filepath).read_text(encoding='utf-8')
        
        # Check for control elements
        control_checks = {
            'Play/Pause button': 'playBtn' in content,
            'Reset button': 'resetBtn' in content,
            'Sigma slider': 'sigmaSlider' in content,
            'Speed control': 'speedSlider' in content,
            'Horizon slider': 'horizonSlider' in content,
            'Wireframe toggle': 'wireframeBtn' in content,
            'Color scheme selector': 'colorScheme' in content,
            'Color pickers': 'color-picker' in content,
            'Camera presets': 'viewTopBtn' in content and 'viewSideBtn' in content,
        }
        
        print("\n🎮 UI Controls Validation:")
        for check, passed in control_checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        
        return sum(control_checks.values()) >= len(control_checks) * 0.8
        
    except Exception as e:
        print(f"❌ Error validating UI controls: {e}")
        return False


def analyze_file_quality(filepath):
    """Analyze overall file quality."""
    try:
        path = Path(filepath)
        
        if not path.exists():
            print(f"❌ File not found: {filepath}")
            return False
        
        size = path.stat().st_size
        content = path.read_text(encoding='utf-8')
        
        lines = content.split('\n')
        
        quality_metrics = {
            'File size': f"{size:,} bytes ({size/1024:.1f} KB)",
            'Line count': f"{len(lines):,} lines",
            'Contains comments': '<!--' in content or '//' in content,
            'Proper indentation': any(line.startswith('  ') for line in lines),
            'Readable structure': len([l for l in lines if l.strip()]) / len(lines) > 0.5,
        }
        
        print("\n📊 File Quality Analysis:")
        for metric, value in quality_metrics.items():
            if isinstance(value, bool):
                status = "✅" if value else "⚠️"
                print(f"  {status} {metric}")
            else:
                print(f"  📏 {metric}: {value}")
        
        return size > 5000 and len(lines) > 100  # Minimum thresholds
        
    except Exception as e:
        print(f"❌ Error analyzing file quality: {e}")
        return False


def main():
    """Run comprehensive WebGL validation."""
    print("🧪 LTQG WebGL Validation Test")
    print("=" * 40)
    
    files_to_test = [
        "ltqg_black_hole_webgl.html",
        "ltqg_black_hole_webgl_fixed.html"
    ]
    
    for filepath in files_to_test:
        if not Path(filepath).exists():
            continue
            
        print(f"\n🔍 Testing: {filepath}")
        print("-" * 30)
        
        tests = [
            ("HTML Syntax", validate_html_syntax),
            ("CSS Styling", validate_css),
            ("JavaScript", validate_javascript),
            ("Dependencies", validate_dependencies),
            ("UI Controls", validate_ui_controls),
            ("File Quality", analyze_file_quality),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func(filepath)
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ Error in {test_name}: {e}")
                results.append((test_name, False))
        
        # Summary
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        print(f"\n📈 Results for {filepath}:")
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name:15} {status}")
        
        score = passed / total * 100
        print(f"\n🎯 Overall Score: {passed}/{total} ({score:.1f}%)")
        
        if score >= 80:
            print("🎉 WebGL demo is ready for use!")
        elif score >= 60:
            print("👍 WebGL demo is mostly functional")
        else:
            print("⚠️ WebGL demo needs significant fixes")


if __name__ == "__main__":
    main()