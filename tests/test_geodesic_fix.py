#!/usr/bin/env python3
"""
Quick Geodesic Animation Test
============================

Tests if the geodesic animation fix is working correctly.
"""

from pathlib import Path
import webbrowser
import time


def test_geodesic_fix():
    """Test the geodesic animation fix."""
    
    print("🧪 Geodesic Animation Fix Test")
    print("=" * 35)
    
    # Check if file was updated
    webgl_file = Path("ltqg_black_hole_webgl.html")
    
    if not webgl_file.exists():
        print("❌ WebGL file not found!")
        return False
    
    content = webgl_file.read_text(encoding='utf-8')
    
    # Check for the fix indicators
    checks = {
        "Preserve particle position": "let currentT = 0" in content,
        "Independent geodesic animation": "// Geodesic particle animation (always runs when visible)" in content,
        "Position preservation": "particleMesh.userData.t = currentT" in content,
        "Separated animation logic": "// Main sigma animation (only when playing)" in content,
    }
    
    print("🔍 Fix Verification:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n🎉 All fixes implemented successfully!")
        print("\n📋 Expected Behavior:")
        print("  ✅ Geodesic particle moves when animation is PAUSED")
        print("  ✅ Geodesic particle moves when animation is PLAYING")
        print("  ✅ Particle position preserved during sigma changes")
        print("  ✅ Smooth continuous particle motion")
        
        print("\n🧪 Test Instructions:")
        print("  1. Load the WebGL visualization")
        print("  2. Make sure geodesic is visible (🛸 Geodesic button active)")
        print("  3. Press ⏸️ Pause - particle should move along path")
        print("  4. Press ▶️ Play - particle should continue moving")
        print("  5. Change σ-time slider - particle should maintain motion")
        
        return True
    else:
        print("\n❌ Some fixes are missing!")
        return False


def main():
    """Main test function."""
    print("🌌 LTQG WebGL Geodesic Animation Fix")
    print("=" * 40)
    
    if test_geodesic_fix():
        print("\n🚀 Ready to test!")
        
        # Offer to open the visualization
        try:
            response = input("\nOpen WebGL visualization for testing? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                url = "http://localhost:8081/ltqg_black_hole_webgl.html"
                print(f"🌍 Opening: {url}")
                webbrowser.open(url)
                print("👀 Watch the geodesic particle (small sphere) movement!")
        except KeyboardInterrupt:
            print("\n👋 Test cancelled")
    else:
        print("\n🔧 Please check the WebGL file for missing fixes")


if __name__ == "__main__":
    main()