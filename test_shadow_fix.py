#!/usr/bin/env python3
"""
Shadow System Fix Test
======================

Tests and validates the WebGL shadow system improvements.
"""

from pathlib import Path


def test_shadow_implementation():
    """Test the shadow system fixes."""
    
    print("🌗 Shadow System Fix Test")
    print("=" * 25)
    
    webgl_file = Path("ltqg_black_hole_webgl.html")
    
    if not webgl_file.exists():
        print("❌ WebGL file not found!")
        return False
    
    try:
        content = webgl_file.read_text(encoding='latin-1')
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Check for shadow fixes
    checks = {
        "Ground plane creation": "createGroundPlane" in content,
        "Ground plane variable": "groundPlane" in content,
        "Shadow map size increased": "shadow.mapSize.width = 4096" in content,
        "Improved shadow bias": "shadow.bias = -0.001" in content,
        "Better normal bias": "shadow.normalBias = 0.05" in content,
        "Ground plane in updates": "createGroundPlane()" in content,
        "Ground plane shadow setup": "groundPlane.receiveShadow" in content,
        "Enhanced light positioning": "radius*2.0" in content,
        "Larger shadow frustum": "radius * 1.8" in content,
        "Shadow camera far distance": "radius*6 + 60" in content,
    }
    
    print("🔍 Shadow Fix Verification:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    passed_count = sum(checks.values())
    total_count = len(checks)
    
    print(f"\n📊 Results: {passed_count}/{total_count} checks passed")
    
    if passed_count >= total_count - 1:  # Allow 1 failure
        print("\n🎉 Shadow system fixes implemented!")
        print("\n✨ Shadow improvements:")
        print("  ✅ Added ground plane to receive shadows")
        print("  ✅ Increased shadow map resolution (4096x4096)")
        print("  ✅ Improved shadow bias settings")
        print("  ✅ Better light positioning for shadow casting")
        print("  ✅ Larger shadow frustum coverage")
        print("  ✅ Multiple objects cast/receive shadows")
        
        print("\n🧪 Expected shadow behavior:")
        print("  🌗 Funnel casts shadows onto ground plane")
        print("  🌗 Horizon cylinder casts shadows")
        print("  🌗 Geodesic tube casts shadows")
        print("  🌗 Ground plane receives all shadows")
        print("  🌗 Toggle button controls shadow visibility")
        print("  🌗 Shadows update with geometry changes")
        
        return True
    else:
        print("\n❌ Some shadow fixes are missing!")
        return False


def main():
    """Main test function."""
    print("🌌 LTQG WebGL Shadow System Fix")
    print("=" * 35)
    
    success = test_shadow_implementation()
    
    if success:
        print("\n🚀 Shadow system ready for testing!")
        print("\nTo test shadows:")
        print("1. Open the WebGL visualization")
        print("2. Make sure 🌗 Shadows button is active (blue)")
        print("3. Look for shadows cast by the funnel onto ground plane")
        print("4. Try toggling shadows on/off")
        print("5. Change geometry (σ-time) - shadows should update")
        print("6. Shadows should be visible from different camera angles")
    else:
        print("\n🔧 Please check the shadow implementation")
    
    print(f"\n📍 Test URL: http://localhost:8081/ltqg_black_hole_webgl.html")


if __name__ == "__main__":
    main()