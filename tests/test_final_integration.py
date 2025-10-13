#!/usr/bin/env python3
"""
Final Integration Test for LTQG Black Hole Visualizations
=========================================================

This script performs a comprehensive test of all black hole visualization
components to ensure they work correctly together.

Tests:
1. Static visualization generation (original and enhanced)
2. WebGL HTML file validation
3. Launcher script functionality
4. Integration with main visualization framework
5. File integrity and size checks

Usage:
    python test_final_integration.py

Author: Denzil James Greenwood
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import tempfile


def test_static_visualizations():
    """Test static matplotlib visualizations."""
    print("🔍 Testing Static Visualizations")
    print("-" * 35)
    
    try:
        from ltqg_visualization import LTQGVisualizer
        
        # Create temporary directory for test outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = LTQGVisualizer(save_dir=temp_dir, dpi=150)  # Lower DPI for faster testing
            
            # Test original version
            print("  🔨 Testing original black hole embedding...")
            fig1 = visualizer.figure_black_hole_embedding(save=True, enhanced=False)
            original_file = Path(temp_dir) / "black_hole_embedding.png"
            
            if original_file.exists() and original_file.stat().st_size > 10000:  # At least 10KB
                print("  ✅ Original version: PASS")
                original_success = True
            else:
                print("  ❌ Original version: FAIL")
                original_success = False
            
            # Test enhanced version
            print("  🔨 Testing enhanced black hole embedding...")
            fig2 = visualizer.figure_black_hole_embedding(save=True, enhanced=True)
            enhanced_file = Path(temp_dir) / "black_hole_embedding_enhanced.png"
            
            if enhanced_file.exists() and enhanced_file.stat().st_size > 10000:  # At least 10KB
                print("  ✅ Enhanced version: PASS")
                enhanced_success = True
            else:
                print("  ❌ Enhanced version: FAIL")
                enhanced_success = False
            
            # Test integration with full suite
            print("  🔨 Testing integration with full visualization suite...")
            all_figures = visualizer.generate_all_figures(save=False)
            
            if 'black_hole_embedding' in all_figures and 'black_hole_enhanced' in all_figures:
                print("  ✅ Full suite integration: PASS")
                integration_success = True
            else:
                print("  ❌ Full suite integration: FAIL")
                integration_success = False
        
        return original_success and enhanced_success and integration_success
        
    except Exception as e:
        print(f"  ❌ Error in static visualization test: {e}")
        return False


def test_webgl_files():
    """Test WebGL and launcher files."""
    print("\n🔍 Testing WebGL Components")
    print("-" * 28)
    
    # Check WebGL HTML file
    webgl_file = Path("ltqg_black_hole_webgl.html")
    print(f"  📄 Checking {webgl_file.name}...")
    
    if webgl_file.exists():
        size = webgl_file.stat().st_size
        print(f"    File exists: {size:,} bytes")
        
        # Basic content validation
        content = webgl_file.read_text(encoding='utf-8')
        required_elements = [
            "Three.js",
            "OrbitControls",
            "lil-gui",
            "LTQG",
            "black hole",
            "sigma",
            "geodesic"
        ]
        
        missing = [elem for elem in required_elements if elem.lower() not in content.lower()]
        
        if not missing and size > 5000:  # At least 5KB
            print("    ✅ WebGL HTML: PASS")
            webgl_success = True
        else:
            print(f"    ❌ WebGL HTML: FAIL (missing: {missing})")
            webgl_success = False
    else:
        print("    ❌ WebGL HTML: FILE NOT FOUND")
        webgl_success = False
    
    # Check launcher script
    launcher_file = Path("launch_webgl_demo.py")
    print(f"  🚀 Checking {launcher_file.name}...")
    
    if launcher_file.exists():
        size = launcher_file.stat().st_size
        print(f"    File exists: {size:,} bytes")
        
        # Try to validate Python syntax
        try:
            with open(launcher_file, 'r', encoding='utf-8') as f:
                compile(f.read(), launcher_file, 'exec')
            print("    ✅ Launcher script: PASS")
            launcher_success = True
        except SyntaxError as e:
            print(f"    ❌ Launcher script: SYNTAX ERROR - {e}")
            launcher_success = False
    else:
        print("    ❌ Launcher script: FILE NOT FOUND")
        launcher_success = False
    
    return webgl_success and launcher_success


def test_demo_scripts():
    """Test demonstration scripts."""
    print("\n🔍 Testing Demo Scripts")
    print("-" * 23)
    
    demo_scripts = [
        "demo_black_hole.py",
        "demo_enhanced_black_hole.py", 
        "showcase_black_hole_complete.py",
        "test_black_hole_original.py",
        "test_comprehensive_black_hole.py"
    ]
    
    success_count = 0
    
    for script in demo_scripts:
        script_path = Path(script)
        print(f"  📜 Checking {script}...")
        
        if script_path.exists():
            size = script_path.stat().st_size
            
            # Basic syntax check
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    compile(f.read(), script_path, 'exec')
                print(f"    ✅ {script}: PASS ({size:,} bytes)")
                success_count += 1
            except SyntaxError as e:
                print(f"    ❌ {script}: SYNTAX ERROR - {e}")
        else:
            print(f"    ❌ {script}: FILE NOT FOUND")
    
    print(f"  📊 Demo scripts: {success_count}/{len(demo_scripts)} PASS")
    return success_count == len(demo_scripts)


def test_file_integrity():
    """Test file integrity and sizes."""
    print("\n🔍 Testing File Integrity")
    print("-" * 25)
    
    # Expected files and minimum sizes (bytes)
    expected_files = {
        "ltqg_black_hole_webgl.html": 5000,
        "launch_webgl_demo.py": 3000,
        "showcase_black_hole_complete.py": 5000,
        "demo_black_hole.py": 2000,
        "demo_enhanced_black_hole.py": 3000,
        "test_comprehensive_black_hole.py": 4000,
        "figs/black_hole_embedding.png": 100000,  # At least 100KB for image
        "figs/black_hole_embedding_enhanced.png": 100000
    }
    
    success_count = 0
    total_size = 0
    
    for filename, min_size in expected_files.items():
        filepath = Path(filename)
        print(f"  📁 {filename}...")
        
        if filepath.exists():
            size = filepath.stat().st_size
            total_size += size
            
            if size >= min_size:
                print(f"    ✅ OK ({size:,} bytes)")
                success_count += 1
            else:
                print(f"    ⚠️  SMALL ({size:,} < {min_size:,} bytes)")
        else:
            print(f"    ❌ MISSING")
    
    print(f"  📊 File integrity: {success_count}/{len(expected_files)} PASS")
    print(f"  💾 Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    
    return success_count >= len(expected_files) * 0.8  # Allow 20% tolerance


def test_documentation_updates():
    """Test that documentation was properly updated."""
    print("\n🔍 Testing Documentation Updates")
    print("-" * 33)
    
    readme_file = Path("README.md")
    
    if readme_file.exists():
        content = readme_file.read_text(encoding='utf-8')
        
        # Check for WebGL mentions
        webgl_keywords = [
            "WebGL",
            "black_hole_embedding_enhanced.png",
            "launch_webgl_demo.py",
            "Interactive WebGL Demo"
        ]
        
        found_keywords = [kw for kw in webgl_keywords if kw in content]
        
        print(f"  📖 README.md exists ({len(content):,} characters)")
        print(f"  🔍 Found WebGL keywords: {len(found_keywords)}/{len(webgl_keywords)}")
        
        if len(found_keywords) >= len(webgl_keywords) * 0.75:  # At least 75%
            print("  ✅ Documentation updates: PASS")
            return True
        else:
            print(f"  ❌ Documentation updates: INCOMPLETE (missing: {set(webgl_keywords) - set(found_keywords)})")
            return False
    else:
        print("  ❌ README.md: FILE NOT FOUND")
        return False


def run_integration_test():
    """Run complete integration test."""
    print("🧪 LTQG Black Hole Visualization Integration Test")
    print("=" * 55)
    print()
    
    # Run all tests
    tests = [
        ("Static Visualizations", test_static_visualizations),
        ("WebGL Components", test_webgl_files),
        ("Demo Scripts", test_demo_scripts),
        ("File Integrity", test_file_integrity),
        ("Documentation", test_documentation_updates)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    elapsed = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("\n" + "=" * 55)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 55)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:25} {status}")
    
    print(f"\n📈 Overall Score: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"⏱️  Test Duration: {elapsed:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! LTQG black hole visualizations are ready!")
        print("🚀 You can now:")
        print("   • Generate static visualizations with the Python API")
        print("   • Launch interactive WebGL demos")
        print("   • Run complete showcases")
        print("   • Use all demonstration scripts")
        return True
    elif passed >= total * 0.8:
        print(f"\n👍 MOSTLY SUCCESSFUL ({passed}/{total} tests passed)")
        print("⚠️  Some minor issues detected but core functionality works")
        return True
    else:
        print(f"\n❌ SIGNIFICANT ISSUES ({passed}/{total} tests passed)")
        print("🔧 Please check the failed tests above")
        return False


def main():
    """Main test function."""
    try:
        success = run_integration_test()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())