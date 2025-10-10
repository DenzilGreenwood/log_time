#!/usr/bin/env python3
"""
LTQG Black Hole Visualization Launcher
======================================

Complete launcher for all LTQG black hole visualizations:
- Static matplotlib visualizations
- Interactive WebGL visualization
- Validation and testing tools
"""

import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def print_header():
    """Print the launcher header."""
    print("🌌 LTQG Black Hole Visualization Suite")
    print("=" * 50)
    print("Complete visualization suite for Log-Time Quantum Gravity")
    print("black hole embeddings and σ-time dynamics")
    print()


def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    # Check Python packages
    try:
        import numpy
        import matplotlib
        import scipy
        print("  ✅ Python packages (numpy, matplotlib, scipy)")
    except ImportError as e:
        print(f"  ❌ Missing Python package: {e}")
        return False
    
    # Check LTQG modules
    try:
        import ltqg_core
        import ltqg_visualization
        print("  ✅ LTQG modules (core, visualization)")
    except ImportError as e:
        print(f"  ❌ Missing LTQG module: {e}")
        return False
    
    # Check WebGL files
    webgl_files = [
        "ltqg_black_hole_webgl_fixed.html",
        "serve_webgl.py"
    ]
    
    for file in webgl_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ Missing: {file}")
            return False
    
    print("  🎉 All dependencies available!")
    return True


def launch_static_visualization():
    """Launch static matplotlib visualization."""
    print("\n🎨 Launching Static Visualization...")
    print("-" * 30)
    
    try:
        from ltqg_visualization import LTQGVisualization
        
        # Create visualization instance
        viz = LTQGVisualization()
        
        print("Creating black hole embedding plots...")
        
        # Generate both versions
        viz.figure_black_hole_embedding(enhanced=False)
        print("  ✅ Original version created")
        
        viz.figure_black_hole_embedding(enhanced=True)
        print("  ✅ Enhanced version created")
        
        print("  📊 Static visualizations complete!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error creating static visualization: {e}")
        return False


def launch_webgl_visualization():
    """Launch interactive WebGL visualization."""
    print("\n🌐 Launching WebGL Visualization...")
    print("-" * 30)
    
    try:
        # Start server in background
        server_process = subprocess.Popen([
            sys.executable, "serve_webgl.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("  🚀 Starting WebGL server...")
        time.sleep(2)  # Give server time to start
        
        # Open browser
        url = "http://localhost:8080/ltqg_black_hole_webgl_fixed.html"
        print(f"  🌍 Opening: {url}")
        webbrowser.open(url)
        
        print("  ✅ WebGL visualization launched!")
        print("  📋 Use Ctrl+C to stop the server")
        
        # Wait for user to stop
        try:
            server_process.wait()
        except KeyboardInterrupt:
            server_process.terminate()
            print("\n  🛑 Server stopped")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error launching WebGL: {e}")
        return False


def run_validation_tests():
    """Run validation tests."""
    print("\n🧪 Running Validation Tests...")
    print("-" * 30)
    
    try:
        # Run WebGL validation
        subprocess.run([sys.executable, "-c", 
                       "import test_webgl_validation; test_webgl_validation.main()"],
                      check=True)
        
        print("  ✅ WebGL validation complete!")
        return True
        
    except Exception as e:
        print(f"  ❌ Validation error: {e}")
        return False


def show_demo_options():
    """Show available demo options."""
    print("\n📋 Available Demonstrations:")
    print("-" * 30)
    print("1. 🎨 Static Visualizations")
    print("   • Original black hole embedding")
    print("   • Enhanced version with geodesics")
    print("   • Matplotlib-based, publication ready")
    print()
    print("2. 🌐 Interactive WebGL Visualization")
    print("   • 3D interactive black hole funnel")
    print("   • Real-time σ-time evolution")
    print("   • Multiple color schemes")
    print("   • Camera controls and animations")
    print("   • Toggle wireframe, horizon, geodesics")
    print()
    print("3. 🧪 Validation Tests")
    print("   • HTML/CSS/JavaScript validation")
    print("   • WebGL capability testing")
    print("   • Dependency checking")
    print()
    print("4. 📚 Educational Demo")
    print("   • Complete physics demonstration")
    print("   • Step-by-step explanations")
    print("   • Parameter exploration")
    print()


def launch_educational_demo():
    """Launch the educational demonstration."""
    print("\n📚 Launching Educational Demo...")
    print("-" * 30)
    
    try:
        subprocess.run([sys.executable, "ltqg_demo.py"], check=True)
        print("  ✅ Educational demo complete!")
        return True
        
    except Exception as e:
        print(f"  ❌ Demo error: {e}")
        return False


def main():
    """Main launcher interface."""
    print_header()
    
    if not check_dependencies():
        print("\n❌ Dependencies missing. Please install required packages.")
        return
    
    while True:
        show_demo_options()
        
        print("\n🚀 Choose visualization:")
        print("1 - Static matplotlib plots")
        print("2 - Interactive WebGL demo")
        print("3 - Run validation tests")
        print("4 - Educational demonstration")
        print("0 - Exit")
        
        try:
            choice = input("\nEnter choice (0-4): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                break
            elif choice == "1":
                launch_static_visualization()
            elif choice == "2":
                launch_webgl_visualization()
            elif choice == "3":
                run_validation_tests()
            elif choice == "4":
                launch_educational_demo()
            else:
                print("❌ Invalid choice. Please try again.")
                
            input("\nPress Enter to continue...")
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()