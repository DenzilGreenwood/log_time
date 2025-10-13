#!/usr/bin/env python3
"""
LTQG Repository Cleanup Script

This script organizes the LTQG repository by:
1. Moving development artifacts to an archive folder
2. Organizing test files
3. Cleaning up temporary files
4. Creating a clean project structure

Run this script to maintain a clean, professional repository structure.
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_repository():
    """Clean up the LTQG repository structure."""
    
    print("🧹 LTQG Repository Cleanup")
    print("=" * 40)
    
    # Get repository root
    repo_root = Path(__file__).parent
    
    # Create archive and test directories
    archive_dir = repo_root / "archive"
    test_dir = repo_root / "tests"
    
    archive_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Files to archive (development artifacts and status reports)
    archive_patterns = [
        "*_COMPLETE.md",
        "*_FIX*.md", 
        "*_STATUS*.md",
        "*_GUIDE.md",
        "*_UPDATE*.md",
        "REPOSITORY_STATUS.md",
        "PROJECT_COMPLETE.md",
        "SIGMA_JACOBIAN_IMPLEMENTATION_COMPLETE.md",
        "WEBGL_*.md",
        "SHADOW_SYSTEM_FIX.md",
        "GEODESIC_ANIMATION_FIX.md",
        "DROPDOWN_FIX_COMPLETE.md",
        "PEER_REVIEW_IMPLEMENTATION_COMPLETE.md",
        "LTQG_EDUCATIONAL_FEATURES_COMPLETE.md"
    ]
    
    # Test files to move to tests directory
    test_patterns = [
        "test_*.py",
        "*_test.py",
        "final_codebase_review.py"
    ]
    
    print("📁 Moving development artifacts to archive/")
    archived_count = 0
    for pattern in archive_patterns:
        for file_path in repo_root.glob(pattern):
            if file_path.is_file():
                dest_path = archive_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                print(f"  ✓ {file_path.name} → archive/")
                archived_count += 1
    
    print(f"\n🧪 Moving test files to tests/")
    test_count = 0
    for pattern in test_patterns:
        for file_path in repo_root.glob(pattern):
            if file_path.is_file():
                dest_path = test_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                print(f"  ✓ {file_path.name} → tests/")
                test_count += 1
    
    # Clean up __pycache__ directories
    print(f"\n🗑️  Cleaning Python cache files")
    cache_count = 0
    for cache_dir in repo_root.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            print(f"  ✓ Removed {cache_dir}")
            cache_count += 1
    
    # Clean up .pyc files
    pyc_count = 0
    for pyc_file in repo_root.rglob("*.pyc"):
        if pyc_file.is_file():
            pyc_file.unlink()
            pyc_count += 1
    
    # Clean up temporary image files (keep organized ones in figs/)
    temp_image_patterns = [
        "problem_of_time_demo.png",
        "*_temp.png",
        "*_test.png"
    ]
    
    print(f"\n🖼️  Cleaning temporary image files")
    temp_img_count = 0
    for pattern in temp_image_patterns:
        for img_file in repo_root.glob(pattern):
            if img_file.is_file() and img_file.parent != repo_root / "figs":
                img_file.unlink()
                print(f"  ✓ Removed {img_file.name}")
                temp_img_count += 1
    
    # Create README for archive
    archive_readme = archive_dir / "README.md"
    if not archive_readme.exists():
        with open(archive_readme, 'w') as f:
            f.write("""# LTQG Development Archive

This directory contains development artifacts, status reports, and 
temporary files from the LTQG project development process.

These files document the development history but are not part of 
the main theoretical framework.

## Contents:
- Development status reports (*.md)
- Implementation completion markers
- Fix and update documentation
- Temporary development artifacts

## Note:
These files are preserved for development history but can be 
safely ignored for understanding the LTQG framework itself.
""")
    
    # Create README for tests
    test_readme = test_dir / "README.md"
    if not test_readme.exists():
        with open(test_readme, 'w') as f:
            f.write("""# LTQG Tests

This directory contains test files and validation scripts for the LTQG framework.

## Test Categories:
- Unit tests for core mathematical functions
- Integration tests for complete workflows  
- Validation tests for theoretical consistency
- Performance benchmarks

## Running Tests:
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_specific_module.py
```

## Adding Tests:
Follow the naming convention `test_*.py` for new test files.
Include docstrings explaining what each test validates.
""")
    
    # Summary
    print(f"\n📊 Cleanup Summary:")
    print(f"  ✓ {archived_count} development artifacts archived")
    print(f"  ✓ {test_count} test files organized")
    print(f"  ✓ {cache_count} cache directories cleaned")
    print(f"  ✓ {pyc_count} .pyc files removed")
    print(f"  ✓ {temp_img_count} temporary images cleaned")
    
    print(f"\n🎯 Repository Structure Optimized!")
    print(f"📁 Main directory now contains only essential project files")
    print(f"📁 Development artifacts preserved in archive/")
    print(f"📁 Test files organized in tests/")
    
    # List remaining main files
    main_files = []
    for item in repo_root.iterdir():
        if item.is_file() and not item.name.startswith('.'):
            main_files.append(item.name)
    
    print(f"\n📋 Main Project Files Remaining ({len(main_files)}):")
    core_files = [f for f in main_files if f.endswith('.py') and not f.startswith('demo_')]
    demo_files = [f for f in main_files if f.startswith('demo_') or f.startswith('launch_')]
    web_files = [f for f in main_files if f.endswith('.html')]
    doc_files = [f for f in main_files if f.endswith('.md') or f.endswith('.txt') or f == 'LICENSE']
    
    if core_files:
        print(f"  🔬 Core Framework: {', '.join(sorted(core_files))}")
    if demo_files:
        print(f"  🎮 Demos: {', '.join(sorted(demo_files))}")
    if web_files:
        print(f"  🌐 Web: {', '.join(sorted(web_files))}")
    if doc_files:
        print(f"  📚 Documentation: {', '.join(sorted(doc_files))}")
    
    print(f"\n✅ Repository cleanup completed successfully!")

if __name__ == "__main__":
    cleanup_repository()