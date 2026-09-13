"""
Initialize project structure and verify dependencies.
"""

import sys
from pathlib import Path
import subprocess
import json

def check_python_version():
    """Check Python version >= 3.9."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, but found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def create_directory_structure():
    """Create all necessary directories."""
    dirs = [
        "data/raw/pdfs",
        "data/raw/tables",
        "data/raw/notes",
        "data/interim/extracted",
        "data/interim/validated",
        "data/interim/normalized",
        "data/processed/splits",
        "results/figures",
        "results/tables",
        "results/reports",
        "logs",
    ]
    
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create .gitkeep to track empty directories
        gitkeep = Path(directory) / ".gitkeep"
        gitkeep.touch()
    
    print("✓ Directory structure created")

def verify_dependencies():
    """Check if required packages are installed."""
    required = [
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "chemdata",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} (missing)")
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("Run: pip install -e .")
        return False
    
    print("✓ All dependencies installed")
    return True

def create_project_metadata():
    """Create project metadata file."""
    metadata = {
        "project_name": "photocatalysis-research-benchmark",
        "version": "0.1.0",
        "created_date": "2024-09-26",
        "target_journal": "Scientific Data",
        "status": "in_development",
        "data_points_target": 500,
        "papers_target": 50,
    }
    
    metadata_file = Path("project_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Project metadata created: {metadata_file}")

def initialize_git():
    """Initialize git repository if not already done."""
    if not Path(".git").exists():
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "add", ".gitignore"], check=True)
        subprocess.run(
            ["git", "commit", "-m", "build: initialize project structure"],
            check=True
        )
        print("✓ Git repository initialized")
    else:
        print("✓ Git repository already exists")

def main():
    """Run all setup steps."""
    print("="*60)
    print("PHOTOCATALYSIS RESEARCH BENCHMARK - PROJECT SETUP")
    print("="*60 + "\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directory_structure()
    
    # Create metadata
    create_project_metadata()
    
    # Verify dependencies
    dependencies_ok = verify_dependencies()
    
    # Initialize git
    try:
        initialize_git()
    except Exception as e:
        print(f"⚠️ Git initialization failed: {e}")
    
    # Final status
    print("\n" + "="*60)
    if dependencies_ok:
        print("✅ PROJECT SETUP COMPLETE")
        print("\nNext steps:")
        print("1. Run: python scripts/02_import_existing_data.py")
        print("2. Check: notebooks/exploration/01_initial_data_exploration.ipynb")
    else:
        print("⚠️ SETUP INCOMPLETE - Install missing dependencies")
        print("Run: pip install -e .")
    print("="*60)

if __name__ == "__main__":
    main()