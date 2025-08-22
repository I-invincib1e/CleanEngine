#!/usr/bin/env python3
"""
CleanEngine Installation Script

This script provides a quick way to install CleanEngine dependencies
and set up the development environment.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(
        f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected"
    )
    return True


def install_dependencies():
    """Install project dependencies"""
    print("\n📦 Installing dependencies...")

    # Upgrade pip first
    if not run_command(
        f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"
    ):
        return False

    # Install requirements
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        if not run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing requirements",
        ):
            return False
    else:
        print("⚠️  requirements.txt not found, skipping dependency installation")

    return True


def install_development_dependencies():
    """Install development dependencies"""
    print("\n🔧 Installing development dependencies...")

    dev_deps = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
    ]

    for dep in dev_deps:
        if not run_command(
            f"{sys.executable} -m pip install {dep}", f"Installing {dep}"
        ):
            print(f"⚠️  Failed to install {dep}, continuing...")

    return True


def install_package():
    """Install the package in development mode"""
    print("\n📦 Installing CleanEngine in development mode...")

    if not run_command(
        f"{sys.executable} -m pip install -e .", "Installing CleanEngine"
    ):
        return False

    return True


def run_tests():
    """Run the test suite to verify installation"""
    print("\n🧪 Running tests to verify installation...")

    if not run_command(f"{sys.executable} -m pytest tests/ -v", "Running tests"):
        print("⚠️  Tests failed, but installation may still be functional")
        return False

    return True


def show_usage():
    """Show usage information"""
    print("\n🎉 CleanEngine installation completed!")
    print("\n📖 Usage:")
    print("   # Clean a dataset")
    print("   cleanengine clean data.csv")
    print("   # or")
    print("   python main.py clean data.csv")
    print("\n   # Create sample data")
    print("   cleanengine samples")
    print("\n   # Run tests")
    print("   cleanengine test")
    print("\n   # Launch GUI")
    print("   cleanengine gui")
    print("\n   # Show help")
    print("   cleanengine --help")
    print("\n📚 Documentation: https://github.com/I-invincib1e/CleanEngine#readme")


def main():
    """Main installation function"""
    print("🧹 CleanEngine Installation Script")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)

    # Install development dependencies (optional)
    install_dev = (
        input("\n🔧 Install development dependencies? (y/N): ").lower().strip()
    )
    if install_dev in ["y", "yes"]:
        install_development_dependencies()

    # Install package
    if not install_package():
        print("❌ Failed to install CleanEngine")
        sys.exit(1)

    # Run tests
    run_tests_input = (
        input("\n🧪 Run tests to verify installation? (Y/n): ").lower().strip()
    )
    if run_tests_input not in ["n", "no"]:
        run_tests()

    # Show usage
    show_usage()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
