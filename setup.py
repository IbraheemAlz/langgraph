#!/usr/bin/env python3
"""
Setup script for the Multi-Agent AI Hiring System.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages."""
    logger.info("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install requirements: {e}")
        return False

def setup_environment():
    """Setup environment variables."""
    logger.info("Setting up environment...")
    
    if not os.path.exists('.env'):
        logger.info("Creating .env file...")
        with open('.env', 'w') as f:
            f.write("# Google Generative AI API Key\n")
            f.write("GOOGLE_API_KEY=your_api_key_here\n")
        logger.info("✅ .env file created")
        logger.warning("⚠️  Please edit .env file and add your Google API key")
    else:
        logger.info("✅ .env file already exists")
    
    return True

def verify_setup():
    """Verify the setup is working."""
    logger.info("Verifying setup...")
    
    try:
        # Check if we can import the main modules
        from src.config import Config
        from src.main import create_hiring_workflow
        logger.info("✅ Core modules can be imported")
        
        # Check if environment variables are set
        from dotenv import load_dotenv
        load_dotenv()
        
        if os.getenv('GOOGLE_API_KEY') and os.getenv('GOOGLE_API_KEY') != 'your_api_key_here':
            logger.info("✅ API key is configured")
        else:
            logger.warning("⚠️  API key not configured in .env file")
        
        return True
    
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Setup verification issue: {e}")
        return True  # Don't fail setup for minor issues

def main():
    """Main setup function."""
    logger.info("🚀 Setting up Multi-Agent AI Hiring System...")
    logger.info("=" * 50)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing requirements", install_requirements),
        ("Setting up environment", setup_environment),
        ("Verifying setup", verify_setup)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\n📋 {step_name}...")
        if not step_func():
            logger.error(f"❌ Setup failed at: {step_name}")
            logger.error("Please fix the errors above and run setup again.")
            sys.exit(1)
    
    logger.info("\n🎉 Setup completed successfully!")
    logger.info("=" * 50)
    logger.info("Next steps:")
    logger.info("1. Edit .env file and add your Google API key")
    logger.info("2. Run: python run.py (to test the system)")
    logger.info("3. Run: python batch_processor.py (for batch processing)")
    logger.info("4. Run: python chart_generator.py (for analytics)")
    logger.info("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
