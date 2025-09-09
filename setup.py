#!/usr/bin/env python3
"""Setup script for the multi-agent system."""

import os
import sys
import subprocess
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_dependencies():
    """Install Python dependencies."""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")


def create_directories():
    """Create necessary directories."""
    directories = ["logs", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    return True


def setup_environment():
    """Setup environment file."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        logger.info("✅ Created .env file from template")
        logger.info("⚠️  Please edit .env file and add your API keys")
    elif env_file.exists():
        logger.info("✅ .env file already exists")
    else:
        logger.warning("⚠️  No .env file found. Please create one with your API keys")
    
    return True


def check_redis():
    """Check if Redis is available."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        logger.info("✅ Redis is running and accessible")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Redis not available: {e}")
        logger.info("💡 Redis is optional - SQLite will be used as fallback")
        return True


def main():
    """Main setup function."""
    logger.info("🚀 Setting up Multi-Agent Supervisor Framework...")
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Setting up environment", setup_environment),
        ("Checking Redis", check_redis),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        logger.info(f"\n📋 {step_name}...")
        if not step_func():
            failed_steps.append(step_name)
    
    logger.info("\n" + "="*50)
    logger.info("📊 SETUP SUMMARY")
    logger.info("="*50)
    
    if failed_steps:
        logger.error(f"❌ Setup failed. Failed steps: {', '.join(failed_steps)}")
        logger.info("\n💡 Next steps:")
        logger.info("1. Fix the failed steps above")
        logger.info("2. Edit .env file with your API keys")
        logger.info("3. Run: python run_supervisor.py")
        logger.info("4. Run: python run_agents.py")
        return False
    else:
        logger.info("🎉 Setup completed successfully!")
        logger.info("\n💡 Next steps:")
        logger.info("1. Edit .env file with your API keys")
        logger.info("2. Start Redis (optional): redis-server")
        logger.info("3. Run: python run_supervisor.py")
        logger.info("4. Run: python run_agents.py")
        logger.info("5. Test: python test_system.py")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
