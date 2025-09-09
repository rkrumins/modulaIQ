#!/usr/bin/env python3
"""Script to run the Supervisor agent."""

import asyncio
import uvicorn
import logging
from pathlib import Path

from supervisor.api import app
from core.config import Config


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/supervisor.log', mode='a')
        ]
    )


async def main():
    """Main function to run the supervisor."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config.load_from_yaml()
        logger.info(f"Starting Supervisor on {config.supervisor.host}:{config.supervisor.port}")
        
        # Create uvicorn config
        uvicorn_config = uvicorn.Config(
            "supervisor.api:app",
            host=config.supervisor.host,
            port=config.supervisor.port,
            reload=True,
            log_level="info"
        )
        
        # Create and run server
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
    except Exception as e:
        logger.error(f"Failed to start supervisor: {e}")
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
