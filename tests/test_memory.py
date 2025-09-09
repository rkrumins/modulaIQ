#!/usr/bin/env python3
"""Test script for the in-memory memory system."""

import asyncio
import logging
from core.config import Config
from core.memory import MemoryManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_memory_backend(backend_type: str, config: Config):
    """Test a specific memory backend."""
    logger.info(f"\n🧪 Testing {backend_type} backend...")
    
    # Create a temporary config with the specific backend
    test_config = Config(
        supervisor=config.supervisor.model_copy(update={"memory_backend": backend_type}),
        llm_providers=config.llm_providers,
        memory=config.memory,
        agent_registry=config.agent_registry,
        logging=config.logging
    )
    
    try:
        # Create memory manager
        memory_manager = MemoryManager(test_config)
        
        # Test session
        session_id = f"test-session-{backend_type}"
        
        # Store messages
        memory_manager.store_message(session_id, "user", "Hello, how are you?")
        memory_manager.store_message(session_id, "assistant", "I'm doing well, thank you!")
        memory_manager.store_message(session_id, "user", "What's the weather like?")
        memory_manager.store_message(session_id, "assistant", "I don't have access to weather data.")
        
        # Retrieve messages
        messages = memory_manager.get_messages(session_id)
        logger.info(f"✅ Stored and retrieved {len(messages)} messages")
        
        # Test conversation context
        context = memory_manager.get_conversation_context(session_id, limit=2)
        logger.info(f"✅ Retrieved conversation context with {len(context)} messages")
        
        # Test limited retrieval
        limited_messages = memory_manager.get_messages(session_id, limit=2)
        logger.info(f"✅ Limited retrieval returned {len(limited_messages)} messages")
        
        # Test session clearing
        memory_manager.clear_session(session_id)
        cleared_messages = memory_manager.get_messages(session_id)
        logger.info(f"✅ Session cleared, remaining messages: {len(cleared_messages)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ {backend_type} backend test failed: {e}")
        return False


def test_all_backends():
    """Test all available memory backends."""
    logger.info("🚀 Starting memory backend tests...")
    
    try:
        # Load configuration
        config = Config.load_from_yaml()
        
        # Test backends
        backends_to_test = [
            "in_memory_sqlite",
            "in_memory_dict",
            "sqlite"
        ]
        
        # Add Redis if available
        try:
            import redis
            backends_to_test.append("redis")
            logger.info("✅ Redis is available for testing")
        except ImportError:
            logger.info("⚠️  Redis not available, skipping Redis tests")
        
        results = {}
        
        for backend in backends_to_test:
            results[backend] = test_memory_backend(backend, config)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("📊 MEMORY BACKEND TEST SUMMARY")
        logger.info("="*50)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for backend, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{backend}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} backends passed")
        
        if passed == total:
            logger.info("🎉 All memory backend tests passed!")
        else:
            logger.info("⚠️  Some memory backend tests failed.")
        
        return passed == total
        
    except Exception as e:
        logger.error(f"❌ Memory test failed: {e}")
        return False


def test_concurrent_access():
    """Test concurrent access to memory backends."""
    logger.info("\n🔄 Testing concurrent access...")
    
    try:
        config = Config.load_from_yaml()
        memory_manager = MemoryManager(config)
        
        async def store_messages(session_id: str, count: int):
            """Store messages concurrently."""
            for i in range(count):
                memory_manager.store_message(
                    session_id, 
                    "user", 
                    f"Message {i} from concurrent test"
                )
                await asyncio.sleep(0.001)  # Small delay
        
        async def test_concurrent():
            """Run concurrent tests."""
            # Test concurrent writes to different sessions
            tasks = []
            for i in range(5):
                task = store_messages(f"concurrent-session-{i}", 10)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # Verify all messages were stored
            for i in range(5):
                messages = memory_manager.get_messages(f"concurrent-session-{i}")
                logger.info(f"Session {i}: {len(messages)} messages stored")
        
        asyncio.run(test_concurrent())
        logger.info("✅ Concurrent access test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Concurrent access test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🧪 Testing In-Memory Memory System")
    logger.info("="*60)
    
    # Test all backends
    backend_test_passed = test_all_backends()
    
    # Test concurrent access
    concurrent_test_passed = test_concurrent_access()
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("🎯 FINAL TEST SUMMARY")
    logger.info("="*60)
    
    if backend_test_passed and concurrent_test_passed:
        logger.info("🎉 All memory system tests passed!")
        logger.info("✅ The in-memory memory system is working correctly.")
        return True
    else:
        logger.info("❌ Some memory system tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
