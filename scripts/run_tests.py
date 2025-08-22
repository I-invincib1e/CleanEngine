#!/usr/bin/env python3
"""
Test Runner for Advanced Dataset Cleaner
Runs all unit tests and generates coverage report.
"""

import unittest
import sys
import os
from pathlib import Path
import time

def run_tests():
    """Run all unit tests"""
    print("🧪 Running Advanced Dataset Cleaner Test Suite")
    print("=" * 60)
    
    # Add current directory to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / 'src'))
    
    # Discover and run tests from the project root
    start_dir = project_root / 'tests'
    
    if not start_dir.exists():
        print("❌ Tests directory not found!")
        return False
    
    # Load test suite
    loader = unittest.TestLoader()
    suite = loader.discover(str(start_dir), pattern='test_*.py')
    
    # Count tests
    test_count = suite.countTestCases()
    print(f"📊 Found {test_count} test cases")
    print("-" * 60)
    
    # Run tests
    start_time = time.time()
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"⏱️  Duration: {end_time - start_time:.2f} seconds")
    print(f"🧪 Tests Run: {result.testsRun}")
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    # Print failures and errors
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    # Overall result
    success = result.wasSuccessful()
    if success:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {len(result.failures) + len(result.errors)} TEST(S) FAILED")
    
    return success

def run_specific_test(test_name):
    """Run a specific test module"""
    print(f"🧪 Running specific test: {test_name}")
    print("=" * 60)
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir.parent / 'src'))
    
    try:
        # Import and run specific test
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(f'tests.{test_name}')
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    except ImportError as e:
        print(f"❌ Could not import test module: {e}")
        return False

def main():
    """Main test runner function"""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        # Run all tests
        success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()