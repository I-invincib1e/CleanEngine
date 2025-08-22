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

def run_test_suite(verbose: bool = False, coverage: bool = False) -> dict:
    """Run all unit tests and return results dictionary"""
    if verbose:
        print("🧪 Running CleanEngine Test Suite")
        print("=" * 60)
    
    # Add current directory to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / 'src'))
    
    # Discover and run tests from the project root
    start_dir = project_root / 'tests'
    
    if not start_dir.exists():
        print("❌ Tests directory not found!")
        return {
            "success": False,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0
        }
    
    # Load test suite
    loader = unittest.TestLoader()
    suite = loader.discover(str(start_dir), pattern='test_*.py')
    
    # Count tests
    test_count = suite.countTestCases()
    if verbose:
        print(f"📊 Found {test_count} test cases")
        print("-" * 60)
    
    # Run tests
    start_time = time.time()
    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        stream=sys.stdout,
        buffer=True
    )
    
    result = runner.run(suite)
    end_time = time.time()
    
    # Calculate results
    passed = result.testsRun - len(result.failures) - len(result.errors)
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    
    # Print summary if verbose
    if verbose:
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        print(f"⏱️  Duration: {end_time - start_time:.2f} seconds")
        print(f"🧪 Tests Run: {result.testsRun}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Errors: {errors}")
        print(f"⏭️  Skipped: {skipped}")
        
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
            print(f"\n⚠️  {failed + errors} TEST(S) FAILED")
    
    # Return results dictionary
    return {
        "success": result.wasSuccessful(),
        "total": result.testsRun,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "duration": end_time - start_time
    }

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
        results = run_test_suite(verbose=True)
        success = results["success"]
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()