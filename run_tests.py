#!/usr/bin/env python3
"""Test runner script for Project Maestro."""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
        
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
        
    if result.returncode != 0:
        print(f"❌ {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        return True


def run_unit_tests():
    """Run unit tests."""
    cmd = [
        "python", "-m", "pytest", 
        "-m", "unit",
        "--tb=short",
        "--cov=src/project_maestro",
        "--cov-report=term-missing",
        "tests/"
    ]
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "integration", 
        "--tb=short",
        "tests/"
    ]
    return run_command(cmd, "Integration Tests")


def run_performance_tests():
    """Run performance tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "performance",
        "--tb=short",
        "tests/"
    ]
    return run_command(cmd, "Performance Tests")


def run_api_tests():
    """Run API tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "api",
        "--tb=short", 
        "tests/"
    ]
    return run_command(cmd, "API Tests")


def run_all_tests():
    """Run all tests."""
    cmd = [
        "python", "-m", "pytest",
        "--tb=short",
        "--cov=src/project_maestro",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--durations=10",
        "tests/"
    ]
    return run_command(cmd, "All Tests")


def run_specific_test(test_path):
    """Run a specific test file or test function."""
    cmd = [
        "python", "-m", "pytest",
        "--tb=short",
        "-v",
        test_path
    ]
    return run_command(cmd, f"Specific Test: {test_path}")


def run_linting():
    """Run code linting."""
    print("\n" + "="*60)
    print("Running Code Quality Checks")
    print("="*60)
    
    # Check if flake8 is available
    try:
        subprocess.run(["flake8", "--version"], capture_output=True, check=True)
        flake8_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        flake8_available = False
        
    # Check if black is available
    try:
        subprocess.run(["black", "--version"], capture_output=True, check=True)
        black_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        black_available = False
    
    success = True
    
    if flake8_available:
        cmd = ["flake8", "src/", "tests/", "--max-line-length=100", "--ignore=E203,W503"]
        if not run_command(cmd, "Flake8 Linting"):
            success = False
    else:
        print("⚠️ Flake8 not available, skipping linting")
    
    if black_available:
        cmd = ["black", "--check", "--diff", "src/", "tests/"]
        if not run_command(cmd, "Black Code Formatting Check"):
            success = False
    else:
        print("⚠️ Black not available, skipping formatting check")
    
    return success


def generate_test_report():
    """Generate detailed test report."""
    cmd = [
        "python", "-m", "pytest",
        "--tb=long",
        "--cov=src/project_maestro",
        "--cov-report=html:test_reports/coverage",
        "--cov-report=xml:test_reports/coverage.xml", 
        "--junitxml=test_reports/junit.xml",
        "--durations=0",
        "tests/"
    ]
    
    # Create reports directory
    Path("test_reports").mkdir(exist_ok=True)
    
    return run_command(cmd, "Generate Test Report")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Project Maestro Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                          # Run all tests
  python run_tests.py --unit                   # Run unit tests only
  python run_tests.py --integration            # Run integration tests only
  python run_tests.py --performance            # Run performance tests only
  python run_tests.py --api                    # Run API tests only
  python run_tests.py --lint                   # Run linting only
  python run_tests.py --report                 # Generate detailed report
  python run_tests.py --test tests/test_api.py # Run specific test file
        """
    )
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--api", action="store_true", help="Run API tests")
    parser.add_argument("--lint", action="store_true", help="Run code linting")
    parser.add_argument("--report", action="store_true", help="Generate detailed test report")
    parser.add_argument("--test", type=str, help="Run specific test file or function")
    parser.add_argument("--no-cov", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    
    args = parser.parse_args()
    
    print("🚀 Project Maestro Test Runner")
    print("="*60)
    
    success = True
    
    # Run specific test if requested
    if args.test:
        success = run_specific_test(args.test)
    
    # Run linting if requested
    elif args.lint:
        success = run_linting()
    
    # Run test report generation if requested
    elif args.report:
        success = generate_test_report()
    
    # Run specific test types
    elif args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.performance:
        success = run_performance_tests()
    elif args.api:
        success = run_api_tests()
        
    # Run all tests by default
    else:
        print("Running full test suite...")
        
        # Run linting first
        print("\n📋 Step 1: Code Quality Checks")
        if not run_linting():
            print("⚠️ Linting issues found, but continuing with tests...")
            
        # Run tests
        print("\n🧪 Step 2: Running Tests")
        if not run_all_tests():
            success = False
        
        # Generate report
        print("\n📊 Step 3: Generating Reports")
        generate_test_report()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if success:
        print("✅ All tests completed successfully!")
        print("\n📊 Test reports generated in:")
        print("   - HTML Coverage: test_reports/coverage/index.html")
        print("   - XML Coverage: test_reports/coverage.xml") 
        print("   - JUnit XML: test_reports/junit.xml")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        print("Please check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()