# Contributing to CleanEngine

Thank you for your interest in contributing to CleanEngine! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Git
- pip or conda

### Local Development Setup
```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/CleanEngine.git
cd CleanEngine

# 2. Install dependencies
python setup.py

# 3. Verify installation
python main.py -t
```

## 📝 Code Style

### Python Style Guide
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (compatible with black)
- Use descriptive variable and function names

### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np

# Local imports
from .utils.config_manager import ConfigManager
```

### Documentation
- All public functions must have docstrings
- Use Google-style docstring format
- Include type hints for function parameters and return values

```python
def clean_dataset(self, file_path: str, output_path: str = None) -> pd.DataFrame:
    """Clean a dataset using the configured pipeline.
    
    Args:
        file_path: Path to the input dataset file
        output_path: Optional path for the cleaned output
        
    Returns:
        Cleaned pandas DataFrame
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file format is unsupported
    """
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python main.py -t

# Run specific test file
python -m pytest tests/test_data_analyzer.py -v

# Run with coverage
python -m pytest --cov=src/dataset_cleaner tests/
```

### Writing Tests
- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Use descriptive test names that explain the expected behavior
- Include both positive and negative test cases
- Mock external dependencies when appropriate

```python
def test_cleaner_handles_missing_values_correctly():
    """Test that missing values are handled according to config."""
    # Arrange
    df = pd.DataFrame({'col1': [1, None, 3], 'col2': ['a', 'b', None]})
    cleaner = DatasetCleaner()
    
    # Act
    result = cleaner.handle_missing_values(df)
    
    # Assert
    assert result.isnull().sum().sum() == 0
```

### Test Coverage
- Maintain test coverage above 80%
- Focus on critical business logic
- Test edge cases and error conditions

## 🔄 Pull Request Process

### Before Submitting
1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Update documentation** if needed
4. **Run tests locally** to ensure they pass
5. **Check code style** with your IDE or linter

### Branch Naming
Use descriptive branch names:
- `feature/add-new-analysis-method`
- `bugfix/fix-correlation-calculation`
- `docs/update-api-documentation`

### Commit Messages
Follow conventional commit format:
```
type(scope): description

feat(analysis): add DBSCAN clustering support
fix(core): resolve memory leak in large dataset processing
docs(readme): update installation instructions
style(utils): format code according to PEP 8
```

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new functionality
- [ ] Updated existing tests if needed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes introduced
```

## 🐛 Bug Reports

### Before Reporting
1. Check if the issue has already been reported
2. Try to reproduce the issue with the latest version
3. Check if it's a configuration issue

### Bug Report Template
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 10, macOS 12.0]
- Python Version: [e.g., 3.9.7]
- CleanEngine Version: [e.g., 0.1.0]

## Additional Context
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template
```markdown
## Feature Description
Clear description of the feature

## Use Case
Why this feature is needed

## Proposed Implementation
How you think it should work

## Alternatives Considered
Other approaches you've considered
```

## 📚 Documentation

### Updating Documentation
- Update README.md for user-facing changes
- Update docstrings for API changes
- Update config examples if configuration changes
- Keep documentation in sync with code changes

## 🔒 Security

### Reporting Security Issues
- **DO NOT** create a public issue for security vulnerabilities
- Email security reports to: [security@cleanengine.com]
- Include detailed description and steps to reproduce
- Allow time for response before public disclosure

## 🎯 Contribution Areas

### High Priority
- Bug fixes
- Performance improvements
- Test coverage improvements
- Documentation updates

### Medium Priority
- New analysis methods
- Additional file format support
- UI/UX improvements
- Configuration enhancements

### Low Priority
- Code refactoring
- Style improvements
- Additional examples

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Provide constructive feedback
- Celebrate contributions and achievements

## 📞 Getting Help

- Create an issue for questions
- Join our community discussions
- Check existing documentation
- Review closed issues and PRs

## 📄 License

By contributing to CleanEngine, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to CleanEngine! 🎉
