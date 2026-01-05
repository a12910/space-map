# Contributing to Space-map

Thank you for your interest in contributing to Space-map! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/space-map.git
   cd space-map
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e .
   pip install -e ".[dev]"
   ```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, package versions)
- Code snippets or error messages

### Suggesting Features

Feature requests are welcome! Please:
- Check if the feature already exists or is planned
- Clearly describe the feature and its use case
- Explain why it would be useful to the community

### Code Contributions

1. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Test your changes**:
   ```bash
   python -c "import space_map; print('Import successful')"
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## Coding Standards

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex logic

### Docstring Format

Use NumPy-style docstrings:

```python
def my_function(param1, param2):
    """
    Brief description of function.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2

    Returns
    -------
    type
        Description of return value

    Examples
    --------
    >>> my_function(1, 2)
    3
    """
    return param1 + param2
```

## Documentation

When adding new features:
- Update relevant documentation in `docs/`
- Add examples if appropriate
- Update README.md if needed

To build documentation locally:
```bash
pip install -e ".[docs]"
mkdocs serve
# View at http://127.0.0.1:8000
```

## Pull Request Process

1. Ensure your code follows the coding standards
2. Update documentation as needed
3. Make sure your branch is up to date with master
4. Create a descriptive PR title and description
5. Link any related issues

### PR Title Format

- `Add: new feature description`
- `Fix: bug description`
- `Update: documentation/dependency updates`
- `Refactor: code refactoring description`

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Give constructive feedback
- Celebrate successes together

## Questions?

If you have questions:
- Check existing issues and discussions
- Create a new discussion on GitHub
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for contributing to Space-map and helping advance spatial biology research!
