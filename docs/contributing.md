# Contributing to Space-map

Thank you for your interest in contributing to Space-map! We welcome contributions from the community to help improve this project.

## How You Can Contribute

There are many ways to contribute to Space-map:

- 🐛 **Report bugs** - Help us identify and fix issues
- 💡 **Suggest features** - Share ideas for new functionality
- 📖 **Improve documentation** - Enhance guides, fix typos, add examples
- 🔬 **Share use cases** - Tell us how you're using Space-map
- 💻 **Submit code** - Fix bugs or implement new features
- 🧪 **Test releases** - Help validate new versions
- 💬 **Help others** - Answer questions in discussions

## Getting Started

### Prerequisites

Before contributing, make sure you have:

- Python 3.7+ installed
- Git installed
- A GitHub account
- Basic Python programming knowledge

### Development Setup

1. **Fork the repository** on GitHub by clicking the "Fork" button at [github.com/a12910/space-map](https://github.com/a12910/space-map)

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/space-map.git
   cd space-map
   ```

3. **Add upstream remote** to stay in sync:
   ```bash
   git remote add upstream https://github.com/a12910/space-map.git
   ```

4. **Create virtual environment** and install:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

5. **Verify installation**:
   ```bash
   python -c "import space_map; print(space_map.__version__)"
   ```

## Reporting Bugs

Found a bug? Please help us fix it by creating a detailed bug report.

### Before Reporting

- Search [existing issues](https://github.com/a12910/space-map/issues) to avoid duplicates
- Try to reproduce the bug with the latest version
- Collect relevant information about your environment

### Creating a Bug Report

Open a [new issue](https://github.com/a12910/space-map/issues/new) and include:

**Required Information:**
- **Clear title** - Brief description of the bug
- **Description** - What happened vs. what you expected
- **Steps to reproduce** - Minimal code example that triggers the bug
- **Environment details**:
  - Space-map version: `python -c "import space_map; print(space_map.__version__)"`
  - Python version: `python --version`
  - Operating system: e.g., "Ubuntu 22.04", "macOS 13.0", "Windows 11"
  - GPU info (if relevant): `nvidia-smi` output

**Optional but Helpful:**
- Screenshots or visualizations
- Error messages and stack traces
- Sample data (if shareable)

**Example Bug Report:**

```markdown
### Bug: LDDMM fails with GPU out of memory

**Description**: When processing 10 sections with ~500K cells each, LDDMM
registration fails with CUDA out of memory error.

**Steps to reproduce**:
1. Load 10 sections with 500K cells each
2. Run affine registration (succeeds)
3. Run LDDMM with `mgr.ldm_pair(...)` (fails)

**Environment**:
- Space-map: 0.1.0
- Python: 3.9.7
- OS: Ubuntu 20.04
- GPU: NVIDIA RTX 3080 (10GB VRAM)

**Error message**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.5 GiB...
```
```

## Suggesting Features

Have an idea to improve Space-map? We'd love to hear it!

### Before Suggesting

- Check if the feature already exists in the latest version
- Search [existing issues](https://github.com/a12910/space-map/issues) for similar suggestions
- Consider if the feature fits Space-map's scope (3D tissue reconstruction)

### Creating a Feature Request

Open a [new issue](https://github.com/a12910/space-map/issues/new) with:

- **Clear title** - Brief feature description
- **Problem statement** - What problem does this solve?
- **Proposed solution** - How should it work?
- **Use cases** - Who would benefit and how?
- **Alternatives** - Other approaches you've considered

**Example Feature Request:**

```markdown
### Feature: Support for multi-channel image alignment

**Problem**: Currently Space-map only uses cell coordinates. I have
histology images for each section that could improve alignment.

**Proposed solution**: Add option to provide images alongside coordinates:
```python
flowImport.init_with_images(xys, images, ids=layer_ids)
```

**Use cases**: Researchers with H&E or immunofluorescence images of
sections could achieve better registration accuracy.

**Alternatives**: Manual pre-alignment of images before using Space-map.
```

## Contributing Code

Ready to contribute code? Here's our workflow:

### 1. Create a Branch

```bash
# Update your fork
git checkout master
git pull upstream master

# Create a feature branch
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add comments for complex logic
- Keep changes focused on one issue/feature

### 3. Test Your Changes

```bash
# Test import
python -c "import space_map"

# Test basic functionality
python examples/01_quickstart.py  # If you have test data

# Check for obvious errors
python -m py_compile space_map/**/*.py
```

### 4. Commit Your Changes

Write clear commit messages:

```bash
git add .
git commit -m "Add: support for multi-channel image alignment

- Add init_with_images() method to FlowImport
- Update registration to use image features
- Add example notebook demonstrating usage
"
```

**Commit Message Format:**
- **Add:** for new features
- **Fix:** for bug fixes
- **Update:** for improvements to existing features
- **Docs:** for documentation changes
- **Refactor:** for code restructuring

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request" button
3. Select your branch
4. Fill out the PR template with:
   - Description of changes
   - Related issue number (if applicable)
   - Testing performed
   - Screenshots (if relevant)

### 7. Code Review

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged

## Coding Standards

### Style Guidelines

- **PEP 8**: Follow Python's style guide
- **Line length**: Max 100 characters (flexible for readability)
- **Imports**: Group by standard library, third-party, local
- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`

### Documentation

Add docstrings for all public functions and classes:

```python
def register_sections(sections, method="auto"):
    """
    Register multiple tissue sections using specified method.

    Parameters
    ----------
    sections : list of Slice
        Tissue sections to register
    method : str, optional
        Registration method ('auto', 'sift', 'loftr'), by default 'auto'

    Returns
    -------
    list of Slice
        Registered sections with aligned coordinates

    Examples
    --------
    >>> sections = load_sections()
    >>> aligned = register_sections(sections, method='auto')
    """
    pass
```

## Documentation Contributions

Improving documentation is incredibly valuable!

### What to Document

- **Fix typos and errors** - Even small fixes help
- **Add examples** - Show how to use features
- **Clarify explanations** - Make concepts easier to understand
- **Add tutorials** - Create guides for common workflows
- **Update outdated info** - Keep docs current with code

### Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/
│   ├── installation.md         # Installation guide
│   └── quickstart.md          # Quick start tutorial
└── contributing.md            # This file
```

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material

# Serve documentation locally
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. We expect all contributors to:

- **Be respectful** - Treat everyone with respect and kindness
- **Be constructive** - Provide helpful, actionable feedback
- **Be collaborative** - Work together towards common goals
- **Be patient** - Remember that everyone is learning
- **Be inclusive** - Welcome people of all backgrounds and skill levels

### Communication

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - Questions and community discussion
- **Pull Requests** - Code contributions and reviews
- **Email** - Direct contact: a12910@qq.com

## Recognition

Contributors are recognized in several ways:

- Listed in commit history
- Mentioned in release notes
- Acknowledged in documentation
- Added to CONTRIBUTORS file (for significant contributions)

## Questions?

Not sure how to get started? Have questions about contributing?

- Check [GitHub Discussions](https://github.com/a12910/space-map/discussions)
- Open an [issue](https://github.com/a12910/space-map/issues) with your question
- Email us at a12910@qq.com

## License

By contributing to Space-map, you agree that your contributions will be licensed under the MIT License.

---

## Quick Reference

**Setting up development environment:**
```bash
git clone https://github.com/YOUR_USERNAME/space-map.git
cd space-map
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

**Creating a contribution:**
```bash
git checkout master
git pull upstream master
git checkout -b feature/my-feature
# Make changes
git add .
git commit -m "Add: my feature"
git push origin feature/my-feature
# Create PR on GitHub
```

**Building documentation:**
```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

---

Thank you for contributing to Space-map and helping advance spatial biology research! 🔬
