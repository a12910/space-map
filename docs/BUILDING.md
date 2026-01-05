# Building and Testing the Documentation Website

This guide explains how to build and test the Space-map documentation website locally.

## Prerequisites

Install the required dependencies:

```bash
pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python mkdocs-git-revision-date-localized-plugin
```

## Building the Website Locally

### 1. Test the Build

Build the static website:

```bash
mkdocs build
```

This will generate the site in the `site/` directory.

### 2. Preview Locally

Start a local development server with live reload:

```bash
mkdocs serve
```

Then open your browser to http://127.0.0.1:8000

The server will automatically reload when you make changes to the documentation files.

## Deployment

The website is automatically deployed to GitHub Pages when you push to the `master` branch. The GitHub Actions workflow (`.github/workflows/docs.yml`) handles this automatically.

### Manual Deployment

If you need to manually deploy:

```bash
mkdocs gh-deploy
```

This will build the site and push it to the `gh-pages` branch.

## Project Structure

```
spacemap/
├── docs/                          # Documentation source files
│   ├── index.md                   # Homepage
│   ├── overview/                  # Getting started guides
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   └── ...
│   ├── examples/                  # Example gallery
│   │   ├── examples.md
│   │   └── ...
│   ├── api/                       # API reference
│   └── assets/                    # Images and other assets
├── examples/                      # Jupyter notebook examples
│   ├── 01_quickstart.ipynb
│   ├── 02_advanced_registration.ipynb
│   └── raw.ipynb
├── mkdocs.yml                     # MkDocs configuration
└── .github/workflows/docs.yml     # GitHub Actions workflow
```

## Documentation Updates

### Adding New Examples

1. Create a new Jupyter notebook in `examples/`
2. Add a link to it in `docs/examples/examples.md`
3. Update the navigation in `mkdocs.yml` if needed

### Updating API Documentation

The API documentation is generated automatically from docstrings using mkdocstrings. Make sure your Python code has proper docstrings in NumPy format.

### Adding Images

Place images in `docs/assets/images/` and reference them in markdown:

```markdown
![Alt text](../assets/images/your-image.png)
```

## What's New in This Update

This documentation update includes:

1. **New Example Notebooks**:
   - `01_quickstart.ipynb`: Complete beginner-friendly tutorial
   - `02_advanced_registration.ipynb`: Advanced techniques and parameter tuning

2. **Updated Documentation**:
   - `docs/overview/quickstart.md`: Comprehensive quick start guide with real API
   - `docs/examples/examples.md`: Multiple practical examples using correct API
   - `docs/index.md`: Updated homepage with accurate quick start

3. **Corrections**:
   - All code examples now use the actual Space-map API
   - Fixed incorrect function calls and module paths
   - Added proper imports and parameter names

## Testing Checklist

Before committing changes:

- [ ] Run `mkdocs build` to check for errors
- [ ] Run `mkdocs serve` and manually browse all pages
- [ ] Verify all code examples use correct API
- [ ] Check that all links work
- [ ] Verify images display correctly
- [ ] Test on different screen sizes if possible

## Troubleshooting

### Build Errors

If you get errors about missing files:
- Check that all referenced files exist
- Verify file paths in markdown links

### Plugin Errors

If mkdocstrings or other plugins fail:
- Make sure Space-map is installed: `pip install -e .`
- Check that Python imports work correctly

### Styling Issues

If the site looks wrong:
- Clear your browser cache
- Rebuild with `mkdocs build --clean`

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)

## GitHub Pages URL

The live documentation is available at:
https://a12910.github.io/space-map
