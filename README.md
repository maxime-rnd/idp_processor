# IDP Extractor

A Python project for Intelligent Document Processing (IDP) extraction from French passports using Multimodal Large Language Models (MLLM).

## Features

- Extract structured information from French passport images using MLLM
- Pydantic BaseModel for data validation
- Performance tracking: processing time, energy consumption, cost, confidence
- Document quality assessment module

## Installation

Using uv:

```bash
pip install uv
./scripts/setup.sh
```

## Configuration

Create a `.env` file in the project root with your settings:

```bash
cp .env.example .env
# Edit .env with your API key and other settings
```

Required settings:
- `API_KEY`: Your API key for LLM services
- `BASE_URL`: Base URL for API endpoint (optional, defaults to OpenAI)

Optional settings:
- `MODEL`: Default model (default: gpt-4o)
- `COST_PER_TOKEN`: Cost per token in euros (default: 0.0000255)
- `DEFAULT_MAX_TOKENS`: Maximum tokens for API calls (default: 500)

## Usage

### Python API

```python
from idp_extractor import extract_passport_info

result = extract_passport_info("path/to/passport.jpg")
# Also supports PDFs
result = extract_passport_info("path/to/passport.pdf", api_key="your_key")
print(result.model_dump())
```

### Command Line Interface

The package includes a modern CLI built with Typer:

```bash
# Process a single file (uses API key from .env)
uv run idp-extractor process path/to/passport.jpg --output result.json

# Process a single file with custom API key
uv run idp-extractor process path/to/passport.jpg --api-key your_key --output result.json

# Process all files in a folder
uv run idp-extractor process-folder /path/to/folder --output results.json

# Get help
uv run idp-extractor --help
```

## Development

### Setup

```bash
./scripts/setup.sh
```

### Scripts

- `./scripts/setup.sh` - Set up development environment
- `./scripts/clean.sh` - Clean build artifacts and cache files
- `./scripts/format.sh` - Format code with ruff
- `./scripts/lint.sh` - Lint code with ruff
- `./scripts/test.sh` - Run tests

### CLI Commands

- `uv run idp-extractor process <file>` - Process a single file
- `uv run idp-extractor process-folder <folder>` - Process all files in a folder

### Code Quality

This project uses:
- **ruff** for linting and formatting
- **mypy** for type checking
- **pre-commit** for git hooks
- **pytest** for testing

Run all checks:

```bash
./scripts/lint.sh
./scripts/test.sh
```

### CI/CD

GitHub Actions CI runs on every push and pull request, testing against Python 3.10, 3.11, and 3.12.