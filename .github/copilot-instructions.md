# Copilot Instructions for IDP Extractor

## Project Overview
This project is an Intelligent Document Processing (IDP) extractor that processes French passport images to extract structured data using Multimodal Large Language Models (MLLM) via OpenAI-compatible endpoints.

## Architecture
- **Models**: Pydantic BaseModel for French passport data (`src/idp_extractor/document/passport.py`)
- **Extractor**: MLLM-based extraction using connector abstraction (`src/idp_extractor/extractor.py`)
- **Connector**: Abstract LLM interface with implementations for OpenAI-compatible (`src/idp_extractor/connector/llmaas.py`) and local VLM via vLLM (`src/idp_extractor/connector/local_connector.py`); supports image and PDF file processing
- **Prompts**: Versioned prompt templates with system and user prompts (`src/idp_extractor/prompts/passeport/v1.py`)
- **Quality**: Document image quality assessment (`src/idp_extractor/quality.py`)
- Data flows from image input → quality check → connector-based MLLM extraction → Pydantic validation → metrics tracking

## Key Workflows
- **Install**: `uv sync` (uses `pyproject.toml` for dependencies)
- **Run**: `uv run python -c "from idp_extractor import extract_passport_info; print(extract_passport_info('path/to/image.jpg', api_key='your_key'))"`
- **Test**: `uv run pytest` (tests in `tests/`)
- **Debug**: Check logs from codecarbon/emissions; validate Pydantic models for extraction accuracy

## Conventions
- Use Pydantic BaseModel for all extracted data structures (e.g., `Passeport` in `document/passport.py`)
- Implement LLM connectors inheriting from `LLMConnector` abstract base class
- Track performance with `PerformanceTracker` class wrapping extraction calls
- Assess document quality before processing using `assess_document_quality()`
- Use versioned prompt templates from `prompts/` with system and user prompts; user prompts support .format() for dynamic variables
- Dependencies managed via uv in `pyproject.toml`; prefer OpenAI client for MLLM calls, vLLM for local inference

## Integration Points
- External API: OpenAI-compatible endpoint for MLLM (vision models)
- Metrics: codecarbon for energy tracking, time module for processing time
- Quality: OpenCV for image metrics (blur, contrast, brightness)

## Patterns
- Modular: Each feature in separate module under `src/idp_extractor/`
- Example: Extraction pipeline: `quality = assess_document_quality(image); tracker.start_tracking(); result = extract_passport_info(image, api_key); metrics = tracker.stop_tracking()` as in hypothetical `main.py`