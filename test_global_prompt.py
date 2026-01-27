#!/usr/bin/env python3
"""Test the global prompt generator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from idp_extractor.prompts.global import generate_document_extraction_prompt
from idp_extractor.document.passport import Passeport

# Test the global prompt generator
prompt = generate_document_extraction_prompt(Passeport, "French passport")

print("Generated Prompt:")
print("=" * 50)
print(prompt)
print("=" * 50)

print("\nPrompt length:", len(prompt))
print("Contains JSON schema:", "{" in prompt and "}" in prompt)