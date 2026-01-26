#!/usr/bin/env python3
"""Quick test of the DocumentQualityAssessment functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from idp_extractor.quality.quality import DocumentQualityAssessment
    print("✓ DocumentQualityAssessment imported successfully")

    # Test creating an instance
    assessment = DocumentQualityAssessment(
        blur_score=0.8,
        contrast=0.7,
        brightness=0.6,
        resolution_dpi=300,
        noise_level=0.2,
        skew_angle=1.5,
        text_confidence=0.9,
        edge_sharpness=0.8,
        illumination_uniformity=0.7,
        focus_measure=450.0
    )
    print("✓ DocumentQualityAssessment instance created")

    # Test the score property
    score = assessment.score
    print(f"✓ Overall score calculated: {score:.3f}")

    print("All tests passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()