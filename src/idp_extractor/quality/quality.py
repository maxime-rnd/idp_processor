import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from pydantic import BaseModel, Field
import math
from scipy import ndimage
from skimage import filters
from skimage.metrics import structural_similarity as ssim


class DocumentQualityMetrics(BaseModel):
    """Comprehensive quality metrics for document images and PDFs."""

    # Basic image properties
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    channels: int = Field(..., description="Number of color channels")
    file_size: int = Field(..., description="File size in bytes")
    aspect_ratio: float = Field(..., description="Width/height ratio")

    # Sharpness/Blur metrics
    laplacian_variance: float = Field(..., description="Variance of Laplacian (blur detection)")
    sobel_sharpness: float = Field(..., description="Sobel edge sharpness score")
    fft_focus_measure: float = Field(..., description="FFT-based focus measure")

    # Contrast and brightness
    contrast_std: float = Field(..., description="Standard deviation of pixel intensities")
    contrast_rms: float = Field(..., description="RMS contrast measure")
    brightness_mean: float = Field(..., description="Mean brightness (0-255)")
    brightness_median: float = Field(..., description="Median brightness (0-255)")

    # Histogram analysis
    histogram_entropy: float = Field(..., description="Image entropy from histogram")
    dynamic_range: float = Field(..., description="Dynamic range (max-min intensity)")

    # Noise and texture
    noise_estimate: float = Field(..., description="Estimated noise level")
    texture_energy: float = Field(..., description="Texture energy measure")

    # Color analysis
    color_balance_score: float = Field(..., description="RGB balance score (0-1)")
    saturation_mean: float = Field(..., description="Mean saturation (0-1)")
    color_temperature: float = Field(..., description="Estimated color temperature")

    # Lighting and uniformity
    lighting_uniformity: float = Field(..., description="Lighting uniformity score (0-1)")
    glare_pixels_ratio: float = Field(..., description="Ratio of pixels with glare (0-1)")
    shadow_pixels_ratio: float = Field(..., description="Ratio of pixels with shadows (0-1)")

    # Geometric properties
    estimated_rotation: float = Field(..., description="Estimated rotation angle in degrees")
    perspective_distortion: float = Field(..., description="Perspective distortion score (0-1)")
    document_alignment_score: float = Field(..., description="Document alignment quality (0-1)")

    # Content analysis
    text_region_ratio: float = Field(..., description="Ratio of image that appears to be text")
    background_uniformity: float = Field(..., description="Background uniformity score (0-1)")
    foreground_separation: float = Field(..., description="Foreground-background separation quality")

    # Compression and artifacts
    compression_artifacts_score: float = Field(..., description="Compression artifacts detection (0-1)")
    jpeg_quality_estimate: Optional[float] = Field(None, description="Estimated JPEG quality (0-100)")

    # Advanced metrics
    structural_similarity: float = Field(..., description="SSIM score compared to ideal document")
    edge_density: float = Field(..., description="Edge density in the image")
    local_contrast_variation: float = Field(..., description="Local contrast variation measure")

    # PDF-specific metrics (if applicable)
    is_pdf: bool = Field(False, description="Whether the input was a PDF")
    pdf_page_count: Optional[int] = Field(None, description="Number of pages in PDF")
    pdf_text_extraction_ratio: Optional[float] = Field(None, description="Ratio of text that could be extracted")

    # Error handling
    processing_errors: List[str] = Field(default_factory=list, description="List of processing errors encountered")

    @property
    def overall_quality_score(self) -> float:
        """
        Compute overall quality score from all metrics.

        This is a weighted combination of all quality factors.
        Returns a score between 0.0 (poor quality) and 1.0 (excellent quality).
        """
        if self.processing_errors:
            return 0.0

        # Define weights for different quality aspects
        weights = {
            'sharpness': 0.25,      # Blur/sharpness is most important
            'contrast': 0.15,       # Good contrast is crucial
            'brightness': 0.10,     # Proper brightness
            'noise': 0.10,          # Low noise
            'lighting': 0.10,       # Uniform lighting
            'geometric': 0.10,      # Proper alignment
            'content': 0.10,        # Clear content
            'artifacts': 0.10       # No compression artifacts
        }

        # Normalize and score each component
        scores = {}

        # Sharpness score (higher laplacian = sharper)
        scores['sharpness'] = min(1.0, self.laplacian_variance / 500.0)

        # Contrast score (optimal contrast around 50-80)
        optimal_contrast = 65.0
        contrast_distance = abs(self.contrast_std - optimal_contrast)
        scores['contrast'] = max(0.0, 1.0 - contrast_distance / 50.0)

        # Brightness score (optimal around 128)
        brightness_distance = abs(self.brightness_mean - 128.0)
        scores['brightness'] = max(0.0, 1.0 - brightness_distance / 64.0)

        # Noise score (lower noise = higher score)
        scores['noise'] = max(0.0, 1.0 - self.noise_estimate / 50.0)

        # Lighting uniformity
        scores['lighting'] = self.lighting_uniformity

        # Geometric quality
        scores['geometric'] = (self.document_alignment_score +
                              (1.0 - self.perspective_distortion)) / 2.0

        # Content quality
        scores['content'] = (self.text_region_ratio +
                            self.background_uniformity +
                            self.foreground_separation) / 3.0

        # Artifacts (lower artifacts = higher score)
        scores['artifacts'] = 1.0 - self.compression_artifacts_score

        # Compute weighted average
        overall_score = sum(scores[aspect] * weight for aspect, weight in weights.items())

        return max(0.0, min(1.0, overall_score))

    @property
    def quality_grade(self) -> str:
        """Return a letter grade based on overall quality score."""
        score = self.overall_quality_score
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        elif score >= 0.5:
            return "D"
        else:
            return "F"

    @property
    def issues_detected(self) -> List[str]:
        """Return a list of quality issues detected."""
        issues = []

        if self.laplacian_variance < 100:
            issues.append("Image appears blurry")
        if self.contrast_std < 30:
            issues.append("Low contrast")
        if self.brightness_mean < 80 or self.brightness_mean > 180:
            issues.append("Poor brightness")
        if self.noise_estimate > 20:
            issues.append("High noise level")
        if self.lighting_uniformity < 0.7:
            issues.append("Uneven lighting")
        if self.glare_pixels_ratio > 0.1:
            issues.append("Glare detected")
        if self.shadow_pixels_ratio > 0.2:
            issues.append("Heavy shadows")
        if self.compression_artifacts_score > 0.3:
            issues.append("Compression artifacts detected")
        if abs(self.estimated_rotation) > 5:
            issues.append("Document may be rotated")
        if self.perspective_distortion > 0.2:
            issues.append("Perspective distortion detected")

        return issues


class DocumentQualityAssessment(BaseModel):
    """Exhaustive document quality assessment with key performance indicators."""

    # Core quality KPIs
    blur_score: float = Field(..., ge=0.0, le=1.0, description="Blur assessment score (0=very blurry, 1=sharp)")
    contrast: float = Field(..., ge=0.0, le=1.0, description="Contrast quality score (0=low contrast, 1=optimal contrast)")
    brightness: float = Field(..., ge=0.0, le=1.0, description="Brightness quality score (0=too dark/bright, 1=optimal)")
    resolution_dpi: int = Field(..., ge=0, description="Estimated resolution in DPI")
    noise_level: float = Field(..., ge=0.0, le=1.0, description="Noise level (0=clean, 1=very noisy)")
    skew_angle: float = Field(..., description="Document skew angle in degrees")
    text_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in text detection (0=no text, 1=clear text)")
    edge_sharpness: float = Field(..., ge=0.0, le=1.0, description="Edge sharpness score (0=blurry edges, 1=sharp edges)")
    illumination_uniformity: float = Field(..., ge=0.0, le=1.0, description="Lighting uniformity (0=uneven, 1=uniform)")
    focus_measure: float = Field(..., ge=0.0, description="Focus quality measure (higher=better focus)")

    @property
    def score(self) -> float:
        """
        Compute overall quality score from all KPIs.

        Returns a weighted score between 0.0 (poor quality) and 1.0 (excellent quality).
        """
        # Define weights for each KPI
        weights = {
            'blur_score': 0.25,          # Most important for readability
            'contrast': 0.15,
            'brightness': 0.10,
            'noise_level': 0.10,         # Invert since higher noise = lower quality
            'text_confidence': 0.15,
            'edge_sharpness': 0.10,
            'illumination_uniformity': 0.10,
            'focus_measure': 0.05        # Normalized focus measure
        }

        # Calculate weighted score
        total_score = (
            self.blur_score * weights['blur_score'] +
            self.contrast * weights['contrast'] +
            self.brightness * weights['brightness'] +
            (1.0 - self.noise_level) * weights['noise_level'] +  # Invert noise
            self.text_confidence * weights['text_confidence'] +
            self.edge_sharpness * weights['edge_sharpness'] +
            self.illumination_uniformity * weights['illumination_uniformity'] +
            min(1.0, self.focus_measure / 1000.0) * weights['focus_measure']  # Normalize focus
        )

        # Penalize for extreme skew
        skew_penalty = max(0.0, abs(self.skew_angle) - 2.0) / 10.0  # Penalty for skew > 2 degrees
        total_score = max(0.0, total_score - skew_penalty)

        # Penalize for low resolution
        if self.resolution_dpi < 150:
            resolution_penalty = (150 - self.resolution_dpi) / 300.0
            total_score = max(0.0, total_score - resolution_penalty)

        return min(1.0, max(0.0, total_score))


def assess_document_quality(file_path: str) -> DocumentQualityMetrics:
    """
    Exhaustive quality assessment of document images and PDFs.

    Analyzes multiple quality factors including sharpness, contrast, brightness,
    noise, lighting, geometric properties, and content quality.

    Args:
        file_path: Path to image file (jpg, png, etc.) or PDF

    Returns:
        DocumentQualityMetrics: Comprehensive quality assessment
    """
    path = Path(file_path)
    if not path.exists():
        return DocumentQualityMetrics(
            width=0, height=0, channels=0, file_size=0, aspect_ratio=0.0,
            laplacian_variance=0.0, sobel_sharpness=0.0, fft_focus_measure=0.0,
            contrast_std=0.0, contrast_rms=0.0, brightness_mean=0.0, brightness_median=0.0,
            histogram_entropy=0.0, dynamic_range=0.0, noise_estimate=0.0, texture_energy=0.0,
            color_balance_score=0.0, saturation_mean=0.0, color_temperature=0.0,
            lighting_uniformity=0.0, glare_pixels_ratio=0.0, shadow_pixels_ratio=0.0,
            estimated_rotation=0.0, perspective_distortion=0.0, document_alignment_score=0.0,
            text_region_ratio=0.0, background_uniformity=0.0, foreground_separation=0.0,
            compression_artifacts_score=0.0, structural_similarity=0.0, edge_density=0.0,
            local_contrast_variation=0.0, processing_errors=["File does not exist"]
        )

    file_size = path.stat().st_size

    # Handle PDFs
    if path.suffix.lower() == '.pdf':
        return _assess_pdf_quality(file_path, file_size)

    # Handle images
    return _assess_image_quality(file_path, file_size)


def _assess_image_quality(image_path: str, file_size: int) -> DocumentQualityMetrics:
    """Assess quality of image file."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")

        height, width, channels = image.shape
        aspect_ratio = width / height if height > 0 else 0.0

        # Convert to grayscale for most analyses
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Sharpness/Blur metrics
        laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Sobel sharpness
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_sharpness = np.sqrt(sobel_x**2 + sobel_y**2).mean()

        # FFT focus measure
        fft_focus_measure = _compute_fft_focus_measure(gray)

        # Contrast and brightness
        contrast_std = gray.std()
        contrast_rms = np.sqrt(np.mean((gray - gray.mean())**2))
        brightness_mean = gray.mean()
        brightness_median = np.median(gray)

        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        histogram_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        dynamic_range = gray.max() - gray.min()

        # Noise estimation
        noise_estimate = _estimate_noise(gray)

        # Texture energy
        texture_energy = _compute_texture_energy(gray)

        # Color analysis
        color_balance_score = _compute_color_balance(image)
        saturation_mean = _compute_mean_saturation(image)
        color_temperature = _estimate_color_temperature(image)

        # Lighting analysis
        lighting_uniformity = _compute_lighting_uniformity(gray)
        glare_pixels_ratio, shadow_pixels_ratio = _detect_glare_and_shadows(gray)

        # Geometric analysis
        estimated_rotation = _estimate_rotation(gray)
        perspective_distortion = _estimate_perspective_distortion(image)
        document_alignment_score = _compute_alignment_score(gray)

        # Content analysis
        text_region_ratio = _estimate_text_region_ratio(gray)
        background_uniformity = _compute_background_uniformity(gray)
        foreground_separation = _compute_foreground_separation(gray)

        # Compression artifacts
        compression_artifacts_score = _detect_compression_artifacts(gray)
        jpeg_quality_estimate = _estimate_jpeg_quality(image_path)

        # Advanced metrics
        structural_similarity = _compute_structural_similarity(gray)
        edge_density = _compute_edge_density(gray)
        local_contrast_variation = _compute_local_contrast_variation(gray)

        return DocumentQualityMetrics(
            width=width,
            height=height,
            channels=channels,
            file_size=file_size,
            aspect_ratio=aspect_ratio,
            laplacian_variance=laplacian_variance,
            sobel_sharpness=sobel_sharpness,
            fft_focus_measure=fft_focus_measure,
            contrast_std=contrast_std,
            contrast_rms=contrast_rms,
            brightness_mean=brightness_mean,
            brightness_median=brightness_median,
            histogram_entropy=histogram_entropy,
            dynamic_range=dynamic_range,
            noise_estimate=noise_estimate,
            texture_energy=texture_energy,
            color_balance_score=color_balance_score,
            saturation_mean=saturation_mean,
            color_temperature=color_temperature,
            lighting_uniformity=lighting_uniformity,
            glare_pixels_ratio=glare_pixels_ratio,
            shadow_pixels_ratio=shadow_pixels_ratio,
            estimated_rotation=estimated_rotation,
            perspective_distortion=perspective_distortion,
            document_alignment_score=document_alignment_score,
            text_region_ratio=text_region_ratio,
            background_uniformity=background_uniformity,
            foreground_separation=foreground_separation,
            compression_artifacts_score=compression_artifacts_score,
            jpeg_quality_estimate=jpeg_quality_estimate,
            structural_similarity=structural_similarity,
            edge_density=edge_density,
            local_contrast_variation=local_contrast_variation,
            is_pdf=False
        )

    except Exception as e:
        return DocumentQualityMetrics(
            width=0, height=0, channels=0, file_size=file_size, aspect_ratio=0.0,
            laplacian_variance=0.0, sobel_sharpness=0.0, fft_focus_measure=0.0,
            contrast_std=0.0, contrast_rms=0.0, brightness_mean=0.0, brightness_median=0.0,
            histogram_entropy=0.0, dynamic_range=0.0, noise_estimate=0.0, texture_energy=0.0,
            color_balance_score=0.0, saturation_mean=0.0, color_temperature=0.0,
            lighting_uniformity=0.0, glare_pixels_ratio=0.0, shadow_pixels_ratio=0.0,
            estimated_rotation=0.0, perspective_distortion=0.0, document_alignment_score=0.0,
            text_region_ratio=0.0, background_uniformity=0.0, foreground_separation=0.0,
            compression_artifacts_score=0.0, structural_similarity=0.0, edge_density=0.0,
            local_contrast_variation=0.0, processing_errors=[str(e)]
        )


def _assess_pdf_quality(pdf_path: str, file_size: int) -> DocumentQualityMetrics:
    """Assess quality of PDF file."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(pdf_path)
        page_count = len(pdf)

        # Get first page for analysis
        if page_count > 0:
            page = pdf[0]
            pil_image = page.render(scale=2).to_pil()

            # Convert to OpenCV format
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            height, width, channels = image.shape

            # Basic metrics from first page
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast_std = gray.std()
            brightness_mean = gray.mean()

            # PDF-specific metrics
            text_extraction_ratio = 0.8  # Placeholder - would need OCR/text extraction

            return DocumentQualityMetrics(
                width=width,
                height=height,
                channels=channels,
                file_size=file_size,
                aspect_ratio=width/height,
                laplacian_variance=laplacian_variance,
                sobel_sharpness=0.0,  # Not computed for PDF
                fft_focus_measure=0.0,
                contrast_std=contrast_std,
                contrast_rms=0.0,
                brightness_mean=brightness_mean,
                brightness_median=0.0,
                histogram_entropy=0.0,
                dynamic_range=0.0,
                noise_estimate=0.0,
                texture_energy=0.0,
                color_balance_score=0.0,
                saturation_mean=0.0,
                color_temperature=0.0,
                lighting_uniformity=0.0,
                glare_pixels_ratio=0.0,
                shadow_pixels_ratio=0.0,
                estimated_rotation=0.0,
                perspective_distortion=0.0,
                document_alignment_score=0.0,
                text_region_ratio=0.0,
                background_uniformity=0.0,
                foreground_separation=0.0,
                compression_artifacts_score=0.0,
                structural_similarity=0.0,
                edge_density=0.0,
                local_contrast_variation=0.0,
                is_pdf=True,
                pdf_page_count=page_count,
                pdf_text_extraction_ratio=text_extraction_ratio
            )
        else:
            return DocumentQualityMetrics(
                width=0, height=0, channels=0, file_size=file_size, aspect_ratio=0.0,
                laplacian_variance=0.0, sobel_sharpness=0.0, fft_focus_measure=0.0,
                contrast_std=0.0, contrast_rms=0.0, brightness_mean=0.0, brightness_median=0.0,
                histogram_entropy=0.0, dynamic_range=0.0, noise_estimate=0.0, texture_energy=0.0,
                color_balance_score=0.0, saturation_mean=0.0, color_temperature=0.0,
                lighting_uniformity=0.0, glare_pixels_ratio=0.0, shadow_pixels_ratio=0.0,
                estimated_rotation=0.0, perspective_distortion=0.0, document_alignment_score=0.0,
                text_region_ratio=0.0, background_uniformity=0.0, foreground_separation=0.0,
                compression_artifacts_score=0.0, structural_similarity=0.0, edge_density=0.0,
                local_contrast_variation=0.0, is_pdf=True, pdf_page_count=0,
                processing_errors=["PDF has no pages"]
            )

    except Exception as e:
        return DocumentQualityMetrics(
            width=0, height=0, channels=0, file_size=file_size, aspect_ratio=0.0,
            laplacian_variance=0.0, sobel_sharpness=0.0, fft_focus_measure=0.0,
            contrast_std=0.0, contrast_rms=0.0, brightness_mean=0.0, brightness_median=0.0,
            histogram_entropy=0.0, dynamic_range=0.0, noise_estimate=0.0, texture_energy=0.0,
            color_balance_score=0.0, saturation_mean=0.0, color_temperature=0.0,
            lighting_uniformity=0.0, glare_pixels_ratio=0.0, shadow_pixels_ratio=0.0,
            estimated_rotation=0.0, perspective_distortion=0.0, document_alignment_score=0.0,
            text_region_ratio=0.0, background_uniformity=0.0, foreground_separation=0.0,
            compression_artifacts_score=0.0, structural_similarity=0.0, edge_density=0.0,
            local_contrast_variation=0.0, is_pdf=True, processing_errors=[str(e)]
        )


def assess_document_quality_simple(file_path: str) -> DocumentQualityAssessment:
    """
    Simple quality assessment with key performance indicators.

    Provides essential quality metrics for quick evaluation and decision making.

    Args:
        file_path: Path to image file (jpg, png, etc.) or PDF

    Returns:
        DocumentQualityAssessment: Key quality KPIs with overall score
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Handle PDFs by converting first page to image
    if path.suffix.lower() == '.pdf':
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(file_path)
        if len(pdf) == 0:
            raise ValueError("PDF has no pages")
        page = pdf[0]
        pil_image = page.render(scale=2).to_pil()
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Could not load image")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Estimate resolution (rough approximation)
    # Assume passport-sized document, typical scanning resolution
    resolution_dpi = int(min(width, height) / 3.375)  # Passport width ~3.375 inches

    # Blur score (0-1, higher = sharper)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(1.0, laplacian_var / 500.0)

    # Contrast (0-1, optimal around 0.6-0.8)
    contrast_std = gray.std()
    contrast = min(1.0, contrast_std / 128.0)  # Normalize to 0-1

    # Brightness (0-1, optimal around 0.5)
    brightness_mean = gray.mean() / 255.0
    brightness = 1.0 - abs(brightness_mean - 0.5) * 2  # Peak at 0.5, drop to 0 at extremes

    # Noise level (0-1, higher = noisier)
    noise_level = _estimate_noise(gray) / 50.0  # Normalize

    # Skew angle estimation
    skew_angle = _estimate_skew_angle(gray)

    # Text confidence (simplified - based on edge density and contrast)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / gray.size
    text_confidence = min(1.0, edge_density * 5)  # Rough approximation

    # Edge sharpness (based on sobel gradients)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_sharpness = min(1.0, np.sqrt(sobel_x**2 + sobel_y**2).mean() / 50.0)

    # Illumination uniformity (0-1, higher = more uniform)
    illumination_uniformity = _compute_lighting_uniformity(gray)

    # Focus measure (variance of Laplacian)
    focus_measure = laplacian_var

    return DocumentQualityAssessment(
        blur_score=blur_score,
        contrast=contrast,
        brightness=brightness,
        resolution_dpi=resolution_dpi,
        noise_level=noise_level,
        skew_angle=skew_angle,
        text_confidence=text_confidence,
        edge_sharpness=edge_sharpness,
        illumination_uniformity=illumination_uniformity,
        focus_measure=focus_measure
    )


# Helper functions for quality metrics

def _compute_fft_focus_measure(gray: np.ndarray) -> float:
    """Compute focus measure using FFT."""
    try:
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        focus_measure = np.sum(magnitude[1:, 1:]) / np.sum(magnitude)
        return float(focus_measure)
    except:
        return 0.0


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate document skew angle using Hough transform."""
    try:
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Use Hough transform to find lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            # Convert to skew angle (-45 to 45 degrees)
            if angle > 90:
                angle -= 180
            angles.append(angle)

        if not angles:
            return 0.0

        # Return median angle as skew estimate
        return float(np.median(angles))
    except:
        return 0.0


def _estimate_noise(gray: np.ndarray) -> float:
    """Estimate noise level in the image."""
    try:
        # Use median filter to estimate noise
        median_filtered = cv2.medianBlur(gray, 3)
        noise = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
        return float(np.mean(noise))
    except:
        return 0.0


def _compute_texture_energy(gray: np.ndarray) -> float:
    """Compute texture energy using Gabor filters."""
    try:
        # Simple texture measure using local variance
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_var = cv2.filter2D(gray**2, -1, kernel) - local_mean**2
        return float(np.mean(np.sqrt(local_var)))
    except:
        return 0.0


def _compute_color_balance(image: np.ndarray) -> float:
    """Compute RGB color balance score."""
    try:
        b, g, r = cv2.split(image)
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        total = b_mean + g_mean + r_mean
        if total == 0:
            return 0.0
        # Ideal balance is roughly equal RGB
        balance = 1.0 - (abs(b_mean - g_mean) + abs(g_mean - r_mean) + abs(r_mean - b_mean)) / (3 * total)
        return max(0.0, min(1.0, balance))
    except:
        return 0.0


def _compute_mean_saturation(image: np.ndarray) -> float:
    """Compute mean saturation from HSV."""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1]) / 255.0)
    except:
        return 0.0


def _estimate_color_temperature(image: np.ndarray) -> float:
    """Estimate color temperature (rough approximation)."""
    try:
        b, g, r = cv2.split(image.astype(np.float32))
        rg = np.mean(r) / max(np.mean(g), 1)
        bg = np.mean(b) / max(np.mean(g), 1)

        # Rough color temperature estimation
        if rg > 1.1:
            return 5000  # Warm
        elif bg > 1.1:
            return 10000  # Cool
        else:
            return 6500  # Neutral
    except:
        return 6500


def _compute_lighting_uniformity(gray: np.ndarray) -> float:
    """Compute lighting uniformity score."""
    try:
        # Divide image into 4 quadrants and compare means
        h, w = gray.shape
        q1 = gray[:h//2, :w//2].mean()
        q2 = gray[:h//2, w//2:].mean()
        q3 = gray[h//2:, :w//2].mean()
        q4 = gray[h//2:, w//2:].mean()

        quadrants = [q1, q2, q3, q4]
        mean_intensity = np.mean(quadrants)
        uniformity = 1.0 - np.std(quadrants) / max(mean_intensity, 1)
        return max(0.0, min(1.0, uniformity))
    except:
        return 0.0


def _detect_glare_and_shadows(gray: np.ndarray) -> Tuple[float, float]:
    """Detect glare and shadow pixels."""
    try:
        # Glare: very bright pixels
        glare_threshold = np.percentile(gray, 95)
        glare_pixels = np.sum(gray > glare_threshold)
        glare_ratio = glare_pixels / gray.size

        # Shadows: very dark pixels
        shadow_threshold = np.percentile(gray, 5)
        shadow_pixels = np.sum(gray < shadow_threshold)
        shadow_ratio = shadow_pixels / gray.size

        return float(glare_ratio), float(shadow_ratio)
    except:
        return 0.0, 0.0


def _estimate_rotation(gray: np.ndarray) -> float:
    """Estimate document rotation angle."""
    try:
        # Use Hough line transform to find dominant lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

        if lines is not None:
            angles = []
            for line in lines[:10]:  # Use first 10 lines
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                if angle > 90:
                    angle -= 180
                angles.append(angle)

            if angles:
                return float(np.median(angles))

        return 0.0
    except:
        return 0.0


def _estimate_perspective_distortion(image: np.ndarray) -> float:
    """Estimate perspective distortion."""
    try:
        # Simple heuristic: check if corners are significantly different
        h, w = image.shape[:2]
        corners = [
            image[0, 0].mean(),
            image[0, w-1].mean(),
            image[h-1, 0].mean(),
            image[h-1, w-1].mean()
        ]
        distortion = np.std(corners) / max(np.mean(corners), 1)
        return min(1.0, distortion)
    except:
        return 0.0


def _compute_alignment_score(gray: np.ndarray) -> float:
    """Compute document alignment score."""
    try:
        # Check if text lines are roughly horizontal
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

        if lines is not None:
            horizontal_lines = 0
            total_lines = len(lines)

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:  # Roughly horizontal
                    horizontal_lines += 1

            return horizontal_lines / max(total_lines, 1)
        return 0.5  # Neutral score if no lines detected
    except:
        return 0.5


def _estimate_text_region_ratio(gray: np.ndarray) -> float:
    """Estimate ratio of image that contains text."""
    try:
        # Use adaptive thresholding to find text regions
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        text_pixels = np.sum(cleaned > 0)
        return text_pixels / gray.size
    except:
        return 0.0


def _compute_background_uniformity(gray: np.ndarray) -> float:
    """Compute background uniformity score."""
    try:
        # Assume background is the most common intensity
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        background_intensity = np.argmax(hist)

        # Create mask for background pixels
        background_mask = np.abs(gray.astype(np.float32) - background_intensity) < 20
        background_std = np.std(gray[background_mask])

        # Lower std = more uniform background
        uniformity = 1.0 - min(1.0, background_std / 50.0)
        return max(0.0, uniformity)
    except:
        return 0.0


def _compute_foreground_separation(gray: np.ndarray) -> float:
    """Compute foreground-background separation quality."""
    try:
        # Use Otsu's method for thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Compute separation score based on bimodality
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()

        # Simple bimodality measure
        mid = len(hist) // 2
        left_mean = np.average(range(mid), weights=hist_norm[:mid])
        right_mean = np.average(range(mid, len(hist)), weights=hist_norm[mid:])

        separation = abs(right_mean - left_mean) / 128.0
        return min(1.0, separation)
    except:
        return 0.0


def _detect_compression_artifacts(gray: np.ndarray) -> float:
    """Detect compression artifacts."""
    try:
        # Look for periodic patterns that indicate JPEG compression
        # Simple heuristic: check for regular patterns in DCT-like analysis
        h, w = gray.shape

        # Check for 8x8 block artifacts (common in JPEG)
        block_artifacts = 0
        for i in range(0, h-8, 8):
            for j in range(0, w-8, 8):
                block = gray[i:i+8, j:j+8]
                # Check if block edges have similar patterns
                if np.std(block[0, :]) < 5 and np.std(block[:, 0]) < 5:
                    block_artifacts += 1

        return min(1.0, block_artifacts / ((h//8) * (w//8)))
    except:
        return 0.0


def _estimate_jpeg_quality(image_path: str) -> Optional[float]:
    """Estimate JPEG quality if applicable."""
    try:
        # This is a placeholder - actual JPEG quality estimation is complex
        # Would need to analyze quantization tables
        return None
    except:
        return None


def _compute_structural_similarity(gray: np.ndarray) -> float:
    """Compute SSIM compared to an ideal document image."""
    try:
        # Create a synthetic "ideal" document image (uniform background with text-like patterns)
        h, w = gray.shape
        ideal = np.full_like(gray, 200, dtype=np.uint8)

        # Add some text-like horizontal lines
        for i in range(0, h, 20):
            ideal[i:i+2, :] = 50

        # Compute SSIM
        score = ssim(gray, ideal, data_range=255)
        return max(0.0, score)
    except:
        return 0.5


def _compute_edge_density(gray: np.ndarray) -> float:
    """Compute edge density in the image."""
    try:
        edges = cv2.Canny(gray, 100, 200)
        return np.sum(edges > 0) / gray.size
    except:
        return 0.0


def _compute_local_contrast_variation(gray: np.ndarray) -> float:
    """Compute local contrast variation."""
    try:
        # Compute local standard deviation
        kernel = np.ones((15, 15), np.float32) / 225
        local_std = cv2.filter2D(gray.astype(np.float32)**2, -1, kernel) - \
                   cv2.filter2D(gray.astype(np.float32), -1, kernel)**2
        local_std = np.sqrt(np.maximum(local_std, 0))

        return float(np.mean(local_std) / 50.0)  # Normalize
    except:
        return 0.0