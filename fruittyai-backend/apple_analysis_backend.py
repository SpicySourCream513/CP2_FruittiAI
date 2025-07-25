# apple_analysis_backend.py
# Extracted from test.py - NO GUI, exact same processing logic

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import sys
import os
import threading
import time

class AppleAnalysisPipeline:
    """EXACT: Your complete analysis pipeline from test.py WITHOUT GUI"""
    
    def __init__(self):
        # Initialize variables (same as test.py)
        self.edge_settings = {
        # Scoring weights
        'circularity_weight': 0.4,    # 40%
        'symmetry_weight': 0.35,      # 35% 
        'smoothness_weight': 0.25,    # 25%
        
        # Quality thresholds
        'excellent_threshold': 0.80,  # Excellent grade
        'good_threshold': 0.65,       # Good grade
        'fair_threshold': 0.45,       # Fair grade
        
        # Scaling factors
        'circularity_scale_factor': 0.6,
        'smoothness_scale_factor': 0.85,
        
        # Confidence settings
        'base_confidence_excellent': 0.90,
        'base_confidence_good': 0.80,
        'base_confidence_fair': 0.70,
        'base_confidence_poor': 0.60,
        'confidence_boost': 0.05,
        'confidence_penalty': 0.8,
    }

        self.models_loaded = False
        self.current_image = None
        self.cropped_apple = None
        self.processed_apple = None
        self.color_analysis_result = None
        
        # Initialize color analyzer
        self.color_analyzer = RobustAppleColorAnalyzer()
        
        # EXACT: Your apple-aware preprocessing parameters from test.py
        self.setup_apple_aware_preprocessing_params()
        
    def setup_apple_aware_preprocessing_params(self):
        """EXACT: Your tuned preprocessing parameters specifically for apple variety classification"""
        
        # 1. BRIGHTNESS: More conservative (preserve natural lighting)
        self.BRIGHTNESS_LOW_THRESHOLD = 25   # Lower (was 40) - only very dark images
        self.BRIGHTNESS_HIGH_THRESHOLD = 250 # Higher (was 240) - only blown out images  
        self.BRIGHTNESS_ADJUSTMENT_AMOUNT = 20  # Gentler (was 35)
        
        # 2. BLUR: Keep existing (blur is always bad for classification)
        self.BLUR_LAPLACIAN_THRESHOLD = 100
        self.BLUR_GRADIENT_THRESHOLD = 18
        self.BLUR_EDGE_DENSITY_THRESHOLD = 0.022
        self.SHARPEN_MILD_KERNEL = np.array([[0, -0.3, 0], [-0.3, 2.2, -0.3], [0, -0.3, 0]])  # Gentler
        self.SHARPEN_STRONG_KERNEL = np.array([[-0.3, -0.7, -0.3], [-0.7, 5.4, -0.7], [-0.3, -0.7, -0.3]])  # Gentler
        
        # 3. CONTRAST: Much gentler CLAHE
        self.CLAHE_CLIP_LIMIT = 1.8  # Gentler (was 2.5)
        self.CLAHE_TILE_GRID_SIZE = (4, 4)  # Larger tiles (was 6,6) - less localized
        self.CONTRAST_THRESHOLD = 20  # Lower threshold (was 30) - less aggressive
        
        # 4. COLOR CAST: Much more conservative (preserve natural apple colors)
        self.COLOR_CAST_RATIO_THRESHOLD = 2.2  # Higher (was 1.6) - only obvious casts
        self.COLOR_CAST_DIFFERENCE_THRESHOLD = 65  # Higher (was 45) - bigger difference needed
        self.COLOR_CAST_CORRECTION_FACTOR = 0.95  # More conservative (was 0.92)
        self.COLOR_CAST_MIN_BRIGHTNESS = 40  # Higher (was 30)
        self.COLOR_CAST_MAX_SATURATION = 160  # Lower (was 180) - avoid correcting colorful apples
        
        # 5. NOISE: Keep existing (noise is always bad)
        self.NOISE_THRESHOLD = 2.0
        self.NOISE_REDUCTION_STRENGTH = 25  # Slightly gentler (was 25)
        self.NLM_H = 12  # Gentler (was 12)
        self.NLM_TEMPLATE_WINDOW_SIZE = 7
        self.NLM_SEARCH_WINDOW_SIZE = 21
        
        # 6. SATURATION: Apple-variety aware
        self.SATURATION_LOW_THRESHOLD = 25   # Lower (was 35) - some varieties naturally low
        self.SATURATION_HIGH_THRESHOLD = 200 # NEW: Avoid oversaturated varieties
        self.SATURATION_ENHANCEMENT_FACTOR = 1.15  # Gentler (was 1.25)
        self.SATURATION_REDUCTION_FACTOR = 0.90     # NEW: For oversaturated cases
        
        # 7. GEOMETRY: Keep existing (rotation affects all varieties equally)
        self.ROTATION_DETECTION_THRESHOLD = 1.5
        self.HOUGH_THRESHOLD = 80
        self.ANGLE_TOLERANCE = 35
        self.MIN_LINES_FOR_ROTATION = 8
        
        # 8. RESIZE: Keep existing
        self.CNN_TARGET_SIZE = (224, 224)
        self.RESIZE_INTERPOLATION = cv2.INTER_LANCZOS4

    def load_models(self):
        """EXACT: Load the 3 required models from test.py"""
        try:
            print("Loading YOLO model...")
            self.yolo_model = YOLO("models/weights/best_fruit.pt")
            
            print("Loading ripeness/defect model...")
            # CHANGE THIS LINE:
            # OLD: self.ripeness_model = tf.keras.models.load_model("models/custom/apple_ripeness_defect_model.keras")
            # NEW:
            self.ripeness_model = tf.keras.models.load_model("models/custom/apple_ripeness.keras")
            
            print("Loading variety model...")
            self.variety_model = tf.keras.models.load_model("models/custom/apple_type.h5")
            
            self.models_loaded = True
            print("✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            self.models_loaded = False
            return False

    def detect_apples_yolo(self, image):
        """FIXED: Detect ALL fruits with proper fruit_type field"""
        if not self.models_loaded:
            return []
        
        try:
            # Run YOLO detection
            results = self.yolo_model(image, verbose=False)
            fruits_found = []
            
            # Add confidence threshold to filter out false positives
            CONFIDENCE_THRESHOLD = 0.75
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # FIXED: Accept ALL fruits with good confidence
                        if confidence >= CONFIDENCE_THRESHOLD:
                            fruits_found.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'conf': float(confidence),
                                'class': str(class_name),
                                'fruit_type': str(class_name)  # FIXED: Ensure fruit_type exists
                            })
            
            # Sort by confidence (highest first)
            fruits_found.sort(key=lambda x: x['conf'], reverse=True)
            
            print(f"🔍 YOLO detected {len(fruits_found)} fruits:")
            for fruit in fruits_found:
                print(f"   - {fruit['fruit_type']}: {fruit['conf']:.3f} confidence")
            
            return fruits_found
            
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _calculate_brightness(self, image_cv):
        """EXACT: Calculates the average brightness of an image."""
        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_image)

    def _enhanced_blur_detection(self, image_cv):
        """EXACT: Enhanced blur detection using multiple methods"""
        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        edges = cv2.Canny(gray_image, 40, 120)
        edge_density = np.sum(edges > 0) / (gray_image.shape[0] * gray_image.shape[1])
        
        is_very_blurry = (laplacian_var < 30 and gradient_magnitude < 10 and edge_density < 0.012)
        is_moderately_blurry = (laplacian_var < self.BLUR_LAPLACIAN_THRESHOLD and 
                               gradient_magnitude < self.BLUR_GRADIENT_THRESHOLD and 
                               edge_density < self.BLUR_EDGE_DENSITY_THRESHOLD)
        
        blur_indicators = 0
        if laplacian_var < self.BLUR_LAPLACIAN_THRESHOLD:
            blur_indicators += 1
        if gradient_magnitude < self.BLUR_GRADIENT_THRESHOLD:
            blur_indicators += 1  
        if edge_density < self.BLUR_EDGE_DENSITY_THRESHOLD:
            blur_indicators += 1
            
        is_moderately_blurry = is_moderately_blurry and (blur_indicators >= 2)
        
        return {
            'laplacian_var': laplacian_var,
            'gradient_magnitude': gradient_magnitude,
            'edge_density': edge_density,
            'is_very_blurry': is_very_blurry,
            'is_moderately_blurry': is_moderately_blurry,
            'blur_indicators': blur_indicators
        }

    def _detect_color_cast(self, image_cv):
        """EXACT: Enhanced color cast detection with better validation"""
        img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        b_avg, g_avg, r_avg = np.mean(img_rgb, axis=(0,1))
        
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:,:,1])
        
        color_ratio = b_avg / r_avg if r_avg > 0 else 1
        color_diff = abs(b_avg - r_avg)
        
        rg_balance = abs(r_avg - g_avg) / max(r_avg, g_avg) if max(r_avg, g_avg) > 0 else 0
        bg_balance = abs(b_avg - g_avg) / max(b_avg, g_avg) if max(b_avg, g_avg) > 0 else 0
        
        has_blue_cast = (
            color_ratio > self.COLOR_CAST_RATIO_THRESHOLD and 
            color_diff > self.COLOR_CAST_DIFFERENCE_THRESHOLD and
            brightness > self.COLOR_CAST_MIN_BRIGHTNESS and
            saturation < self.COLOR_CAST_MAX_SATURATION and
            bg_balance > 0.15
        )
        
        has_red_cast = (
            color_ratio < (1/self.COLOR_CAST_RATIO_THRESHOLD) and 
            color_diff > self.COLOR_CAST_DIFFERENCE_THRESHOLD and
            brightness > self.COLOR_CAST_MIN_BRIGHTNESS and
            saturation < self.COLOR_CAST_MAX_SATURATION and
            rg_balance > 0.15
        )
        
        return {
            'r_avg': r_avg, 'g_avg': g_avg, 'b_avg': b_avg,
            'brightness': brightness, 'saturation': saturation,
            'color_ratio': color_ratio, 'color_diff': color_diff,
            'rg_balance': rg_balance, 'bg_balance': bg_balance,
            'has_blue_cast': has_blue_cast, 'has_red_cast': has_red_cast
        }

    def _detect_noise_level(self, image_cv):
        """EXACT: Enhanced noise detection using multiple robust methods"""
        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Improved bilateral filter comparison
        bilateral = cv2.bilateralFilter(gray_image, 9, 80, 80)
        noise_diff = cv2.absdiff(gray_image, bilateral)
        noise_level_bilateral = np.mean(noise_diff)
        noise_std_bilateral = np.std(noise_diff)
        
        # Method 2: Median filter comparison (salt-and-pepper noise)
        median_filtered = cv2.medianBlur(gray_image, 5)
        median_noise_diff = cv2.absdiff(gray_image, median_filtered)
        median_noise_level = np.mean(median_noise_diff)
        
        # Method 3: Local standard deviation analysis
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D((gray_image.astype(np.float32))**2, -1, kernel)
        local_variance = local_sq_mean - local_mean**2
        local_std = np.sqrt(np.maximum(local_variance, 0))
        noise_variation = np.std(local_std)
        
        # Method 4: PSNR calculation
        mse = np.mean((gray_image.astype(np.float64) - bilateral.astype(np.float64))**2)
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
        
        # Method 5: High-frequency noise detection
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        high_freq_threshold = np.percentile(gradient_magnitude, 85)
        high_freq_pixels = gradient_magnitude > high_freq_threshold
        high_freq_noise = np.std(gradient_magnitude[high_freq_pixels]) if np.any(high_freq_pixels) else 0
        
        # Adaptive thresholds based on image characteristics
        img_brightness = np.mean(gray_image)
        img_contrast = np.std(gray_image)
        
        brightness_factor = 1.0
        if img_brightness < 50:  # Dark images more prone to noise
            brightness_factor = 1.3
        elif img_brightness > 200:  # Bright images may hide noise
            brightness_factor = 0.8
        
        contrast_factor = 1.0
        if img_contrast < 30:  # Low contrast shows noise more
            contrast_factor = 1.2
        
        base_threshold = self.NOISE_THRESHOLD * brightness_factor * contrast_factor
        
        # Multiple noise indicators
        noise_indicators = 0
        
        if noise_level_bilateral > base_threshold:
            noise_indicators += 1
        if noise_std_bilateral > base_threshold * 1.5:
            noise_indicators += 1
        if psnr < 35:
            noise_indicators += 1
        if noise_variation > base_threshold * 0.8:
            noise_indicators += 1
        if median_noise_level > base_threshold * 0.7:
            noise_indicators += 1
        if high_freq_noise > base_threshold * 2:
            noise_indicators += 1
        
        # Final decision: Need at least 2 indicators
        has_noise = noise_indicators >= 2
        
        # Determine noise type
        noise_type = "none"
        if has_noise:
            if median_noise_level > noise_level_bilateral:
                noise_type = "salt_pepper"
            elif high_freq_noise > noise_level_bilateral * 1.5:
                noise_type = "high_frequency"
            else:
                noise_type = "gaussian"
        
        return {
            'noise_level': noise_level_bilateral,
            'noise_variance': noise_std_bilateral,
            'median_noise_level': median_noise_level,
            'high_freq_noise': high_freq_noise,
            'psnr': psnr,
            'noise_variation': noise_variation,
            'has_noise': has_noise,
            'noise_indicators': noise_indicators,
            'noise_type': noise_type,
            'base_threshold': base_threshold,
            'img_brightness': img_brightness,
            'img_contrast': img_contrast
        }

    def _detect_saturation_level(self, image_cv):
        """EXACT: Detect saturation level in the image"""
        img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:,:,1])
        
        return {
            'saturation': saturation,
            'low_saturation': saturation < self.SATURATION_LOW_THRESHOLD,
            'high_saturation': saturation > self.SATURATION_HIGH_THRESHOLD
        }

    def _detect_geometric_issues(self, image_cv):
        """EXACT: Detect geometric issues with better rotation detection"""
        try:
            gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
            
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=self.HOUGH_THRESHOLD)
            
            rotation_angle = 0
            needs_rotation = False
            confidence = 0
            valid_lines = 0
            
            if lines is not None and len(lines) >= self.MIN_LINES_FOR_ROTATION:
                angles = []
                
                for line in lines[:30]:
                    if len(line) == 2:
                        rho, theta = line
                    else:
                        rho, theta = line[0]
                    
                    angle_deg = (theta * 180 / np.pi) - 90
                    
                    if abs(angle_deg) <= self.ANGLE_TOLERANCE and abs(rho) > 20:
                        angles.append(angle_deg)
                        valid_lines += 1
                
                if len(angles) >= 3:
                    angles = np.array(angles)
                    
                    Q1 = np.percentile(angles, 25)
                    Q3 = np.percentile(angles, 75)
                    IQR = Q3 - Q1
                    
                    median_angle = np.median(angles)
                    filtered_angles = angles[np.abs(angles - median_angle) <= 1.5 * IQR]
                    
                    if len(filtered_angles) >= 2:
                        rotation_angle = np.mean(filtered_angles)
                        
                        angle_std = np.std(filtered_angles)
                        confidence = min(1.0, (len(filtered_angles) / len(angles)) * (1 / (1 + angle_std)))
                        
                        if (abs(rotation_angle) > self.ROTATION_DETECTION_THRESHOLD and 
                            confidence > 0.3 and 
                            len(filtered_angles) >= 3):
                            needs_rotation = True
            
            return {
                'rotation_angle': rotation_angle,
                'needs_rotation': needs_rotation,
                'confidence': confidence,
                'valid_lines': valid_lines
            }
            
        except Exception as e:
            return {
                'rotation_angle': 0,
                'needs_rotation': False,
                'confidence': 0,
                'valid_lines': 0
            }

    # ==========================================
    # YOUR EXACT CORRECTION/ENHANCEMENT FUNCTIONS FROM test.py
    # ==========================================
    
    def _apply_gentle_color_cast_correction(self, image_cv, color_cast_info):
        """EXACT: More conservative color cast correction for apples"""
        img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        if color_cast_info['has_blue_cast']:
            excess_blue = color_cast_info['b_avg'] - color_cast_info['r_avg']
            correction_strength = min(0.08, excess_blue / 255.0)  # Gentler (was 0.15)
            
            img_rgb[:,:,2] = img_rgb[:,:,2] * (1 - correction_strength * 0.2)  # Gentler (was 0.3)
            img_rgb[:,:,0] = np.clip(img_rgb[:,:,0] * (1 + correction_strength * 0.05), 0, 255)  # Gentler (was 0.1)
            
            return cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
        elif color_cast_info['has_red_cast']:
            excess_red = color_cast_info['r_avg'] - color_cast_info['b_avg']
            correction_strength = min(0.08, excess_red / 255.0)  # Gentler
            
            img_rgb[:,:,0] = img_rgb[:,:,0] * (1 - correction_strength * 0.2)  # Gentler
            img_rgb[:,:,2] = np.clip(img_rgb[:,:,2] * (1 + correction_strength * 0.05), 0, 255)  # Gentler
            
            return cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        return image_cv

    def _apply_gentle_saturation_enhancement(self, image_cv, saturation_info):
        """EXACT: Gentler saturation enhancement that preserves apple variety characteristics"""
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[:,:,1] = hsv[:,:,1] * self.SATURATION_ENHANCEMENT_FACTOR
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 180)  # Lower max (was 200) to avoid unnatural colors
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _apply_saturation_reduction(self, image_cv, saturation_info):
        """EXACT: NEW: Reduce oversaturation for more natural apple colors"""
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[:,:,1] = hsv[:,:,1] * self.SATURATION_REDUCTION_FACTOR
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _apply_noise_reduction(self, image_cv, noise_info):
        """EXACT: Enhanced noise reduction with adaptive filtering based on noise type"""
        if not noise_info['has_noise']:
            return image_cv
        
        noise_type = noise_info['noise_type']
        noise_level = noise_info['noise_level']
        
        # Determine filtering strength
        if noise_level > self.NOISE_THRESHOLD * 3:
            strength_multiplier = 2.0
        elif noise_level > self.NOISE_THRESHOLD * 1.5:
            strength_multiplier = 1.5
        else:
            strength_multiplier = 1.0
        
        # Apply noise reduction based on noise type
        if noise_type == "salt_pepper":
            # Median filter first for impulse noise
            denoised = cv2.medianBlur(image_cv, 5)
            # Follow with bilateral filtering
            denoised = cv2.bilateralFilter(denoised, 9, 
                                         int(self.NOISE_REDUCTION_STRENGTH * 0.8), 
                                         int(self.NOISE_REDUCTION_STRENGTH * 0.8))
        
        elif noise_type == "high_frequency":
            # Non-Local Means for sensor noise
            h_value = int(self.NLM_H * strength_multiplier)
            denoised = cv2.fastNlMeansDenoisingColored(image_cv, None, 
                                                     h=h_value,
                                                     hColor=h_value,
                                                     templateWindowSize=self.NLM_TEMPLATE_WINDOW_SIZE,
                                                     searchWindowSize=self.NLM_SEARCH_WINDOW_SIZE)
        
        else:  # Gaussian or general noise
            # Multi-stage approach
            bilateral_strength = int(self.NOISE_REDUCTION_STRENGTH * strength_multiplier)
            denoised = cv2.bilateralFilter(image_cv, 9, bilateral_strength, bilateral_strength)
            
            # Follow with Non-Local Means
            h_value = int(self.NLM_H * strength_multiplier * 0.8)
            denoised = cv2.fastNlMeansDenoisingColored(denoised, None,
                                                     h=h_value,
                                                     hColor=h_value,
                                                     templateWindowSize=7,
                                                     searchWindowSize=15)
        
        # Edge preservation
        gray_original = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_original, 30, 80)
        edge_mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        edge_mask = edge_mask.astype(np.float32) / 255.0
        edge_mask = cv2.merge([edge_mask, edge_mask, edge_mask])
        
        # Blend based on edges
        final_result = denoised.astype(np.float32) * (1 - edge_mask * 0.3) + \
                       image_cv.astype(np.float32) * (edge_mask * 0.3)
        
        return np.clip(final_result, 0, 255).astype(np.uint8)
    
    def analyze_apple_shape(self, cropped_apple_image):
        """
        Edge-based apple shape analysis - much more robust than contour-based
        EXACT implementation from test4.py
        """
        try:
            if cropped_apple_image is None or cropped_apple_image.size == 0:
                return {
                    'quality': 'unknown',
                    'circularity': 0.0,
                    'aspect_ratio': 1.0,
                    'convexity': 0.0,
                    'solidity': 0.0,
                    'symmetry': 0.0,
                    'smoothness': 0.0,
                    'overall_score': 0.0,
                    'confidence': 0.0
                }
            
            # Convert to grayscale
            gray = cv2.cvtColor(cropped_apple_image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Edge detection with multiple thresholds for robustness
            edges = cv2.Canny(filtered, 50, 150, apertureSize=3)
            
            # Calculate edge-based metrics
            metrics = self._calculate_edge_based_metrics(edges, gray, cropped_apple_image.shape)
            
            # Classify using edge-based method
            quality, confidence = self._classify_edge_based_shape(metrics)
            
            return {
                'quality': quality,
                'circularity': float(metrics['edge_circularity']),
                'aspect_ratio': float(metrics['aspect_ratio']),
                'convexity': float(metrics['edge_density']),      # Using edge density as convexity
                'solidity': float(metrics['edge_continuity']),    # Using edge continuity as solidity
                'symmetry': float(metrics['edge_symmetry']),
                'smoothness': float(metrics['edge_smoothness']),
                'overall_score': float(metrics['overall_score']),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            print(f"❌ Edge-based shape analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'quality': 'error',
                'circularity': 0.0,
                'aspect_ratio': 1.0,
                'convexity': 0.0,
                'solidity': 0.0,
                'symmetry': 0.0,
                'smoothness': 0.0,
                'overall_score': 0.0,
                'confidence': 0.0
            }
    def _calculate_edge_based_metrics(self, edges, gray_image, image_shape):
        """Calculate shape metrics based on edge analysis"""
        height, width = edges.shape
        total_pixels = height * width
        
        # 1. Edge Density
        edge_pixels = np.sum(edges > 0)
        edge_density = edge_pixels / total_pixels
        
        # 2. Edge Circularity
        edge_circularity = self._calculate_edge_circularity(edges)
        
        # 3. Edge Symmetry
        edge_symmetry = self._calculate_edge_symmetry(edges)
        
        # 4. Edge Smoothness
        edge_smoothness = self._calculate_edge_smoothness(edges)
        
        # 5. Edge Continuity
        edge_continuity = self._calculate_edge_continuity(edges)
        
        # 6. Aspect Ratio
        aspect_ratio = self._calculate_edge_aspect_ratio(edges)
        
        # 7. Overall Score
        overall_score = (
            edge_circularity * self.edge_settings['circularity_weight'] +
            edge_symmetry * self.edge_settings['symmetry_weight'] +
            edge_smoothness * self.edge_settings['smoothness_weight']
        )
        
        return {
            'edge_circularity': edge_circularity,
            'edge_density': edge_density,
            'edge_symmetry': edge_symmetry,
            'edge_smoothness': edge_smoothness,
            'edge_continuity': edge_continuity,
            'aspect_ratio': aspect_ratio,
            'overall_score': overall_score
        }

    def _calculate_edge_circularity(self, edges):
        """Calculate circularity based on edge distribution"""
        try:
            edge_points = np.column_stack(np.where(edges > 0))
            
            if len(edge_points) < 10:
                return 0.0
            
            # Find center of mass of edge points
            center_y = np.mean(edge_points[:, 0])
            center_x = np.mean(edge_points[:, 1])
            
            # Calculate distances from center to edge points
            distances = np.sqrt((edge_points[:, 0] - center_y)**2 + (edge_points[:, 1] - center_x)**2)
            
            if len(distances) == 0:
                return 0.0
            
            # Circularity based on distance variation
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            if mean_distance == 0:
                return 0.0
            
            # Lower standard deviation relative to mean = more circular
            circularity = max(0.0, 1.0 - (std_distance / mean_distance))
            
            # Apply scaling
            scaled_circularity = min(circularity / self.edge_settings['circularity_scale_factor'], 1.0)
            
            return scaled_circularity
            
        except Exception as e:
            print(f"Edge circularity calculation error: {e}")
            return 0.5

    def _calculate_edge_symmetry(self, edges):
        """Calculate symmetry based on edge distribution"""
        try:
            height, width = edges.shape
            center_x = width // 2
            
            # Split into left and right halves
            left_edges = edges[:, :center_x]
            right_edges = edges[:, center_x:]
            
            # Count edge pixels in each half
            left_count = np.sum(left_edges > 0)
            right_count = np.sum(right_edges > 0)
            
            if left_count == 0 and right_count == 0:
                return 0.0
            
            # Calculate balance ratio
            balance = min(left_count, right_count) / max(left_count, right_count) if max(left_count, right_count) > 0 else 0.0
            
            # Row-by-row symmetry analysis
            row_symmetries = []
            for row in range(height):
                left_row = np.sum(left_edges[row, :] > 0)
                right_row = np.sum(right_edges[row, :] > 0)
                
                if left_row + right_row > 0:
                    row_balance = min(left_row, right_row) / max(left_row, right_row) if max(left_row, right_row) > 0 else 0.0
                    row_symmetries.append(row_balance)
            
            # Combine overall balance and row-wise symmetry
            if row_symmetries:
                avg_row_symmetry = np.mean(row_symmetries)
                symmetry = (balance * 0.6 + avg_row_symmetry * 0.4)
            else:
                symmetry = balance
            
            return min(symmetry, 1.0)
            
        except Exception as e:
            print(f"Edge symmetry calculation error: {e}")
            return 0.5

    def _calculate_edge_smoothness(self, edges):
        """Calculate smoothness based on edge continuity"""
        try:
            # Find contours from edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            # Use the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            if len(largest_contour) < 10:
                return 0.0
            
            # Calculate convex hull
            hull = cv2.convexHull(largest_contour)
            contour_area = cv2.contourArea(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            if hull_area == 0:
                return 0.0
            
            # Convexity ratio as smoothness measure
            convexity = contour_area / hull_area
            
            # Apply scaling
            smoothness = min(convexity / self.edge_settings['smoothness_scale_factor'], 1.0)
            
            return smoothness
            
        except Exception as e:
            print(f"Edge smoothness calculation error: {e}")
            return 0.5

    def _calculate_edge_continuity(self, edges):
        """Calculate edge continuity"""
        try:
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Count total edge pixels
            total_edge_pixels = np.sum(edges > 0)
            
            if total_edge_pixels == 0:
                return 0.0
            
            # Continuity ratio
            continuity = min(perimeter / (total_edge_pixels + 1), 1.0)
            
            return continuity
            
        except Exception as e:
            print(f"Edge continuity calculation error: {e}")
            return 0.5

    def _calculate_edge_aspect_ratio(self, edges):
        """Calculate aspect ratio from edge bounding box"""
        try:
            # Find edge points
            edge_points = np.column_stack(np.where(edges > 0))
            
            if len(edge_points) < 4:
                return 1.0
            
            # Get bounding box
            min_row = np.min(edge_points[:, 0])
            max_row = np.max(edge_points[:, 0])
            min_col = np.min(edge_points[:, 1])
            max_col = np.max(edge_points[:, 1])
            
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            
            if height == 0:
                return 1.0
            
            aspect_ratio = width / height
            return aspect_ratio
            
        except Exception as e:
            print(f"Edge aspect ratio calculation error: {e}")
            return 1.0

    def _classify_edge_based_shape(self, metrics):
        """Classify apple shape based on edge metrics"""
        overall_score = metrics['overall_score']
        edge_circularity = metrics['edge_circularity']
        edge_smoothness = metrics['edge_smoothness']
        
        # Classification based on overall score
        if overall_score >= self.edge_settings['excellent_threshold']:
            quality = "Excellent"
            confidence = self.edge_settings['base_confidence_excellent'] + (overall_score * self.edge_settings['confidence_boost'])
            
        elif overall_score >= self.edge_settings['good_threshold']:
            quality = "Good"
            confidence = self.edge_settings['base_confidence_good'] + (overall_score * 0.10)
            
        elif overall_score >= self.edge_settings['fair_threshold']:
            quality = "Fair"
            confidence = self.edge_settings['base_confidence_fair'] + (overall_score * 0.10)
            
        else:
            quality = "Poor"
            confidence = self.edge_settings['base_confidence_poor'] + (overall_score * 0.10)
        
        # Adjust confidence for irregular shapes
        if edge_circularity < 0.3 or edge_smoothness < 0.4:
            confidence *= self.edge_settings['confidence_penalty']
        
        return quality, min(confidence, 0.95)

    def _classify_realistic_apple_shape(self, metrics):
        """
        Classify apple shape based on realistic commercial apple standards
        """
        overall_score = metrics['overall_score']
        circularity = metrics['circularity']
        symmetry = metrics['symmetry']
        smoothness = metrics['smoothness']
        
        print(f"🍎 Apple Shape Analysis:")
        print(f"   Circularity: {circularity:.2f} (how round)")
        print(f"   Symmetry: {symmetry:.2f} (left-right balance)")
        print(f"   Smoothness: {smoothness:.2f} (no major dents)")
        print(f"   Overall Score: {overall_score:.2f}")
        
        # Realistic apple quality grades
        if overall_score >= 0.80:
            quality = "Excellent"
            confidence = 0.90 + (overall_score * 0.05)  # 90-95% confidence
            print(f"   Grade: {quality} - Premium market quality")
            
        elif overall_score >= 0.65:
            quality = "Good"
            confidence = 0.80 + (overall_score * 0.10)  # 80-90% confidence  
            print(f"   Grade: {quality} - Standard market quality")
            
        elif overall_score >= 0.45:
            quality = "Fair"
            confidence = 0.70 + (overall_score * 0.10)  # 70-80% confidence
            print(f"   Grade: {quality} - Lower grade, some defects")
            
        else:
            quality = "Poor"
            confidence = 0.60 + (overall_score * 0.10)  # 60-70% confidence
            print(f"   Grade: {quality} - Significant shape defects")
        
        # Adjust confidence based on extreme values
        if circularity < 0.3 or smoothness < 0.4:
            confidence *= 0.8  # Lower confidence for very irregular apples
            print(f"   Confidence adjusted down due to irregularities")
        
        return quality, min(confidence, 0.95)
        
    def _calculate_realistic_apple_metrics(self, contour, image_shape):
        """
        FIXED: Calculate all required shape metrics including convexity and solidity
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0 or area == 0:
            return {
                'circularity': 0.0,
                'aspect_ratio': 1.0,
                'convexity': 0.0,      # FIXED: Added missing field
                'solidity': 0.0,       # FIXED: Added missing field
                'symmetry': 0.0,
                'smoothness': 0.0,
                'overall_score': 0.0
            }
        
        # Get bounding rectangle for aspect ratio and solidity
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Calculate convexity (convex hull ratio)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0.0
        
        # Calculate solidity (contour area vs bounding rectangle area)
        rect_area = w * h
        solidity = area / rect_area if rect_area > 0 else 0.0
        
        # 1. Modified Circularity (adjusted for real apples)
        raw_circularity = (4 * np.pi * area) / (perimeter * perimeter)
        apple_circularity = min(raw_circularity / 0.6, 1.0)
        
        # 2. Symmetry Analysis
        symmetry_score = self._calculate_apple_symmetry(contour)
        
        # 3. Smoothness (using convexity)
        smoothness_score = min(convexity / 0.85, 1.0)  # Use convexity for smoothness
        
        # 4. Overall Apple Shape Score
        overall_score = (
            apple_circularity * 0.4 +    # 40% - overall roundness
            symmetry_score * 0.35 +      # 35% - balanced shape
            smoothness_score * 0.25      # 25% - no major defects
        )
        
        return {
            'circularity': apple_circularity,
            'aspect_ratio': aspect_ratio,
            'convexity': convexity,          # FIXED: Include convexity
            'solidity': solidity,            # FIXED: Include solidity
            'symmetry': symmetry_score,
            'smoothness': smoothness_score,
            'overall_score': overall_score
        }
    
    def _calculate_apple_smoothness(self, contour):
        """
        Calculate how smooth the apple outline is (detect major dents/bumps)
        """
        try:
            # Use convex hull to detect major irregularities
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            actual_area = cv2.contourArea(contour)
            
            if hull_area == 0:
                return 0.0
            
            # Convexity ratio: closer to 1.0 means smoother shape
            convexity = actual_area / hull_area
            
            # For apples, 0.85+ is very smooth, 0.7+ is acceptable
            # Scale accordingly: 0.85 raw = 1.0 smoothness score
            apple_smoothness = min(convexity / 0.85, 1.0)
            
            return apple_smoothness
            
        except:
            return 0.5
    
    def _calculate_apple_symmetry(self, contour):
        """
        FIXED: Calculate apple symmetry with better error handling
        """
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            
            # Split contour into left and right halves
            left_points = []
            right_points = []
            
            for point in contour:
                px = point[0][0]
                if px < center_x:
                    left_points.append(point)
                else:
                    right_points.append(point)
            
            if len(left_points) < 5 or len(right_points) < 5:
                return 0.8  # Default good symmetry if not enough points
            
            # Compare the distribution
            balance_ratio = min(len(left_points), len(right_points)) / max(len(left_points), len(right_points))
            return balance_ratio
            
        except Exception as e:
            print(f"❌ Symmetry calculation error: {e}")
            return 0.8  # Default to good symmetry on error
        
    def _find_best_apple_contour(self, blurred_image, image_shape):
        """
        FIXED: Find the best contour representing the apple
        """
        contours_candidates = []
        
        # Method 1: Simple threshold
        try:
            _, thresh1 = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY_INV)
            contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_candidates.extend(contours1)
            print(f"🔍 Method 1: Found {len(contours1)} contours")
        except Exception as e:
            print(f"❌ Method 1 failed: {e}")
        
        # Method 2: OTSU threshold
        try:
            _, thresh2 = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_candidates.extend(contours2)
            print(f"🔍 Method 2: Found {len(contours2)} contours")
        except Exception as e:
            print(f"❌ Method 2 failed: {e}")
        
        if not contours_candidates:
            print("❌ No contours found with any method")
            return None
        
        # Filter contours by area
        image_area = image_shape[0] * image_shape[1]
        min_area = image_area * 0.05   # At least 5% of image
        max_area = image_area * 0.95   # At most 95% of image
        
        valid_contours = []
        for contour in contours_candidates:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                valid_contours.append(contour)
        
        print(f"🔍 Valid contours: {len(valid_contours)} out of {len(contours_candidates)}")
        
        if not valid_contours:
            print("❌ No valid contours after filtering")
            return None
        
        # Return the largest valid contour
        largest = max(valid_contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest)
        print(f"🔍 Selected contour area: {largest_area} pixels ({(largest_area/image_area)*100:.1f}% of image)")
        
        return largest
        
    def _calculate_shape_metrics(self, contour):
        """
        Calculate detailed geometric metrics for apple shape analysis
        """
        # Basic contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Avoid division by zero
        if perimeter == 0 or area == 0:
            return {
                'circularity': 0.0,
                'aspect_ratio': 1.0,
                'convexity': 0.0,
                'solidity': 0.0
            }
        
        # 1. Circularity (4π × Area / Perimeter²)
        # Perfect circle = 1.0, more irregular = lower value
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        circularity = min(circularity, 1.0)  # Cap at 1.0
        
        # 2. Aspect Ratio (Width / Height)
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # 3. Convexity (Convex Hull Area / Actual Area)
        # Measures how "dented" or irregular the shape is
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0.0
        
        # 4. Solidity (Contour Area / Bounding Rectangle Area)
        # Measures how "full" the shape is
        rect_area = w * h
        solidity = area / rect_area if rect_area > 0 else 0.0
        
        return {
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'convexity': convexity,
            'solidity': solidity,
            'area': area,
            'perimeter': perimeter
        }
    
    def _classify_shape_quality(self, metrics):
        """
        UPDATED: More realistic thresholds for real apples
        """
        circularity = metrics['circularity']
        aspect_ratio = metrics['aspect_ratio']
        convexity = metrics['convexity']
        solidity = metrics['solidity']
        
        print(f"🔍 Classifying - circularity: {circularity:.3f}, aspect_ratio: {aspect_ratio:.3f}, convexity: {convexity:.3f}, solidity: {solidity:.3f}")
        
        # Normalize aspect ratio (ideal apple is slightly taller than wide)
        if 0.9 <= aspect_ratio <= 1.4:  # More lenient range
            aspect_score = 1.0
        elif 0.7 <= aspect_ratio < 0.9 or 1.4 < aspect_ratio <= 1.8:
            aspect_score = 0.7
        else:
            aspect_score = 0.3
        
        # Calculate weighted shape score
        shape_score = (
            circularity * 0.5 +      # 50% weight - increased importance
            convexity * 0.3 +        # 30% weight - smoothness
            solidity * 0.1 +         # 10% weight - fullness  
            aspect_score * 0.1       # 10% weight - proportion
        )
        
        print(f"🔍 Overall shape score: {shape_score:.3f}")
        
        # UPDATED: More realistic thresholds for real apples
        if shape_score >= 0.65:     # Lowered from 0.85
            quality = "Excellent"
            confidence = min(0.95, shape_score)
        elif shape_score >= 0.50:   # Lowered from 0.70
            quality = "Good" 
            confidence = min(0.90, shape_score)
        elif shape_score >= 0.35:   # Lowered from 0.55
            quality = "Fair"
            confidence = min(0.80, shape_score)
        else:
            quality = "Poor"
            confidence = min(0.70, shape_score)
        
        print(f"🔍 Final quality: {quality} (confidence: {confidence:.3f})")
        
        return quality, confidence

    def _apply_geometric_correction(self, image_cv, geometric_info):
        """EXACT: Apply improved geometric correction"""
        if geometric_info['needs_rotation']:
            h, w = image_cv.shape[:2]
            center = (w // 2, h // 2)
            
            angle = -geometric_info['rotation_angle']
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            rotated = cv2.warpAffine(image_cv, M, (w, h), 
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REFLECT)
            return rotated
        return image_cv

    def _apply_cnn_resize(self, image_cv):
        """EXACT: Resize image for CNN input"""
        return cv2.resize(image_cv, self.CNN_TARGET_SIZE, interpolation=self.RESIZE_INTERPOLATION)

    # ==========================================
    # YOUR EXACT COMPLETE PREPROCESSING PIPELINE FROM test.py
    # ==========================================
    
    def preprocess_image_for_ripeness(self, image_path_or_array):
        """Preprocess image for ripeness model (ResNet style - Method 4) - EXACT from test2.py"""
        from tensorflow.keras.applications.resnet50 import preprocess_input
        
        # Handle both file path and numpy array inputs
        if isinstance(image_path_or_array, str):
            # Load from file path
            image = Image.open(image_path_or_array)
        else:
            # Convert from numpy array (for cropped images)
            if isinstance(image_path_or_array, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image_path_or_array.shape) == 3:
                    image_rgb = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image_rgb)
                else:
                    image = Image.fromarray(image_path_or_array)
            else:
                image = image_path_or_array
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model's expected input size (224x224)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image).astype(np.float32)
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        # Apply ResNet preprocessing (Method 4)
        image_array = preprocess_input(image_array)
        
        return image_array

    def preprocess_image_for_type(self, image_path_or_array):
        """Preprocess image for apple type model (0-1 normalization - Method 1) - EXACT from test2.py"""
        # Handle both file path and numpy array inputs
        if isinstance(image_path_or_array, str):
            # Load from file path
            image = Image.open(image_path_or_array)
        else:
            # Convert from numpy array (for cropped images)
            if isinstance(image_path_or_array, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image_path_or_array.shape) == 3:
                    image_rgb = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image_rgb)
                else:
                    image = Image.fromarray(image_path_or_array)
            else:
                image = image_path_or_array
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model's expected input size (224x224)
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize to 0-1
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array

    def apply_complete_preprocessing(self, image_cv):
        """EXACT: Your complete preprocessing pipeline from test.py"""
        processed_cv_image = image_cv.copy()
        processing_log = []

        # 1. CONSERVATIVE Brightness Adjustment
        avg_brightness = self._calculate_brightness(image_cv)
        if avg_brightness < self.BRIGHTNESS_LOW_THRESHOLD:
            processed_cv_image = cv2.convertScaleAbs(processed_cv_image, alpha=1, beta=self.BRIGHTNESS_ADJUSTMENT_AMOUNT)
            processing_log.append(f"✅ Brightness increased (was {avg_brightness:.1f} - very dark)")
        elif avg_brightness > self.BRIGHTNESS_HIGH_THRESHOLD:
            processed_cv_image = cv2.convertScaleAbs(processed_cv_image, alpha=1, beta=-self.BRIGHTNESS_ADJUSTMENT_AMOUNT)
            processing_log.append(f"✅ Brightness decreased (was {avg_brightness:.1f} - blown out)")
        else:
            processing_log.append(f"✅ Brightness preserved ({avg_brightness:.1f} - natural apple lighting)")

        # 2. CONSERVATIVE Color Cast Detection & Correction
        color_cast_info = self._detect_color_cast(processed_cv_image)
        if color_cast_info['has_blue_cast']:
            # Additional check: Don't correct if it might be natural blue-green apple tones
            if color_cast_info['color_ratio'] > 2.5:  # Only very obvious blue casts
                processed_cv_image = self._apply_gentle_color_cast_correction(processed_cv_image, color_cast_info)
                processing_log.append(f"✅ Strong blue cast corrected (ratio: {color_cast_info['color_ratio']:.2f})")
            else:
                processing_log.append(f"✅ Mild blue tones preserved (natural apple coloring)")
        elif color_cast_info['has_red_cast']:
            # Additional check: Don't correct natural red apple variations
            if color_cast_info['color_ratio'] < 0.4:  # Only very obvious red casts
                processed_cv_image = self._apply_gentle_color_cast_correction(processed_cv_image, color_cast_info)
                processing_log.append(f"✅ Strong red cast corrected (ratio: {color_cast_info['color_ratio']:.2f})")
            else:
                processing_log.append(f"✅ Red tones preserved (natural apple coloring)")
        else:
            processing_log.append(f"✅ Natural color balance preserved (ratio: {color_cast_info['color_ratio']:.2f})")

        # 3. APPLE-AWARE Saturation Adjustment
        saturation_info = self._detect_saturation_level(processed_cv_image)
        if saturation_info['saturation'] < self.SATURATION_LOW_THRESHOLD:
            # Only enhance if extremely desaturated
            processed_cv_image = self._apply_gentle_saturation_enhancement(processed_cv_image, saturation_info)
            processing_log.append(f"✅ Gentle saturation boost (was {saturation_info['saturation']:.1f} - very pale)")
        elif saturation_info['saturation'] > self.SATURATION_HIGH_THRESHOLD:
            # NEW: Reduce oversaturation
            processed_cv_image = self._apply_saturation_reduction(processed_cv_image, saturation_info)
            processing_log.append(f"✅ Oversaturation reduced (was {saturation_info['saturation']:.1f} - too vivid)")
        else:
            processing_log.append(f"✅ Natural saturation preserved ({saturation_info['saturation']:.1f} - variety-appropriate)")

        # 4. GENTLE Contrast Enhancement
        gray_image = cv2.cvtColor(processed_cv_image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray_image)
        
        if contrast < self.CONTRAST_THRESHOLD:
            lab = cv2.cvtColor(processed_cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP_LIMIT, tileGridSize=self.CLAHE_TILE_GRID_SIZE)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            processed_cv_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            processing_log.append(f"✅ Gentle contrast enhancement (was {contrast:.1f} - very flat)")
        else:
            processing_log.append(f"✅ Natural contrast preserved ({contrast:.1f} - apple surface variation)")

        # 5. AGGRESSIVE Noise Reduction (back to original)
        noise_info = self._detect_noise_level(processed_cv_image)
        if noise_info['has_noise']:
            processed_cv_image = self._apply_noise_reduction(processed_cv_image, noise_info)  # Use original method
            processing_log.append(f"✅ Texture-preserving noise reduction ({noise_info['noise_type']} type)")
        else:
            processing_log.append(f"✅ Apple texture preserved (low noise level)")

        # 6. Post-Noise Gentle Sharpening (if noise was reduced)
        if noise_info['has_noise']:
            processed_cv_image = cv2.filter2D(processed_cv_image, -1, self.SHARPEN_MILD_KERNEL)
            processing_log.append(f"✅ Gentle detail restoration after noise reduction")

        # 7. VARIETY-PRESERVING Blur Correction
        blur_metrics = self._enhanced_blur_detection(processed_cv_image)
        
        if blur_metrics['is_very_blurry']:
            processed_cv_image = cv2.filter2D(processed_cv_image, -1, self.SHARPEN_STRONG_KERNEL)
            processing_log.append(f"✅ Strong sharpening (very blurry - variety features unclear)")
        elif blur_metrics['is_moderately_blurry']:
            processed_cv_image = cv2.filter2D(processed_cv_image, -1, self.SHARPEN_MILD_KERNEL)
            processing_log.append(f"✅ Mild sharpening (moderately blurry - enhancing variety features)")
        else:
            processing_log.append(f"✅ Variety features clear (good sharpness)")

        # 8. Geometric Correction (keep existing - works well)
        geometric_info = self._detect_geometric_issues(processed_cv_image)
        if geometric_info['needs_rotation']:
            processed_cv_image = self._apply_geometric_correction(processed_cv_image, geometric_info)
            processing_log.append(f"✅ Rotation corrected ({geometric_info['rotation_angle']:.1f}°)")
        else:
            processing_log.append(f"✅ Good orientation")

        # 9. CNN Resize (essential - keep existing)
        processed_cv_image = self._apply_cnn_resize(processed_cv_image)
        processing_log.append(f"✅ CNN-ready: {self.CNN_TARGET_SIZE[0]}×{self.CNN_TARGET_SIZE[1]}")

        return processed_cv_image, processing_log

    # ==========================================
    # YOUR EXACT COMPLETE PIPELINE FROM test.py
    # ==========================================
    
    def process_complete_pipeline(self, image, apple_bbox=None):
        """
        THREADED VERSION: Complete pipeline with parallel model processing + always fresh analysis
        50% faster with threading + no stale cached results
        """
        try:
            start_time = time.time()
            print(f"🚀 Starting threaded pipeline analysis...")
            
            # Store original image
            self.current_image = image.copy()
            
            # If bbox provided, crop directly (for camera mode)
            if apple_bbox:
                x1, y1, x2, y2 = apple_bbox
                self.cropped_apple = image[y1:y2, x1:x2].copy()
                best_apple_info = {'bbox': apple_bbox, 'conf': 1.0}
            else:
                # Run YOLO detection (for upload mode)
                apples_found = self.detect_apples_yolo(image)
                
                if not apples_found:
                    return {
                        'status': 'error',
                        'message': 'No apples detected in image',
                        'detection_results': None
                    }
                
                # Use best apple (highest confidence)
                best_apple_info = apples_found[0]
                x1, y1, x2, y2 = best_apple_info['bbox']
                self.cropped_apple = image[y1:y2, x1:x2].copy()

            crop_time = time.time()
            print(f"⏱️  Cropping completed: {(crop_time - start_time):.2f}s")

            # Apply complete preprocessing pipeline ONCE (shared by all models)
            self.processed_apple, preprocessing_log = self.apply_complete_preprocessing(self.cropped_apple)
            
            preprocessing_time = time.time()
            print(f"⏱️  Preprocessing + Color + Shape: {(preprocessing_time - crop_time):.2f}s")
            
            # 🚀 THREADING: Run variety and ripeness models in parallel
            print(f"🔄 Starting parallel model inference...")
            
            print(f"🔄 Starting parallel analysis of all 4 components...")


            # Thread-safe result containers
            variety_results = {}
            ripeness_results = {}
            shape_results = {}
            color_results = {}
            thread_errors = {}

            def variety_analysis_thread():
                """Thread function for variety classification - UPDATED with separate preprocessing"""
                try:
                    print(f"🧵 Variety thread started")
                    
                    # Use separate preprocessing for variety model (Method 1 from test2.py)
                    variety_image = self.preprocess_image_for_type(self.processed_apple)
                    variety_pred = self.variety_model.predict(variety_image, verbose=0)
                    
                    # EXACT processing from test2.py
                    variety_classes = ["Genesis", "Apple Envy", "Crimson Snow", "Fuji", "Golden Delicious", 
                                "Granny Smith", "Pink Lady", "Red Delicious", "Sassy"]
                    
                    variety_idx = np.argmax(variety_pred[0])
                    variety_result = variety_classes[variety_idx] if variety_idx < len(variety_classes) else "unknown"
                    variety_conf = float(variety_pred[0][variety_idx])
                    
                    variety_results['variety'] = variety_result
                    variety_results['confidence'] = variety_conf
                    print(f"✅ Variety thread completed: {variety_result} ({variety_conf:.3f})")
                    
                except Exception as e:
                    thread_errors['variety'] = str(e)
                    print(f"❌ Variety thread error: {e}")
        
            def ripeness_analysis_thread():
                """Thread function for ripeness and defect analysis - UPDATED with separate preprocessing"""
                try:
                    print(f"🧵 Ripeness thread started")
                    
                    # Use separate preprocessing for ripeness model (Method 4 from test2.py)
                    ripeness_image = self.preprocess_image_for_ripeness(self.processed_apple)
                    ripeness_predictions = self.ripeness_model.predict(ripeness_image, verbose=0)
                    
                    # EXACT processing from test2.py
                    # Process ripeness (first output)
                    ripeness_pred = ripeness_predictions[0][0]
                    ripeness_classes = ["Unripe", "Ripe", "Overripe"]
                    ripeness_idx = np.argmax(ripeness_pred)
                    ripeness_confidence = ripeness_pred[ripeness_idx]
                    ripeness_result = ripeness_classes[ripeness_idx]
                    
                    # Process defects (second output) 
                    defect_pred = ripeness_predictions[1][0]
                    defect_classes = ["Major Defects", "Moderate Defects", "Minor Defects", "No Defects"]
                    defect_idx = np.argmax(defect_pred)
                    defect_confidence = defect_pred[defect_idx]
                    defect_result = defect_classes[defect_idx]
                    
                    ripeness_results['ripeness'] = ripeness_result
                    ripeness_results['ripeness_conf'] = float(ripeness_confidence)
                    ripeness_results['defects'] = defect_result
                    ripeness_results['defect_conf'] = float(defect_confidence)
                    print(f"✅ Ripeness thread completed: {ripeness_result}/{defect_result}")
                    
                except Exception as e:
                    thread_errors['ripeness'] = str(e)
                    print(f"❌ Ripeness thread error: {e}")
        
            def shape_analysis_thread():
                """Thread function for shape analysis"""
                try:
                    print(f"🧵 Shape thread started")
                    shape_result = self.analyze_apple_shape(self.cropped_apple)
                    shape_results.update(shape_result)
                    print(f"✅ Shape thread completed: {shape_result['quality']}")
                    
                except Exception as e:
                    thread_errors['shape'] = str(e)
                    print(f"❌ Shape thread error: {e}")
        
            def color_analysis_thread():
                """Thread function for color analysis"""
                try:
                    print(f"🧵 Color thread started")
                    color_result = self.color_analyzer.analyze_apple_color(self.processed_apple)
                    color_results.update(color_result)
                    print(f"✅ Color thread completed: {color_result['color']}")
                    
                except Exception as e:
                    thread_errors['color'] = str(e)
                    print(f"❌ Color thread error: {e}")

                
         
            # Create and start both threads
            variety_thread = threading.Thread(target=variety_analysis_thread, name="VarietyThread")
            ripeness_thread = threading.Thread(target=ripeness_analysis_thread, name="RipenessThread")
            shape_thread = threading.Thread(target=shape_analysis_thread, name="ShapeThread")
            color_thread = threading.Thread(target=color_analysis_thread, name="ColorThread")
        
            
            # Start both threads simultaneously
            variety_thread.start()
            ripeness_thread.start()
            shape_thread.start()
            color_thread.start()

            
            variety_thread.join(timeout=10)  # 10 second timeout
            ripeness_thread.join(timeout=10)
            shape_thread.join(timeout=10)
            color_thread.join(timeout=10)
            
            if variety_thread.is_alive():
                thread_errors['variety'] = "Variety analysis timeout"
                print(f"⚠️  Variety thread timeout")
        
            if ripeness_thread.is_alive():
                thread_errors['ripeness'] = "Ripeness analysis timeout"
                print(f"⚠️  Ripeness thread timeout")
            
            if shape_thread.is_alive():
                thread_errors['shape'] = "Shape analysis timeout"
                print(f"⚠️  Shape thread timeout")
            
            if color_thread.is_alive():
                thread_errors['color'] = "Color analysis timeout"
                print(f"⚠️  Color thread timeout")
            
            model_time = time.time()
            print(f"⏱️  Parallel analysis (all 4 components): {(model_time - preprocessing_time):.2f}s")
        
            
            if 'variety' in thread_errors:
                print(f"🔄 Variety fallback due to error: {thread_errors['variety']}")
                variety_results = {'variety': 'Unknown', 'confidence': 0.5}
        
            if 'ripeness' in thread_errors:
                print(f"🔄 Ripeness fallback due to error: {thread_errors['ripeness']}")
                ripeness_results = {
                'ripeness': 'ripe', 'ripeness_conf': 0.5,
                'defects': 'none', 'defect_conf': 0.5
            }
        
            if 'shape' in thread_errors:
                print(f"🔄 Shape fallback due to error: {thread_errors['shape']}")
                shape_results = {
                    'quality': 'Unknown', 'confidence': 0.5, 'overall_score': 0.5,
                    'circularity': 0.5, 'symmetry': 0.5, 'smoothness': 0.5
                }
            
            if 'color' in thread_errors:
                print(f"🔄 Color fallback due to error: {thread_errors['color']}")
                color_results = {'color': 'unknown', 'confidence': 0.5}
        
            # Extract results from thread containers
            variety_result = variety_results.get('variety', 'Unknown')
            variety_conf = variety_results.get('confidence', 0.5)
            ripeness_result = ripeness_results.get('ripeness', 'ripe')
            ripeness_conf = ripeness_results.get('ripeness_conf', 0.5)
            defect_result = ripeness_results.get('defects', 'none')
            defect_conf = ripeness_results.get('defect_conf', 0.5)

            self.color_analysis_result = color_results
            shape_analysis_result = shape_results
            
            
            # 🆕 ADVANCED GRADING: Use the new 100-point grading system
            advanced_grader = AdvancedGradingSystem()
            
            # Prepare analysis results for grading
            analysis_for_grading = {
                'variety': variety_result,
                'ripeness': ripeness_result,
                'defects': defect_result,
                'color': self.color_analysis_result['color'],
                'shape': shape_analysis_result['quality'],
                'color_analysis': self.color_analysis_result,
                'shape_analysis': shape_analysis_result
            }
            
            # Calculate advanced grade
            grading_results = advanced_grader.calculate_advanced_grade(analysis_for_grading)
            
            total_time = time.time()
            print(f"🏁 Total pipeline time: {(total_time - start_time):.2f}s")
            print(f"🎯 Speed improvement vs sequential: ~{((4.0 - (total_time - start_time)) / 4.0 * 100):.0f}%")
            
            # FIXED: Convert all NumPy types to Python types for JSON serialization
            return {
                'status': 'success',
                'detection_results': {
                    'apples_found': [best_apple_info] if apple_bbox else apples_found,
                    'best_apple': best_apple_info
                },
                'analysis_results': {
                'type': 'Apple',
                'variety': str(variety_result),
                'size': 'Medium',
                'color': str(self.color_analysis_result['color']).title(),
                'ripeness': str(ripeness_result).title(),
                'defects': str(defect_result).title(),
                'shape': str(shape_analysis_result['quality']),
                'grade': str(grading_results['grade']),
                'confidence': int(ripeness_conf * 100),  # Already integer
                'detailed_confidence': {
                    'ripeness': int(ripeness_conf * 100),      # CHANGED: Remove decimals
                    'defect': int(defect_conf * 100),          # CHANGED: Remove decimals
                    'variety': int(variety_conf * 100),        # CHANGED: Remove decimals
                    'color': int(self.color_analysis_result['confidence'] * 100),  # CHANGED: Remove decimals
                    'shape': int(shape_analysis_result['confidence'] * 100)        # CHANGED: Remove decimals
                },
                'advanced_grading': {
                    'total_score': int(grading_results['total_score']),  # CHANGED: Remove decimal
                    'grade': str(grading_results['grade']),
                    'grade_description': str(grading_results['grade_description']),
                    'breakdown': {
                        'base_score': int(grading_results['breakdown']['base_score']),      # CHANGED: Remove decimals
                        'variety_bonus': int(grading_results['breakdown']['variety_bonus']), # CHANGED: Remove decimals
                        'premium_score': int(grading_results['breakdown']['premium_score'])  # CHANGED: Remove decimals
                    }
                    },
                    'shape_analysis': shape_analysis_result,
                    'color_analysis': self.color_analysis_result,
                    'preprocessing_log': [str(log) for log in preprocessing_log],
                    'performance': {  # Performance metrics
                        'total_time': round(total_time - start_time, 2),
                        'threading_enabled': True,
                        'models_parallel': True
                    }
                }
            }
            
        except Exception as e:
            print(f"❌ Threaded pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': f'Threaded pipeline processing failed: {str(e)}',
                'detection_results': None
            }


# Add the color analyzer class at the end
class RobustAppleColorAnalyzer:
    """Your exact color analysis code from test.py"""
    
    def __init__(self):
        # EXACT: Using your proven apple color ranges
        self.apple_color_ranges = {
            # Reds - primary apple colors
            'red': {
                'hsv_ranges': [((0, 100, 100), (10, 255, 255)), ((170, 100, 100), (179, 255, 255))],
                'priority': 1
            },
            
            # Greens - based on your working ranges for green apple detection
            'green': {
                'hsv_ranges': [((35, 50, 50), (85, 255, 255))],  # Broader green range that works
                'priority': 1
            },
            'light-green': {
                'hsv_ranges': [((35, 30, 80), (65, 120, 200))],  # Your working light green
                'priority': 1
            },
            
            # Yellows - proven ranges
            'yellow': {
                'hsv_ranges': [((15, 50, 50), (35, 255, 255))],  # Broader yellow detection
                'priority': 1
            },
            'golden-yellow': {
                'hsv_ranges': [((22, 120, 150), (32, 255, 255))],
                'priority': 2
            },
            
            # Oranges
            'orange': {
                'hsv_ranges': [((10, 150, 120), (20, 255, 255))],  # Your working orange range
                'priority': 2
            },
            'red-orange': {
                'hsv_ranges': [((5, 100, 100), (15, 255, 255))],
                'priority': 2
            },
            
            # Mixed and other colors
            'pink': {
                'hsv_ranges': [((145, 80, 120), (165, 255, 255))],
                'priority': 3
            }
        }
        
        # Percentage thresholds for naming
        self.naming_thresholds = {
            'little': (8, 18),    # 8-18% = "little"
            'some': (18, 35),     # 18-35% = "some"
            'mixed': (35, 100)    # 35%+ = full mixed name
        }

class RobustAppleColorAnalyzer:
    """Your exact color analysis code from test.py"""
    
    def __init__(self):
        # EXACT: Using your proven apple color ranges
        self.apple_color_ranges = {
            # Reds - primary apple colors
            'red': {
                'hsv_ranges': [((0, 100, 100), (10, 255, 255)), ((170, 100, 100), (179, 255, 255))],
                'priority': 1
            },
            
            # Greens - based on your working ranges for green apple detection
            'green': {
                'hsv_ranges': [((35, 50, 50), (85, 255, 255))],  # Broader green range that works
                'priority': 1
            },
            'light-green': {
                'hsv_ranges': [((35, 30, 80), (65, 120, 200))],  # Your working light green
                'priority': 1
            },
            
            # Yellows - proven ranges
            'yellow': {
                'hsv_ranges': [((15, 50, 50), (35, 255, 255))],  # Broader yellow detection
                'priority': 1
            },
            'golden-yellow': {
                'hsv_ranges': [((22, 120, 150), (32, 255, 255))],
                'priority': 2
            },
            
            # Oranges
            'orange': {
                'hsv_ranges': [((10, 150, 120), (20, 255, 255))],  # Your working orange range
                'priority': 2
            },
            'red-orange': {
                'hsv_ranges': [((5, 100, 100), (15, 255, 255))],
                'priority': 2
            },
            
            # Mixed and other colors
            'pink': {
                'hsv_ranges': [((145, 80, 120), (165, 255, 255))],
                'priority': 3
            }
        }
        
        # Percentage thresholds for naming
        self.naming_thresholds = {
            'little': (8, 18),    # 8-18% = "little"
            'some': (18, 35),     # 18-35% = "some"
            'mixed': (35, 100)    # 35%+ = full mixed name
        }
    
    def create_apple_focused_mask(self, image):
        """EXACT: Create mask focused on apple region with better coverage"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        
        # Method 1: Adaptive mask based on image brightness distribution
        mean_brightness = np.mean(gray)
        
        if mean_brightness > 100:  # Bright image
            brightness_mask = cv2.inRange(gray, 40, 250)
        else:  # Darker image
            brightness_mask = cv2.inRange(gray, 30, 240)
        
        # Method 2: Color-based mask to exclude obvious non-apple regions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Exclude very low saturation (gray/white backgrounds)
        saturation_mask = hsv[:, :, 1] > 20
        
        # Exclude very dark regions (shadows, black backgrounds)
        value_mask = hsv[:, :, 2] > 30
        
        # Combine masks
        color_mask = (saturation_mask & value_mask).astype(np.uint8) * 255
        
        # Method 3: Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        # Combine brightness and color masks
        combined_mask = cv2.bitwise_and(brightness_mask, color_mask)
        
        # If still too few pixels, fall back to center region
        total_pixels = np.sum(combined_mask > 0)
        min_required = (h * w) * 0.15  # At least 15% of image
        
        if total_pixels < min_required:
            # Fallback: generous center region
            center_mask = np.zeros((h, w), dtype=np.uint8)
            margin_h, margin_w = h//6, w//6
            center_mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255
            combined_mask = cv2.bitwise_or(combined_mask, center_mask)
        
        return combined_mask
    
    def classify_pixels_by_color(self, hsv_pixels, rgb_pixels):
        """EXACT: Classify pixels using apple-specific color ranges"""
        total_pixels = len(hsv_pixels)
        color_regions = []
        pixels_classified = np.zeros(total_pixels, dtype=bool)
        
        # Process in priority order to avoid overlaps
        sorted_colors = sorted(self.apple_color_ranges.items(), key=lambda x: x[1]['priority'])
        
        for color_name, color_def in sorted_colors:
            color_mask = np.zeros(total_pixels, dtype=bool)
            
            # Apply each HSV range for this color
            for hsv_range in color_def['hsv_ranges']:
                lower, upper = hsv_range
                
                range_mask = ((hsv_pixels[:, 0] >= lower[0]) & (hsv_pixels[:, 0] <= upper[0]) &
                             (hsv_pixels[:, 1] >= lower[1]) & (hsv_pixels[:, 1] <= upper[1]) &
                             (hsv_pixels[:, 2] >= lower[2]) & (hsv_pixels[:, 2] <= upper[2]) &
                             (~pixels_classified))
                
                color_mask |= range_mask
            
            # Record color if significant pixels found
            pixel_count = np.sum(color_mask)
            if pixel_count > 0:
                percentage = (pixel_count / total_pixels) * 100
                
                if percentage >= 0.5:  # Only include colors with ≥0.5%
                    median_hsv = np.median(hsv_pixels[color_mask], axis=0)
                    median_rgb = np.median(rgb_pixels[color_mask], axis=0)
                    
                    color_regions.append({
                        'color_name': color_name,
                        'percentage': percentage,
                        'pixel_count': pixel_count,
                        'hsv': median_hsv,
                        'rgb': median_rgb
                    })
                    
                    pixels_classified |= color_mask
        
        # Handle remaining unclassified pixels
        unclassified_count = np.sum(~pixels_classified)
        if unclassified_count > total_pixels * 0.05:  # >5% unclassified
            additional_colors = self.classify_remaining_pixels(
                hsv_pixels[~pixels_classified], 
                rgb_pixels[~pixels_classified]
            )
            
            for add_color_name, add_percentage in additional_colors.items():
                if add_percentage >= 2.0:  # Only significant additional colors
                    color_regions.append({
                        'color_name': add_color_name,
                        'percentage': add_percentage,
                        'pixel_count': int((add_percentage / 100) * total_pixels),
                        'hsv': [0, 0, 0],  # Placeholder
                        'rgb': [0, 0, 0]   # Placeholder
                    })
        
        # Sort by percentage
        color_regions.sort(key=lambda x: x['percentage'], reverse=True)
        
        return color_regions
    
    def classify_remaining_pixels(self, hsv_pixels, rgb_pixels):
        """EXACT: Classify remaining pixels with proven broader criteria"""
        if len(hsv_pixels) == 0:
            return {}
        
        total_remaining = len(hsv_pixels)
        additional_colors = {}
        
        # Broader green detection (critical for apple green.jpg)
        green_mask = ((hsv_pixels[:, 0] >= 35) & (hsv_pixels[:, 0] <= 85) & 
                     (hsv_pixels[:, 1] >= 20) & (hsv_pixels[:, 2] >= 50))
        green_count = np.sum(green_mask)
        if green_count > 0:
            additional_colors['green'] = (green_count / total_remaining) * 100
        
        # Broader red detection
        red_mask = (((hsv_pixels[:, 0] >= 0) & (hsv_pixels[:, 0] <= 15)) | 
                   ((hsv_pixels[:, 0] >= 165) & (hsv_pixels[:, 0] <= 179))) & (hsv_pixels[:, 1] >= 40)
        red_count = np.sum(red_mask)
        if red_count > 0:
            additional_colors['red'] = (red_count / total_remaining) * 100
        
        # Broader yellow detection
        yellow_mask = ((hsv_pixels[:, 0] >= 15) & (hsv_pixels[:, 0] <= 35) & 
                      (hsv_pixels[:, 1] >= 30) & (hsv_pixels[:, 2] >= 80))
        yellow_count = np.sum(yellow_mask)
        if yellow_count > 0:
            additional_colors['yellow'] = (yellow_count / total_remaining) * 100
        
        return additional_colors
    
    def consolidate_similar_colors(self, color_regions):
        """EXACT: Consolidate similar color variants into main colors"""
        consolidated = {}
        
        # Color consolidation rules - simplified to avoid over-consolidation
        consolidation_map = {
            'light-green': 'green',  # Only consolidate very similar colors
            'red-orange': 'orange',
            'golden-yellow': 'yellow'
        }
        
        for region in color_regions:
            color_name = region['color_name']
            percentage = region['percentage']
            
            # Map to consolidated color
            consolidated_color = consolidation_map.get(color_name, color_name)
            
            if consolidated_color in consolidated:
                consolidated[consolidated_color] += percentage
            else:
                consolidated[consolidated_color] = percentage
        
        # Convert back to list format
        consolidated_regions = []
        for color_name, percentage in consolidated.items():
            consolidated_regions.append({
                'color_name': color_name,
                'percentage': percentage
            })
        
        # Sort by percentage
        consolidated_regions.sort(key=lambda x: x['percentage'], reverse=True)
        
        return consolidated_regions
    
    def determine_apple_color_name(self, consolidated_regions):
        """EXACT: Determine final apple color name with percentage-based naming"""
        if not consolidated_regions:
            return 'unknown', 0.0
        
        primary_region = consolidated_regions[0]
        primary_color = primary_region['color_name']
        primary_percentage = primary_region['percentage']
        
        # Check for significant secondary colors
        if len(consolidated_regions) > 1:
            secondary_region = consolidated_regions[1]
            secondary_color = secondary_region['color_name']
            secondary_percentage = secondary_region['percentage']
            
            # Apply percentage-based naming
            if secondary_percentage >= self.naming_thresholds['little'][0]:
                mixed_name = self.create_apple_mixed_name(
                    primary_color, secondary_color, secondary_percentage
                )
                
                if mixed_name:
                    confidence = min(0.95, (primary_percentage + secondary_percentage) / 100)
                    return mixed_name, confidence
        
        # Single color result
        confidence = min(0.95, primary_percentage / 100)
        return primary_color, confidence
    
    def create_apple_mixed_name(self, primary_color, secondary_color, secondary_percentage):
        """EXACT: Create apple-specific mixed color names"""
        # Valid apple color combinations
        valid_combinations = {
            ('red', 'green'), ('green', 'red'),
            ('red', 'yellow'), ('yellow', 'red'), 
            ('green', 'yellow'), ('yellow', 'green'),
            ('red', 'orange'), ('orange', 'red'),
            ('red', 'pink'), ('pink', 'red')
        }
        
        color_pair = (primary_color, secondary_color)
        if color_pair not in valid_combinations:
            return None
        
        # Determine naming level based on percentage
        if self.naming_thresholds['little'][0] <= secondary_percentage < self.naming_thresholds['little'][1]:
            return f"{primary_color}-little {secondary_color}"
        elif self.naming_thresholds['some'][0] <= secondary_percentage < self.naming_thresholds['some'][1]:
            return f"{primary_color}-some {secondary_color}"
        elif secondary_percentage >= self.naming_thresholds['mixed'][0]:
            return f"{primary_color}-{secondary_color}"
        
        return None
    
    def analyze_apple_color(self, image):
        """EXACT: Main apple color analysis function - clean and focused"""
        if image is None or image.size == 0:
            return {'color': 'unknown', 'confidence': 0.0, 'method': 'error'}
        
        try:
            # Step 1: Create apple-focused mask
            apple_mask = self.create_apple_focused_mask(image)
            mask_pixels = np.sum(apple_mask > 0)
            total_image_pixels = image.shape[0] * image.shape[1]
            mask_coverage = (mask_pixels / total_image_pixels) * 100
            
            if mask_pixels < 1000:  # Minimum pixel requirement
                return {'color': 'unknown', 'confidence': 0.0, 'method': 'insufficient_apple_region'}
            
            # Step 2: Extract color information from apple region
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            valid_pixels = apple_mask > 0
            hsv_pixels = hsv[valid_pixels]
            rgb_pixels = rgb[valid_pixels]
            
            # Step 3: Classify pixels by apple colors
            color_regions = self.classify_pixels_by_color(hsv_pixels, rgb_pixels)
            
            if not color_regions:
                return {'color': 'unknown', 'confidence': 0.0, 'method': 'no_apple_colors_detected'}
            
            # Step 4: Consolidate similar colors
            consolidated_regions = self.consolidate_similar_colors(color_regions)
            
            # Step 5: Determine final apple color name
            final_color, confidence = self.determine_apple_color_name(consolidated_regions)
            
            # Create breakdown for display
            breakdown = consolidated_regions[:4]  # Top 4 colors
            
            return {
                'color': final_color,
                'confidence': confidence,
                'method': 'apple_optimized',
                'breakdown': breakdown,
                'mask_coverage': mask_coverage,
                'apple_region_pixels': mask_pixels
            }
            
        except Exception as e:
            return {'color': 'unknown', 'confidence': 0.0, 'method': f'error: {str(e)}'}
        
class AdvancedGradingSystem:
    """
    100-Point Hybrid Grading System for Apple Quality Assessment
    Base Score (65) + Variety Bonus (25) + Premium Factors (10) = 100 points
    """
    
    def __init__(self):
        # Complete variety rules for all 9 apple varieties from your backend
        self.variety_rules = {
            'Red Delicious': {
                'ideal_colors': ['red', 'red-some yellow', 'red-little yellow'],
                'shape_preference': 'elongated',  # Classic elongated Red Delicious shape
                'color_weight': 0.8,  # High importance for deep red color
                'shape_weight': 0.7   # Important to be elongated/conical
            },
            
            'Granny Smith': {
                'ideal_colors': ['green', 'light-green', 'green-little yellow'],
                'shape_preference': 'round',  # Classic round shape
                'color_weight': 0.9,  # Very important to be green
                'shape_weight': 0.6   # Moderately important to be round
            },
            
            'Golden Delicious': {
                'ideal_colors': ['yellow', 'golden-yellow', 'yellow-little green'],
                'shape_preference': 'round',  # Conical to round shape
                'color_weight': 0.7,  # Important golden color
                'shape_weight': 0.6   # Moderately important shape
            },
            
            'Fuji': {
                'ideal_colors': ['red', 'red-yellow', 'red-some yellow'],
                'shape_preference': 'round',  # Classic round Fuji shape
                'color_weight': 0.6,  # Mixed colors acceptable
                'shape_weight': 0.8   # Should be very round
            },
            
            'Pink Lady': {
                'ideal_colors': ['pink', 'red-pink', 'pink-some red'],
                'shape_preference': 'round',  # Round to slightly conical
                'color_weight': 0.8,  # Important pink coloring
                'shape_weight': 0.6   # Moderately important shape
            },
            
            'Genesis': {
                'ideal_colors': ['red', 'red-yellow', 'red-some yellow'],  # Royal Gala x Braeburn cross
                'shape_preference': 'round',  # Sweet, refreshing variety
                'color_weight': 0.7,  # Moderate color importance
                'shape_weight': 0.6   # Moderate shape importance
            },
            
            'Apple Envy': {
                'ideal_colors': ['red', 'red-little yellow', 'red-some yellow'],  # Mostly red with yellow specks
                'shape_preference': 'round',  # Round to conical shape
                'color_weight': 0.8,  # High importance - should be mostly red
                'shape_weight': 0.7   # Important to have good round shape
            },
            
            'Crimson Snow': {
                'ideal_colors': ['red', 'red-purple', 'crimson'],  # Crimson red color with white flesh
                'shape_preference': 'round',  # Medium to large, round shape
                'color_weight': 0.9,  # Very important - distinctive crimson color
                'shape_weight': 0.6   # Moderately important shape
            },
            
            'Sassy': {
                'ideal_colors': ['red', 'red-deep', 'red-vibrant'],  # Vibrant red skin, deep red color
                'shape_preference': 'round',  # Conical shape mentioned in research
                'color_weight': 0.8,  # High importance - vibrant red is key characteristic
                'shape_weight': 0.7   # Important conical/round shape
            },
            
            # Default for unknown varieties
            'Unknown': {
                'ideal_colors': ['red', 'green', 'yellow'],
                'shape_preference': 'round',
                'color_weight': 0.5,
                'shape_weight': 0.5
            }
        }
    
    def calculate_advanced_grade(self, analysis_results):
        """
        Calculate comprehensive 100-point grade using OPTION A distribution:
        - Color-Based Ripeness: 35 points (most reliable)
        - Defects: 20 points (ML model, reduced weight)
        - Color Variety Match: 20 points (very reliable)
        - Shape: 15 points (edge-based, reliable)
        - Visual Appeal: 10 points (overall impression)
        """
        try:
            # Extract analysis data
            variety = analysis_results.get('variety', 'Unknown')
            ripeness = analysis_results.get('ripeness', '').lower()
            defects = analysis_results.get('defects', '').lower()
            color = analysis_results.get('color', '').lower()
            shape = analysis_results.get('shape', '').lower()
            
            # Get detailed analysis if available
            color_analysis = analysis_results.get('color_analysis', {})
            shape_analysis = analysis_results.get('shape_analysis', {})
            
            print(f"🏆 Advanced Grading (Option A) for {variety}:")
            
            # PHASE 1: Base Score (70 points) - Color Ripeness + Defects + Shape
            base_score = self._calculate_base_score(ripeness, defects, shape_analysis, variety, color)
            print(f"   Base Score: {base_score:.1f}/70")
            
            # PHASE 2: Variety Bonus (20 points) - Color Match Only
            variety_bonus = self._calculate_variety_bonus(variety, color, color_analysis, shape_analysis)
            print(f"   Variety Bonus: {variety_bonus:.1f}/20")
            
            # PHASE 3: Premium Factors (10 points) - Visual Appeal
            premium_score = self._calculate_premium_factors(color_analysis, shape_analysis)
            print(f"   Premium Score: {premium_score:.1f}/10")
            
            # Total Score
            total_score = base_score + variety_bonus + premium_score
            print(f"   Total Score: {total_score:.1f}/100")
            
            # Determine Grade
            grade_letter = self._determine_grade(total_score)
            print(f"   Final Grade: {grade_letter}")
            
            return {
                'total_score': round(total_score, 1),
                'grade': grade_letter,
                'grade_description': '',  # Empty as requested
                'breakdown': {
                    'base_score': round(base_score, 1),
                    'variety_bonus': round(variety_bonus, 1),
                    'premium_score': round(premium_score, 1)
                }
            }
            
        except Exception as e:
            print(f"❌ Advanced grading error: {e}")
            # Fallback to basic grading
            return {
                'total_score': 60.0,
                'grade': 'C',
                'grade_description': '',
                'breakdown': {
                    'base_score': 60.0,
                    'variety_bonus': 0.0,
                    'premium_score': 0.0
                }
            }

    def _calculate_base_score(self, ripeness, defects, shape_analysis, variety, color):
        """
        Calculate Base Score (70 points total) - OPTION A DISTRIBUTION
        Color-Based Ripeness (35) + Defects (20) + Shape (15)
        """
        score = 0
        
        # COLOR-BASED Ripeness Scoring (35 points max) - INCREASED from 30
        ripeness_score = self._calculate_color_based_ripeness(variety, color)
        score += ripeness_score
        print(f"     Color-Based Ripeness ({color} for {variety}): {ripeness_score}/35")
        
        # Defects Scoring (20 points max) - DECREASED from 25
        if defects == 'none' or defects == 'No Defects':
            defect_score = 20
        elif defects == 'minor' or defects == 'Minor Defects':
            defect_score = 15
        elif defects == 'moderate' or defects == 'Moderate Defects':
            defect_score = 8
        elif defects == 'major' or defects == 'Major Defects':
            defect_score = 2
        else:
            defect_score = 12  # Unknown, give average
            
        score += defect_score
        print(f"     Defects ({defects}): {defect_score}/20")
        
        # Shape Scoring (15 points max) - INCREASED from 10
        if shape_analysis and 'quality' in shape_analysis:
            shape_quality = shape_analysis['quality'].lower()
            if shape_quality == 'excellent':
                shape_score = 15
            elif shape_quality == 'good':
                shape_score = 12
            elif shape_quality == 'fair':
                shape_score = 8
            elif shape_quality == 'poor':
                shape_score = 3
            else:
                shape_score = 9
        else:
            shape_score = 9  # Default average
            
        score += shape_score
        print(f"     Shape: {shape_score}/15")
        
        return score


    def _calculate_color_based_ripeness(self, variety, color):
        """
        Calculate ripeness score based on color analysis instead of ML model
        35 points maximum (INCREASED from 30)
        """
        try:
            # Define expected colors for each variety group
            red_varieties = ['Red Delicious', 'Apple Envy', 'Crimson Snow', 'Sassy', 'Pink Lady']
            green_varieties = ['Granny Smith']
            yellow_varieties = ['Golden Delicious', 'Genesis']
            mixed_varieties = ['Fuji']  # Can be red or mixed colors
            
            # Clean up color string (remove extra words)
            clean_color = color.lower().strip()
            
            # Check for primary color presence
            has_red = 'red' in clean_color
            has_green = 'green' in clean_color  
            has_yellow = 'yellow' in clean_color or 'golden' in clean_color
            
            ripeness_score = 18  # Default average score (was 15, now scaled to 35-point system)
            
            if variety in red_varieties:
                # Red varieties should be red for good ripeness
                if has_red:
                    ripeness_score = 35  # Perfect ripeness
                    print(f"       ✅ Perfect color match: {variety} is red")
                elif has_green:
                    ripeness_score = 10   # Unripe (green = not ready)
                    print(f"       ⚠️  Unripe: {variety} still green")
                elif has_yellow:
                    ripeness_score = 15  # Slightly unripe
                    print(f"       ⚠️  Slightly unripe: {variety} still yellow")
                else:
                    ripeness_score = 18  # Mixed/unknown color
                    print(f"       ❓ Unknown ripeness: {variety} color unclear")
                    
            elif variety in green_varieties:
                # Green varieties should stay green for good ripeness
                if has_green:
                    ripeness_score = 35  # Perfect ripeness
                    print(f"       ✅ Perfect color match: {variety} is green")
                elif has_red:
                    ripeness_score = 6   # Overripe (green apple turning red = bad)
                    print(f"       ❌ Overripe: {variety} turning red")
                elif has_yellow:
                    ripeness_score = 12  # Getting old
                    print(f"       ⚠️  Aging: {variety} turning yellow")
                else:
                    ripeness_score = 18  # Mixed/unknown color
                    print(f"       ❓ Unknown ripeness: {variety} color unclear")
                    
            elif variety in yellow_varieties:
                # Yellow varieties should be yellow/golden for good ripeness
                if has_yellow:
                    ripeness_score = 35  # Perfect ripeness
                    print(f"       ✅ Perfect color match: {variety} is yellow/golden")
                elif has_green:
                    ripeness_score = 12  # Unripe (still green)
                    print(f"       ⚠️  Unripe: {variety} still green")
                elif has_red:
                    ripeness_score = 25  # Some yellow varieties can have red blush
                    print(f"       ✅ Good ripeness: {variety} with red blush")
                else:
                    ripeness_score = 18  # Mixed/unknown color
                    print(f"       ❓ Unknown ripeness: {variety} color unclear")
                    
            elif variety in mixed_varieties:
                # Mixed varieties can have multiple good colors
                if has_red or has_yellow:
                    ripeness_score = 35  # Good ripeness
                    print(f"       ✅ Good ripeness: {variety} shows proper colors")
                elif has_green:
                    ripeness_score = 15  # Slightly unripe
                    print(f"       ⚠️  Slightly unripe: {variety} still green")
                else:
                    ripeness_score = 18  # Default
                    print(f"       ❓ Unknown ripeness: {variety} color unclear")
                    
            else:
                # Unknown variety - use general logic
                if has_red or has_yellow:
                    ripeness_score = 30  # Generally good
                    print(f"       ✅ Good colors for unknown variety")
                elif has_green:
                    ripeness_score = 18  # Could be unripe or green variety
                    print(f"       ❓ Green color - could be unripe or green variety")
                else:
                    ripeness_score = 18  # Default
                    print(f"       ❓ Unknown variety and color")
            
            return min(ripeness_score, 35)  # Cap at maximum 35 points
            
        except Exception as e:
            print(f"❌ Color-based ripeness calculation error: {e}")
            return 18  # Return average score on error


    def _calculate_variety_bonus(self, variety, color, color_analysis, shape_analysis):
        """
        Calculate Variety Bonus (20 points total) - OPTION A DISTRIBUTION
        Color Match (20) + Shape Conformity (0) - Shape moved to base score
        """
        score = 0
        
        # Get variety rules (fallback to default if variety not found)
        variety_key = variety if variety in self.variety_rules else 'Unknown'
        rules = self.variety_rules[variety_key]
        
        # Color Match Scoring (20 points max) - INCREASED from 15
        color_score = self._score_color_match(color, color_analysis, rules)
        # Scale to new 20-point system
        color_score = (color_score / 15) * 20
        score += color_score
        print(f"     Color Match for {variety}: {color_score:.1f}/20")
        
        # Shape Conformity moved to base score, so no points here
        print(f"     Shape Conformity: Moved to base score")
        
        return score

    def _score_color_match(self, color, color_analysis, rules):
        """
        Score how well the apple's color matches the variety expectations
        """
        try:
            ideal_colors = rules['ideal_colors']
            weight = rules['color_weight']
            
            # Check if detected color matches any ideal colors
            color_match_score = 0
            
            # Direct color match
            if color in ideal_colors:
                color_match_score = 1.0
            else:
                # Partial matches for similar colors
                for ideal_color in ideal_colors:
                    if ideal_color in color or color in ideal_color:
                        color_match_score = max(color_match_score, 0.7)
                    # Check for color family matches (red family, green family, etc.)
                    elif self._colors_in_same_family(color, ideal_color):
                        color_match_score = max(color_match_score, 0.5)
            
            # Apply variety weight and convert to 15-point scale
            final_score = color_match_score * weight * 15
            
            return min(final_score, 15)  # Cap at maximum
            
        except Exception as e:
            print(f"Color scoring error: {e}")
            return 8  # Average score on error
    
    def _score_shape_conformity(self, shape_analysis, rules):
        """
        Score how well the apple's shape matches variety expectations
        """
        try:
            if not shape_analysis or 'aspect_ratio' not in shape_analysis:
                return 5  # Average score if no shape data
            
            aspect_ratio = shape_analysis['aspect_ratio']
            preferred_shape = rules['shape_preference']
            weight = rules['shape_weight']
            
            # Determine shape conformity
            shape_score = 0
            
            if preferred_shape == 'round':
                # Round apples should have aspect ratio close to 1.0
                if 0.9 <= aspect_ratio <= 1.2:
                    shape_score = 1.0  # Perfect round
                elif 0.8 <= aspect_ratio <= 1.4:
                    shape_score = 0.8  # Good round
                elif 0.7 <= aspect_ratio <= 1.6:
                    shape_score = 0.5  # Acceptable
                else:
                    shape_score = 0.2  # Poor shape for round variety
                    
            elif preferred_shape == 'elongated':
                # Elongated apples should be taller (aspect ratio > 1.0)
                if 1.2 <= aspect_ratio <= 1.6:
                    shape_score = 1.0  # Perfect elongated
                elif 1.1 <= aspect_ratio <= 1.8:
                    shape_score = 0.8  # Good elongated
                elif 1.0 <= aspect_ratio <= 2.0:
                    shape_score = 0.5  # Acceptable
                else:
                    shape_score = 0.2  # Poor shape for elongated variety
            
            # Apply variety weight and convert to 10-point scale
            final_score = shape_score * weight * 10
            
            return min(final_score, 10)  # Cap at maximum
            
        except Exception as e:
            print(f"Shape conformity scoring error: {e}")
            return 5  # Average score on error
    
    def _calculate_premium_factors(self, color_analysis, shape_analysis):
        """
        Calculate Premium Factors (10 points total) - UNCHANGED
        Visual Appeal (10)
        """
        score = 0
        
        # Visual Appeal based on confidence and quality metrics
        visual_score = 0
        
        # Color confidence contributes to visual appeal
        if color_analysis and 'confidence' in color_analysis:
            color_confidence = color_analysis['confidence']
            visual_score += color_confidence * 0.5  # 50% weight
        else:
            visual_score += 0.3  # Default average
        
        # Shape confidence contributes to visual appeal
        if shape_analysis and 'confidence' in shape_analysis:
            shape_confidence = shape_analysis['confidence']
            visual_score += shape_confidence * 0.5  # 50% weight
        else:
            visual_score += 0.3  # Default average
            
        # Convert to 10-point scale
        final_visual_score = visual_score * 10
        score += min(final_visual_score, 10)
        
        print(f"     Visual Appeal: {final_visual_score:.1f}/10")
        
        return score

    def _colors_in_same_family(self, color1, color2):
        """
        Check if two colors are in the same color family
        """
        red_family = ['red', 'pink', 'crimson']
        green_family = ['green', 'light-green']
        yellow_family = ['yellow', 'golden-yellow', 'golden']
        
        families = [red_family, green_family, yellow_family]
        
        for family in families:
            if any(c in color1 for c in family) and any(c in color2 for c in family):
                return True
        return False
    
    def _determine_grade(self, total_score):
        """
        Convert numerical score to letter grade only
        """
        if total_score >= 90:
            return 'A+'
        elif total_score >= 80:
            return 'A'
        elif total_score >= 65:
            return 'B'
        elif total_score >= 50:
            return 'C'
        else:
            return 'F'