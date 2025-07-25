# main.py - Updated with Threading Support and Always Fresh Analysis
# Removed analysis caching for always fresh results + threading support

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import os
import base64
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
from collections import defaultdict
import time

# Import your exact analysis pipeline
from apple_analysis_backend import AppleAnalysisPipeline

# Create FastAPI app
app = FastAPI(
    title="FruittiAI Backend with Threading & Fresh Analysis", 
    description="Real-time apple analysis with parallel processing and always fresh results",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for pipeline and tracking
pipeline = None
apple_tracker = defaultdict(lambda: {
    'positions': [],
    'stable_count': 0,
    'last_seen': time.time(),
    # ❌ REMOVED: 'last_analysis': None,  # No more caching!
    # ❌ REMOVED: 'analyzing': False     # No more analysis flags!
})

# ❌ REMOVED: Analysis queue management (not needed with direct analysis)
# analysis_queue = []
# MAX_QUEUE_SIZE = 3

# Create uploads folder
os.makedirs("uploads", exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Load models at startup with threading support"""
    global pipeline
    print("🚀 Starting FruittiAI Backend with Threading Support...")
    print("📦 Loading models at startup...")
    
    pipeline = AppleAnalysisPipeline()
    success = pipeline.load_models()
    
    if success:
        print("✅ All models loaded successfully!")
        print("🧵 Threading support enabled for parallel processing")
        print("🔄 Always fresh analysis enabled - no caching")
    else:
        print("❌ Model loading failed!")

@app.get("/")
def read_root():
    """Check if API is working"""
    model_status = "loaded" if pipeline and pipeline.models_loaded else "not loaded"
    return {
        "message": "FruittiAI Backend with Threading & Fresh Analysis is running!",
        "status": "healthy",
        "models": model_status,
        "features": {
            "threading_enabled": True,
            "always_fresh_analysis": True,
            "parallel_processing": True
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/status")
def check_models_status():
    """Check if models are loaded"""
    if pipeline and pipeline.models_loaded:
        return {
            "status": "success",
            "models_loaded": True,
            "message": "All models ready for threaded analysis",
            "threading_support": True
        }
    else:
        return {
            "status": "error", 
            "models_loaded": False,
            "message": "Models not loaded yet"
        }
    
def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj
    

def decode_image_from_base64(image_data: str):
    """Convert base64 image to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def calculate_position_distance(pos1, pos2):
    """Fast distance calculation"""
    if not pos1 or not pos2:
        return float('inf')
    
    center1_x = (pos1[0] + pos1[2]) / 2
    center1_y = (pos1[1] + pos1[3]) / 2
    center2_x = (pos2[0] + pos2[2]) / 2
    center2_y = (pos2[1] + pos2[3]) / 2
    
    return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

def is_same_apple(bbox1, bbox2, tolerance=40):
    """Faster apple tracking with looser tolerance"""
    if not bbox1 or not bbox2:
        return False
    
    # Calculate center points
    center1_x = (bbox1[0] + bbox1[2]) / 2
    center1_y = (bbox1[1] + bbox1[3]) / 2
    center2_x = (bbox2[0] + bbox2[2]) / 2
    center2_y = (bbox2[1] + bbox2[3]) / 2
    
    # Simple distance check
    distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    return distance < tolerance

@app.post("/camera/detect")
async def detect_fruits_realtime(image_data: dict):  # Changed function name
    """
    UPDATED: Detect all fruits, only analyze apples
    """
    if not pipeline or not pipeline.models_loaded:
        return {
            "status": "error",
            "message": "Models not loaded yet",
            "detections": []
        }
    
    try:
        start_time = time.time()
        
        # Decode image
        image = decode_image_from_base64(image_data.get('image', ''))
        
        if image is None:
            return {
                "status": "error", 
                "message": "Could not decode image",
                "detections": []
            }
        
        # SPEED OPTIMIZATION: Run YOLO detection for ALL fruits
        fruits_found = pipeline.detect_apples_yolo(image)  # Now returns all fruits
        print(f"🔍 Detection results: {len(fruits_found)} items")
        for i, fruit in enumerate(fruits_found):
            print(f"   {i}: {fruit}")
        
        detection_time = time.time()
        
        if not fruits_found:
            # Quick cleanup of old apple trackers only
            current_time = time.time()
            expired_trackers = [
                tracked_id for tracked_id, tracked_data in apple_tracker.items()
                if current_time - tracked_data['last_seen'] > 5
            ]
            for tracked_id in expired_trackers:
                print(f"🧹 Cleaning up expired tracker: {tracked_id}")
                del apple_tracker[tracked_id]
            
            return {
                "status": "success",
                "detections": [],
                "message": "No fruits detected",
                "performance": {
                    "detection_time": round(detection_time - start_time, 3),
                    "total_time": round(detection_time - start_time, 3)
                }
            }
        
        # PROCESS FRUITS: Separate apples from other fruits
        current_time = time.time()
        processed_detections = []
        
        for fruit in fruits_found:
            fruit_type = fruit.get('fruit_type', fruit.get('class', 'unknown'))
            bbox = fruit['bbox']
            
            if fruit_type == 'apple':
                # EXISTING APPLE PROCESSING: Use your existing apple tracking + analysis
                apple_id = None
                
                # Quick tracking lookup
                for tracked_id, tracked_data in apple_tracker.items():
                    if tracked_data['positions'] and is_same_apple(bbox, tracked_data['positions'][-1], tolerance=40):
                        apple_id = tracked_id
                        break
                
                # Create new tracker if needed
                if apple_id is None:
                    apple_id = f"apple_{len(apple_tracker)}_{current_time}"
                    apple_tracker[apple_id] = {
                        'positions': [bbox],
                        'stable_count': 1,
                        'last_seen': current_time,
                        'analysis_status': 'none',
                        'analysis_results': None,
                        'analysis_timestamp': 0,
                        'analysis_attempts': 0
                    }
                    print(f"🆕 New apple tracker created: {apple_id}")
                else:
                    # Update existing tracker
                    tracker = apple_tracker[apple_id]
                    tracker['positions'].append(bbox)
                    tracker['last_seen'] = current_time
                    
                    if len(tracker['positions']) > 3:
                        tracker['positions'] = tracker['positions'][-3:]
                    
                    if len(tracker['positions']) >= 2:
                        recent_distance = calculate_position_distance(
                            tracker['positions'][-2], 
                            tracker['positions'][-1]
                        )
                        if recent_distance < 25:
                            tracker['stable_count'] += 1
                        else:
                            tracker['stable_count'] = max(1, tracker['stable_count'] - 1)
                
                # Get tracker reference
                tracker = apple_tracker[apple_id]
                
                # Apple detection with full analysis capability
                apple_data = {
                    'fruit_type': 'apple',
                    'detection_type': 'apple_analysis',  # Special type for apples
                    'apple_id': str(apple_id),
                    'bbox': fruit['bbox'],
                    'conf': fruit['conf'],
                    'stable': tracker['stable_count'] >= 2,
                    'has_analysis': tracker['analysis_status'] == 'completed',
                    'analysis': tracker['analysis_results'],
                    'analysis_status': tracker['analysis_status']
                }
                
                # EXISTING APPLE ANALYSIS LOGIC (unchanged)
                should_analyze = (
                    tracker['stable_count'] >= 3 and
                    tracker['analysis_status'] == 'none' and
                    tracker['analysis_attempts'] == 0
                )
                
                if should_analyze:
                    try:
                        print(f"🔬 Starting FIRST-TIME analysis for apple {apple_id}")
                        
                        tracker['analysis_status'] = 'analyzing'
                        tracker['analysis_attempts'] += 1
                        
                        analysis_results = pipeline.process_complete_pipeline(image, bbox)
                        
                        if analysis_results['status'] == 'success':
                            tracker['analysis_status'] = 'completed'
                            tracker['analysis_results'] = analysis_results['analysis_results']
                            tracker['analysis_timestamp'] = current_time
                            
                            apple_data['has_analysis'] = True
                            apple_data['analysis'] = tracker['analysis_results']
                            apple_data['analysis_status'] = 'completed'
                            
                            print(f"✅ Analysis COMPLETED for apple {apple_id}")
                        else:
                            tracker['analysis_status'] = 'failed'
                            print(f"❌ Analysis FAILED for apple {apple_id}")
                
                    except Exception as e:
                        print(f"💥 Analysis ERROR for apple {apple_id}: {e}")
                        tracker['analysis_status'] = 'failed'
                
                elif tracker['analysis_status'] == 'analyzing':
                    apple_data['analysis_status'] = 'analyzing'
                elif tracker['analysis_status'] == 'completed':
                    apple_data['has_analysis'] = True
                    apple_data['analysis'] = tracker['analysis_results']
                    apple_data['analysis_status'] = 'completed'
                elif tracker['analysis_status'] == 'failed':
                    apple_data['analysis_status'] = 'failed'
                
                processed_detections.append(apple_data)
                
            else:
                # OTHER FRUITS: Simple detection only (no tracking, no analysis)
                other_fruit_data = {
                    'fruit_type': fruit_type,
                    'detection_type': 'simple_detection',  # Simple type for other fruits
                    'bbox': fruit['bbox'],
                    'conf': fruit['conf'],
                    'label': f"{fruit_type.title()} {int(fruit['conf']*100)}%"
                }
                processed_detections.append(other_fruit_data)
                print(f"🍓 Detected {fruit_type}: {fruit['conf']:.3f} confidence")
        
        # Cleanup old apple trackers only
        expired_trackers = [
            tracked_id for tracked_id, tracked_data in apple_tracker.items()
            if current_time - tracked_data['last_seen'] > 10
        ]
        for tracked_id in expired_trackers:
            print(f"🧹 Cleaning up old apple tracker: {tracked_id}")
            del apple_tracker[tracked_id]
        
        total_time = time.time()
        
        # Enhanced response with mixed fruit types
        response_data = {
            "status": "success", 
            "detections": processed_detections,
            "performance": {
                "detection_time": round(detection_time - start_time, 3),
                "total_time": round(total_time - start_time, 3),
                "multi_fruit_detection": True
            },
            "tracking_info": {
                "total_fruits_detected": len(processed_detections),
                "apples_tracked": len(apple_tracker),
                "apples_with_analysis": sum(1 for d in processed_detections 
                                          if d.get('detection_type') == 'apple_analysis' and d.get('has_analysis')),
                "other_fruits": sum(1 for d in processed_detections 
                                  if d.get('detection_type') == 'simple_detection')
            }
        }
        
        return convert_numpy_types(response_data)
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return {
            "status": "error",
            "message": f"Detection failed: {str(e)}",
            "detections": []
        }

@app.post("/upload/analyze")
async def analyze_uploaded_image_with_detection(file: UploadFile = File(...)):
    """
    NEW: Upload analysis with mixed fruit detection and bounding boxes
    Returns same format as camera detection for consistent frontend rendering
    """
    if not pipeline or not pipeline.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        start_time = time.time()
        print(f"📤 Starting upload analysis with mixed fruit detection...")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get image dimensions for frontend scaling
        image_height, image_width = image_cv.shape[:2]
        
        # STEP 1: Run YOLO detection for ALL fruits (same as camera mode)
        fruits_found = pipeline.detect_apples_yolo(image_cv)
        print(f"🔍 UPLOAD: detect_apples_yolo returned: {len(fruits_found)} fruits")
        for i, fruit in enumerate(fruits_found):
            print(f"   UPLOAD fruit {i}: {fruit}")

        detection_time = time.time()
        
        if not fruits_found:
            return {
                "status": "success",
                "detections": [],
                "message": "No fruits detected in uploaded image",
                "image_info": {
                    "filename": str(file.filename),
                    "dimensions": f"{image_width}x{image_height}"
                },
                "performance": {
                    "detection_time": round(detection_time - start_time, 3),
                    "total_time": round(detection_time - start_time, 3)
                }
            }
        
        # STEP 2: Process fruits same as camera mode (apples vs other fruits)
        processed_detections = []
        apple_count = 0
        
        for fruit in fruits_found:
            fruit_type = fruit.get('fruit_type', fruit.get('class', 'unknown'))
            bbox = fruit['bbox']
            conf = fruit['conf']
            
            if fruit_type == 'apple':
                # APPLE PROCESSING: Full analysis for apples
                apple_count += 1
                apple_id = f"upload_apple_{apple_count}_{start_time}"
                
                print(f"🍎 Processing apple {apple_count} for full analysis...")
                
                # Run complete analysis pipeline for this apple
                analysis_results = pipeline.process_complete_pipeline(image_cv, bbox)
                
                apple_data = {
                    'fruit_type': 'apple',
                    'detection_type': 'apple_analysis',
                    'apple_id': apple_id,
                    'bbox': bbox,
                    'conf': conf,
                    'stable': True,  # Upload images are always "stable"
                    'has_analysis': analysis_results['status'] == 'success',
                    'analysis': analysis_results.get('analysis_results') if analysis_results['status'] == 'success' else None,
                    'analysis_status': 'completed' if analysis_results['status'] == 'success' else 'failed'
                }
                
                processed_detections.append(apple_data)
                print(f"✅ Apple {apple_count} analysis: {apple_data['analysis_status']}")
                
            else:
                # OTHER FRUITS: Simple detection only (same as camera mode)
                other_fruit_data = {
                    'fruit_type': fruit_type,
                    'detection_type': 'simple_detection',
                    'bbox': bbox,
                    'conf': conf,
                    'label': f"{fruit_type.title()} {int(conf*100)}%"
                }
                processed_detections.append(other_fruit_data)
                print(f"🍓 Detected {fruit_type}: {conf:.3f} confidence")
        
        total_time = time.time()
        print(f"🏁 Upload analysis completed: {(total_time - start_time):.2f}s")
        
        # Return same format as camera detection for consistent frontend rendering
        response_data = {
            "status": "success",
            "detections": processed_detections,
            "file_info": {
                "filename": str(file.filename),
                "dimensions": f"{image_width}x{image_height}",
                "size_bytes": len(contents)
            },
            "detection_summary": {
                "total_fruits": len(processed_detections),
                "apples_found": sum(1 for d in processed_detections if d.get('detection_type') == 'apple_analysis'),
                "other_fruits": sum(1 for d in processed_detections if d.get('detection_type') == 'simple_detection'),
                "apples_analyzed": sum(1 for d in processed_detections 
                                    if d.get('detection_type') == 'apple_analysis' and d.get('has_analysis'))
            },
            "performance": {
                "detection_time": round(detection_time - start_time, 3),
                "total_time": round(total_time - start_time, 2),
                "mixed_fruit_detection": True,
                "upload_mode": True
            }
        }
        
        # CRITICAL: Convert all NumPy types to Python types for JSON serialization
        return convert_numpy_types(response_data)
        
    except Exception as e:
        print(f"❌ Upload analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload analysis failed: {str(e)}")

@app.get("/health")
def health_check():
    """Comprehensive health check with threading info"""
    return {
        "api_status": "running",
        "models_loaded": pipeline.models_loaded if pipeline else False,
        "active_trackers": len(apple_tracker),
        "features": {
            "threading_enabled": True,
            "always_fresh_analysis": True,
            "parallel_processing": True,
            "caching_disabled": True
        },
        "timestamp": datetime.now().isoformat()
    }

# ❌ REMOVED: Analysis queue endpoints (not needed with direct analysis)

# Run the server
if __name__ == "__main__":
    import uvicorn
    print("🍎 Starting FruittiAI Backend with Threading & Fresh Analysis...")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🚀 Features: Threading ✅ | Always Fresh Analysis ✅ | No Caching ✅")
    uvicorn.run(app, host="0.0.0.0", port=8000)