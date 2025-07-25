import React, { useState, useRef, useEffect } from 'react';

// Updated styles - NO FUNCTION CHANGES, ONLY LAYOUT
const styles = {
  btn: {
    padding: '12px 24px',
    border: 'none',
    borderRadius: '8px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '14px',
  },

  toastContainer: {
    position: 'fixed',
    top: '20px',
    right: '20px',
    zIndex: 9999,
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },

  toast: {
    background: 'white',
    padding: '12px 16px',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    minWidth: '300px',
    fontSize: '14px',
    fontWeight: '500',
    animation: 'slideIn 0.3s ease-out',
    borderLeft: '4px solid',
  },

  toastError: {
    borderLeftColor: '#dc3545',
    color: '#721c24',
    background: '#f8d7da',
  },

  toastWarning: {
    borderLeftColor: '#ffc107',
    color: '#856404',
    background: '#fff3cd',
  },

  btnPrimary: {
    background: 'linear-gradient(135deg, #007bff, #0056b3)',
    color: 'white',
  },

  btnSuccess: {
    background: 'linear-gradient(135deg, #28a745, #1e7e34)',
    color: 'white',
  },

  gradingContainer: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    padding: '20px',
  },

  gradingCard: {
    maxWidth: '1400px', // Increased max width
    margin: '0 auto',
    background: 'white',
    borderRadius: '20px',
    boxShadow: '0 20px 60px rgba(0,0,0,0.1)',
    overflow: 'hidden',
  },

  gradingHeader: {
    background: 'linear-gradient(135deg, #2c3e50 0%, #34495e 100%)',
    color: 'white',
    padding: '30px',
    textAlign: 'center',
  },

  // NEW: Single column layout
  gradingContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '30px',
    padding: '40px',
  },

  // NEW: Camera/Upload section (top part)
  displaySection: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
  },

  // UPDATED: Much larger camera container to match results width
  // UPDATED: Smaller camera container
  cameraContainer: {
    position: 'relative',
    width: '80%', // Reduced from 100%
    maxWidth: '800px', // Add max width constraint
    background: '#f8f9fa',
    borderRadius: '15px',
    overflow: 'hidden',
    border: '3px dashed #dee2e6',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: '20px',
    margin: '0 auto 20px auto', // Center the camera
  },

  video: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },

  cameraPlaceholder: {
    textAlign: 'center',
    color: '#6c757d',
  },

  detectionBox: {
    position: 'absolute',
    border: '3px solid #28a745',
    borderRadius: '8px',
    background: 'rgba(40, 167, 69, 0.1)',
    cursor: 'pointer',
    transition: 'all 0.3s ease', // Smooth movement animation
  },

  detectionLabel: {
    position: 'absolute',
    top: '-30px',
    left: '0',
    background: '#28a745',
    color: 'white',
    padding: '5px 10px',
    borderRadius: '5px',
    fontSize: '12px',
    fontWeight: '600',
  },

  // UPDATED: Controls now below the large display
  controls: {
    display: 'flex',
    gap: '15px',
    marginBottom: '20px',
    justifyContent: 'center', // Center the controls
    flexWrap: 'wrap', // Allow wrapping on smaller screens
  },

  // UPDATED: Results section now full width at bottom
  resultsSection: {
    background: '#f8f9fa',
    borderRadius: '15px',
    padding: '30px 40px', // Equal horizontal padding
    width: '100%', // Full width
    margin: '0', // Removed margin-top to fix uneven spacing
    boxSizing: 'border-box', // Include padding in width calculation
  },

  resultCard: {
    background: 'white',
    borderRadius: '12px',
    padding: '20px',
    borderLeft: '4px solid #007bff',
    boxShadow: '0 2px 10px rgba(0,0,0,0.05)',
    marginBottom: '20px',
  },

  resultLabel: {
    fontSize: '14px',
    color: '#6c757d',
    marginBottom: '5px',
    fontWeight: '500',
  },

  resultValue: {
    fontSize: '18px',
    fontWeight: '700',
    color: '#2c3e50',
  },

  confidenceBar: {
    width: '100%',
    height: '8px',
    background: '#e9ecef',
    borderRadius: '4px',
    marginTop: '10px',
    overflow: 'hidden',
  },

  confidenceFill: {
    height: '100%',
    background: 'linear-gradient(90deg, #28a745, #20c997)',
    borderRadius: '4px',
    transition: 'width 0.5s ease',
  },

  gradeBadge: {
    display: 'inline-block',
    padding: '8px 16px',
    borderRadius: '20px',
    fontWeight: '600',
    fontSize: '14px',
    marginTop: '10px',
  },

  gradeA: { background: '#d4edda', color: '#155724' },
  gradeB: { background: '#fff3cd', color: '#856404' },
  gradeC: { background: '#f8d7da', color: '#721c24' },

  noResults: {
    textAlign: 'center',
    color: '#6c757d',
    padding: '40px 20px',
  },

  // NEW: Mode selection styling
  modeSelection: {
    display: 'flex',
    gap: '10px',
    marginBottom: '30px',
    background: '#f8f9fa',
    padding: '10px',
    borderRadius: '8px',
    justifyContent: 'center',
  },
};

const GradingSystem = ({ onBackToHome }) => {
  const [toasts, setToasts] = useState([]);

  // Camera states
  const [stream, setStream] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState(null);
  const [uploadMode, setUploadMode] = useState('camera');
  const videoRef = useRef(null);
  const [uploadDetections, setUploadDetections] = useState([]);
  const [selectedUploadAppleId, setSelectedUploadAppleId] = useState(null);
  const [uploadImageDimensions, setUploadImageDimensions] = useState({ width: 0, height: 0 });

  // Two-stage detection states
  const [clientSideDetecting, setClientSideDetecting] = useState(false);
  const [objectsPresent, setObjectsPresent] = useState(false);
  const [detections, setDetections] = useState([]);
  const [selectedAppleId, setSelectedAppleId] = useState(null);

  // Analysis states
  const [results, setResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Intervals
  const clientDetectionIntervalRef = useRef(null);
  const backendDetectionIntervalRef = useRef(null);

  // Client-side object detection using simple motion/change detection

  const showToast = (message, type = 'error') => {
    const id = Date.now();
    const newToast = { id, message, type };

    setToasts(prev => [...prev, newToast]);

    // Auto remove after 4 seconds
    setTimeout(() => {
      setToasts(prev => prev.filter(toast => toast.id !== id));
    }, 4000);
  };

  const removeToast = (id) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };
  const detectObjectsClientSide = () => {
    const video = videoRef.current;
    if (!video || !video.srcObject) return false;

    try {
      // Create canvas for frame analysis
      const canvas = document.createElement('canvas');
      canvas.width = 320; // Smaller for faster processing
      canvas.height = 240;
      const ctx = canvas.getContext('2d');

      // Draw current frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const pixels = imageData.data;

      // Simple object detection: look for color variations that suggest objects
      let colorVariation = 0;
      let edgeCount = 0;

      // Sample pixels in a grid pattern for speed
      for (let y = 10; y < canvas.height - 10; y += 20) {
        for (let x = 10; x < canvas.width - 10; x += 20) {
          const i = (y * canvas.width + x) * 4;
          const r = pixels[i];
          const g = pixels[i + 1];
          const b = pixels[i + 2];

          // Check color variation (non-uniform backgrounds suggest objects)
          const brightness = (r + g + b) / 3;

          // Compare with neighboring pixels for edge detection
          const rightI = (y * canvas.width + (x + 20)) * 4;
          const downI = ((y + 20) * canvas.width + x) * 4;

          if (rightI < pixels.length && downI < pixels.length) {
            const rightBrightness = (pixels[rightI] + pixels[rightI + 1] + pixels[rightI + 2]) / 3;
            const downBrightness = (pixels[downI] + pixels[downI + 1] + pixels[downI + 2]) / 3;

            if (Math.abs(brightness - rightBrightness) > 30 ||
              Math.abs(brightness - downBrightness) > 30) {
              edgeCount++;
            }
          }

          // Color saturation check (objects usually have more color than backgrounds)
          const maxColor = Math.max(r, g, b);
          const minColor = Math.min(r, g, b);
          colorVariation += maxColor - minColor;
        }
      }

      // Thresholds for object detection
      const avgColorVariation = colorVariation / ((canvas.width / 20) * (canvas.height / 20));
      const edgeDensity = edgeCount / ((canvas.width / 20) * (canvas.height / 20));

      // Simple heuristic: if there's enough color variation and edges, objects are likely present
      const objectsDetected = avgColorVariation > 20 && edgeDensity > 0.3;

      console.log(`🔍 Client detection - Color: ${avgColorVariation.toFixed(1)}, Edges: ${edgeDensity.toFixed(2)}, Objects: ${objectsDetected}`);

      return objectsDetected;

    } catch (error) {
      console.error('Client-side detection error:', error);
      return false;
    }
  };

  // Start client-side detection loop

  // Backend detection (only when objects are present)
  const startBackendDetection = () => {
    if (backendDetectionIntervalRef.current) return; // Already running

    console.log('🍎 Starting backend apple detection...');

    backendDetectionIntervalRef.current = setInterval(async () => {
      try {
        const video = videoRef.current;

        if (!video || !video.srcObject || !objectsPresent) {
          return;
        }

        if (video.readyState !== 4 || video.videoWidth === 0) {
          return;
        }

        console.log('📸 Sending frame to backend for apple detection...');

        // Capture frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        // Send to backend
        const response = await fetch('http://localhost:8000/camera/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
          console.error('❌ Backend response error:', response.status);
          return;
        }

        const result = await response.json();

        console.log('🔍 RAW BACKEND RESPONSE:', JSON.stringify(result, null, 2));
        if (result.detections) {
          result.detections.forEach((detection, idx) => {
            console.log(`Detection ${idx}:`, {
              fruit_type: detection.fruit_type,
              detection_type: detection.detection_type,
              bbox: detection.bbox,
              conf: detection.conf
            });
          });
        }
        console.log('📋 Backend result:', result.status, 'detections:', result.detections?.length || 0);

        if (result.status === 'success') {
          setDetections(result.detections || []);
          if (result.detections && result.detections.length > 0) {
            console.log('🍎 Apple(s) detected!', result.detections.length);
          }
        } else {
          console.log('❌ Apple detection failed:', result.message);
          setDetections([]);
        }

      } catch (error) {
        console.error('💥 Backend detection error:', error);
      }
    }, 3000); // Check every 3 seconds when objects are present
  };

  // Stop backend detection
  const stopBackendDetection = () => {
    if (backendDetectionIntervalRef.current) {
      clearInterval(backendDetectionIntervalRef.current);
      backendDetectionIntervalRef.current = null;
      setDetections([]);
      console.log('🛑 Backend detection stopped');
    }
  };

  const stopAllDetection = () => {
    console.log('🛑 Stopping all detection...');

    // Clear detection interval immediately
    if (backendDetectionIntervalRef.current) {
      clearInterval(backendDetectionIntervalRef.current);
      backendDetectionIntervalRef.current = null;
    }

    // Clear states
    setDetections([]);
    setSelectedAppleId(null);
    setResults(null);
    setIsAnalyzing(false);
  };

  // Analysis polling - UPDATED for new direct analysis API

  const handleAppleSelection = async (appleId) => {
    console.log(`🍎 Manual apple selection: ${appleId}`);
    setSelectedAppleId(appleId);

    // Find the selected apple in current detections
    const selectedApple = detections.find(detection => detection.apple_id === appleId);

    if (selectedApple) {
      console.log('🔍 Selected apple details:', {
        id: selectedApple.apple_id,
        stable: selectedApple.stable,
        has_analysis: selectedApple.has_analysis,
        analysis_exists: !!selectedApple.analysis
      });

      if (selectedApple.has_analysis && selectedApple.analysis) {
        console.log('✅ Immediate results available for selected apple');
        console.log('📊 Setting results:', selectedApple.analysis);
        setResults(selectedApple.analysis);
        setIsAnalyzing(false);
      } else {
        console.log('⏳ No analysis yet - will wait for results');
        setResults(null);
        setIsAnalyzing(true);
      }
    } else {
      console.log('❌ Selected apple not found in current detections');
      setResults(null);
      setIsAnalyzing(true);
    }
  };

  const startCamera = async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera API not supported in this browser');
      }

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 }, // Good quality
          height: { ideal: 720 },
          frameRate: { ideal: 30 } // Smooth video for better detection
        }
      });
      setStream(mediaStream);

      console.log('📷 Camera stream started with optimized settings!');

      // Start detection as soon as possible
      setTimeout(() => {
        console.log('⚡ Starting IMMEDIATE detection in 200ms...');
        startImmediateYOLODetection();
      }, 200); // Minimal delay for camera initialization

    } catch (err) {
      let errorMessage = 'Camera Error: ';

      if (err.name === 'NotAllowedError') {
        errorMessage += 'Permission denied. Please allow camera access.';
      } else if (err.name === 'NotFoundError') {
        errorMessage += 'No camera found.';
      } else if (err.name === 'NotSupportedError') {
        errorMessage += 'Camera not supported in this browser.';
      } else if (err.name === 'NotReadableError') {
        errorMessage += 'Camera is busy. Close other apps using the camera.';
      } else {
        errorMessage += err.message;
      }

      console.error('Camera error:', err);
      alert(errorMessage);
    }
  };

  const startImmediateYOLODetection = () => {
    console.log('⚡ Starting ULTRA-FAST YOLO detection with analysis caching!');

    backendDetectionIntervalRef.current = setInterval(async () => {
      try {
        const video = videoRef.current;

        if (!video || !video.srcObject) {
          return;
        }

        if (video.readyState !== 4 || video.videoWidth === 0) {
          return;
        }

        // Capture frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        const imageData = canvas.toDataURL('image/jpeg', 0.7);

        const response = await fetch('http://localhost:8000/camera/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: imageData })
        });

        if (!response.ok) {
          console.error('❌ Backend response error:', response.status);
          return;
        }

        const result = await response.json();

        if (result.status === 'success') {
          setDetections(result.detections || []);

          if (result.detections && result.detections.length > 0) {
            // Log analysis status for debugging
            result.detections.forEach((detection, idx) => {
              const status = detection.analysis_status || 'unknown';
              console.log(`📊 Apple ${idx + 1} (${detection.apple_id}): ${status}${status === 'completed' && detection.analysis ?
                ` - Grade ${detection.analysis.advanced_grading?.grade || detection.analysis.grade}` : ''
                }`);
            });

            // Auto-select apple with completed analysis
            const completedApple = result.detections.find(d =>
              d.analysis_status === 'completed' && d.analysis
            );

            if (completedApple && completedApple.apple_id !== selectedAppleId) {
              console.log('🎯 Auto-selecting apple with completed analysis:', completedApple.apple_id);
              setSelectedAppleId(completedApple.apple_id);
              setResults(completedApple.analysis);
              setIsAnalyzing(false);
            } else if (!selectedAppleId) {
              // Select first stable apple if none selected
              const stableApple = result.detections.find(d => d.stable);
              if (stableApple) {
                console.log('🔄 Auto-selecting stable apple:', stableApple.apple_id);
                setSelectedAppleId(stableApple.apple_id);

                if (stableApple.analysis_status === 'completed' && stableApple.analysis) {
                  setResults(stableApple.analysis);
                  setIsAnalyzing(false);
                } else {
                  setResults(null);
                  setIsAnalyzing(stableApple.analysis_status === 'analyzing');
                }
              }
            }

            // Update results for currently selected apple
            if (selectedAppleId) {
              const currentApple = result.detections.find(d => d.apple_id === selectedAppleId);
              if (currentApple && currentApple.detection_type === 'apple_analysis') {
                if (currentApple.analysis_status === 'completed' && currentApple.analysis) {
                  console.log('✅ Updated results for selected apple:', selectedAppleId);
                  setResults(currentApple.analysis);
                  setIsAnalyzing(false);
                } else if (currentApple.analysis_status === 'analyzing') {
                  console.log('⏳ Selected apple still analyzing...');
                  setIsAnalyzing(true);
                } else if (currentApple.analysis_status === 'failed') {
                  console.log('❌ Selected apple analysis failed');
                  setIsAnalyzing(false);
                  setResults(null); // Clear results on failed analysis
                }
              } else {
                // Selected item is not an apple or apple no longer detected
                setResults(null);
                setIsAnalyzing(false);
                setSelectedAppleId(null);
              }
            }

          } else {
            // No apples detected
            setDetections([]);
            if (selectedAppleId) {
              setSelectedAppleId(null);
              setResults(null);
              setIsAnalyzing(false);
            }
          }

          setTimeout(() => {
            const hasApples = result.detections.some(d => d.detection_type === 'apple_analysis');
            const hasOtherFruits = result.detections.some(d => d.detection_type === 'simple_detection');

            if (hasOtherFruits && !hasApples) {
              console.log('🧹 DELAYED CLEAR - non-apple fruit detected!');
              setResults(null);
              setIsAnalyzing(false);
              setSelectedAppleId(null);
            }
          }, 50);

          // Log tracking info
          if (result.tracking_info) {
            const info = result.tracking_info;
            console.log(`📋 Tracking: ${info.total_tracked} total, ${info.stable_apples} stable, ${info.analyzed_apples} analyzed, ${info.analyzing_apples || 0} analyzing`);
          }

        } else {
          console.log('❌ Detection failed:', result.message);
          setDetections([]);
        }

      } catch (error) {
        console.error('💥 Detection error:', error);
      }
    }, 300); // Continue with 300ms for responsiveness
  };


  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      stopAllDetection();
      setResults(null);

      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }
  };

  const calculateAutoZoom = (detections, imageDimensions) => {
    if (!detections || detections.length === 0 || !imageDimensions.width) {
      return { scale: 1, marginX: 0, marginY: 0 };
    }

    let maxX = 0, maxY = 0;

    // Check all bounding boxes to find the maximum extent
    detections.forEach(detection => {
      const [x1, y1, x2, y2] = detection.bbox;
      maxX = Math.max(maxX, x2 / imageDimensions.width);
      maxY = Math.max(maxY, y2 / imageDimensions.height);
    });

    // If any bounding box extends beyond 90% of the image, we need to zoom out
    const needsZoom = maxX > 0.9 || maxY > 0.9;

    if (!needsZoom) {
      return { scale: 1, marginX: 0, marginY: 0 };
    }

    // Calculate scale factor - aim to keep bounding boxes within 85% of container
    const scaleX = 0.85 / maxX;
    const scaleY = 0.85 / maxY;
    const scale = Math.min(scaleX, scaleY, 0.8); // Cap at 0.8x minimum

    // Calculate margins to center the scaled image
    const marginX = (100 - (scale * 100)) / 2;
    const marginY = (100 - (scale * 100)) / 2;

    return { scale, marginX, marginY };
  };

  const handleUploadAppleSelection = (appleId) => {
    console.log(`🍎 Upload apple selection: ${appleId}`);
    setSelectedUploadAppleId(appleId);

    // Find selected apple in upload detections
    const selectedApple = uploadDetections.find(d => d.apple_id === appleId);

    if (selectedApple && selectedApple.has_analysis && selectedApple.analysis) {
      console.log('✅ Displaying analysis for selected apple');
      setResults(selectedApple.analysis);
      setIsAnalyzing(false);
    } else {
      console.log('❌ No analysis available for selected apple');
      setResults(null);
      setIsAnalyzing(false);
    }
  };

  const renderUploadBoundingBoxes = () => {
    if (!uploadDetections || uploadDetections.length === 0) {
      return null;
    }

    if (!uploadedImageUrl || uploadImageDimensions.width === 0) {
      return null;
    }

    // Get auto-zoom settings
    const autoZoom = calculateAutoZoom(uploadDetections, uploadImageDimensions);

    return uploadDetections.map((detection, index) => {
      const [x1, y1, x2, y2] = detection.bbox;

      // Calculate the actual display dimensions within the 16:9 container
      const containerAspectRatio = 16 / 9;
      const imageAspectRatio = uploadImageDimensions.width / uploadImageDimensions.height;

      let displayWidth, displayHeight, offsetX, offsetY;

      if (imageAspectRatio > containerAspectRatio) {
        // Image is wider - fit to width, letterbox top/bottom
        displayWidth = 100;
        displayHeight = (containerAspectRatio / imageAspectRatio) * 100;
        offsetX = 0;
        offsetY = (100 - displayHeight) / 2;
      } else {
        // Image is taller - fit to height, pillarbox left/right
        displayHeight = 100;
        displayWidth = (imageAspectRatio / containerAspectRatio) * 100;
        offsetY = 0;
        offsetX = (100 - displayWidth) / 2;
      }

      // Apply auto-zoom scaling and margins
      displayWidth *= autoZoom.scale;
      displayHeight *= autoZoom.scale;
      offsetX = autoZoom.marginX + offsetX * autoZoom.scale;
      offsetY = autoZoom.marginY + offsetY * autoZoom.scale;

      // Calculate bounding box position relative to the actual displayed image
      const leftPercent = offsetX + (x1 / uploadImageDimensions.width) * displayWidth;
      const topPercent = offsetY + (y1 / uploadImageDimensions.height) * displayHeight;
      const widthPercent = ((x2 - x1) / uploadImageDimensions.width) * displayWidth;
      const heightPercent = ((y2 - y1) / uploadImageDimensions.height) * displayHeight;

      // Apple analysis detection
      if (detection.detection_type === 'apple_analysis') {
        const isSelected = detection.apple_id === selectedUploadAppleId;
        const analysisStatus = detection.analysis_status || 'none';

        let labelText = `Apple ${detection.conf.toFixed(2)}`;
        let themeColor = '#28a745';
        let borderStyle = 'solid';
        let borderWidth = '2px';
        let backgroundColor = 'rgba(40, 167, 69, 0.05)';

        if (analysisStatus === 'completed' && detection.analysis) {
          const grade = detection.analysis.advanced_grading?.grade || detection.analysis.grade || 'Unknown';
          const score = detection.analysis.advanced_grading?.total_score;
          labelText = score ? `Grade ${grade} (${score})` : `Grade ${grade}`;
          themeColor = '#28a745';
          backgroundColor = 'rgba(40, 167, 69, 0.08)';
        }

        if (isSelected) {
          borderWidth = '4px';
          backgroundColor = backgroundColor.replace('0.08', '0.15');
        }

        return (
          <div
            key={detection.apple_id || index}
            style={{
              position: 'absolute',
              left: `${leftPercent}%`,
              top: `${topPercent}%`,
              width: `${widthPercent}%`,
              height: `${heightPercent}%`,
              border: `${borderWidth} ${borderStyle} ${themeColor}`,
              borderRadius: '8px',
              background: backgroundColor,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              pointerEvents: 'auto',
              zIndex: 10,
            }}
            onClick={() => handleUploadAppleSelection(detection.apple_id)}
          >
            <div style={{
              position: 'absolute',
              top: '-25px',
              left: '0',
              background: themeColor,
              color: 'white',
              padding: '3px 8px',
              borderRadius: '4px',
              fontSize: '11px',
              fontWeight: '600',
              whiteSpace: 'nowrap',
              boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
            }}>
              {labelText}
            </div>

            <div style={{
              position: 'absolute',
              top: '5px',
              right: '5px',
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: themeColor,
              boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
            }} />
          </div>
        );

      } else if (detection.detection_type === 'simple_detection') {
        // Other fruits detection
        const getFruitTheme = (fruitType) => {
          switch (fruitType) {
            case 'banana':
              return { color: '#FFD700', emoji: '🍌', bgColor: 'rgba(255, 215, 0, 0.15)' };
            case 'grape':
              return { color: '#8A2BE2', emoji: '🍇', bgColor: 'rgba(138, 43, 226, 0.15)' };
            case 'guava':
              return { color: '#90EE90', emoji: '🥭', bgColor: 'rgba(144, 238, 144, 0.15)' };
            case 'mango':
              return { color: '#FF8C00', emoji: '🥭', bgColor: 'rgba(255, 140, 0, 0.15)' };
            case 'orange':
              return { color: '#FF4500', emoji: '🍊', bgColor: 'rgba(255, 69, 0, 0.15)' };
            case 'pineapple':
              return { color: '#DAA520', emoji: '🍍', bgColor: 'rgba(218, 165, 32, 0.15)' };
            default:
              return { color: '#FF6347', emoji: '🍎', bgColor: 'rgba(255, 99, 71, 0.15)' };
          }
        };

        const fruitTheme = getFruitTheme(detection.fruit_type);

        return (
          <div
            key={`${detection.fruit_type}_${index}`}
            style={{
              position: 'absolute',
              left: `${leftPercent}%`,
              top: `${topPercent}%`,
              width: `${widthPercent}%`,
              height: `${heightPercent}%`,
              border: `3px solid ${fruitTheme.color}`,
              borderRadius: '8px',
              background: fruitTheme.bgColor,
              pointerEvents: 'none',
              transition: 'all 0.2s ease',
              zIndex: 9,
            }}
          >
            <div style={{
              position: 'absolute',
              top: '-28px',
              left: '0',
              background: fruitTheme.color,
              color: 'white',
              padding: '4px 10px',
              borderRadius: '4px',
              fontSize: '12px',
              fontWeight: '600',
              whiteSpace: 'nowrap',
              boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}>
              <span>{fruitTheme.emoji}</span>
              <span>{detection.label}</span>
            </div>

            <div style={{
              position: 'absolute',
              top: '5px',
              right: '5px',
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              background: fruitTheme.color,
              boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
            }} />

            <div style={{
              position: 'absolute',
              bottom: '5px',
              left: '5px',
              background: 'rgba(0,0,0,0.7)',
              color: 'white',
              padding: '2px 6px',
              borderRadius: '3px',
              fontSize: '9px',
              fontWeight: '500',
            }}>
              DETECT
            </div>
          </div>
        );
      }

      return null;
    });
  };

  useEffect(() => {
    if (uploadedImageUrl && uploadDetections.length > 0) {
      // Small delay to ensure image is fully loaded and measured
      const timer = setTimeout(() => {
        console.log('🔄 Forcing bounding box re-render after image load');
        // Force a re-render by updating a dummy state or just log
        setUploadDetections([...uploadDetections]);
      }, 100);

      return () => clearTimeout(timer);
    }
  }, [uploadedImageUrl, uploadDetections.length]);

  // Upload functionality (unchanged)
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    // Reset all states
    setResults(null);
    setIsAnalyzing(true);
    setUploadDetections([]);
    setSelectedUploadAppleId(null);
    setUploadImageDimensions({ width: 0, height: 0 });

    try {
      if (uploadedImageUrl) {
        URL.revokeObjectURL(uploadedImageUrl);
      }

      const imageUrl = URL.createObjectURL(file);
      setUploadedImageUrl(imageUrl);
      setUploadedImage(file);
      setUploadMode('upload');

      // Get image dimensions for bounding box scaling
      const img = new Image();
      img.onload = () => {
        setUploadImageDimensions({ width: img.width, height: img.height });
      };
      img.src = imageUrl;

      console.log('📤 Uploading image for mixed fruit detection...');

      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/upload/analyze', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      console.log('📋 Upload result:', result);

      if (result.status === 'success') {
        // Store detections for bounding box rendering
        setUploadDetections(result.detections || []);

        console.log('📋 Full detections array:', result.detections);
        console.log('📋 Setting uploadDetections to:', result.detections || []);
        console.log('🔄 About to check apples and other fruits...');

        // Check what was detected
        const apples = result.detections?.filter(d => d.detection_type === 'apple_analysis') || [];
        const otherFruits = result.detections?.filter(d => d.detection_type === 'simple_detection') || [];

        console.log(`🍎 Found ${apples.length} apples and ${otherFruits.length} other fruits`);

        if (apples.length > 0) {
          // Auto-select first apple with analysis
          const appleWithAnalysis = apples.find(apple => apple.has_analysis && apple.analysis);

          if (appleWithAnalysis) {
            console.log('✅ Auto-selecting analyzed apple:', appleWithAnalysis.apple_id);
            setSelectedUploadAppleId(appleWithAnalysis.apple_id);
            setResults(appleWithAnalysis.analysis);
            setIsAnalyzing(false);
          } else {
            // Select first apple even if no analysis yet
            console.log('🔄 Selecting first apple:', apples[0].apple_id);
            setSelectedUploadAppleId(apples[0].apple_id);
            setResults(apples[0].analysis || null);
            setIsAnalyzing(!apples[0].has_analysis);
          }
        } else if (otherFruits.length > 0) {
          // Only other fruits detected - no analysis needed
          console.log('🍓 Only other fruits detected - no apple analysis');
          setResults(null);
          setIsAnalyzing(false);
        } else {
          // No fruits detected
          console.log('❌ No fruits detected in uploaded image');
          setResults(null);
          setIsAnalyzing(false);
          showToast('⚠️ No fruits detected in image', 'warning');
        }

      } else {
        throw new Error(result.message || 'Analysis failed');
      }

    } catch (error) {
      console.error('❌ Upload analysis failed:', error);
      alert(`Analysis failed: ${error.message}`);
      setIsAnalyzing(false);
      setUploadDetections([]);
    }
  };

  const resetUpload = () => {
    if (uploadedImageUrl) {
      URL.revokeObjectURL(uploadedImageUrl);
    }
    setUploadedImage(null);
    setUploadedImageUrl(null);
    setResults(null);

    // ADD this line to reset the message:
    setUploadDetections([]);  // This will clear the detections and reset the message

    const fileInput = document.getElementById('file-upload-controls');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const reAnalyzeImage = () => {
    if (!uploadedImage) return;

    // Check if there are any apples to re-analyze
    const hasApples = uploadDetections.some(d => d.detection_type === 'apple_analysis');

    if (!hasApples) {
      showToast('❌ No apples found to re-analyze', 'error');
      return;
    }

    setIsAnalyzing(true);
    setResults(null);

    // Trigger re-upload
    const formData = new FormData();
    formData.append('file', uploadedImage);

    fetch('http://localhost:8000/upload/analyze', {
      method: 'POST',
      body: formData,
    })
      .then(response => response.json())
      .then(result => {
        if (result.status === 'success') {
          setResults(result.analysis);
        }
        setIsAnalyzing(false);
      })
      .catch(error => {
        console.error('Re-analysis failed:', error);
        setIsAnalyzing(false);
      });
  };

  // Utility functions

  const smoothAnimationStyles = `
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
  }
  
  @keyframes smoothEntry {
    0% { 
      opacity: 0; 
      transform: scale(0.8); 
    }
    100% { 
      opacity: 1; 
      transform: scale(1); 
    }
  }

  @keyframes slideIn {
    0% {
      opacity: 0;
      transform: translateX(100%);
    }
    100% {
      opacity: 1;
      transform: translateX(0);
    }
  }
`;

  // Add this to your component (inject styles)
  useEffect(() => {
    const styleSheet = document.createElement('style');
    styleSheet.textContent = smoothAnimationStyles;
    document.head.appendChild(styleSheet);

    return () => {
      document.head.removeChild(styleSheet);
    };
  }, []);


  // Auto-assign stream to video element
  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play().catch(err => {
          console.error('Error playing video:', err);
        });
      };
    }
  }, [stream]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      stopAllDetection();
    };
  }, [stream]);

  return (
    <div style={styles.gradingContainer}>
      <div style={styles.gradingCard}>
        {/* Header - unchanged */}
        <div style={styles.gradingHeader}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
            <button
              onClick={onBackToHome}
              style={{ ...styles.btn, ...styles.btnPrimary }}
            >
              ← Back to Home
            </button>
          </div>
          <h1 style={{ fontSize: '2.5rem', fontWeight: '700', marginBottom: '10px' }}>
            🍎 AI Fruit Grading System
          </h1>
          <p style={{ fontSize: '1.1rem', opacity: 0.9 }}>
            Smart detection - only analyzes when objects are present
          </p>
        </div>

        {/* NEW LAYOUT: Single column with large display area */}
        <div style={styles.gradingContent}>

          {/* Display Section - Camera or Upload */}
          <div style={styles.displaySection}>

            {/* Mode Selection */}
            <div style={styles.modeSelection}>
              <button
                onClick={() => {
                  setUploadMode('camera');
                  resetUpload();
                }}
                style={{
                  ...styles.btn,
                  background: uploadMode === 'camera' ? '#007bff' : '#6c757d',
                  color: 'white',
                  flex: 1,
                  minWidth: '150px',
                  whiteSpace: 'nowrap',
                  fontSize: '14px',
                  padding: '12px 20px'
                }}
              >
                📷 Smart Camera
              </button>

              <button
                onClick={() => {
                  setUploadMode('upload');
                  stopCamera();
                }}
                style={{
                  ...styles.btn,
                  background: uploadMode === 'upload' ? '#007bff' : '#6c757d',
                  color: 'white',
                  flex: 1,
                  minWidth: '150px',
                  whiteSpace: 'nowrap',
                  fontSize: '14px',
                  padding: '12px 20px'
                }}
              >
                📁 Upload Mode
              </button>
            </div>

            {/* Large Display Container - Same for both modes */}
            <div style={styles.cameraContainer}>
              {uploadMode === 'camera' ? (
                <>
                  {stream ? (
                    <>
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        style={styles.video}
                      />
                      {/* Complete Real bounding boxes with improved tracking - Mixed fruit detection */}
                      {detections.map((detection, index) => {
                        // Check detection type
                        if (detection.detection_type === 'apple_analysis') {
                          // APPLE LOGIC - UPDATED with percentage positioning
                          const [x1, y1, x2, y2] = detection.bbox;

                          // GET VIDEO DIMENSIONS for percentage calculation
                          const videoElement = videoRef.current;
                          if (!videoElement) return null;

                          const videoActualWidth = videoElement.videoWidth;
                          const videoActualHeight = videoElement.videoHeight;

                          if (videoActualWidth === 0 || videoActualHeight === 0) return null;

                          // PERCENTAGE-BASED POSITIONING (same as upload mode)
                          const leftPercent = (x1 / videoActualWidth) * 100;
                          const topPercent = (y1 / videoActualHeight) * 100;
                          const widthPercent = ((x2 - x1) / videoActualWidth) * 100;
                          const heightPercent = ((y2 - y1) / videoActualHeight) * 100;

                          console.log(`🎯 Camera Apple: bbox [${x1},${y1},${x2},${y2}] → [${leftPercent.toFixed(1)}%,${topPercent.toFixed(1)}%,${widthPercent.toFixed(1)}%,${heightPercent.toFixed(1)}%]`);

                          const isSelected = detection.apple_id === selectedAppleId;
                          const analysisStatus = detection.analysis_status || 'none';

                          let labelText = `Apple ${detection.conf.toFixed(2)}`;
                          let themeColor = '#28a745';
                          let borderStyle = 'solid';
                          let borderWidth = '2px';
                          let showPulse = false;
                          let backgroundColor = 'rgba(40, 167, 69, 0.05)';

                          switch (analysisStatus) {
                            case 'completed':
                              if (detection.analysis) {
                                const grade = detection.analysis.advanced_grading?.grade || detection.analysis.grade || 'Unknown';
                                const score = detection.analysis.advanced_grading?.total_score;
                                labelText = score ? `Grade ${grade} (${score})` : `Grade ${grade}`;
                                themeColor = '#28a745';
                                backgroundColor = 'rgba(40, 167, 69, 0.08)';
                                borderWidth = isSelected ? '3px' : '2px';
                              }
                              break;

                            case 'analyzing':
                              labelText = 'Analyzing...';
                              themeColor = '#ffc107';
                              backgroundColor = 'rgba(255, 193, 7, 0.08)';
                              borderStyle = 'dashed';
                              borderWidth = '3px';
                              showPulse = true;
                              break;

                            case 'failed':
                              labelText = 'Analysis Failed';
                              themeColor = '#dc3545';
                              backgroundColor = 'rgba(220, 53, 69, 0.08)';
                              borderStyle = 'dotted';
                              borderWidth = '2px';
                              break;

                            default:
                              if (detection.stable) {
                                labelText = 'Ready';
                                themeColor = '#17a2b8';
                                backgroundColor = 'rgba(23, 162, 184, 0.08)';
                              } else {
                                labelText = `${detection.conf.toFixed(2)}`;
                                themeColor = '#6c757d';
                                backgroundColor = 'rgba(108, 117, 125, 0.08)';
                                borderStyle = 'dotted';
                              }
                          }

                          if (isSelected) {
                            borderWidth = '4px';
                            backgroundColor = backgroundColor.replace('0.08', '0.12');
                          }

                          return (
                            <div
                              key={detection.apple_id || index}
                              style={{
                                position: 'absolute',
                                left: `${leftPercent}%`,
                                top: `${topPercent}%`,
                                width: `${widthPercent}%`,
                                height: `${heightPercent}%`,
                                border: `${borderWidth} ${borderStyle} ${themeColor}`,
                                borderRadius: '8px',
                                background: backgroundColor,
                                cursor: 'pointer',
                                transition: 'all 0.08s ease-out',
                                pointerEvents: 'auto',
                                willChange: 'transform, left, top, width, height',
                                transform: 'translateZ(0)',
                                animation: showPulse ? 'pulse 1.5s infinite' : 'none',
                                boxShadow: isSelected ? `0 0 0 1px ${themeColor}` : 'none',
                              }}
                              onClick={() => handleAppleSelection(detection.apple_id)}
                            >
                              <div style={{
                                position: 'absolute',
                                top: '-25px',
                                left: '0',
                                background: themeColor,
                                color: 'white',
                                padding: '3px 8px',
                                borderRadius: '4px',
                                fontSize: '11px',
                                fontWeight: '600',
                                whiteSpace: 'nowrap',
                                transition: 'background-color 0.1s ease',
                                willChange: 'background-color',
                                animation: showPulse ? 'pulse 1.5s infinite' : 'none',
                                boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                              }}>
                                {labelText}
                              </div>

                              <div style={{
                                position: 'absolute',
                                top: '5px',
                                right: '5px',
                                width: '8px',
                                height: '8px',
                                borderRadius: '50%',
                                background: themeColor,
                                animation: showPulse ? 'pulse 1s infinite' : 'none',
                                boxShadow: '0 1px 2px rgba(0,0,0,0.2)',
                              }} />
                            </div>
                          );

                        } else if (detection.detection_type === 'simple_detection') {
                          // OTHER FRUITS - UPDATED with percentage positioning
                          const [x1, y1, x2, y2] = detection.bbox;

                          const videoElement = videoRef.current;
                          if (!videoElement) return null;

                          const videoActualWidth = videoElement.videoWidth;
                          const videoActualHeight = videoElement.videoHeight;

                          if (videoActualWidth === 0 || videoActualHeight === 0) return null;

                          // PERCENTAGE-BASED POSITIONING (same as upload mode)
                          const leftPercent = (x1 / videoActualWidth) * 100;
                          const topPercent = (y1 / videoActualHeight) * 100;
                          const widthPercent = ((x2 - x1) / videoActualWidth) * 100;
                          const heightPercent = ((y2 - y1) / videoActualHeight) * 100;

                          // Fruit-specific colors and emojis
                          const getFruitTheme = (fruitType) => {
                            switch (fruitType) {
                              case 'banana':
                                return {
                                  color: '#FFD700',
                                  emoji: '🍌',
                                  bgColor: 'rgba(255, 215, 0, 0.15)'
                                };
                              case 'grape':
                                return {
                                  color: '#8A2BE2',
                                  emoji: '🍇',
                                  bgColor: 'rgba(138, 43, 226, 0.15)'
                                };
                              case 'guava':
                                return {
                                  color: '#90EE90',
                                  emoji: '🥭',
                                  bgColor: 'rgba(144, 238, 144, 0.15)'
                                };
                              case 'mango':
                                return {
                                  color: '#FF8C00',
                                  emoji: '🥭',
                                  bgColor: 'rgba(255, 140, 0, 0.15)'
                                };
                              case 'orange':
                                return {
                                  color: '#FF4500',
                                  emoji: '🍊',
                                  bgColor: 'rgba(255, 69, 0, 0.15)'
                                };
                              case 'pineapple':
                                return {
                                  color: '#DAA520',
                                  emoji: '🍍',
                                  bgColor: 'rgba(218, 165, 32, 0.15)'
                                };
                              default:
                                return {
                                  color: '#FF6347',
                                  emoji: '🍎',
                                  bgColor: 'rgba(255, 99, 71, 0.15)'
                                };
                            }
                          };

                          const fruitTheme = getFruitTheme(detection.fruit_type);

                          return (
                            <div
                              key={`${detection.fruit_type}_${index}`}
                              style={{
                                position: 'absolute',
                                left: `${leftPercent}%`,
                                top: `${topPercent}%`,
                                width: `${widthPercent}%`,
                                height: `${heightPercent}%`,
                                border: `3px solid ${fruitTheme.color}`,
                                borderRadius: '8px',
                                background: fruitTheme.bgColor,
                                pointerEvents: 'none', // No clicking for other fruits
                                transition: 'all 0.1s ease',
                                boxShadow: `0 2px 8px ${fruitTheme.color}30`,
                              }}
                            >
                              {/* Fruit label */}
                              <div style={{
                                position: 'absolute',
                                top: '-28px',
                                left: '0',
                                background: fruitTheme.color,
                                color: 'white',
                                padding: '4px 10px',
                                borderRadius: '4px',
                                fontSize: '12px',
                                fontWeight: '600',
                                whiteSpace: 'nowrap',
                                boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '4px',
                              }}>
                                <span>{fruitTheme.emoji}</span>
                                <span>{detection.label}</span>
                              </div>

                              {/* Confidence indicator */}
                              <div style={{
                                position: 'absolute',
                                top: '5px',
                                right: '5px',
                                width: '10px',
                                height: '10px',
                                borderRadius: '50%',
                                background: fruitTheme.color,
                                boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
                              }} />

                              {/* Detection type indicator */}
                              <div style={{
                                position: 'absolute',
                                bottom: '5px',
                                left: '5px',
                                background: 'rgba(0,0,0,0.7)',
                                color: 'white',
                                padding: '2px 6px',
                                borderRadius: '3px',
                                fontSize: '9px',
                                fontWeight: '500',
                              }}>
                                DETECT
                              </div>
                            </div>
                          );
                        }

                        return null; // Fallback for unknown detection types
                      })}
                    </>
                  ) : (
                    <div style={styles.cameraPlaceholder}>
                      <div style={{ fontSize: '4rem', marginBottom: '15px', opacity: 0.5 }}>🎯</div>
                      <h3>Smart Detection Ready</h3>
                      <p>Uses two-stage detection - only analyzes when objects are present</p>
                    </div>
                  )}
                </>
              ) : (
                <>
                  {uploadedImageUrl ? (
                    <div
                      className="upload-image-container"
                      style={{
                        position: 'relative',
                        width: '100%',
                        height: '0',
                        paddingBottom: '56.25%', // 16:9 aspect ratio
                        background: '#f8f9fa',
                        borderRadius: '15px',
                        overflow: 'hidden'
                      }}
                    >
                      {(() => {
                        // Calculate auto-zoom
                        const autoZoom = calculateAutoZoom(uploadDetections, uploadImageDimensions);

                        return (
                          <img
                            src={uploadedImageUrl}
                            alt="Uploaded fruit"
                            style={{
                              position: 'absolute',
                              top: `${autoZoom.marginY}%`,
                              left: `${autoZoom.marginX}%`,
                              width: `${autoZoom.scale * 100}%`,
                              height: `${autoZoom.scale * 100}%`,
                              objectFit: 'contain',
                              background: '#fff'
                            }}
                            onLoad={() => {
                              console.log('📏 Image loaded with auto-zoom! Scale:', autoZoom.scale);
                              setTimeout(() => {
                                setUploadDetections(prev => [...prev]);
                              }, 50);
                            }}
                          />
                        );
                      })()}

                      {/* ⭐ ADD THIS: Render bounding boxes */}
                      {renderUploadBoundingBoxes()}

                      {/* ⭐ KEEP THIS: Analysis status overlay */}
                      {isAnalyzing && (
                        <div style={{
                          position: 'absolute',
                          top: '50%',
                          left: '50%',
                          transform: 'translate(-50%, -50%)',
                          background: 'rgba(0,0,0,0.8)',
                          color: 'white',
                          padding: '10px 20px',
                          borderRadius: '8px',
                          fontSize: '14px',
                          zIndex: 20  // ⭐ ADD zIndex
                        }}>
                          Analyzing fruits...
                        </div>
                      )}

                      {/* ⭐ REPLACE the results overlay with detection summary */}
                      {uploadDetections.length > 0 && (
                        <div style={{
                          position: 'absolute',
                          top: '10px',
                          right: '10px',
                          background: 'rgba(0,0,0,0.8)',
                          color: 'white',
                          padding: '8px 12px',
                          borderRadius: '6px',
                          fontSize: '12px',
                          fontWeight: '600',
                          zIndex: 20
                        }}>
                          {uploadDetections.length} fruit{uploadDetections.length > 1 ? 's' : ''} detected
                        </div>
                      )}
                    </div>
                  ) : (
                    <div style={styles.cameraPlaceholder}>
                      <div style={{ fontSize: '4rem', marginBottom: '15px', opacity: 0.5 }}>📁</div>
                      <h3>Upload Image</h3>
                      <p>Click "Choose File" to upload a fruit image</p>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Controls - Now below the large display */}
            <div style={styles.controls}>
              {uploadMode === 'camera' ? (
                // Camera Controls
                <>
                  <button
                    onClick={startCamera}
                    disabled={!!stream}
                    style={{
                      ...styles.btn,
                      ...styles.btnPrimary,
                      opacity: stream ? 0.5 : 1,
                      cursor: stream ? 'not-allowed' : 'pointer'
                    }}
                  >
                    🎯 {stream ? 'Smart Detection Active' : 'Start Smart Detection'}
                  </button>

                  <button
                    onClick={stopCamera}
                    disabled={!stream}
                    style={{
                      ...styles.btn,
                      ...styles.btnPrimary,
                      opacity: !stream ? 0.5 : 1,
                      cursor: !stream ? 'not-allowed' : 'pointer',
                      background: 'linear-gradient(135deg, #dc3545, #c82333)'
                    }}
                  >
                    🛑 Stop Detection
                  </button>
                </>
              ) : (
                // Upload Controls
                <>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                    id="file-upload-controls"
                  />

                  <label
                    htmlFor="file-upload-controls"
                    style={{
                      ...styles.btn,
                      ...styles.btnPrimary,
                      cursor: 'pointer'
                    }}
                  >
                    📁 {uploadedImage ? 'Choose Different File' : 'Choose File'}
                  </label>

                  {uploadedImage && (
                    <button
                      onClick={resetUpload}
                      style={{
                        ...styles.btn,
                        ...styles.btnPrimary,
                        background: 'linear-gradient(135deg, #dc3545, #c82333)'
                      }}
                    >
                      🗑️ Remove Image
                    </button>
                  )}

                  {uploadedImage && !isAnalyzing && (
                    <button
                      onClick={reAnalyzeImage}
                      style={{
                        ...styles.btn,
                        ...styles.btnSuccess
                      }}
                    >
                      📊 Re-analyze
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Results Section - Now full width at bottom */}
          <div style={styles.resultsSection}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '30px' }}>
              <h2 style={{ fontSize: '1.8rem', color: '#2c3e50', margin: 0, fontWeight: '700' }}>Analysis Results</h2>
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: (isAnalyzing || results) ? '#28a745' : '#dc3545',
                animation: (isAnalyzing || results) ? 'pulse 2s infinite' : 'none'
              }} />
            </div>

            {/* REMOVED: All debug info panels */}

            {results ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '25px' }}>
                {/* REMOVED: Debug logging console.log statements from JSX */}

                {/* Primary Result - Fruit Type */}
                <div style={{ ...styles.resultCard, marginBottom: 0 }}>
                  <div style={styles.resultLabel}>Fruit Type</div>
                  <div style={styles.resultValue}>
                    {results.type || 'Unknown'} ({results.variety || 'Unknown variety'})
                  </div>
                  <div style={styles.confidenceBar}>
                    <div
                      style={{
                        ...styles.confidenceFill,
                        width: `${results.confidence || 0}%`
                      }}
                    />
                  </div>
                  <small style={{ color: '#6c757d', fontSize: '12px', marginTop: '5px', display: 'block' }}>
                    {results.confidence || 0}% confidence
                  </small>
                </div>

                {/* Secondary Results Grid */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                  gap: '20px'
                }}>
                  <div style={{ ...styles.resultCard, marginBottom: 0 }}>
                    <div style={styles.resultLabel}>Shape Quality</div>
                    <div style={styles.resultValue}>
                      {results.shape_analysis?.quality || results.shape || 'Unknown'}
                    </div>
                  </div>
                  <div style={{ ...styles.resultCard, marginBottom: 0 }}>
                    <div style={styles.resultLabel}>Color Analysis</div>
                    <div style={styles.resultValue}>{results.color || 'Unknown'}</div>
                  </div>
                  <div style={{ ...styles.resultCard, marginBottom: 0 }}>
                    <div style={styles.resultLabel}>Ripeness Status</div>
                    <div style={styles.resultValue}>{results.ripeness || 'Unknown'}</div>
                  </div>
                  <div style={{ ...styles.resultCard, marginBottom: 0 }}>
                    <div style={styles.resultLabel}>Defect Detection</div>
                    <div style={styles.resultValue}>{results.defects || 'Unknown'}</div>
                  </div>
                </div>

                {/* Final Grade */}
                <div style={{
                  ...styles.resultCard,
                  marginBottom: 0,
                  background: 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                  border: 'none',
                  borderLeft: '6px solid #28a745'
                }}>
                  <div style={styles.resultLabel}>Quality Grade</div>
                  <div style={{
                    ...styles.resultValue,
                    fontSize: '2rem',
                    marginBottom: '10px'
                  }}>
                    {results.advanced_grading
                      ? `Grade ${results.advanced_grading.grade}`
                      : `Grade ${results.grade || 'Unknown'}`
                    }
                  </div>

                </div>

                {/* REMOVED: Debug raw results viewer */}
              </div>
            ) : (
              <div style={{
                textAlign: 'center',
                padding: '60px 40px',
                background: 'white',
                borderRadius: '15px',
                border: '2px dashed #dee2e6',
                margin: '0',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '200px'
              }}>
                <div style={{ fontSize: '3rem', marginBottom: '20px', opacity: 0.5 }}>📊</div>
                <h3 style={{
                  color: '#6c757d',
                  marginBottom: '15px',
                  fontSize: '1.2rem',
                  fontWeight: '600',
                  margin: '0 0 15px 0'
                }}>
                  {isAnalyzing ? 'Analyzing apple...' : 'No apple analysis available'}
                </h3>
                <p style={{
                  color: '#6c757d',
                  margin: 0,
                  fontSize: '14px',
                  maxWidth: '400px',
                  lineHeight: '1.5'
                }}>
                  {uploadMode === 'camera'
                    ? (() => {
                      // Check what's currently detected
                      const hasApples = detections.some(d => d.detection_type === 'apple_analysis');
                      const hasOtherFruits = detections.some(d => d.detection_type === 'simple_detection');

                      if (hasOtherFruits && !hasApples) {
                        return 'Other fruits detected - only apples get detailed analysis';
                      } else if (hasApples) {
                        return 'Point camera at an apple - detection and analysis will start automatically';
                      } else {
                        return 'Point camera at an apple for detailed analysis';
                      }
                    })()
                    : (() => {
                      // Check upload detections (same logic as camera)
                      const hasUploadApples = uploadDetections.some(d => d.detection_type === 'apple_analysis');
                      const hasUploadOtherFruits = uploadDetections.some(d => d.detection_type === 'simple_detection');

                      if (hasUploadOtherFruits && !hasUploadApples) {
                        return 'Other fruits detected - only apples get detailed analysis';
                      } else if (hasUploadApples) {
                        return 'Apple detected - analysis results shown above';
                      } else {
                        return 'Upload an apple image to see detailed analysis';
                      }
                    })()
                  }
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
      {/* 🔥 ADD THIS TOAST CONTAINER HERE 🔥 */}
      {toasts.length > 0 && (
        <div style={styles.toastContainer}>
          {toasts.map(toast => (
            <div
              key={toast.id}
              style={{
                ...styles.toast,
                ...(toast.type === 'error' ? styles.toastError : styles.toastWarning)
              }}
              onClick={() => removeToast(toast.id)}
            >
              <span>{toast.message}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  removeToast(toast.id);
                }}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '16px',
                  cursor: 'pointer',
                  color: 'inherit',
                  marginLeft: 'auto',
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default GradingSystem;