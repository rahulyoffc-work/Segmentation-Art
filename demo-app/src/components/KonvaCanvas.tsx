import { useRef, useState, useEffect } from 'react';
import { Stage, Layer, Image as KonvaImage, Line, Circle, Rect, Transformer } from 'react-konva';
import type { KonvaEventObject } from 'konva/lib/Node';

interface Point {
  x: number;
  y: number;
}

interface BrushLine {
  points: number[];
  color: string;
  size: number;
  opacity: number;
  mode: string;
  type: 'normal' | 'soft' | 'stroke';
  hardness: number;
  spacing: number;
  innerRadius?: number;
  outerRadius?: number;
}

interface Selection {
  startX: number;
  startY: number;
  width?: number;
  height?: number;
  points?: number[];
  rotation?: number;
  scaleX?: number;
  scaleY?: number;
}

interface DetectedRegion {
  label: string;
  mask: string; // base64 PNG image of the mask
  bounds: { xmin: number; ymin: number; xmax: number; ymax: number };
}

interface Layer {
  id: string;
  name: string;
  url: string;
  thumbnail: string;
  visible: boolean;
  type: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface KonvaCanvasProps {
  width: number;
  height: number;
  brushSize: number;
  brushColor: string;
  mode: 'select' | 'rectangle' | 'lasso' | 'brush' | 'circle' | 'eraser';
  opacity: number;
  image: string | null;
  onExtract: (dataUrl: string, type: string, position?: { x: number; y: number; width: number; height: number }) => void;
  onImageUpdate?: (dataUrl: string) => void;
  brushType?: 'normal' | 'soft' | 'stroke';
  brushHardness?: number;
  brushSpacing?: number;
  featherAmount?: number;
  detectedRegions?: DetectedRegion[];
  hoveredRegion?: number | null;
  onHoverRegion?: (index: number | null) => void;
  onSelectionChange?: (hasSelection: boolean) => void;
  onExtractSelection?: () => void;
  layers?: Layer[];
  baseLayerVisible?: boolean;
}

export function KonvaCanvas({
  width,
  height,
  brushSize,
  brushColor,
  mode,
  opacity,
  image,
  onExtract,
  onImageUpdate,
  brushType = 'normal',
  brushHardness = 0.5,
  brushSpacing = 0.1,
  featherAmount = 0,
  detectedRegions = [],
  hoveredRegion = null,
  onHoverRegion,
  onSelectionChange,
  onExtractSelection,
  layers = [],
  baseLayerVisible = true
}: KonvaCanvasProps) {
  const [lines, setLines] = useState<BrushLine[]>([]);
  const [imageElement, setImageElement] = useState<HTMLImageElement | null>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [selection, setSelection] = useState<Selection | null>(null);
  const [maskImages, setMaskImages] = useState<Map<number, HTMLImageElement>>(new Map());
  const [overlayImages, setOverlayImages] = useState<Map<number, HTMLImageElement>>(new Map());
  const [outlineImages, setOutlineImages] = useState<Map<number, HTMLImageElement>>(new Map());
  const [layerImages, setLayerImages] = useState<Map<string, HTMLImageElement>>(new Map());
  const isDrawing = useRef(false);
  const lastPoint = useRef<Point | null>(null);
  const lastStrokeTime = useRef<number>(0);
  const stageRef = useRef<any>(null);
  const brushLayerRef = useRef<any>(null);
  const selectionLayerRef = useRef<any>(null);
  const transformerRef = useRef<any>(null);
  const shapeRef = useRef<any>(null);
  const currentLine = useRef<BrushLine | null>(null);
  const hoverCheckCanvas = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (image) {
      const img = new window.Image();
      img.src = image;
      img.onload = () => {
        const scale = Math.min(width / img.width, height / img.height);
        setImageSize({
          width: img.width * scale,
          height: img.height * scale
        });
        setImageElement(img);
      };
    }
  }, [image, width, height]);

  // Load mask images and pre-generate dashed outlines
  useEffect(() => {
    const loadMaskImages = async () => {
      const newMaskImages = new Map<number, HTMLImageElement>();
      const newOutlineImages = new Map<number, HTMLImageElement>();

      for (let i = 0; i < detectedRegions.length; i++) {
        const region = detectedRegions[i];
        if (region.mask) {
          const img = new window.Image();
          img.src = `data:image/png;base64,${region.mask}`;

          // Wait for image to load
          await new Promise<void>((resolve) => {
            img.onload = () => {
              newMaskImages.set(i, img);

              // Pre-generate blue dashed outline
              const canvas = document.createElement('canvas');
              canvas.width = img.width;
              canvas.height = img.height;
              const ctx = canvas.getContext('2d');

              if (ctx) {
                ctx.drawImage(img, 0, 0);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const data = imageData.data;

                // Edge detection: Find boundary pixels
                const edgeData = new Uint8ClampedArray(data.length);
                for (let y = 1; y < canvas.height - 1; y++) {
                  for (let x = 1; x < canvas.width - 1; x++) {
                    const idx = (y * canvas.width + x) * 4;
                    const isWhite = data[idx] > 200;

                    if (isWhite) {
                      // Check neighbors for edge detection
                      const neighbors = [
                        data[((y - 1) * canvas.width + x) * 4],     // top
                        data[((y + 1) * canvas.width + x) * 4],     // bottom
                        data[(y * canvas.width + (x - 1)) * 4],     // left
                        data[(y * canvas.width + (x + 1)) * 4]      // right
                      ];

                      const hasBlackNeighbor = neighbors.some(n => n < 200);

                      if (hasBlackNeighbor) {
                        // This is an edge pixel - make it blue
                        edgeData[idx] = 0;       // R
                        edgeData[idx + 1] = 150; // G
                        edgeData[idx + 2] = 255; // B
                        edgeData[idx + 3] = 255; // Alpha
                      }
                    }
                  }
                }

                // Apply dashed pattern (every 5 pixels)
                for (let i = 0; i < edgeData.length; i += 4) {
                  const x = (i / 4) % canvas.width;
                  const dashPattern = Math.floor(x / 5) % 2 === 0;
                  if (!dashPattern && edgeData[i + 3] > 0) {
                    edgeData[i + 3] = 0; // Make this part transparent for dash effect
                  }
                }

                const edgeImageData = new ImageData(edgeData, canvas.width, canvas.height);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.putImageData(edgeImageData, 0, 0);

                const outlineImg = new window.Image();
                outlineImg.src = canvas.toDataURL();
                outlineImg.onload = () => {
                  newOutlineImages.set(i, outlineImg);
                };
              }

              resolve();
            };
            img.onerror = () => {
              console.error(`Failed to load mask for region ${i}: ${region.label}`);
              resolve();
            };
          });
        }
      }

      setMaskImages(newMaskImages);
      // Wait a bit for outline images to load
      setTimeout(() => setOutlineImages(newOutlineImages), 100);
    };

    if (detectedRegions.length > 0) {
      loadMaskImages();
    } else {
      setMaskImages(new Map());
      setOutlineImages(new Map());
    }
  }, [detectedRegions]);

  // Listen for extract region event
  useEffect(() => {
    const handleExtractEvent = (e: any) => {
      const regionIndex = e.detail;
      if (typeof regionIndex === 'number') {
        extractRegion(regionIndex);
      }
    };

    window.addEventListener('extractRegion', handleExtractEvent);
    return () => window.removeEventListener('extractRegion', handleExtractEvent);
  }, [detectedRegions, maskImages, imageElement, imageSize]);

  // Load layer images when layers change
  useEffect(() => {
    const loadLayerImages = async () => {
      const newLayerImages = new Map<string, HTMLImageElement>();

      for (const layer of layers) {
        const img = new window.Image();
        img.src = layer.url;

        await new Promise<void>((resolve) => {
          img.onload = () => {
            newLayerImages.set(layer.id, img);
            resolve();
          };
          img.onerror = () => {
            console.error(`Failed to load layer image: ${layer.name}`);
            resolve();
          };
        });
      }

      setLayerImages(newLayerImages);
    };

    if (layers.length > 0) {
      loadLayerImages();
    } else {
      setLayerImages(new Map());
    }
  }, [layers]);

  const getImagePosition = () => {
    const offsetX = (width - imageSize.width) / 2;
    const offsetY = (height - imageSize.height) / 2;
    return { offsetX, offsetY };
  };

  const isPointInImage = (point: Point): boolean => {
    const { offsetX, offsetY } = getImagePosition();
    const x = point.x - offsetX;
    const y = point.y - offsetY;
    return x >= 0 && x <= imageSize.width && y >= 0 && y <= imageSize.height;
  };

  const handleDrawStart = (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
    e.evt.preventDefault();
    const stage = e.target.getStage();
    if (!stage) return;

    const pos = stage.getPointerPosition();
    if (!pos || !imageElement || !isPointInImage(pos)) return;

    // Handle select mode - extract hovered region
    if (mode === 'select' && hoveredRegion !== null && detectedRegions[hoveredRegion]) {
      extractRegion(hoveredRegion);
      return;
    }

    isDrawing.current = true;
    lastPoint.current = pos;
    lastStrokeTime.current = performance.now();

    if (mode === 'lasso') {
      setSelection({
        startX: pos.x,
        startY: pos.y,
        points: [pos.x, pos.y]
      });
      if (onSelectionChange) onSelectionChange(true);
    } else if (mode === 'brush' || mode === 'eraser') {
      const newLine = {
        points: [pos.x, pos.y],
        color: brushColor,
        size: brushSize,
        opacity: opacity,
        mode: mode,
        type: brushType,
        hardness: brushHardness,
        spacing: brushSpacing
      };
      currentLine.current = newLine;
      setLines(prev => [...prev, newLine]);
    }
  };

  const extractRegion = (regionIndex: number) => {
    const region = detectedRegions[regionIndex];
    const maskImg = maskImages.get(regionIndex);

    if (!region || !imageElement || !maskImg) {
      console.error('Cannot extract region: missing data');
      return;
    }

    console.log(`[Extract] Extracting "${region.label}"...`);
    console.log(`[Extract] Original image size: ${imageElement.width}x${imageElement.height}`);
    console.log(`[Extract] Mask size: ${maskImg.width}x${maskImg.height}`);

    // Use ORIGINAL image dimensions, not scaled display size
    const originalWidth = imageElement.width;
    const originalHeight = imageElement.height;

    // Create canvas for the full image at ORIGINAL size
    const canvas = document.createElement('canvas');
    canvas.width = originalWidth;
    canvas.height = originalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw the image at original size
    ctx.drawImage(imageElement, 0, 0, originalWidth, originalHeight);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Load and apply mask
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = maskImg.width;
    maskCanvas.height = maskImg.height;
    const maskCtx = maskCanvas.getContext('2d');

    if (!maskCtx) return;

    maskCtx.drawImage(maskImg, 0, 0);
    const maskData = maskCtx.getImageData(0, 0, maskImg.width, maskImg.height);

    // Debug: Analyze mask to understand format
    let whitePixels = 0, blackPixels = 0;
    for (let i = 0; i < maskData.data.length; i += 4) {
      const r = maskData.data[i];
      if (r > 128) whitePixels++;
      else blackPixels++;
    }
    console.log(`[Extract] Mask analysis for "${region.label}": white=${whitePixels}, black=${blackPixels}`);

    // Auto-detect if mask appears inverted (background labels should have more white than black)
    const isBackgroundRegion = region.label.toLowerCase().includes('background') ||
                                region.label.toLowerCase().includes('bg') ||
                                whitePixels > blackPixels * 3; // If >75% white, likely background

    if (isBackgroundRegion) {
      console.log(`[Extract] ⚠ Detected inverted mask for "${region.label}" - will invert logic`);
    }

    // Apply mask as alpha channel (only keep pixels where mask matches the region type)
    for (let y = 0; y < originalHeight; y++) {
      for (let x = 0; x < originalWidth; x++) {
        const imageIdx = (y * originalWidth + x) * 4;

        // Map image coords to mask coords
        const maskX = Math.floor((x / originalWidth) * maskImg.width);
        const maskY = Math.floor((y / originalHeight) * maskImg.height);
        const maskIdx = (maskY * maskImg.width + maskX) * 4;

        // Check if mask pixel is white (bright)
        const isWhite = maskData.data[maskIdx] > 128;

        // For normal regions: keep white pixels, remove black pixels
        // For inverted regions (background): keep black pixels, remove white pixels
        const shouldKeep = isBackgroundRegion ? !isWhite : isWhite;

        if (!shouldKeep) {
          imageData.data[imageIdx + 3] = 0; // Set alpha to 0 (transparent)
        }
      }
    }

    // Debug: Count resulting transparent vs opaque pixels
    let transparentPixels = 0, opaquePixels = 0;
    for (let i = 3; i < imageData.data.length; i += 4) {
      if (imageData.data[i] === 0) transparentPixels++;
      else opaquePixels++;
    }
    console.log(`[Extract] Result: transparent=${transparentPixels}, opaque=${opaquePixels}`);

    ctx.putImageData(imageData, 0, 0);

    // Export the FULL-SIZE layer (like Photoshop) - region stays in original position
    console.log(`[Extract] ✓ Extracted "${region.label}" (full layer: ${originalWidth}x${originalHeight}, ${opaquePixels} visible pixels)`);
    onExtract(canvas.toDataURL(), region.label, {
      x: 0,
      y: 0,
      width: originalWidth,
      height: originalHeight
    });

    // Now update the main canvas to remove the extracted region (make it transparent)
    const mainCanvas = document.createElement('canvas');
    mainCanvas.width = originalWidth;
    mainCanvas.height = originalHeight;
    const mainCtx = mainCanvas.getContext('2d');

    if (mainCtx && onImageUpdate) {
      // Draw current image at original size
      mainCtx.drawImage(imageElement, 0, 0, originalWidth, originalHeight);
      const mainImageData = mainCtx.getImageData(0, 0, mainCanvas.width, mainCanvas.height);

      // Make extracted region transparent
      for (let y = 0; y < originalHeight; y++) {
        for (let x = 0; x < originalWidth; x++) {
          const imageIdx = (y * originalWidth + x) * 4;

          // Map image coords to mask coords
          const maskX = Math.floor((x / originalWidth) * maskImg.width);
          const maskY = Math.floor((y / originalHeight) * maskImg.height);
          const maskIdx = (maskY * maskImg.width + maskX) * 4;

          // If mask pixel is white (part of extracted region), make transparent
          const isWhite = maskData.data[maskIdx] > 200;

          if (isWhite) {
            mainImageData.data[imageIdx + 3] = 0; // Set alpha to 0 (transparent)
          }
        }
      }

      mainCtx.putImageData(mainImageData, 0, 0);
      console.log(`[Extract] ✓ Updated main canvas - made "${region.label}" transparent`);

      // Update the main image
      onImageUpdate(mainCanvas.toDataURL());
    }
  };

  const handleDrawMove = (e: KonvaEventObject<MouseEvent | TouchEvent>) => {
    e.evt.preventDefault();
    const stage = e.target.getStage();

    const pos = stage?.getPointerPosition();
    if (!pos || !imageElement) return;

    // Handle select mode - detect hovered region by checking actual mask pixels
    if (mode === 'select' && detectedRegions.length > 0 && onHoverRegion) {
      if (!isPointInImage(pos)) {
        onHoverRegion(null);
        return;
      }

      const { offsetX, offsetY } = getImagePosition();

      // Convert mouse position to image coordinates
      const imageX = pos.x - offsetX;
      const imageY = pos.y - offsetY;

      // Find which region contains this point by checking actual mask pixels
      // Check in reverse order so smaller parts (nose) are prioritized over larger ones (skin)
      let foundRegion: number | null = null;
      for (let i = detectedRegions.length - 1; i >= 0; i--) {
        const maskImg = maskImages.get(i);
        if (!maskImg) continue;

        // Use cached canvas for faster pixel checking
        if (!hoverCheckCanvas.current) {
          hoverCheckCanvas.current = document.createElement('canvas');
        }
        const canvas = hoverCheckCanvas.current;
        canvas.width = maskImg.width;
        canvas.height = maskImg.height;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });

        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(maskImg, 0, 0);

          // Convert image coordinates to mask coordinates
          const maskX = Math.floor((imageX / imageSize.width) * maskImg.width);
          const maskY = Math.floor((imageY / imageSize.height) * maskImg.height);

          // Check if this pixel is part of the mask (white/opaque)
          if (maskX >= 0 && maskX < maskImg.width && maskY >= 0 && maskY < maskImg.height) {
            const pixelData = ctx.getImageData(maskX, maskY, 1, 1).data;
            const isWhite = pixelData[0] > 200 && pixelData[1] > 200 && pixelData[2] > 200;

            if (isWhite) {
              foundRegion = i;
              break;
            }
          }
        }
      }

      onHoverRegion(foundRegion);
      return;
    }

    if (!isDrawing.current || !lastPoint.current) return;
    if (!isPointInImage(pos)) return;

    if (mode === 'lasso') {
      setSelection(prev => {
        if (!prev || !prev.points) return prev;
        return {
          ...prev,
          points: [...prev.points, pos.x, pos.y]
        };
      });
    } else if (mode === 'brush' || mode === 'eraser') {
      if (!currentLine.current) return;

      setLines(prev => {
        const lastLine = prev[prev.length - 1];
        return [
          ...prev.slice(0, -1),
          {
            ...lastLine,
            points: [...lastLine.points, pos.x, pos.y]
          }
        ];
      });

      // Extract mask after each stroke
      if (brushLayerRef.current) {
        const canvas = brushLayerRef.current.getCanvas()._canvas;
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = width;
        maskCanvas.height = height;
        const ctx = maskCanvas.getContext('2d');
        if (ctx) {
          ctx.fillStyle = 'black';
          ctx.fillRect(0, 0, width, height);
          ctx.drawImage(canvas, 0, 0);
          onExtract(maskCanvas.toDataURL(), 'mask');
        }
      }
    }
  };

  const handleDrawEnd = () => {
    isDrawing.current = false;
    lastPoint.current = null;
    currentLine.current = null;
  };

  // Extract selection function (called externally)
  useEffect(() => {
    if (onExtractSelection) {
      (window as any).extractLassoSelection = () => {
        if (!selection || !selection.points || selection.points.length < 6) {
          console.warn('No valid lasso selection');
          return;
        }

        extractLassoSelection();
      };
    }
  }, [selection, onExtractSelection]);

  const extractLassoSelection = () => {
    if (!selection || !selection.points || !imageElement) return;

    const { offsetX, offsetY } = getImagePosition();

    // Use ORIGINAL image dimensions
    const originalWidth = imageElement.width;
    const originalHeight = imageElement.height;

    // Calculate scale factor between display size and original size
    const scaleX = originalWidth / imageSize.width;
    const scaleY = originalHeight / imageSize.height;

    console.log(`[Extract] Lasso extraction - Display: ${imageSize.width}x${imageSize.height}, Original: ${originalWidth}x${originalHeight}, Scale: ${scaleX}x${scaleY}`);

    // Create mask canvas at ORIGINAL size
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = originalWidth;
    maskCanvas.height = originalHeight;
    const maskCtx = maskCanvas.getContext('2d');

    if (!maskCtx) return;

    // Draw the lasso selection path as a mask, scaled to original dimensions
    maskCtx.fillStyle = 'white';
    maskCtx.beginPath();

    const points = selection.points;
    for (let i = 0; i < points.length; i += 2) {
      // Convert screen coordinates to image coordinates, then scale to original
      const x = (points[i] - offsetX) * scaleX;
      const y = (points[i + 1] - offsetY) * scaleY;
      if (i === 0) {
        maskCtx.moveTo(x, y);
      } else {
        maskCtx.lineTo(x, y);
      }
    }
    maskCtx.closePath();
    maskCtx.fill();

    // Get mask data at original size
    const maskData = maskCtx.getImageData(0, 0, originalWidth, originalHeight);

    // Create result canvas at ORIGINAL size
    const resultCanvas = document.createElement('canvas');
    resultCanvas.width = originalWidth;
    resultCanvas.height = originalHeight;
    const resultCtx = resultCanvas.getContext('2d');

    if (!resultCtx) return;

    // Draw the original image at original size
    resultCtx.drawImage(imageElement, 0, 0, originalWidth, originalHeight);
    const imageData = resultCtx.getImageData(0, 0, originalWidth, originalHeight);

    // Apply the mask - only keep pixels inside the lasso selection
    for (let i = 0; i < imageData.data.length; i += 4) {
      const maskAlpha = maskData.data[i]; // White = 255, Black = 0
      if (maskAlpha === 0) {
        // Outside selection - make transparent
        imageData.data[i + 3] = 0;
      }
    }

    resultCtx.putImageData(imageData, 0, 0);

    // Export the full-size image with transparency outside selection
    console.log(`[Extract] ✓ Lasso selection extracted (full layer: ${originalWidth}x${originalHeight})`);
    onExtract(resultCanvas.toDataURL(), 'Lasso Selection', {
      x: 0,
      y: 0,
      width: originalWidth,
      height: originalHeight
    });

    // Now update the main image to remove the extracted region (make it transparent)
    if (onImageUpdate) {
      const mainCanvas = document.createElement('canvas');
      mainCanvas.width = originalWidth;
      mainCanvas.height = originalHeight;
      const mainCtx = mainCanvas.getContext('2d');

      if (mainCtx) {
        // Draw current image at original size
        mainCtx.drawImage(imageElement, 0, 0, originalWidth, originalHeight);
        const mainImageData = mainCtx.getImageData(0, 0, mainCanvas.width, mainCanvas.height);

        // Make extracted region transparent using the same mask
        for (let i = 0; i < mainImageData.data.length; i += 4) {
          const maskAlpha = maskData.data[i]; // White = 255, Black = 0
          if (maskAlpha > 0) {
            // Inside selection - make transparent
            mainImageData.data[i + 3] = 0;
          }
        }

        mainCtx.putImageData(mainImageData, 0, 0);
        console.log('[Extract] ✓ Updated main canvas - removed lasso selection');

        // Update the main image
        onImageUpdate(mainCanvas.toDataURL());
      }
    }

    // Clear selection
    setSelection(null);
    if (onSelectionChange) onSelectionChange(false);
  };

  return (
    <Stage
      ref={stageRef}
      width={width}
      height={height}
      onMouseDown={handleDrawStart}
      onMouseMove={handleDrawMove}
      onMouseUp={handleDrawEnd}
      onMouseLeave={handleDrawEnd}
      onTouchStart={handleDrawStart}
      onTouchMove={handleDrawMove}
      onTouchEnd={handleDrawEnd}
      style={{
        background: `repeating-conic-gradient(#CCCCCC 0% 25%, #FFFFFF 0% 50%) 
                    50% / 16px 16px`,
        touchAction: 'none'
      }}
    >
      {/* Base Image Layer */}
      <Layer listening={false}>
        {imageElement && baseLayerVisible && (
          <KonvaImage
            image={imageElement}
            width={imageSize.width}
            height={imageSize.height}
            x={(width - imageSize.width) / 2}
            y={(height - imageSize.height) / 2}
            listening={false}
            perfectDrawEnabled={false}
          />
        )}
      </Layer>

      {/* Extracted Layers - Displayed above base image */}
      <Layer name="extracted-layers">
        {layers.map((layer) => {
          if (!layer.visible) return null;

          const layerImg = layerImages.get(layer.id);
          if (!layerImg) return null;

          const { offsetX, offsetY } = getImagePosition();

          return (
            <KonvaImage
              key={layer.id}
              image={layerImg}
              width={imageSize.width}
              height={imageSize.height}
              x={offsetX}
              y={offsetY}
              listening={false}
              perfectDrawEnabled={false}
            />
          );
        })}
      </Layer>
      <Layer ref={brushLayerRef} name="brush-layer">
        {lines.map((line, i) => (
          <Line
            key={i}
            points={line.points}
            stroke={line.color}
            strokeWidth={line.size}
            tension={0.5}
            lineCap="round"
            lineJoin="round"
            globalCompositeOperation={
              line.mode === 'eraser' ? 'destination-out' : 'source-over'
            }
            opacity={line.opacity}
            perfectDrawEnabled={false}
            listening={false}
          />
        ))}
      </Layer>
      {mode === 'lasso' && selection && selection.points && selection.points.length > 0 && (
        <Layer name="lasso-layer">
          <Line
            points={selection.points}
            stroke="#3b82f6"
            strokeWidth={2}
            dash={[5, 5]}
            closed={false}
            listening={false}
            perfectDrawEnabled={false}
          />
          <Line
            points={selection.points}
            fill="rgba(59, 130, 246, 0.3)"
            closed={true}
            listening={false}
            perfectDrawEnabled={false}
          />
        </Layer>
      )}
      {mode === 'select' && detectedRegions.length > 0 && hoveredRegion !== null && (
        <Layer name="regions-layer">
          {(() => {
            const { offsetX, offsetY } = getImagePosition();
            const outlineImg = outlineImages.get(hoveredRegion);

            if (!outlineImg) return null;

            return (
              <KonvaImage
                key={`mask-outline-${hoveredRegion}`}
                image={outlineImg}
                x={offsetX}
                y={offsetY}
                width={imageSize.width}
                height={imageSize.height}
                opacity={1}
                listening={false}
                perfectDrawEnabled={false}
              />
            );
          })()}
        </Layer>
      )}
    </Stage>
  );
}