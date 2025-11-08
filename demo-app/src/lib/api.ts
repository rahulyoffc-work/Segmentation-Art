// API Configuration
const HF_API_URL = 'https://api-inference.huggingface.co/models';
const HF_API_KEY = import.meta.env.VITE_HUGGING_FACE_API_KEY;
const PHOTOROOM_API_URL = 'https://sdk.photoroom.com/v1/segment';
const PHOTOROOM_API_KEY = import.meta.env.VITE_PHOTOROOM_API_KEY;
const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY;

// Model Endpoints - Only models actually used in Segment Art feature
const MODEL_ENDPOINTS = {
  'object-detection': 'facebook/detr-resnet-50',
  'face-parsing': 'jonathandinu/face-parsing',
  'face-detection': 'Xenova/detr-resnet-50', // For detecting if image contains faces
  'panoptic-segmentation': 'facebook/mask2former-swin-base-coco-panoptic' // For landscape/object segmentation
} as const;

// Enhanced rate limiting for dedicated endpoint (much more generous)
const API_CONFIG = {
  maxRetries: 3,
  retryDelay: 1000, // 1 second between retries
  maxRequestsPerMinute: 20, // More generous for dedicated endpoint
  requests: [] as number[],
  requestWindow: 60000, // 1 minute window
  timeout: 300000, // 5 minutes timeout for image generation
};

// Helper functions
function canMakeRequest(): boolean {
  const now = Date.now();
  API_CONFIG.requests = API_CONFIG.requests.filter(
    time => now - time < API_CONFIG.requestWindow
  );
  return API_CONFIG.requests.length < API_CONFIG.maxRequestsPerMinute;
}

function addRequest(): void {
  API_CONFIG.requests.push(Date.now());
}

async function waitForRateLimit(): Promise<void> {
  while (!canMakeRequest()) {
    console.log('Rate limit reached, waiting 5 seconds...');
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
}

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// API key validation
function validateApiKeys(): { hf: boolean; photoroom: boolean; openai: boolean } {
  const hfValid = HF_API_KEY &&
    HF_API_KEY.startsWith('hf_') &&
    HF_API_KEY.length > 30 &&
    !HF_API_KEY.includes('your_huggingface_api_key_here') &&
    !HF_API_KEY.includes('your_key_here');

  const photoroomValid = PHOTOROOM_API_KEY &&
    PHOTOROOM_API_KEY.length > 10 &&
    !PHOTOROOM_API_KEY.includes('your_key_here');

  const openaiValid = OPENAI_API_KEY &&
    OPENAI_API_KEY.startsWith('sk-') &&
    OPENAI_API_KEY.length > 10 &&
    !OPENAI_API_KEY.includes('your_key_here');

  console.log('API Key Validation:', {
    hf: hfValid ? 'Valid' : 'Invalid/Missing',
    hfKey: HF_API_KEY ? `${HF_API_KEY.substring(0, 8)}...` : 'Not found',
    photoroom: photoroomValid ? 'Valid' : 'Invalid/Missing',
    openai: openaiValid ? 'Valid' : 'Invalid/Missing'
  });

  return {
    hf: hfValid,
    photoroom: photoroomValid,
    openai: openaiValid
  };
}

// Object detection function
export async function detectObjects(imageBlob: Blob): Promise<Array<{ label: string; box: { xmin: number; ymin: number; xmax: number; ymax: number } }>> {
  const apiKeys = validateApiKeys();
  
  if (!apiKeys.hf) {
    throw new Error('Hugging Face API key is not configured or invalid. Please ensure it starts with "hf_" and has inference permissions.');
  }

  let retries = API_CONFIG.maxRetries;
  let lastError: Error | null = null;

  while (retries > 0) {
    try {
      await waitForRateLimit();
      addRequest();

      const response = await fetch(`${HF_API_URL}/${MODEL_ENDPOINTS['object-detection']}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${HF_API_KEY}`,
          'Content-Type': 'application/octet-stream'
        },
        body: imageBlob
      });

      if (!response.ok) {
        const error = await response.text();
        let errorMessage = 'Failed to detect objects';
        
        try {
          const errorData = JSON.parse(error);
          if (errorData.error) {
            if (errorData.error.includes('loading')) {
              throw new Error('Model is currently loading, please wait a moment and try again');
            }
            if (errorData.error.includes('rate limit') || errorData.error.includes('Rate limit')) {
              throw new Error('Rate limit exceeded, please wait before trying again');
            }
            if (errorData.error.includes('unauthorized') || errorData.error.includes('Invalid token') || errorData.error.includes('Invalid credentials')) {
              throw new Error('Invalid Hugging Face API token. Please check your token configuration and ensure it has inference permissions.');
            }
            errorMessage = errorData.error;
          }
        } catch (parseError) {
          if (response.status === 503) {
            throw new Error('Model is currently loading, please wait a moment and try again');
          }
          if (response.status === 429) {
            throw new Error('Rate limit exceeded, please wait before trying again');
          }
          if (response.status === 401 || response.status === 403) {
            throw new Error('Invalid Hugging Face API token. Please check your token configuration and ensure it has inference permissions.');
          }
        }

        throw new Error(errorMessage);
      }

      const detections = await response.json();
      
      if (!Array.isArray(detections)) {
        throw new Error('Invalid response format from object detection API');
      }
      
      return detections.map(detection => ({
        label: detection.label || 'unknown',
        box: {
          xmin: detection.box?.xmin || 0,
          ymin: detection.box?.ymin || 0,
          xmax: detection.box?.xmax || 1,
          ymax: detection.box?.ymax || 1
        }
      }));
    } catch (error) {
      console.error('Object detection attempt failed:', error);
      lastError = error instanceof Error ? error : new Error('An unexpected error occurred');
      
      if (error instanceof Error && error.message.includes('loading') && retries > 1) {
        console.log(`Model loading, waiting ${API_CONFIG.retryDelay}ms before retry...`);
        await delay(API_CONFIG.retryDelay);
        retries--;
        continue;
      } else if (error instanceof Error && error.message.includes('Rate limit') && retries > 1) {
        console.log(`Rate limited, waiting ${API_CONFIG.retryDelay}ms before retry...`);
        await delay(API_CONFIG.retryDelay);
        retries--;
        continue;
      }
      
      if (retries <= 1) {
        throw lastError;
      }
      
      retries--;
      await delay(API_CONFIG.retryDelay);
    }
  }

  throw lastError || new Error('Failed to detect objects after multiple retries');
}

// Face parsing using pixel-perfect segmentation masks
export async function segmentImage(imageBlob: Blob): Promise<{ masks: Array<{ label: string; mask: string; bounds: { xmin: number; ymin: number; xmax: number; ymax: number } }> }> {
  const apiKeys = validateApiKeys();

  if (!apiKeys.hf) {
    throw new Error('Hugging Face API key is not configured or invalid');
  }

  let retries = API_CONFIG.maxRetries;
  let lastError: Error | null = null;

  while (retries > 0) {
    try {
      await waitForRateLimit();
      addRequest();

      console.log('Calling face parsing API...');

      // Use face parsing model to get pixel-perfect masks
      const response = await fetch(`${HF_API_URL}/${MODEL_ENDPOINTS['face-parsing']}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${HF_API_KEY}`,
          'Content-Type': 'application/octet-stream'
        },
        body: imageBlob
      });

      if (!response.ok) {
        const error = await response.text();
        let errorMessage = 'Failed to parse face';

        try {
          const errorData = JSON.parse(error);
          if (errorData.error) {
            if (errorData.error.includes('loading')) {
              throw new Error('Model is currently loading, please wait a moment and try again');
            }
            if (errorData.error.includes('rate limit')) {
              throw new Error('Rate limit exceeded, please wait before trying again');
            }
            errorMessage = errorData.error;
          }
        } catch (parseError) {
          if (response.status === 503) {
            throw new Error('Model is currently loading, please wait a moment and try again');
          }
          if (response.status === 429) {
            throw new Error('Rate limit exceeded, please wait before trying again');
          }
        }

        throw new Error(errorMessage);
      }

      const result = await response.json();

      console.log('Face parsing response:', result);

      if (!Array.isArray(result)) {
        throw new Error('Invalid response format from face parsing API');
      }

      // Process the results - each item has label, score, and mask (base64 PNG)
      const masks = await Promise.all(result.map(async (item: any) => {
        const { label, mask } = item;

        if (!mask) {
          console.warn(`No mask for ${label}`);
          return null;
        }

        // mask is base64 encoded PNG - convert to data URL for bounds calculation
        const maskDataUrl = `data:image/png;base64,${mask}`;

        // Calculate bounds from mask image
        const bounds = await calculateBoundsFromMaskImage(maskDataUrl);

        return {
          label: label || 'unknown',
          mask: mask, // Return just the base64 string (no data URL prefix)
          bounds
        };
      }));

      // Filter out null entries
      const validMasks = masks.filter(m => m !== null) as Array<{ label: string; mask: string; bounds: { xmin: number; ymin: number; xmax: number; ymax: number } }>;

      console.log(`Successfully parsed ${validMasks.length} face regions`);

      return { masks: validMasks };
    } catch (error) {
      console.error('Face parsing attempt failed:', error);
      lastError = error instanceof Error ? error : new Error('Face parsing failed');

      if (error instanceof Error && error.message.includes('loading') && retries > 1) {
        console.log(`Model loading, waiting ${API_CONFIG.retryDelay * 2}ms before retry...`);
        await delay(API_CONFIG.retryDelay * 2);
        retries--;
        continue;
      }

      if (retries > 1) {
        await delay(API_CONFIG.retryDelay);
        retries--;
        continue;
      }

      throw lastError;
    }
  }

  throw lastError || new Error('Failed to segment image');
}

// Detect if image contains faces (to determine if we should use face parsing or general segmentation)
export async function detectImageType(imageBlob: Blob): Promise<'face' | 'landscape'> {
  const apiKeys = validateApiKeys();

  if (!apiKeys.hf) {
    // If no API key, default to landscape mode
    console.warn('No HF API key, defaulting to landscape mode');
    return 'landscape';
  }

  // Retry logic for object detection
  let retries = 2;
  while (retries > 0) {
    try {
      console.log('Detecting image type (face vs landscape)...');
      console.log(`Image blob size: ${imageBlob.size} bytes, type: ${imageBlob.type}`);

      // Add rate limiting before calling
      await waitForRateLimit();
      addRequest();

      // Use DETR object detection to find "person" objects
      // Convert blob to ArrayBuffer for HF API
      const arrayBuffer = await imageBlob.arrayBuffer();

      const response = await fetch(`${HF_API_URL}/${MODEL_ENDPOINTS['object-detection']}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${HF_API_KEY}`,
          'Content-Type': imageBlob.type || 'image/jpeg'
        },
        body: arrayBuffer
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Object detection failed with status ${response.status}:`, errorText);

        // Check if it's a model loading error
        if (response.status === 503 || (errorText && errorText.includes('loading'))) {
          console.log('Model is loading, retrying in 3 seconds...');
          await delay(3000);
          retries--;
          continue;
        }

        console.warn('Defaulting to landscape mode due to detection failure');
        return 'landscape';
      }

      const detections = await response.json();
      console.log('Raw detection response:', detections);

      console.log(`Object detection found ${detections.length} objects total`);

      // Check if any "person" detected with good confidence
      // Lower threshold to 0.7 for better face detection
      const personDetections = detections.filter((d: any) =>
        d.label === 'person' && d.score > 0.7
      );

      console.log(`Found ${personDetections.length} person detection(s)`);

      if (personDetections.length > 0) {
        // Check if person box is large (indicates portrait/face shot)
        const largestPerson = personDetections.reduce((largest: any, current: any) => {
          const currentSize = (current.box.xmax - current.box.xmin) * (current.box.ymax - current.box.ymin);
          const largestSize = (largest.box.xmax - largest.box.xmin) * (largest.box.ymax - largest.box.ymin);
          return currentSize > largestSize ? current : largest;
        });

        const personArea = (largestPerson.box.xmax - largestPerson.box.xmin) *
                           (largestPerson.box.ymax - largestPerson.box.ymin);

        console.log(`Largest person area: ${(personArea * 100).toFixed(1)}% of image`);
        console.log(`Person box: [${largestPerson.box.xmin.toFixed(3)}, ${largestPerson.box.ymin.toFixed(3)}, ${largestPerson.box.xmax.toFixed(3)}, ${largestPerson.box.ymax.toFixed(3)}]`);
        console.log(`Person confidence: ${(largestPerson.score * 100).toFixed(1)}%`);

        // Lower threshold to 20% for better face detection
        // Portraits typically have person taking 30-80% of image
        if (personArea > 0.20) {
          console.log('Detected FACE image (person area > 20%)');
          return 'face';
        } else {
          console.log(`Person too small (${(personArea * 100).toFixed(1)}% < 20%) - treating as LANDSCAPE`);
        }
      } else {
        console.log('No person detected - treating as LANDSCAPE');
      }

      console.log('Final classification: LANDSCAPE');
      return 'landscape';
    } catch (error) {
      console.error('Image type detection failed:', error);
      retries--;
      if (retries === 0) {
        return 'landscape'; // Default to landscape on error after all retries
      }
      console.log(`Retrying... (${retries} attempts left)`);
      await delay(1000);
    }
  }

  // If all retries failed
  return 'landscape';
}

// Panoptic segmentation for landscape/object images
export async function segmentLandscape(imageBlob: Blob): Promise<{ masks: Array<{ label: string; mask: string; bounds: { xmin: number; ymin: number; xmax: number; ymax: number } }> }> {
  const apiKeys = validateApiKeys();

  if (!apiKeys.hf) {
    throw new Error('Hugging Face API key is not configured or invalid');
  }

  let retries = API_CONFIG.maxRetries;
  let lastError: Error | null = null;

  while (retries > 0) {
    try {
      await waitForRateLimit();
      addRequest();

      console.log('Calling panoptic segmentation API for landscape/objects...');

      // Use Mask2Former for panoptic segmentation
      const response = await fetch(`${HF_API_URL}/${MODEL_ENDPOINTS['panoptic-segmentation']}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${HF_API_KEY}`,
          'Content-Type': 'application/octet-stream'
        },
        body: imageBlob
      });

      if (!response.ok) {
        const error = await response.text();
        let errorMessage = 'Failed to segment landscape';

        try {
          const errorData = JSON.parse(error);
          if (errorData.error) {
            if (errorData.error.includes('loading')) {
              throw new Error('Model is currently loading, please wait a moment and try again');
            }
            if (errorData.error.includes('rate limit')) {
              throw new Error('Rate limit exceeded, please wait before trying again');
            }
            errorMessage = errorData.error;
          }
        } catch (parseError) {
          if (response.status === 503) {
            throw new Error('Model is currently loading, please wait a moment and try again');
          }
          if (response.status === 429) {
            throw new Error('Rate limit exceeded, please wait before trying again');
          }
        }

        throw new Error(errorMessage);
      }

      const result = await response.json();

      console.log('Panoptic segmentation response:', result);

      // Mask2Former returns array of segments with label, score, and mask
      if (!Array.isArray(result)) {
        throw new Error('Invalid response format from panoptic segmentation API');
      }

      // Process the results - convert masks to our format
      const masks = await Promise.all(result.map(async (item: any) => {
        const { label, mask, score } = item;

        // Filter low confidence detections
        if (score < 0.5) {
          return null;
        }

        if (!mask) {
          console.warn(`No mask for ${label}`);
          return null;
        }

        // mask is base64 encoded PNG
        const maskDataUrl = `data:image/png;base64,${mask}`;

        // Calculate bounds from mask image
        const bounds = await calculateBoundsFromMaskImage(maskDataUrl);

        return {
          label: label || 'unknown',
          mask: mask, // Return just the base64 string
          bounds
        };
      }));

      // Filter out null entries
      const validMasks = masks.filter(m => m !== null) as Array<{ label: string; mask: string; bounds: { xmin: number; ymin: number; xmax: number; ymax: number } }>;

      console.log(`Successfully segmented ${validMasks.length} objects in landscape`);

      return { masks: validMasks };
    } catch (error) {
      console.error('Landscape segmentation attempt failed:', error);
      lastError = error instanceof Error ? error : new Error('Landscape segmentation failed');

      if (error instanceof Error && error.message.includes('loading') && retries > 1) {
        console.log(`Model loading, waiting ${API_CONFIG.retryDelay * 2}ms before retry...`);
        await delay(API_CONFIG.retryDelay * 2);
        retries--;
        continue;
      }

      if (retries > 1) {
        await delay(API_CONFIG.retryDelay);
        retries--;
        continue;
      }

      throw lastError;
    }
  }

  throw lastError || new Error('Failed to segment landscape');
}

// Smart segmentation that automatically detects image type and uses appropriate model
// Workflow:
// 1. Use object detection to classify image as face or landscape
// 2. If face → Use Jonathan's face parsing model (jonathandinu/face-parsing)
// 3. If landscape → Use panoptic segmentation model (Mask2Former)
export async function smartSegment(imageBlob: Blob): Promise<{
  masks: Array<{ label: string; mask: string; bounds: { xmin: number; ymin: number; xmax: number; ymax: number } }>;
  imageType: 'face' | 'landscape';
}> {
  console.log('Starting smart segmentation...');

  // STEP 1: Detect image type using object detection API
  console.log('Step 1: Running object detection to classify image type...');
  const imageType = await detectImageType(imageBlob);
  console.log(`Image classified as: ${imageType.toUpperCase()}`);

  let result;

  // STEP 2: Use appropriate segmentation model based on classification
  if (imageType === 'face') {
    // FACE MODE: Use Jonathan's face parsing model
    console.log('Step 2: Using face parsing model (jonathandinu/face-parsing)...');
    result = await segmentImage(imageBlob);
    console.log(`Face parsing complete! Found ${result.masks.length} facial features`);
  } else {
    // LANDSCAPE MODE: Use panoptic segmentation model
    console.log('Step 2: Using panoptic segmentation model (Mask2Former)...');
    result = await segmentLandscape(imageBlob);
    console.log(`Panoptic segmentation complete! Found ${result.masks.length} objects`);
  }

  console.log(`Smart segmentation complete: ${result.masks.length} regions found (${imageType} mode)`);

  return {
    masks: result.masks,
    imageType
  };
}

// Helper function to calculate bounding box from mask image
async function calculateBoundsFromMaskImage(maskDataUrl: string): Promise<{ xmin: number; ymin: number; xmax: number; ymax: number }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        resolve({ xmin: 0, ymin: 0, xmax: 1, ymax: 1 });
        return;
      }

      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, img.width, img.height);
      const pixels = imageData.data;

      let xmin = img.width, ymin = img.height, xmax = 0, ymax = 0;
      let hasPixels = false;

      // Find bounding box of non-transparent pixels
      for (let y = 0; y < img.height; y++) {
        for (let x = 0; x < img.width; x++) {
          const i = (y * img.width + x) * 4;
          const alpha = pixels[i + 3];

          if (alpha > 0) {
            hasPixels = true;
            xmin = Math.min(xmin, x);
            xmax = Math.max(xmax, x);
            ymin = Math.min(ymin, y);
            ymax = Math.max(ymax, y);
          }
        }
      }

      if (!hasPixels) {
        resolve({ xmin: 0, ymin: 0, xmax: 1, ymax: 1 });
        return;
      }

      // Normalize to 0-1 range
      resolve({
        xmin: xmin / img.width,
        ymin: ymin / img.height,
        xmax: xmax / img.width,
        ymax: ymax / img.height
      });
    };

    img.onerror = () => {
      resolve({ xmin: 0, ymin: 0, xmax: 1, ymax: 1 });
    };

    img.src = maskDataUrl;
  });
}

// Helper function to pre-fill transparent areas with average surrounding color
function preFillTransparentAreas(imageData: ImageData): ImageData {
  const { width, height, data } = imageData;
  const output = new ImageData(new Uint8ClampedArray(data), width, height);

  // Find transparent pixels and calculate average color from surroundings
  const transparentPixels: number[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      if (data[idx + 3] < 200) {
        transparentPixels.push(idx);
      }
    }
  }

  if (transparentPixels.length === 0) {
    return output;
  }

  console.log(`Pre-filling ${transparentPixels.length} transparent pixels with surrounding colors for DALL-E context`);

  // Sample surrounding colors
  const surroundingColors: { r: number; g: number; b: number }[] = [];

  for (const idx of transparentPixels.slice(0, Math.min(1000, transparentPixels.length))) {
    const x = (idx / 4) % width;
    const y = Math.floor(idx / 4 / width);

    // Sample in a ring around the pixel
    for (let radius = 3; radius <= 10; radius++) {
      for (let i = 0; i < 8; i++) {
        const angle = (Math.PI * 2 * i) / 8;
        const sampleX = Math.round(x + Math.cos(angle) * radius);
        const sampleY = Math.round(y + Math.sin(angle) * radius);

        if (sampleX >= 0 && sampleX < width && sampleY >= 0 && sampleY < height) {
          const sampleIdx = (sampleY * width + sampleX) * 4;
          if (data[sampleIdx + 3] >= 200) {
            surroundingColors.push({
              r: data[sampleIdx],
              g: data[sampleIdx + 1],
              b: data[sampleIdx + 2]
            });
          }
        }
      }

      if (surroundingColors.length > 50) break;
    }
  }

  // Calculate average color
  let fillColor = { r: 255, g: 255, b: 255 }; // Default white

  if (surroundingColors.length > 0) {
    fillColor = {
      r: Math.round(surroundingColors.reduce((sum, c) => sum + c.r, 0) / surroundingColors.length),
      g: Math.round(surroundingColors.reduce((sum, c) => sum + c.g, 0) / surroundingColors.length),
      b: Math.round(surroundingColors.reduce((sum, c) => sum + c.b, 0) / surroundingColors.length)
    };
    console.log(`Calculated average fill color: RGB(${fillColor.r}, ${fillColor.g}, ${fillColor.b})`);
  }

  // Fill transparent pixels with average color
  for (const idx of transparentPixels) {
    output.data[idx] = fillColor.r;
    output.data[idx + 1] = fillColor.g;
    output.data[idx + 2] = fillColor.b;
    output.data[idx + 3] = 255; // Make fully opaque
  }

  return output;
}

// Helper function to resize image to nearest valid DALL-E size (256, 512, 1024)
async function resizeImageForDallE(blob: Blob, preFill: boolean = true): Promise<{ blob: Blob; size: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(blob);

    img.onload = () => {
      URL.revokeObjectURL(url);

      // Determine the best DALL-E size (256, 512, or 1024)
      const maxDimension = Math.max(img.width, img.height);
      let targetSize: number;

      if (maxDimension <= 256) {
        targetSize = 256;
      } else if (maxDimension <= 512) {
        targetSize = 512;
      } else {
        targetSize = 1024;
      }

      // Create canvas and resize
      const canvas = document.createElement('canvas');
      canvas.width = targetSize;
      canvas.height = targetSize;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        reject(new Error('Failed to get canvas context'));
        return;
      }

      // Draw image centered and scaled to fit
      const scale = targetSize / maxDimension;
      const scaledWidth = img.width * scale;
      const scaledHeight = img.height * scale;
      const x = (targetSize - scaledWidth) / 2;
      const y = (targetSize - scaledHeight) / 2;

      if (preFill) {
        // SMART PRE-FILL: Don't fill with white first!
        // Draw image with transparency intact so we can analyze it
        ctx.drawImage(img, x, y, scaledWidth, scaledHeight);

        // Get imageData while transparency is still present
        const imageData = ctx.getImageData(0, 0, targetSize, targetSize);

        // Pre-fill transparent areas with average surrounding color
        const preFilledData = preFillTransparentAreas(imageData);

        // Clear canvas and fill background with white
        ctx.clearRect(0, 0, targetSize, targetSize);
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, targetSize, targetSize);

        // Put back the pre-filled image (now fully opaque with smart colors)
        ctx.putImageData(preFilledData, 0, 0);
      } else {
        // NO PRE-FILL: Just fill white and draw normally
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, targetSize, targetSize);
        ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
      }

      canvas.toBlob((resizedBlob) => {
        if (resizedBlob) {
          resolve({ blob: resizedBlob, size: targetSize });
        } else {
          reject(new Error('Failed to create resized blob'));
        }
      }, 'image/png');
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to load image'));
    };

    img.src = url;
  });
}

// Content-aware fill using OpenAI DALL-E Image Edit API
export async function contentAwareFill(params: {
  image: Blob;
  mask: Blob;
  prompt?: string;
  debug?: boolean; // Enable debug mode to return inspection data
}): Promise<string | { resultUrl: string; debugData: any }> {
  const apiKeys = validateApiKeys();

  if (!apiKeys.openai) {
    throw new Error('OpenAI API key is not configured or invalid. Please check your .env file.');
  }

  let retries = API_CONFIG.maxRetries;
  let lastError: Error | null = null;

  while (retries > 0) {
    try {
      // Resize images to valid DALL-E sizes
      // Pre-fill the image with surrounding colors for context
      const resizedImage = await resizeImageForDallE(params.image, true);
      // Don't pre-fill the mask - keep it as pure mask
      const resizedMask = await resizeImageForDallE(params.mask, false);

      const finalPrompt = params.prompt || 'smooth natural skin, realistic skin texture, seamless blend';

      // Create debug blob URLs if debug mode enabled
      const debugData = params.debug ? {
        imageUrl: URL.createObjectURL(resizedImage.blob),
        maskUrl: URL.createObjectURL(resizedMask.blob),
        size: `${resizedImage.size}x${resizedImage.size}`,
        prompt: finalPrompt,
        timestamp: new Date().toISOString()
      } : null;

      if (params.debug) {
        console.log(`DALL-E Input: ${resizedImage.size}x${resizedImage.size} | "${finalPrompt}"`);
        console.log('Image:', debugData.imageUrl);
        console.log('Mask:', debugData.maskUrl);
      }

      // Create FormData for multipart upload
      const formData = new FormData();
      formData.append('image', resizedImage.blob, 'image.png');
      formData.append('mask', resizedMask.blob, 'mask.png');
      formData.append('prompt', finalPrompt);
      formData.append('n', '1');
      formData.append('size', `${resizedImage.size}x${resizedImage.size}`);

      const response = await fetch('https://api.openai.com/v1/images/edits', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${OPENAI_API_KEY}`
        },
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('OpenAI DALL-E API error:', errorText);

        let errorMessage = 'Failed to fill transparent areas';

        try {
          const errorData = JSON.parse(errorText);
          if (errorData.error) {
            const errMsg = errorData.error.message || errorData.error;

            if (errMsg.includes('rate limit') || errMsg.includes('Rate limit')) {
              throw new Error('OpenAI rate limit exceeded. Please wait before trying again.');
            }
            if (errMsg.includes('quota') || errMsg.includes('billing')) {
              throw new Error('OpenAI quota exceeded. Please check your billing details at https://platform.openai.com/account/billing');
            }
            if (errMsg.includes('invalid') && errMsg.includes('key')) {
              throw new Error('Invalid OpenAI API key. Please check your configuration.');
            }
            if (errMsg.includes('content_policy')) {
              throw new Error('Content policy violation. Please try a different prompt or image.');
            }

            errorMessage = errMsg;
          }
        } catch (parseError) {
          if (response.status === 401) {
            throw new Error('Invalid OpenAI API key. Please check your configuration.');
          }
          if (response.status === 429) {
            throw new Error('OpenAI rate limit exceeded. Please wait before trying again.');
          }
          if (response.status === 402) {
            throw new Error('OpenAI quota exceeded. Please check your billing details.');
          }
        }

        throw new Error(errorMessage);
      }

      const data = await response.json();

      if (!data.data || !data.data[0] || !data.data[0].url) {
        throw new Error('Invalid response from OpenAI DALL-E API');
      }

      // Download the image from OpenAI and convert to blob URL to avoid CORS issues
      const imageUrl = data.data[0].url;
      console.log('Downloading image from OpenAI and converting to blob URL...');

      try {
        // Use fetch to download the image
        // Note: We can't use crossOrigin here because Azure doesn't send CORS headers
        // But we CAN download the blob and create a same-origin blob URL
        const imageResponse = await fetch(imageUrl);

        if (!imageResponse.ok) {
          console.error('Failed to download image from OpenAI:', imageResponse.status);
          // Fallback to direct URL
          console.log('Returning direct URL as fallback');
          return imageUrl;
        }

        const imageBlob = await imageResponse.blob();
        const blobUrl = URL.createObjectURL(imageBlob);

        // Return debug data if debug mode enabled
        if (params.debug && debugData) {
          return {
            resultUrl: blobUrl,
            debugData: debugData
          };
        }

        return blobUrl;
      } catch (fetchError) {
        console.error('Fetch error:', fetchError);

        // Return debug data if debug mode enabled
        if (params.debug && debugData) {
          return {
            resultUrl: imageUrl,
            debugData: debugData
          };
        }

        return imageUrl;
      }

    } catch (error) {
      console.error('Content-aware fill attempt failed:', error);
      lastError = error instanceof Error ? error : new Error('Failed to fill transparent areas');

      if (error instanceof Error && error.message.includes('rate limit') && retries > 1) {
        console.log(`Rate limited, waiting ${API_CONFIG.retryDelay * 2}ms before retry...`);
        await delay(API_CONFIG.retryDelay * 2);
        retries--;
        continue;
      }

      if (retries > 1) {
        await delay(API_CONFIG.retryDelay);
        retries--;
        continue;
      }

      throw lastError;
    }
  }

  throw lastError || new Error('Failed to fill transparent areas after multiple retries');
}

// Helper function to convert Blob to base64
async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// Helper function to apply Gaussian blur to mask for smoother edges
function applyGaussianBlur(imageData: ImageData, radius: number): ImageData {
  const { width, height, data } = imageData;
  const output = new ImageData(width, height);

  // Create Gaussian kernel
  const kernel: number[] = [];
  const sigma = radius / 3;
  let kernelSum = 0;

  for (let x = -radius; x <= radius; x++) {
    const value = Math.exp(-(x * x) / (2 * sigma * sigma));
    kernel.push(value);
    kernelSum += value;
  }

  // Normalize kernel
  for (let i = 0; i < kernel.length; i++) {
    kernel[i] /= kernelSum;
  }

  // Horizontal pass
  const temp = new Uint8ClampedArray(data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0, g = 0, b = 0, a = 0;

      for (let kx = -radius; kx <= radius; kx++) {
        const px = Math.min(Math.max(x + kx, 0), width - 1);
        const idx = (y * width + px) * 4;
        const weight = kernel[kx + radius];

        r += data[idx] * weight;
        g += data[idx + 1] * weight;
        b += data[idx + 2] * weight;
        a += data[idx + 3] * weight;
      }

      const idx = (y * width + x) * 4;
      temp[idx] = r;
      temp[idx + 1] = g;
      temp[idx + 2] = b;
      temp[idx + 3] = a;
    }
  }

  // Vertical pass
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let r = 0, g = 0, b = 0, a = 0;

      for (let ky = -radius; ky <= radius; ky++) {
        const py = Math.min(Math.max(y + ky, 0), height - 1);
        const idx = (py * width + x) * 4;
        const weight = kernel[ky + radius];

        r += temp[idx] * weight;
        g += temp[idx + 1] * weight;
        b += temp[idx + 2] * weight;
        a += temp[idx + 3] * weight;
      }

      const idx = (y * width + x) * 4;
      output.data[idx] = r;
      output.data[idx + 1] = g;
      output.data[idx + 2] = b;
      output.data[idx + 3] = a;
    }
  }

  return output;
}

// Helper function to dilate mask (expand edges)
function dilateMask(imageData: ImageData, iterations: number = 2): ImageData {
  const { width, height } = imageData;
  let current = new ImageData(new Uint8ClampedArray(imageData.data), width, height);

  for (let iter = 0; iter < iterations; iter++) {
    const output = new ImageData(width, height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;

        // Check 3x3 neighborhood
        let maxValue = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            const ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const nIdx = (ny * width + nx) * 4;
              maxValue = Math.max(maxValue, current.data[nIdx]);
            }
          }
        }

        output.data[idx] = maxValue;
        output.data[idx + 1] = maxValue;
        output.data[idx + 2] = maxValue;
        output.data[idx + 3] = 255;
      }
    }

    current = output;
  }

  return current;
}

// Create improved mask with feathering and smooth edges
export function createImprovedMask(params: {
  imageData: ImageData;
  featherRadius?: number;
  expandPixels?: number;
}): ImageData {
  const { imageData, featherRadius = 5, expandPixels = 2 } = params;
  const { width, height, data } = imageData;

  console.log(`Creating improved mask with feather radius: ${featherRadius}, expand: ${expandPixels}px`);

  // Create initial binary mask
  const maskData = new ImageData(width, height);

  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3];
    const isTransparent = alpha < 200;

    // White for transparent areas (to be filled), black for opaque
    if (isTransparent) {
      maskData.data[i] = 255;
      maskData.data[i + 1] = 255;
      maskData.data[i + 2] = 255;
    } else {
      maskData.data[i] = 0;
      maskData.data[i + 1] = 0;
      maskData.data[i + 2] = 0;
    }
    maskData.data[i + 3] = 255; // Full opacity
  }

  // Step 1: Expand mask slightly for better context
  let processedMask = maskData;
  if (expandPixels > 0) {
    processedMask = dilateMask(processedMask, expandPixels);
    console.log(`Expanded mask by ${expandPixels} pixels`);
  }

  // Step 2: Apply Gaussian blur for smooth feathered edges
  if (featherRadius > 0) {
    processedMask = applyGaussianBlur(processedMask, featherRadius);
    console.log(`Applied Gaussian blur with radius ${featherRadius}`);
  }

  return processedMask;
}

// Local content-aware fill using average color from surrounding pixels
export async function localContentAwareFill(imageUrl: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      try {
        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
          reject(new Error('Failed to get canvas context'));
          return;
        }

        // Draw image
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const pixels = imageData.data;

        // Find transparent regions and their surrounding colors
        const transparentPixels: number[] = [];
        const surroundingColors: { r: number; g: number; b: number }[] = [];

        // First pass: identify transparent pixels
        for (let y = 0; y < canvas.height; y++) {
          for (let x = 0; x < canvas.width; x++) {
            const idx = (y * canvas.width + x) * 4;
            const alpha = pixels[idx + 3];

            if (alpha < 200) {
              transparentPixels.push(idx);
            }
          }
        }

        if (transparentPixels.length === 0) {
          console.log('No transparent pixels found');
          resolve(imageUrl);
          return;
        }

        // Second pass: sample colors around each transparent pixel
        for (const idx of transparentPixels) {
          const x = (idx / 4) % canvas.width;
          const y = Math.floor(idx / 4 / canvas.width);

          // Sample pixels in a ring around the transparent pixel (radius 3-10 pixels)
          for (let radius = 3; radius <= 10; radius++) {
            const samples = 8; // Sample 8 points around the circle
            for (let i = 0; i < samples; i++) {
              const angle = (Math.PI * 2 * i) / samples;
              const sampleX = Math.round(x + Math.cos(angle) * radius);
              const sampleY = Math.round(y + Math.sin(angle) * radius);

              // Check bounds
              if (sampleX >= 0 && sampleX < canvas.width && sampleY >= 0 && sampleY < canvas.height) {
                const sampleIdx = (sampleY * canvas.width + sampleX) * 4;
                const sampleAlpha = pixels[sampleIdx + 3];

                // Only sample opaque pixels
                if (sampleAlpha >= 200) {
                  surroundingColors.push({
                    r: pixels[sampleIdx],
                    g: pixels[sampleIdx + 1],
                    b: pixels[sampleIdx + 2]
                  });
                }
              }
            }

            // If we found enough samples, stop
            if (surroundingColors.length > 50) break;
          }
        }

        if (surroundingColors.length === 0) {
          console.log('No surrounding colors found, using white');
          // Fill with white as fallback
          for (const idx of transparentPixels) {
            pixels[idx] = 255;
            pixels[idx + 1] = 255;
            pixels[idx + 2] = 255;
            pixels[idx + 3] = 255;
          }
        } else {
          // Calculate average color
          const avgR = Math.round(surroundingColors.reduce((sum, c) => sum + c.r, 0) / surroundingColors.length);
          const avgG = Math.round(surroundingColors.reduce((sum, c) => sum + c.g, 0) / surroundingColors.length);
          const avgB = Math.round(surroundingColors.reduce((sum, c) => sum + c.b, 0) / surroundingColors.length);

          console.log(`Filling ${transparentPixels.length} transparent pixels with average color RGB(${avgR}, ${avgG}, ${avgB})`);

          // Fill all transparent pixels with average color
          for (const idx of transparentPixels) {
            pixels[idx] = avgR;
            pixels[idx + 1] = avgG;
            pixels[idx + 2] = avgB;
            pixels[idx + 3] = 255; // Full opacity
          }
        }

        // Put modified image data back
        ctx.putImageData(imageData, 0, 0);

        // Convert to data URL
        const dataUrl = canvas.toDataURL('image/png');
        console.log('Local content-aware fill complete!');
        resolve(dataUrl);

      } catch (error) {
        reject(error);
      }
    };

    img.onerror = () => {
      reject(new Error('Failed to load image'));
    };

    img.src = imageUrl;
  });
}

// Background removal function
export async function removeBackground(imageBlob: Blob): Promise<string> {
  const apiKeys = validateApiKeys();
  
  if (!apiKeys.photoroom) {
    throw new Error('PhotoRoom API key is not configured or invalid');
  }

  try {
    const formData = new FormData();
    formData.append('image_file', imageBlob);

    const response = await fetch(PHOTOROOM_API_URL, {
      method: 'POST',
      headers: {
        'x-api-key': PHOTOROOM_API_KEY
      },
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('PhotoRoom API error:', errorText);
      
      if (response.status === 401) {
        throw new Error('Invalid PhotoRoom API key. Please check your API key configuration.');
      }
      if (response.status === 429) {
        throw new Error('PhotoRoom API rate limit exceeded. Please wait before trying again.');
      }
      if (response.status === 413) {
        throw new Error('Image file too large. Please use a smaller image.');
      }
      
      throw new Error(`PhotoRoom API error: ${response.status} - ${errorText}`);
    }

    const outputBlob = await response.blob();
    return URL.createObjectURL(outputBlob);
  } catch (error) {
    console.error('Background removal error:', error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('Failed to remove background. Please try again.');
  }
}

// Smart segmentation prompt parsing with OpenAI
export async function parseSegmentationPrompt(
  userPrompt: string,
  availableLabels: string[]
): Promise<{ labels: string[]; confidence: 'high' | 'medium' | 'low' | 'fallback' }> {
  const apiKeys = validateApiKeys();

  // If OpenAI is not available, return fallback signal
  if (!apiKeys.openai) {
    console.log('OpenAI API not available, using fallback matching');
    return { labels: [], confidence: 'fallback' };
  }

  try {
    const systemPrompt = `You are an AI assistant that helps segment facial features from images.
Available feature labels are: ${availableLabels.join(', ')}

Common mappings:
- "eyes" → l_eye, r_eye
- "eyebrows" or "brows" → l_brow, r_brow
- "ears" → l_ear, r_ear
- "lips" or "mouth" → mouth, u_lip, l_lip
- "face" → skin, nose, l_eye, r_eye, l_brow, r_brow, l_ear, r_ear, mouth, u_lip, l_lip
- "head" → everything including hair

Your task is to parse the user's natural language request and return ONLY the feature labels they want to segment.
Return your response as a JSON object with this structure:
{
  "labels": ["label1", "label2", ...],
  "confidence": "high" | "medium" | "low"
}

Examples:
Input: "segment the eyes"
Output: {"labels": ["l_eye", "r_eye"], "confidence": "high"}

Input: "remove everything except hair"
Output: {"labels": ["skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "neck", "cloth"], "confidence": "medium"}

Input: "get the nose and both ears"
Output: {"labels": ["nose", "l_ear", "r_ear"], "confidence": "high"}

Be intelligent about understanding variations, synonyms, and context.`;

    const response = await fetch(OPENAI_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: 'gpt-3.5-turbo', // Using 3.5-turbo for cost efficiency
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        temperature: 0.3, // Lower temperature for more consistent results
        max_tokens: 200,
        response_format: { type: 'json_object' } // Force JSON response
      })
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('OpenAI API error:', error);

      // Return fallback on API errors
      if (response.status === 401) {
        console.warn('Invalid OpenAI API key, using fallback');
      } else if (response.status === 429) {
        console.warn('OpenAI rate limit exceeded, using fallback');
      }
      return { labels: [], confidence: 'fallback' };
    }

    const data = await response.json();

    if (!data.choices || !data.choices[0] || !data.choices[0].message) {
      console.warn('Invalid OpenAI response format, using fallback');
      return { labels: [], confidence: 'fallback' };
    }

    const result = JSON.parse(data.choices[0].message.content);

    // Validate the result
    if (!result.labels || !Array.isArray(result.labels)) {
      console.warn('Invalid result format from OpenAI, using fallback');
      return { labels: [], confidence: 'fallback' };
    }

    // Filter to only include valid labels
    const validLabels = result.labels.filter((label: string) =>
      availableLabels.includes(label)
    );

    console.log(`OpenAI parsed prompt: "${userPrompt}" → [${validLabels.join(', ')}] (confidence: ${result.confidence})`);

    return {
      labels: validLabels,
      confidence: result.confidence || 'medium'
    };

  } catch (error) {
    console.error('Error parsing segmentation prompt with OpenAI:', error);
    return { labels: [], confidence: 'fallback' };
  }
}

