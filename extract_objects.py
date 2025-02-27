#!/usr/bin/env python3
import os
import json
import logging
import traceback
import re
import concurrent.futures
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from rembg import remove, new_session

# Import Google Gemini generative AI SDK
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_api_key_from_zshrc():
    """Try to extract the Google API key from .zshrc file if it exists."""
    zshrc_path = os.path.expanduser("~/.zshrc")
    if not os.path.exists(zshrc_path):
        return None
    
    try:
        with open(zshrc_path, 'r') as f:
            content = f.read()
            # Look for export GOOGLE_API_KEY=value
            match = re.search(r'export\s+GOOGLE_API_KEY=([^\s]+)', content)
            if match:
                api_key = match.group(1)
                # Remove quotes if present
                api_key = api_key.strip('\'"')
                return api_key
    except Exception as e:
        logging.error(f"Error reading .zshrc file: {e}")
    
    return None

def sanitize_filename(name):
    """Convert object name into a safe filename (lowercase, no spaces/special chars)."""
    fname = name.strip().lower()
    # Replace spaces with underscores
    fname = fname.replace(" ", "_")
    # Remove any characters that are not alphanumeric or underscore
    fname = "".join(ch for ch in fname if ch.isalnum() or ch == "_")
    if not fname:
        fname = "object"  # fallback name if name becomes empty
    return fname

def ensure_unique_filename(directory, base_name, extension=".png"):
    """If a file name already exists in the directory, append a number to make it unique."""
    candidate = base_name + extension
    index = 2
    # Loop until a name is found that isn't taken
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{base_name}_{index}{extension}"
        index += 1
    return candidate

def detect_objects_with_gemini(image_path, debug=False):
    """
    Use Google Gemini to detect objects in the image and return a list of 
    objects with names and bounding boxes. Requires a configured API key.
    """
    # Load the image using PIL
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Prepare an improved prompt for Gemini with clear instructions for structured output
    prompt = [
        "Identify every distinct object in this image and return a JSON array of their bounding boxes. "
        "For each object detected, include: "
        "\"object\": a short descriptive name (1-2 words) that clearly identifies what the object is, and "
        "\"bbox\": [ymin, xmin, ymax, xmax] as a simple array of 4 integers representing coordinates. "
        "The coordinates should be in the range 0-1000, where (0,0) is the top-left corner and (1000,1000) is the bottom-right corner. "
        "Make the bounding box slightly larger than the object itself to include some surrounding context. "
        "Example of expected output format: "
        "```json\n"
        "[{\"object\": \"coffee cup\", \"bbox\": [100, 150, 200, 250]}, "
        "{\"object\": \"laptop\", \"bbox\": [300, 400, 600, 800]}]\n"
        "```\n"
        "Return ONLY the JSON array with no additional text or explanation.",
        img  # the image is provided to the model
    ]
    
    logging.info(f"Sending prompt to Gemini for {os.path.basename(image_path)}")
    
    try:
        # Configure the model to use a low temperature for more deterministic results
        # and request JSON output format
        generation_config = genai.GenerationConfig(
            temperature=0.1,  # Low temperature for more deterministic results
            response_mime_type="application/json",  # Request JSON format
            top_p=0.95,  # High top_p for more focused responses
        )
        
        # Use Gemini 2.0 Flash for faster processing
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response = model.generate_content(prompt, generation_config=generation_config)
    except Exception as e:
        logging.error(f"Gemini API call failed for image {os.path.basename(image_path)}: {e}")
        return None  # indicating failure
    
    # Extract the text content from the response
    try:
        # The response structure has a content field with parts that contain text
        if hasattr(response, 'text'):
            # For newer API versions that have a text property
            content = response.text
        elif hasattr(response, 'parts'):
            # For some API versions
            content = ''.join([part.text for part in response.parts])
        elif hasattr(response, 'content') and hasattr(response.content, 'parts'):
            # For the structure seen in the error logs
            content = ''.join([part.get('text', '') for part in response.content.parts])
        else:
            # Fallback to string representation
            content = str(response)
        
        # Log the raw response for debugging if debug is enabled
        if debug:
            debug_dir = os.path.join(os.path.dirname(image_path), "processed", "debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_response_path = os.path.join(debug_dir, f"{os.path.basename(image_path)}_gemini_response.txt")
            with open(debug_response_path, 'w') as f:
                f.write(content)
            logging.info(f"Saved raw Gemini response to {debug_response_path}")
        
        # Clean up the JSON response to fix common issues
        # Replace single quotes with double quotes for valid JSON
        content = content.replace("'", "\"")
        # Remove trailing commas which are invalid in JSON
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # Now extract the JSON part from the content
        objects = []
        
        # Look for JSON array in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If not in code block, try to find array directly
            try:
                start = content.find('[')
                end = content.rfind(']')
                if start != -1 and end != -1 and end > start:
                    json_str = content[start:end+1]
                else:
                    logging.error(f"Could not find JSON array in response for {os.path.basename(image_path)}")
                    return None
            except Exception as e:
                logging.error(f"Failed to extract JSON from response: {e}")
                return None
        
        # Try to parse the JSON
        try:
            raw_objects = json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            logging.error(f"Problematic JSON: {json_str}")
            # Try to fix common JSON issues
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            try:
                raw_objects = json.loads(json_str)
            except json.JSONDecodeError:
                logging.error("Failed to parse JSON even after cleanup")
                return None
        
        # Process the raw objects to normalize the bbox format
        for obj in raw_objects:
            name = obj.get("object", "object")
            bbox = obj.get("bbox")
            
            # Handle different bbox formats
            if isinstance(bbox, list):
                # If bbox is already a list of coordinates
                if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                    # Format is already [ymin, xmin, ymax, xmax]
                    normalized_bbox = bbox
                else:
                    # If bbox is a list containing an object with named fields
                    if len(bbox) > 0 and isinstance(bbox[0], dict):
                        bbox_obj = bbox[0]
                        if all(k in bbox_obj for k in ["ymin", "xmin", "ymax", "xmax"]):
                            normalized_bbox = [
                                bbox_obj["ymin"],
                                bbox_obj["xmin"],
                                bbox_obj["ymax"],
                                bbox_obj["xmax"]
                            ]
                        else:
                            logging.warning(f"Skipping object with invalid bbox format: {bbox}")
                            continue
                    else:
                        logging.warning(f"Skipping object with unexpected bbox format: {bbox}")
                        continue
            elif isinstance(bbox, dict):
                # If bbox is directly an object with named fields
                if all(k in bbox for k in ["ymin", "xmin", "ymax", "xmax"]):
                    normalized_bbox = [
                        bbox["ymin"],
                        bbox["xmin"],
                        bbox["ymax"],
                        bbox["xmax"]
                    ]
                else:
                    logging.warning(f"Skipping object with invalid bbox keys: {bbox}")
                    continue
            else:
                logging.warning(f"Skipping object with unexpected bbox type: {type(bbox)}")
                continue
            
            # Add the normalized object to our results
            objects.append({
                "object": name,
                "bbox": normalized_bbox
            })
                
    except Exception as parse_err:
        logging.error(f"Failed to parse response from model for {os.path.basename(image_path)}: {parse_err}")
        logging.debug(f"Raw response: {response}")
        return None
        
    return objects  # list of dicts like {"object": "...", "bbox": [ymin, xmin, ymax, xmax]}

def preprocess_image(image):
    """
    Preprocess the image to improve segmentation results.
    Apply gentle denoising and contrast enhancement.
    """
    # Convert PIL image to OpenCV format
    img_cv = np.array(image)
    img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR for OpenCV
    
    # Apply gentle denoising
    img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 5, 5, 7, 21)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Convert back to PIL image
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil

def remove_background_color_based(image, color_tolerance=30, preserve_outlines=True, gradient_opacity=True):
    """
    Removes background based on color detection rather than AI segmentation.
    This method is better at preserving outlines and text.
    
    Args:
        image: PIL Image to process
        color_tolerance: How much variation in background color to allow (higher = more aggressive)
        preserve_outlines: Whether to specifically preserve dark outlines
        gradient_opacity: Whether to apply gradient-based opacity instead of binary transparency
        
    Returns:
        PIL Image with transparent background
    """
    # Convert PIL to OpenCV
    open_cv_image = np.array(image)
    # Convert RGB to BGR (OpenCV format)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    # Get image dimensions
    height, width = open_cv_image.shape[:2]
    
    # Sample background color from corners of the image
    # We'll take the median of the four corners to determine the background color
    corner_size = 10  # Sample from 10x10 pixel areas in each corner
    corners = [
        open_cv_image[:corner_size, :corner_size],  # top-left
        open_cv_image[:corner_size, -corner_size:],  # top-right
        open_cv_image[-corner_size:, :corner_size],  # bottom-left
        open_cv_image[-corner_size:, -corner_size:]  # bottom-right
    ]
    
    # Calculate median background color (BGR)
    bg_samples = np.vstack([corner.reshape(-1, 3) for corner in corners])
    bg_color = np.median(bg_samples, axis=0).astype(np.int32)
    
    logging.info(f"Detected background color (BGR): {bg_color}")
    
    # Vectorized approach for color distance calculation
    flat_image = open_cv_image.reshape(-1, 3).astype(np.int32)
    
    # Calculate Euclidean distance for each pixel from background color
    distances = np.sqrt(np.sum((flat_image - bg_color)**2, axis=1))
    
    if gradient_opacity:
        # Create a gradient alpha mask based on color distance
        # The further a color is from the background, the more opaque it will be
        
        # Normalize distances to create alpha values (0-255)
        # Set a lower threshold to ensure even slight color variations are preserved
        min_distance = 5  # Even slight deviations from background should have some opacity
        
        # Scale distances to 0-255 range for alpha channel
        # Using a non-linear (sqrt) scaling to emphasize small differences in color
        max_distance = float(np.max(distances)) if np.max(distances) > 0 else color_tolerance * 2
        # Apply non-linear scaling to make even subtle color differences more visible
        alpha_values = np.clip(
            np.power(distances / max(max_distance, color_tolerance), 0.5) * 255, 
            0, 255
        ).astype(np.uint8)
        
        # Set minimum alpha for colors that deviate at all from background
        alpha_values = np.where(
            distances > min_distance,
            np.maximum(alpha_values, 40),  # Minimum alpha for any color variation
            alpha_values
        )
        
        # Reshape alpha values to match image shape
        alpha = alpha_values.reshape(height, width)
    else:
        # Binary approach (original method)
        # Create mask: True for background pixels (where distance < tolerance)
        flat_mask = (distances < color_tolerance).astype(np.uint8) * 255
        
        # Reshape back to original image shape
        mask = flat_mask.reshape(height, width)
        
        if preserve_outlines:
            # Convert to grayscale to find edges/outlines
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            
            # Find edges using Canny edge detector
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to make them more prominent
            kernel = np.ones((2, 2), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Add a check for very dark regions (which are likely outlines/text)
            # This helps preserve text and dark outlines even if not detected by edge detection
            gray_dark_mask = (gray < 50).astype(np.uint8) * 255
            
            # Combine edge detection with dark region detection
            outline_mask = cv2.bitwise_or(dilated_edges, gray_dark_mask)
            
            # Preserve pixels that are part of edges/outlines (set to 0 in mask)
            mask[outline_mask > 0] = 0
        
        # Invert mask (now foreground is white, background is black)
        alpha = cv2.bitwise_not(mask)
    
    # Special handling for watercolor images - preserve even subtle color variations from background
    # Create special mask for darker/colored areas in watercolor/illustration images
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # For watercolor images, additionally increase opacity for:
    # 1. Areas with high saturation (colored areas)
    # 2. Areas significantly darker than background
    
    # First convert to HSV to check saturation
    hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    # Calculate grayscale value of background color
    bg_gray = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    
    # Areas darker than background should be more opaque
    darkness_diff = bg_gray - gray
    darkness_mask = (darkness_diff > 15).astype(np.uint8) * 255
    
    # Areas with saturation should be more opaque
    sat_mask = (saturation > 25).astype(np.uint8) * 255
    
    # Combine the masks
    special_mask = cv2.bitwise_or(darkness_mask, sat_mask)
    
    # Apply the special mask to boost alpha values
    alpha = np.maximum(alpha, special_mask)
    
    # Create alpha channel from the gradient mask
    bgra = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    
    # Convert back to PIL
    result_img = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
    
    # Apply a slight blur to the alpha channel to smooth edges
    r, g, b, a = result_img.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=0.5))
    result_img = Image.merge('RGBA', (r, g, b, a))
    
    return result_img

def is_line_art(image, edge_threshold=0.05, dark_threshold=50):
    """
    Detect if an image is likely to be line art, illustration, or contain significant outlines.
    This helps determine which background removal approach to prioritize.
    
    Args:
        image: PIL Image to check
        edge_threshold: Percentage of pixels that need to be edges to consider line art
        dark_threshold: Pixel value threshold to consider a pixel "dark" (0-255)
        
    Returns:
        Boolean indicating if image appears to be line art
    """
    # Convert to OpenCV format
    img_cv = np.array(image)
    img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Count edge pixels
    edge_count = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_ratio = edge_count / total_pixels
    
    # Count dark pixels (likely lines/text)
    dark_count = np.count_nonzero(gray < dark_threshold)
    dark_ratio = dark_count / total_pixels
    
    # Check if image has high edge density or significant dark lines
    is_illustration = (edge_ratio > edge_threshold) or (dark_ratio > 0.01)
    
    logging.info(f"Image analysis - Edge ratio: {edge_ratio:.4f}, Dark pixel ratio: {dark_ratio:.4f}")
    logging.info(f"Image is {'likely' if is_illustration else 'not likely'} to be line art/illustration")
    
    return is_illustration

def remove_background_hybrid(image, bg_color_tolerance=30, use_ai_mask=True, debug=False, debug_dir=None, name="object"):
    """
    Hybrid background removal that combines AI-based and color-based approaches
    to better preserve outlines while still removing backgrounds effectively.
    
    Args:
        image: PIL Image to process
        bg_color_tolerance: Tolerance for background color detection
        use_ai_mask: Whether to use rembg's AI-based mask
        debug: Whether to save debug information
        debug_dir: Directory to save debug info
        name: Name of the object for debug files
        
    Returns:
        PIL Image with transparent background
    """
    # Check if the image is likely line art or illustration
    line_art_mode = is_line_art(image)
    
    # Adjust color tolerance based on image type
    if line_art_mode:
        # Use higher tolerance for line art - we want to remove more background
        # while still preserving the lines
        bg_color_tolerance += 10
        logging.info(f"Line art detected: Increasing background color tolerance to {bg_color_tolerance}")
    
    # For watercolor images and line art, prioritize the gradient-based color removal
    # This provides smooth transitions at edges and preserves subtle color variations
    color_based_result = remove_background_color_based(
        image, 
        color_tolerance=bg_color_tolerance,
        preserve_outlines=True,
        gradient_opacity=True  # Enable gradient-based opacity
    )
    
    # For line art and watercolors, the color-based approach with gradient opacity 
    # may be sufficient without AI masking
    if line_art_mode and not use_ai_mask:
        return color_based_result
    
    # If we want to use AI (rembg) as well, create a session
    try:
        rembg_session = new_session("isnet-general-use")
    except Exception as e:
        logging.warning(f"Failed to initialize rembg session: {e}")
        return color_based_result
    
    # Use rembg to get an AI-generated mask
    try:
        # Get only the mask from rembg
        ai_mask = remove(
            image,
            session=rembg_session,
            only_mask=True
        )
        
        if debug and debug_dir:
            ai_mask_path = os.path.join(debug_dir, f"{name}_ai_mask.png")
            ai_mask.save(ai_mask_path)
            logging.info(f"Saved AI mask to {ai_mask_path}")
        
        # Convert masks to numpy arrays
        ai_mask_np = np.array(ai_mask)
        color_result_np = np.array(color_based_result)
        
        # Create a hybrid result - use color-based result but only keep pixels where AI says is foreground
        # This preserves outlines from the color-based approach
        hybrid_result = color_result_np.copy()
        
        # Set alpha scaling factor based on image type
        if line_art_mode:
            # For line art and watercolors, prioritize preserving all color variations
            # Don't let the AI mask remove too much
            alpha_scale = 0.9  # Higher preservation for line art/watercolors
            logging.info("Using very high outline preservation for line art/watercolor")
        else:
            # For photographic content, use moderate preservation
            alpha_scale = 0.3
        
        # Create a scaled alpha based on the AI mask
        # Where AI mask is white (255), keep full alpha
        # Where AI mask is black (0), reduce alpha by alpha_scale factor
        # For line art/watercolors, we want to preserve more even where AI disagrees
        hybrid_alpha = np.where(
            ai_mask_np < 128,  # Where AI thinks it's background
            (color_result_np[:, :, 3] * alpha_scale).astype(np.uint8),  # Reduce alpha but don't remove
            color_result_np[:, :, 3]  # Keep original alpha
        )
        
        # Update the alpha channel
        hybrid_result[:, :, 3] = hybrid_alpha
        
        # Convert back to PIL
        final_result = Image.fromarray(hybrid_result)
        
        if debug and debug_dir:
            color_result_path = os.path.join(debug_dir, f"{name}_color_based.png")
            hybrid_result_path = os.path.join(debug_dir, f"{name}_hybrid.png")
            color_based_result.save(color_result_path)
            final_result.save(hybrid_result_path)
            logging.info(f"Saved color-based and hybrid results for comparison")
        
        return final_result
    
    except Exception as e:
        logging.error(f"Error in hybrid background removal: {e}")
        return color_based_result  # Fall back to color-based approach

def process_image(image_path, output_dir, debug=False):
    """Process a single image: detect objects, crop them, remove background, and save results."""
    img_name = os.path.basename(image_path)
    logging.info(f"Processing {img_name}...")
    result_entry = []  # to store objects info for JSON output
    
    # Detect objects using Gemini
    detections = detect_objects_with_gemini(image_path, debug)
    if detections is None:
        # An error occurred (already logged), skip this image
        return None
    
    # Log raw detections for debugging
    logging.info(f"Gemini detected {len(detections)} objects in {img_name}:")
    for i, obj in enumerate(detections):
        logging.info(f"  - Object {i+1}: {obj}")
    
    # Open the image once for cropping (do in RGB mode)
    try:
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size
        logging.info(f"Image dimensions: {img_width}x{img_height}")
        
        # Preprocess the image to improve segmentation results
        preprocessed_image = preprocess_image(image)
        
        # Save original and preprocessed images for debugging if debug is enabled
        if debug:
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_original = os.path.join(debug_dir, f"original_{img_name}")
            image.save(debug_original)
            debug_preprocessed = os.path.join(debug_dir, f"preprocessed_{img_name}")
            preprocessed_image.save(debug_preprocessed)
            logging.info(f"Saved original and preprocessed images for debugging")
    except Exception as e:
        logging.error(f"Failed to open image {img_name}: {e}")
        return None
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create debug directory if debug is enabled
    debug_dir = None
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
    # Process each detected object
    for obj_index, obj in enumerate(detections):
        try:
            # Check if the object has the expected structure
            if not isinstance(obj, dict):
                logging.warning(f"Skipping invalid object format in {img_name}: {obj}")
                continue
                
            name = obj.get("object", "object")
            bbox = obj.get("bbox")
            
            if not bbox:
                logging.warning(f"Skipping object with missing bbox in {img_name}: {obj}")
                continue
                
            # Ensure bbox is a list with 4 elements
            if not isinstance(bbox, list) or len(bbox) != 4:
                logging.warning(f"Skipping object with invalid bbox format in {img_name}: {obj}")
                continue
                
            # Log raw bounding box
            logging.info(f"Object '{name}' raw bbox: {bbox}")
            
            # Convert all bbox values to integers and scale from 1000x1000 to actual image dimensions
            try:
                # Gemini returns coordinates in a 1000x1000 normalized space
                # We need to scale them to the actual image dimensions
                ymin_norm, xmin_norm, ymax_norm, xmax_norm = [float(val) for val in bbox]
                
                # Scale the normalized coordinates (0-1000) to actual image dimensions
                ymin = int((ymin_norm / 1000.0) * img_height)
                xmin = int((xmin_norm / 1000.0) * img_width)
                ymax = int((ymax_norm / 1000.0) * img_height)
                xmax = int((xmax_norm / 1000.0) * img_width)
                
                logging.info(f"Scaled bbox from normalized [ymin={ymin_norm}, xmin={xmin_norm}, ymax={ymax_norm}, xmax={xmax_norm}] to actual [ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax}]")
                
                # Add padding to the bounding box (5% of width/height)
                padding_x = int(0.05 * (xmax - xmin))
                padding_y = int(0.05 * (ymax - ymin))
                
                # Apply padding and ensure within image bounds
                xmin = max(0, xmin - padding_x)
                ymin = max(0, ymin - padding_y)
                xmax = min(img_width, xmax + padding_x)
                ymax = min(img_height, ymax + padding_y)
                
                # Calculate width and height for later use
                width = xmax - xmin
                height = ymax - ymin
                
                # Crop region is (left, top, right, bottom)
                crop_region = (xmin, ymin, xmax, ymax)
                
                logging.info(f"Padded crop region: {crop_region}")
                
            except (ValueError, TypeError):
                logging.warning(f"Skipping object with non-numeric bbox values in {img_name}: {obj}")
                continue
                
            # Skip if resulting crop region is too small
            if width <= 10 or height <= 10:
                logging.warning(f"Skipping {name} in {img_name} due to too small crop dimensions: {width}x{height}")
                continue
            
            # Save a debug visualization of the bounding box on the original image if debug is enabled
            if debug:
                debug_img = image.copy()
                try:
                    draw = ImageDraw.Draw(debug_img)
                    # Draw rectangle with the bounding box
                    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
                    # Write the object name above the box
                    draw.text((xmin, max(0, ymin-15)), name, fill="red")
                    # Save the debug image
                    debug_bbox_img = os.path.join(debug_dir, f"{img_name.split('.')[0]}_{obj_index}_{name}_bbox.png")
                    debug_img.save(debug_bbox_img)
                    logging.info(f"Saved bounding box visualization to {debug_bbox_img}")
                except Exception as e:
                    logging.error(f"Failed to create debug visualization: {e}")
                
            # Crop from the preprocessed image for better segmentation
            cropped_img = preprocessed_image.crop(crop_region)
            
            # Save a debug version of the cropped image if debug is enabled
            if debug:
                debug_crop = os.path.join(debug_dir, f"{img_name.split('.')[0]}_{obj_index}_{name}_crop.png")
                cropped_img.save(debug_crop)
                logging.info(f"Saved crop debug image to {debug_crop}")
            
            # Generate a safe filename for this object
            base_name = sanitize_filename(name)
            # If base_name is empty or generic, fall back to "object"
            if not base_name:
                base_name = "object"
            # Ensure unique naming in the output directory
            obj_filename = ensure_unique_filename(output_dir, base_name, extension=".png")
            obj_base = os.path.splitext(obj_filename)[0]  # filename without extension
            
            # Save the cropped object image (with background)
            cropped_path = os.path.join(output_dir, obj_filename)
            cropped_img.save(cropped_path, format="PNG")
            
            # Use our hybrid background removal method
            debug_id = f"{img_name.split('.')[0]}_{obj_index}_{name}" if debug else None
            output_img = remove_background_hybrid(
                cropped_img, 
                bg_color_tolerance=40,  # Slightly higher tolerance
                use_ai_mask=True,       # Use both AI and color-based approaches
                debug=debug,
                debug_dir=debug_dir,
                name=debug_id
            )
            
            # For comparison, also save the original rembg result if in debug mode
            if debug:
                try:
                    rembg_session = new_session("isnet-general-use")
                    rembg_output = remove(
                        cropped_img,
                        session=rembg_session,
                        alpha_matting=True,
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=10,
                        post_process_mask=True
                    )
                    rembg_path = os.path.join(debug_dir, f"{debug_id}_rembg.png")
                    rembg_output.save(rembg_path)
                    logging.info(f"Saved original rembg output to {rembg_path}")
                except Exception as e:
                    logging.error(f"Failed to save rembg comparison: {e}")
            
            # Save background-removed image if successful
            if output_img:
                final_name = obj_base + "-final.png"
                final_path = os.path.join(output_dir, final_name)
                output_img.save(final_path, format="PNG")
                logging.info(f"Saved background-removed image to {final_path}")
            
            # Record the object info for JSON (use original name and bbox)
            result_entry.append({
                "object": name,
                "bbox": [ymin, xmin, ymax, xmax]  # Store in the new format
            })
            logging.info(f" - Saved {obj_filename} (and background-removed version).")
        except Exception as e:
            logging.error(f"Error processing object in {img_name}: {e}")
            # Log stack trace for debugging
            logging.debug(traceback.format_exc())
            continue  # skip this object and move to next
    
    return (img_name, result_entry)

def main(input_dir, debug=False):
    # Default input directory is current working directory
    directory = input_dir or "."
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return 1  # non-zero exit for error
    
    output_dir = os.path.join(directory, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JPG/PNG images in the directory (non-recursively)
    images = [f for f in os.listdir(directory) 
              if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
    if not images:
        print("No images found in the specified directory.")
        return 0
    
    logging.info(f"Found {len(images)} images in '{directory}'. Beginning processing...")
    logging.info(f"Debug mode: {'enabled' if debug else 'disabled'}")
    
    # Configure Gemini API with the API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # If not found in environment, try to get it from .zshrc
    if not api_key:
        logging.info("GOOGLE_API_KEY not found in environment variables, trying to read from .zshrc...")
        api_key = get_api_key_from_zshrc()
        
        if api_key:
            logging.info("Found Google API key in .zshrc file.")
            # Set it in the environment for this process
            os.environ["GOOGLE_API_KEY"] = api_key
    
    if not api_key:
        logging.error("GOOGLE_API_KEY not found in environment variables or .zshrc file.")
        logging.info("Please set the GOOGLE_API_KEY environment variable or add it to your .zshrc file and restart your terminal.")
        return 1
    
    logging.info("Configuring Google Gemini API with the API key...")
    genai.configure(api_key=api_key)
    
    results = {}
    # Use ThreadPoolExecutor for parallel processing of images
    max_workers = min(os.cpu_count() or 4, 4)  # Limit to 4 workers to avoid API rate limits
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Launch parallel tasks
        future_to_img = {
            executor.submit(process_image, os.path.join(directory, img), output_dir, debug): img 
            for img in images
        }
        for future in concurrent.futures.as_completed(future_to_img):
            img = future_to_img[future]
            try:
                res = future.result()
            except Exception as exc:
                logging.error(f"Unhandled exception processing {img}: {exc}")
                logging.debug(traceback.format_exc())
                continue
            if res is None:
                # Already logged inside process_image if it failed
                continue
            img_name, objects = res
            results[img_name] = objects
    
    # Save bounding box results to JSON file
    if results:
        json_path = os.path.join(output_dir, "bounding_boxes.json")
        try:
            with open(json_path, "w") as jf:
                json.dump(results, jf, indent=4)
            logging.info(f"Bounding box data saved to {json_path}")
        except Exception as e:
            logging.error(f"Failed to write JSON output: {e}")
    else:
        logging.info("No objects detected in any image, JSON output not created.")
    
    # Print summary
    total_objects = sum(len(objs) for objs in results.values())
    logging.info(f"Processing complete. Extracted {total_objects} objects from {len(results)} images.")
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract objects from images using Google Gemini and remove backgrounds.")
    parser.add_argument("-i", "--input", type=str, default=".", help="Input directory containing images (PNG/JPG). Defaults to current directory.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode to save additional debug files and visualizations.")
    args = parser.parse_args()
    exit_code = main(args.input, args.debug)
    exit(exit_code)