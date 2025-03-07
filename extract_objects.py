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
import shutil  # Add shutil for directory operations
import time  # Add time for sleep function

# Import Google Gemini generative AI SDK
import google.generativeai as genai

"""
Extract Objects - Background Removal Tool

This script extracts objects from images using Google Gemini AI for detection and 
specialized background removal techniques. It can operate in three modes:

1. All (default): Detect objects, crop them, and remove backgrounds
2. Detect only: Detect objects and crop them without removing backgrounds
3. Remove only: Remove backgrounds from existing cropped images

Background removal techniques:
1. Watercolor paintings (DEFAULT) - Advanced color-to-alpha transitions with edge preservation
   Optimized for watercolor images with fine line details on white backgrounds
2. Line art/illustrations - Enhanced edge preservation and color-based removal
3. General objects - Using rembg and a color-based hybrid approach

The script uses watercolor processing by default, which provides smooth transparency
transitions and preserves fine details. Use the --no-watercolor flag to disable this
and use content-type detection instead.

Examples:
  # Process all images in the current directory (detect objects and remove backgrounds)
  python extract_objects.py
  
  # Process images in a specific directory
  python extract_objects.py -i /path/to/images
  
  # Only detect and crop objects without removing backgrounds
  python extract_objects.py -i /path/to/images -m detect
  
  # Only remove backgrounds from existing cropped images
  python extract_objects.py -i /path/to/cropped_images -m remove
  
  # Specify a different output directory
  python extract_objects.py -i /path/to/images -o /path/to/output
  
  # Disable watercolor mode and use standard background removal
  python extract_objects.py -i /path/to/images --no-watercolor
  
  # Enable debug mode to save additional debug files and visualizations
  python extract_objects.py -i /path/to/images -d
  
  # Clear the output directory before processing
  python extract_objects.py -i /path/to/images -c
"""

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

def to_snake_case(name):
    """Convert a name to snake_case format."""
    # First lowercase everything
    name = name.lower()
    # Replace spaces and hyphens with underscores
    name = name.replace(" ", "_").replace("-", "_")
    # Remove any characters that are not alphanumeric or underscore
    name = "".join(ch for ch in name if ch.isalnum() or ch == "_")
    # Replace multiple consecutive underscores with a single one
    while "__" in name:
        name = name.replace("__", "_")
    # Remove leading/trailing underscores
    name = name.strip("_")
    if not name:
        name = "object"  # fallback name if name becomes empty
    return name

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
    objects with bounding boxes. Requires a configured API key.
    """
    # Maximum number of retries for rate limit errors
    max_retries = 3
    retry_count = 0
    base_delay = 5  # Base delay in seconds
    
    while retry_count <= max_retries:
        # Load the image using PIL
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Prepare a simplified prompt for Gemini that focuses only on object detection, not naming
        prompt = [
            "Identify every distinct object in this image and return a JSON array of their bounding boxes. "
            "For each object detected, include: "
            "\"bbox\": [ymin, xmin, ymax, xmax] as a simple array of 4 integers representing coordinates. "
            "The coordinates should be in the range 0-1000, where (0,0) is the top-left corner and (1000,1000) is the bottom-right corner. "
            "Make the bounding box slightly larger than the object itself to include some surrounding context. "
            "Example of expected output format: "
            "```json\n"
            "[{\"bbox\": [100, 150, 200, 250]}, "
            "{\"bbox\": [300, 400, 600, 800]}]\n"
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
                for obj_index, obj in enumerate(raw_objects):
                    # Initialize with a generic object name
                    name = f"object_{obj_index+1}"
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
                        "object": name,  # Use a generic name initially
                        "bbox": normalized_bbox
                    })
                
                # Successfully processed, return the objects
                return objects  # list of dicts like {"object": "object_1", "bbox": [ymin, xmin, ymax, xmax]}
                    
            except Exception as parse_err:
                logging.error(f"Failed to parse response from model for {os.path.basename(image_path)}: {parse_err}")
                logging.debug(f"Raw response: {response}")
                return None
                
        except Exception as e:
            error_message = str(e)
            
            # Check if this is a rate limit error (429)
            if "429" in error_message and "Resource has been exhausted" in error_message:
                retry_count += 1
                if retry_count <= max_retries:
                    # Calculate exponential backoff delay: base_delay * 2^retry_count
                    delay = base_delay * (2 ** (retry_count - 1))
                    logging.warning(f"Rate limit exceeded for {os.path.basename(image_path)}. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    logging.error(f"Max retries reached for rate limit on {os.path.basename(image_path)}.")
                    return None
            else:
                # For other errors, log and return None
                logging.error(f"Gemini API call failed for image {os.path.basename(image_path)}: {e}")
                return None  # indicating failure

def generate_object_name_for_image(crop_path, debug=False, debug_dir=None, original_image_path=None):
    """
    Generate a descriptive name for a single cropped image.
    
    Args:
        crop_path: Path to the cropped image
        debug: Whether to save debug information
        debug_dir: Directory to save debug information
        original_image_path: Path to the original image (for debug purposes)
        
    Returns:
        Generated name as string
    """
    # Maximum number of retries for rate limit errors
    max_retries = 3
    retry_count = 0
    base_delay = 5  # Base delay in seconds
    
    while retry_count <= max_retries:
        try:
            # Load the cropped image for better naming
            cropped_img = Image.open(crop_path)
            
            # Prepare a prompt for Gemini to name this specific object
            prompt = [
                "Look at this image and provide a short descriptive name (1-2 words) that clearly identifies what this object is. "
                "Return ONLY the name as plain text, with no additional text, quotes, or formatting.",
                cropped_img  # the cropped image is provided to the model
            ]
            
            # Configure the model
            generation_config = genai.GenerationConfig(
                temperature=0.2,  # Slightly higher temperature for more creative naming
                top_p=0.95,
            )
            
            # Use Gemini 2.0 Flash for faster processing
            model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            response = model.generate_content(prompt, generation_config=generation_config)
            
            # Extract the text content from the response
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'parts'):
                content = ''.join([part.text for part in response.parts])
            elif hasattr(response, 'content') and hasattr(response.content, 'parts'):
                content = ''.join([part.get('text', '') for part in response.content.parts])
            else:
                content = str(response)
            
            # Clean up the response (remove quotes, newlines, etc.)
            name = content.strip().strip('"\'').strip()
            
            # Convert the name to snake_case
            name = to_snake_case(name)
            
            # Log the raw response for debugging if debug is enabled
            if debug and debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                debug_response_path = os.path.join(debug_dir, f"{os.path.basename(crop_path)}_naming_response.txt")
                with open(debug_response_path, 'w') as f:
                    f.write(f"Original: {content}\nSnake case: {name}")
                logging.info(f"Saved naming response to {debug_response_path}")
            
            return name if name else None
                
        except Exception as e:
            error_message = str(e)
            
            # Check if this is a rate limit error (429)
            if "429" in error_message and "Resource has been exhausted" in error_message:
                retry_count += 1
                if retry_count <= max_retries:
                    # Calculate exponential backoff delay: base_delay * 2^retry_count
                    delay = base_delay * (2 ** (retry_count - 1))
                    logging.warning(f"Rate limit exceeded for {crop_path}. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    logging.error(f"Max retries reached for rate limit on {crop_path}. Using fallback name.")
                    # Return a fallback name based on the filename
                    base_name = os.path.basename(crop_path)
                    fallback_name = os.path.splitext(base_name)[0]
                    return fallback_name
            else:
                # For other errors, log and return None
                logging.error(f"Error generating name for {crop_path}: {e}")
                logging.debug(traceback.format_exc())
                return None

def generate_object_names(image_path, cropped_images, debug=False):
    """
    Use Google Gemini to generate descriptive names for detected objects.
    
    Args:
        image_path: Path to the original image
        cropped_images: List of dicts with 'object' (generic name), 'crop_path' (path to cropped image)
        debug: Whether to save debug information
        
    Returns:
        Updated list of cropped_images with descriptive names in 'object' field
    """
    if not cropped_images:
        return cropped_images
    
    logging.info(f"Generating descriptive names for {len(cropped_images)} objects...")
    
    # Create debug directory if needed
    debug_dir = None
    if debug:
        debug_dir = os.path.join(os.path.dirname(image_path), "processed", "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
    # Use ThreadPoolExecutor for parallel processing of naming
    # Reduce workers to 2 to help prevent rate limiting
    max_workers = 2  # Reduced from 4 to 2 to avoid API rate limits
    
    # Prepare a list of tasks for parallel execution
    tasks = []
    for obj in cropped_images:
        crop_path = obj.get("crop_path")
        if not crop_path or not os.path.exists(crop_path):
            logging.warning(f"Skipping naming for object with missing or invalid crop path: {obj}")
            continue
        tasks.append((obj, crop_path))
    
    # Process naming in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a mapping of futures to their corresponding objects
        future_to_obj = {
            executor.submit(
                generate_object_name_for_image, 
                crop_path, 
                debug, 
                debug_dir, 
                image_path
            ): (i, obj) 
            for i, (obj, crop_path) in enumerate(tasks)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_obj):
            i, obj = future_to_obj[future]
            try:
                name = future.result()
                if name:
                    obj["object"] = name
                    # Also update original_name if it exists
                    if "original_name" in obj:
                        obj["original_name"] = name
                    logging.info(f"Named object {i+1}: {name}")
                else:
                    logging.warning(f"Empty name received for object {i+1}")
            except Exception as e:
                logging.error(f"Error in naming task for object {i+1}: {e}")
    
    return cropped_images

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

def remove_background_color_based(
    image,
    color_tolerance=30,
    preserve_outlines=True,
    gradient_opacity=True,
    center_feather=True,
    feather_strength=0.4
):
    """
    Removes background based on color detection rather than AI segmentation,
    with an optional smooth feathering from the center outward for watercolor edges.

    Args:
        image: PIL Image to process
        color_tolerance: How much variation in background color to allow (higher = more aggressive)
        preserve_outlines: Whether to specifically preserve dark outlines
        gradient_opacity: Whether to apply gradient-based opacity (based on color difference)
        center_feather: Whether to fade alpha outward from the center of the image (for watercolors)
        feather_strength: How strong the center-based fade is (0 = no fade, higher = more fade)

    Returns:
        PIL Image (RGBA) with transparent background
    """
    # Convert PIL to OpenCV
    open_cv_image = np.array(image)
    # Convert RGB to BGR (OpenCV format)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    # Get image dimensions
    height, width = open_cv_image.shape[:2]
    
    # Sample background color from corners: median BGR value
    corner_size = 10
    corners = [
        open_cv_image[:corner_size, :corner_size],       # top-left
        open_cv_image[:corner_size, -corner_size:],      # top-right
        open_cv_image[-corner_size:, :corner_size],      # bottom-left
        open_cv_image[-corner_size:, -corner_size:]      # bottom-right
    ]
    bg_samples = np.vstack([corner.reshape(-1, 3) for corner in corners])
    bg_color = np.median(bg_samples, axis=0).astype(np.int32)

    # Flatten and compute color distances
    flat_image = open_cv_image.reshape(-1, 3).astype(np.int32)
    distances = np.sqrt(np.sum((flat_image - bg_color)**2, axis=1))

    if gradient_opacity:
        # Create alpha channel from distance to background color
        max_distance = float(np.max(distances)) if np.max(distances) > 0 else color_tolerance * 2
        alpha_values = np.power(distances / max(max_distance, color_tolerance), 0.5) * 255
        alpha_values = np.clip(alpha_values, 0, 255).astype(np.uint8)

        # (Optional) Ensure we never go fully transparent for slight color differences
        min_distance = 5
        alpha_values = np.where(
            distances > min_distance,
            np.maximum(alpha_values, 40),
            alpha_values
        )
        alpha = alpha_values.reshape(height, width)
    else:
        # Simple binary mask if gradient_opacity is False
        flat_mask = (distances < color_tolerance).astype(np.uint8) * 255
        mask = flat_mask.reshape(height, width)
        if preserve_outlines:
            # Same preserve-outlines logic as before...
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((2, 2), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            gray_dark_mask = (gray < 50).astype(np.uint8) * 255
            outline_mask = cv2.bitwise_or(dilated_edges, gray_dark_mask)
            mask[outline_mask > 0] = 0
        alpha = cv2.bitwise_not(mask)

    # Extra watercolor logic: find pixels with significant saturation or that are much darker
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    bg_gray = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    darkness_diff = bg_gray - gray
    darkness_mask = (darkness_diff > 15).astype(np.uint8) * 255
    sat_mask = (saturation > 25).astype(np.uint8) * 255
    special_mask = cv2.bitwise_or(darkness_mask, sat_mask)
    alpha = np.maximum(alpha, special_mask)

    # Convert to BGRA
    bgra = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    # -----------------------------------
    # 1) Optional: Feather from center outward
    # -----------------------------------
    if center_feather and feather_strength > 0:
        # For simplicity, use the middle of the entire image as the "center"
        # If you have an object bounding box, you could use the center of that instead
        center_x, center_y = width // 2, height // 2

        # Create meshgrid of pixel coordinates
        y_indices, x_indices = np.indices((height, width))
        dist_from_center = np.sqrt(
            (x_indices - center_x)**2 + (y_indices - center_y)**2
        )
        max_dist = np.sqrt(
            (width/2.0)**2 + (height/2.0)**2
        )

        # This factor goes from 1.0 at the center to (1 - feather_strength) at the edges
        # If feather_strength=0.4, we get ~0.6 at the edge. You can tweak the exponent for a steeper curve
        # E.g. factor = 1.0 - feather_strength * (dist_from_center / max_dist)**1.3
        factor = 1.0 - feather_strength * (dist_from_center / max_dist)
        factor = np.clip(factor, 0, 1)

        # Multiply existing alpha by this radial factor
        # Cast alpha to float for multiplication, then back to uint8
        alpha_float = bgra[:, :, 3].astype(np.float32)
        alpha_float *= factor
        bgra[:, :, 3] = alpha_float.clip(0, 255).astype(np.uint8)

    # Convert back to PIL RGBA
    result_img = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))

    # Slight blur on alpha channel
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

def is_watercolor(image, saturation_threshold=10, color_variation_threshold=0.10):
    """
    Detect if an image is likely to be a watercolor painting.
    Optimized for watercolor images with fine line details on white backgrounds.
    
    Watercolors typically have:
    1. Moderate saturation values (not too high like digital art, not too low like grayscale)
    2. Smooth color transitions (less sharp edges than line art)
    3. Areas with subtle color variations
    4. Often have fine line details (pencil outlines)

    Args:
        image: PIL Image to check
        saturation_threshold: Minimum average saturation to consider as watercolor
        color_variation_threshold: Threshold for color variation (standard deviation)

    Returns:
        Boolean indicating if image appears to be a watercolor
    """
    # Convert to OpenCV format
    img_cv = np.array(image)
    img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Extract saturation channel
    saturation = hsv[:, :, 1]
    
    # Calculate mean saturation (excluding background)
    # Assuming background is near-white, we'll consider only pixels with some minimal saturation
    mask = saturation > 5  # Minimal saturation to exclude white background
    if np.sum(mask) > 0:  # If there are non-background pixels
        mean_saturation = np.mean(saturation[mask])
    else:
        mean_saturation = 0
    
    # Calculate color variation (standard deviation in each channel)
    rgb_std = np.std(img_cv, axis=(0, 1))
    avg_color_variation = np.mean(rgb_std) / 255.0  # Normalize to 0-1 range
    
    # Check for edge smoothness (watercolors have fewer sharp edges than line art)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_ratio = edge_count / total_pixels
    
    # Get lightness channel to check for soft transitions
    lightness = hsv[:, :, 2]
    lightness_gradient = cv2.Sobel(lightness, cv2.CV_64F, 1, 1, ksize=3)
    mean_gradient = np.mean(np.abs(lightness_gradient))
    
    # Check for fine line details (pencil outlines)
    # These are typically dark, thin lines
    dark_pixels = np.count_nonzero(gray < 100)  # Count pixels darker than 100 (0-255 scale)
    dark_ratio = dark_pixels / total_pixels
    
    # Calculate percentage of white/near-white pixels (background)
    white_pixels = np.count_nonzero(gray > 240)  # Count pixels lighter than 240
    white_ratio = white_pixels / total_pixels
    
    # For the specific type of watercolor images described (with fine line details on white background):
    # 1. Should have some moderate saturation
    # 2. Should have some color variation
    # 3. Should have a significant amount of white/near-white background
    # 4. May have some fine dark lines
    
    # Relaxed criteria specifically for watercolor illustrations on white background
    is_watercolor_image = (
        # Some color saturation present
        (mean_saturation > saturation_threshold) and
        # Some color variation
        (avg_color_variation > color_variation_threshold) and
        # Significant white/near-white background
        (white_ratio > 0.4) and
        # Either has smooth transitions OR has fine line details
        ((mean_gradient < 5.0) or (dark_ratio > 0.005 and dark_ratio < 0.1))
    )
    
    # For the image you shared, we want to be more inclusive
    # If it has white background, some color, and either smooth transitions or fine lines,
    # we'll consider it a watercolor
    
    logging.info(f"Watercolor analysis - Mean saturation: {mean_saturation:.2f}, Color variation: {avg_color_variation:.4f}")
    logging.info(f"Watercolor analysis - Edge ratio: {edge_ratio:.4f}, Mean gradient: {mean_gradient:.2f}")
    logging.info(f"Watercolor analysis - White ratio: {white_ratio:.4f}, Dark ratio: {dark_ratio:.4f}")
    logging.info(f"Image is {'likely' if is_watercolor_image else 'not likely'} to be a watercolor")
    
    return is_watercolor_image

def remove_watercolor_background(image, debug=False, debug_dir=None, name="watercolor"):
    """
    Remove background from a watercolor image with smooth transparency and edge preservation.
    Specifically optimized for watercolor images with fine line details on white backgrounds.
    
    Args:
        image: PIL Image to process
        debug: Whether to save debug images
        debug_dir: Directory to save debug images
        name: Base name for debug images
        
    Returns:
        PIL Image (RGBA) with transparent background
    """
    # Ensure we have an RGBA image to work with
    image = image.convert("RGBA")
    arr = np.array(image)
    h, w, channels = arr.shape
    
    # Separate color and alpha components
    rgb = arr[:, :, :3].astype(np.float32)
    base_alpha = arr[:, :, 3].astype(np.float32) / 255.0
    
    # --- Improved Background Sampling (corners and edges) ---
    # Sample multiple regions that are likely background (assumed near-white paper)
    sample_regions = []
    patch = min(10, h, w)  # size of the sample patch (10x10 or smaller if image is tiny)
    # Four corners
    sample_regions.append(rgb[0:patch, 0:patch, :])
    sample_regions.append(rgb[0:patch, w-patch:w, :])
    sample_regions.append(rgb[h-patch:h, 0:patch, :])
    sample_regions.append(rgb[h-patch:h, w-patch:w, :])
    # Center of each edge (if image is large enough)
    if h > patch and w > patch:
        sample_regions.append(rgb[0:patch, (w//2 - patch//2):(w//2 + patch//2), :])
        sample_regions.append(rgb[h-patch:h, (w//2 - patch//2):(w//2 + patch//2), :])
        sample_regions.append(rgb[(h//2 - patch//2):(h//2 + patch//2), 0:patch, :])
        sample_regions.append(rgb[(h//2 - patch//2):(h//2 + patch//2), w-patch:w, :])
    # Compute the mean color of each sample region
    samples = [np.mean(region.reshape(-1, 3), axis=0) for region in sample_regions if region.size > 0]
    if len(samples) == 0:
        # Fallback to pure white if something went wrong with sampling
        bg_color = np.array([255.0, 255.0, 255.0], np.float32)
    else:
        samples = np.array(samples)
        # Use median to reduce influence of any outlier (in case a sample included some paint)
        bg_color = np.median(samples, axis=0)
    
    logging.info(f"Sampled background color: {bg_color}")
    
    # --- Smooth Alpha Transitions via Color Distance with Exponential Curve ---
    # Compute Euclidean distance of each pixel's color from the background color
    diff = rgb - bg_color  # difference from background for each channel
    dist = np.linalg.norm(diff, axis=2)  # distance map (shape h x w)
    
    # Calculate the maximum possible distance (from background color to black)
    # This helps normalize our distance values
    max_possible_dist = np.linalg.norm(bg_color - np.array([0.0, 0.0, 0.0], np.float32))
    
    # Normalize distances to 0-1 range for easier curve application
    normalized_dist = dist / max_possible_dist
    
    # Apply an exponential curve to create smoother transitions
    # This will make pixels close to background very transparent,
    # but quickly increase opacity as color difference increases
    
    # Parameters to control the exponential curve shape
    # Lower values of base_threshold make more pixels transparent
    # Higher values of exponent make the transition more abrupt
    base_threshold = 0.03  # Minimum normalized distance to start becoming visible
    exponent = 2.5        # Controls how quickly transparency changes (higher = more abrupt)
    
    # Apply the exponential curve
    # First, shift the normalized distance by the base threshold
    shifted_dist = np.maximum(0, normalized_dist - base_threshold)
    # Then apply exponential curve and rescale to 0-1
    scale_factor = 1.0 / (1.0 - base_threshold) if base_threshold < 1.0 else 1.0
    alpha_map = np.power(shifted_dist * scale_factor, exponent)
    # Clip to ensure we stay in 0-1 range
    alpha_map = np.clip(alpha_map, 0.0, 1.0)
    
    # --- Special handling for dark lines and high-contrast details ---
    # Detect dark lines and high contrast areas that should remain fully opaque
    # This helps preserve pencil lines and fine details
    
    # Convert to grayscale for line detection
    gray = cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Calculate darkness relative to background
    bg_gray = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]  # Convert bg to grayscale
    darkness = bg_gray - gray  # How much darker than background
    
    # Create a mask for dark lines (significantly darker than background)
    dark_line_threshold = 30  # Pixels this much darker than background are likely lines
    dark_line_mask = (darkness > dark_line_threshold).astype(np.float32)
    
    # Detect edges for fine details
    edges = cv2.Canny(gray, 50, 150)
    edge_mask = (edges > 0).astype(np.float32)
    
    # Dilate the edge mask slightly to ensure we catch all of the line
    kernel = np.ones((2, 2), np.uint8)
    dilated_edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
    
    # Combine dark lines and edges
    detail_mask = np.maximum(dark_line_mask, dilated_edge_mask)
    
    # Boost alpha for detailed areas
    # Use a weighted combination to preserve details while maintaining smooth transitions
    detail_weight = 0.7  # How much to prioritize details (higher = more emphasis on lines)
    alpha_map = np.maximum(alpha_map, detail_mask * detail_weight)
    
    # Multiply by any existing alpha (if the image already had transparency)
    alpha_map *= base_alpha
    
    # --- Hybrid AI-based Segmentation (using rembg) ---
    try:
        # Use rembg to get a segmentation result
        rembg_session = new_session("isnet-general-use")
        seg_image = remove(
            Image.fromarray(arr[:, :, :3].astype(np.uint8)),
            session=rembg_session,
            only_mask=True
        )
        
        if debug and debug_dir:
            ai_mask_path = os.path.join(debug_dir, f"{name}_ai_mask.png")
            seg_image.save(ai_mask_path)
            logging.info(f"Saved AI mask to {ai_mask_path}")
        
        # Convert mask to numpy array
        seg_alpha = np.array(seg_image).astype(np.float32) / 255.0
        
        # Combine color-based alpha and AI mask to preserve all foreground regions
        # For watercolors, we want a gentler combination that preserves our smooth transitions
        
        # Where AI is confident about foreground (>0.7), boost our alpha
        # Where AI is confident about background (<0.1), reduce our alpha but don't eliminate
        # Otherwise, use a weighted blend
        
        # Define confidence thresholds
        ai_bg_thresh = 0.1
        ai_fg_thresh = 0.7
        
        # Create the final alpha
        final_alpha = alpha_map.copy()
        
        # Where AI is confident about foreground, boost alpha
        final_alpha = np.where(
            seg_alpha > ai_fg_thresh,
            np.maximum(alpha_map, seg_alpha),  # Use maximum
            final_alpha
        )
        
        # Where AI is confident about background, reduce alpha but maintain some
        # of our color-based alpha to preserve subtle transitions
        preservation_factor = 0.5  # How much to preserve our color-based alpha in background
        final_alpha = np.where(
            seg_alpha < ai_bg_thresh,
            alpha_map * preservation_factor,  # Reduce but don't eliminate
            final_alpha
        )
        
        # For the middle range, use a weighted blend
        middle_mask = (seg_alpha >= ai_bg_thresh) & (seg_alpha <= ai_fg_thresh)
        if np.any(middle_mask):
            # Normalize AI confidence to 0-1 within our middle range
            normalized_conf = (seg_alpha[middle_mask] - ai_bg_thresh) / (ai_fg_thresh - ai_bg_thresh)
            # Weighted blend based on AI confidence
            final_alpha[middle_mask] = (
                alpha_map[middle_mask] * (1.0 - normalized_conf) +
                np.maximum(alpha_map[middle_mask], seg_alpha[middle_mask]) * normalized_conf
            )
    except Exception as e:
        logging.warning(f"AI segmentation failed, falling back to color-based alpha: {e}")
        final_alpha = alpha_map
    
    # --- Edge Preservation and Color Adjustment ---
    # To avoid any white fringes on the edges, adjust pixel colors where alpha is partial.
    # We remove the contribution of the background color from these pixels (like GIMP's Color-to-Alpha).
    new_rgb = rgb.copy()
    # Mask for different alpha conditions
    fully_transparent = final_alpha < 1e-5    # ~0
    fully_opaque = final_alpha > (1 - 1e-5)   # ~1
    partial = ~(fully_transparent | fully_opaque)
    # For partially transparent pixels, recompute their color so that blending with white yields the original color.
    # new_color = (original_color - (1 - alpha)*bg_color) / alpha
    # This ensures no white tint remains in the color now that it will sit on a new background.
    for c in range(3):  # for R, G, B channels
        orig_c = rgb[:, :, c]
        # Apply the color adjustment only on partial-alpha pixels
        adj_c = orig_c.copy()
        adj_c[partial] = (orig_c[partial] - (1.0 - final_alpha[partial]) * bg_color[c]) / (final_alpha[partial] + 1e-8)
        # For fully transparent pixels, just set color to background (not that it matters, since alpha=0)
        adj_c[fully_transparent] = bg_color[c]
        # For fully opaque pixels, keep original color
        # (No change needed for fully_opaque since orig_c stays same there)
        new_rgb[:, :, c] = adj_c
    # Clip any values outside valid range after adjustment
    new_rgb = np.clip(new_rgb, 0, 255).astype(np.uint8)
    
    # Apply a very slight blur to the alpha channel to smooth any remaining harsh transitions
    final_alpha_img = Image.fromarray((final_alpha * 255).astype(np.uint8), mode="L")
    final_alpha_img = final_alpha_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    final_alpha = np.array(final_alpha_img).astype(np.float32) / 255.0
    
    # Stack color and alpha back into an RGBA image
    result_arr = np.dstack((new_rgb, (final_alpha * 255).astype(np.uint8)))
    result_img = Image.fromarray(result_arr, mode="RGBA")
    
    # Save debug images if requested
    if debug and debug_dir:
        # Save the color-based alpha map
        color_alpha_img = Image.fromarray((alpha_map * 255).astype(np.uint8), mode="L")
        color_alpha_path = os.path.join(debug_dir, f"{name}_color_alpha.png")
        color_alpha_img.save(color_alpha_path)
        
        # Save the final alpha map
        final_alpha_img = Image.fromarray((final_alpha * 255).astype(np.uint8), mode="L")
        final_alpha_path = os.path.join(debug_dir, f"{name}_final_alpha.png")
        final_alpha_img.save(final_alpha_path)
        
        # Save the intermediate result before color adjustment
        before_color_adj = np.dstack((rgb.astype(np.uint8), (final_alpha * 255).astype(np.uint8)))
        before_adj_img = Image.fromarray(before_color_adj, mode="RGBA")
        before_adj_path = os.path.join(debug_dir, f"{name}_before_color_adj.png")
        before_adj_img.save(before_adj_path)
        
        # Save visualization of the exponential curve
        if h > 100 and w > 100:  # Only for reasonably sized images
            # Create a gradient visualization
            gradient = np.linspace(0, 1, 256)
            # Apply our exponential function
            shifted_gradient = np.maximum(0, gradient - base_threshold)
            exp_gradient = np.power(shifted_gradient * scale_factor, exponent)
            exp_gradient = np.clip(exp_gradient, 0, 1)
            
            # Create a visualization image
            viz_height = 50
            viz_width = 256
            viz_img = np.zeros((viz_height, viz_width), dtype=np.uint8)
            for x in range(viz_width):
                alpha_val = int(exp_gradient[x] * 255)
                viz_img[:, x] = alpha_val
                
            viz_pil = Image.fromarray(viz_img, mode="L")
            viz_path = os.path.join(debug_dir, f"{name}_alpha_curve.png")
            viz_pil.save(viz_path)
            logging.info(f"Saved alpha curve visualization to {viz_path}")
        
        logging.info(f"Saved watercolor debug images to {debug_dir}")
    
    return result_img

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
    # First, check if the image is likely a watercolor
    is_watercolor_image = is_watercolor(image)
    
    # If it's a watercolor, use our specialized watercolor background removal
    if is_watercolor_image:
        logging.info(f"Using specialized watercolor background removal for {name}")
        return remove_watercolor_background(image, debug=debug, debug_dir=debug_dir, name=name)
    
    # Otherwise, check if the image is likely line art or illustration
    line_art_mode = is_line_art(image)
    
    # Adjust color tolerance based on image type
    if line_art_mode:
        # Use higher tolerance for line art - we want to remove more background
        # while still preserving the lines
        bg_color_tolerance += 10
        logging.info(f"Line art detected: Increasing background color tolerance to {bg_color_tolerance}")
    
    # For line art, prioritize the gradient-based color removal
    # This provides smooth transitions at edges and preserves subtle color variations
    color_based_result = remove_background_color_based(
        image, 
        color_tolerance=bg_color_tolerance,
        preserve_outlines=True,
        gradient_opacity=True  # Enable gradient-based opacity
    )
    
    # For line art, the color-based approach with gradient opacity 
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
            # For line art, prioritize preserving all color variations
            # Don't let the AI mask remove too much
            alpha_scale = 0.9  # Higher preservation for line art
            logging.info("Using very high outline preservation for line art")
        else:
            # For photographic content, use moderate preservation
            alpha_scale = 0.3
        
        # Create a scaled alpha based on the AI mask
        # Where AI mask is white (255), keep full alpha
        # Where AI mask is black (0), reduce alpha by alpha_scale factor
        # For line art, we want to preserve more even where AI disagrees
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

def detect_and_crop_objects(image_path, output_dir, debug=False):
    """Detect objects in an image, crop them, and save the cropped images.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save cropped images
        debug: Whether to save debug information
        
    Returns:
        Tuple of (image_name, list of detected objects with their bounding boxes)
        Each object is a dict with 'object' (name), 'bbox' (coordinates), and 'crop_path' (path to cropped image)
    """
    img_name = os.path.basename(image_path)
    logging.info(f"Detecting objects in {img_name}...")
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
                
            name = obj.get("object", f"object_{obj_index+1}")
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
            
            # Generate a temporary filename for the cropped image
            # Instead of using the object name, we'll use a generic name with the index
            temp_filename = f"temp_crop_{img_name.split('.')[0]}_{obj_index}.png"
            cropped_path = os.path.join(output_dir, temp_filename)
            cropped_img.save(cropped_path, format="PNG")
            logging.info(f"Saved cropped image to {cropped_path}")
            
            # Record the object info for JSON (use generic name and bbox)
            result_entry.append({
                "object": name,
                "bbox": [ymin, xmin, ymax, xmax],  # Store in the new format
                "crop_path": cropped_path,  # Store the path to the cropped image
                "original_name": name  # Store the original name for later use
            })
            
        except Exception as e:
            logging.error(f"Error processing object in {img_name}: {e}")
            # Log stack trace for debugging
            logging.debug(traceback.format_exc())
            continue  # skip this object and move to next
    
    # Generate descriptive names for the objects
    if result_entry:
        result_entry = generate_object_names(image_path, result_entry, debug)
    
    return (img_name, result_entry)

def remove_backgrounds(cropped_images, output_dir, debug=False, force_watercolor=True):
    """Remove backgrounds from cropped images.
    
    Args:
        cropped_images: List of dicts with 'object' (name), 'crop_path' (path to cropped image),
                        and 'original_name' (original object name from detection)
        output_dir: Directory to save background-removed images
        debug: Whether to save debug information
        force_watercolor: Whether to force watercolor mode (default: True)
        
    Returns:
        List of paths to background-removed images
    """
    logging.info(f"Removing backgrounds from {len(cropped_images)} cropped images...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create debug directory if debug is enabled
    debug_dir = None
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
    result_paths = []
    
    # Process each cropped image
    for obj_index, obj in enumerate(cropped_images):
        try:
            name = obj.get("original_name", obj.get("object", "object"))
            crop_path = obj.get("crop_path")
            
            if not crop_path or not os.path.exists(crop_path):
                logging.warning(f"Skipping object with missing or invalid crop path: {obj}")
                continue
            
            # Load the cropped image
            cropped_img = Image.open(crop_path).convert("RGB")
            
            # Use appropriate background removal method
            img_name = os.path.basename(crop_path)
            debug_id = f"{img_name.split('.')[0]}_{obj_index}_{name}" if debug else None
            
            # If force_watercolor is enabled, use watercolor-specific background removal
            if force_watercolor:
                logging.info(f"Using forced watercolor background removal for {name}")
                output_img = remove_watercolor_background(
                    cropped_img,
                    debug=debug,
                    debug_dir=debug_dir,
                    name=debug_id
                )
            else:
                # Otherwise use the hybrid approach which will auto-detect content type
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
                # Generate a safe filename for this object using the original name
                base_name = sanitize_filename(name)
                # If base_name is empty or generic, fall back to "object"
                if not base_name:
                    base_name = "object"
                # Ensure unique naming in the output directory
                final_name = ensure_unique_filename(output_dir, base_name, extension=".png")
                final_path = os.path.join(output_dir, final_name)
                output_img.save(final_path, format="PNG")
                logging.info(f"Saved background-removed image to {final_path}")
                result_paths.append(final_path)
                
                # Only delete temporary cropped files (those starting with "temp_crop_")
                # and never delete original input files
                if not debug and os.path.exists(crop_path) and os.path.basename(crop_path).startswith("temp_crop_"):
                    try:
                        os.remove(crop_path)
                        logging.debug(f"Deleted temporary crop file: {crop_path}")
                    except Exception as e:
                        logging.warning(f"Failed to delete temporary crop file {crop_path}: {e}")
            
        except Exception as e:
            logging.error(f"Error removing background from {crop_path}: {e}")
    
    return result_paths

def process_cropped_directory(input_dir, output_dir=None, debug=False, force_watercolor=True, clear_output=False):
    """Process a directory of cropped images for background removal.
    
    This function is useful when you have manually cropped images or want to
    apply background removal to images from another source.
    
    Args:
        input_dir: Directory containing cropped images
        output_dir: Directory to save background-removed images (defaults to input_dir)
        debug: Whether to save debug information
        force_watercolor: Whether to force watercolor mode (default: True)
        clear_output: Whether to clear the output directory before processing (default: False)
        
    Returns:
        List of paths to background-removed images
    """
    if not output_dir:
        output_dir = input_dir
    
    # Clear the output directory if requested
    if clear_output and output_dir != input_dir:  # Only clear if output_dir is different from input_dir
        clear_directory(output_dir, preserve_debug=debug)
    
    logging.info(f"Processing cropped images in {input_dir}...")
    
    # Find all image files in the directory
    image_files = [f for f in os.listdir(input_dir) 
                  if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")
                  and not f.endswith("-final.png")]  # Skip already processed images
    
    if not image_files:
        logging.info("No images found in the specified directory.")
        return []
    
    logging.info(f"Found {len(image_files)} images for background removal.")
    
    # Create a list of cropped image objects
    cropped_images = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        crop_path = os.path.join(input_dir, img_file)
        cropped_images.append({
            'object': base_name,
            'crop_path': crop_path,
            'original_name': base_name  # Use the filename as the original name
        })
    
    # Generate descriptive names for all objects using the parallelized function
    cropped_images = generate_object_names(input_dir, cropped_images, debug)
    
    # Process the cropped images - ensure we're passing the correct output directory
    # so original files in input_dir are preserved
    result_paths = remove_backgrounds(cropped_images, output_dir, debug, force_watercolor)
    
    logging.info(f"Background removal complete. Processed {len(result_paths)} images.")
    return result_paths

def process_image(image_path, output_dir, debug=False, force_watercolor=True, mode="all"):
    """Process a single image: detect objects, crop them, remove background, and save results.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save processed images
        debug: Whether to save debug information
        force_watercolor: Whether to force watercolor mode (default: True)
        mode: Processing mode - "all", "detect", or "remove" (default: "all")
        
    Returns:
        Tuple of (image_name, list of detected objects with their bounding boxes)
    """
    if mode not in ["all", "detect", "remove"]:
        logging.error(f"Invalid mode: {mode}. Must be 'all', 'detect', or 'remove'.")
        return None
    
    # For "remove" mode, we need existing cropped images
    if mode == "remove":
        logging.error("'remove' mode cannot be used with process_image directly. Use process_cropped_directory instead.")
        return None
    
    # Detect and crop objects
    result = detect_and_crop_objects(image_path, output_dir, debug)
    if result is None:
        return None
    
    img_name, objects = result
    
    # If mode is "all", also remove backgrounds
    if mode == "all" and objects:
        remove_backgrounds(objects, output_dir, debug, force_watercolor)
    
    return (img_name, objects)

def find_cropped_images(directory):
    """Find all image files in a directory that appear to be cropped images.
    
    Args:
        directory: Directory to search for cropped images
        
    Returns:
        List of dicts with 'object' (name), 'crop_path' (path to cropped image), and 'original_name'
    """
    logging.info(f"Finding cropped images in {directory}...")
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(directory) 
                  if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")
                  and not f.endswith("-final.png")]  # Skip already processed images
    
    if not image_files:
        logging.info("No images found in the specified directory.")
        return []
    
    # Create a list of cropped image objects
    cropped_images = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        crop_path = os.path.join(directory, img_file)
        cropped_images.append({
            'object': base_name,
            'crop_path': crop_path,
            'original_name': base_name  # Use the filename as the original name
        })
    
    logging.info(f"Found {len(cropped_images)} cropped images")
    return cropped_images

def clear_directory(directory, preserve_debug=False):
    """
    Clear all files and subdirectories in the specified directory.
    
    Args:
        directory: Directory to clear
        preserve_debug: If True, preserve the debug subdirectory
        
    Returns:
        None
    """
    if not os.path.exists(directory):
        return
    
    logging.info(f"Clearing output directory: {directory}")
    
    # If preserve_debug is True and there's a debug subdirectory, temporarily move it
    debug_dir = os.path.join(directory, "debug")
    temp_debug_dir = None
    
    if preserve_debug and os.path.exists(debug_dir):
        temp_debug_dir = os.path.join(os.path.dirname(directory), "temp_debug")
        if os.path.exists(temp_debug_dir):
            shutil.rmtree(temp_debug_dir)
        shutil.move(debug_dir, temp_debug_dir)
        logging.info(f"Temporarily moved debug directory to {temp_debug_dir}")
    
    # Clear all files in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            logging.error(f"Error while clearing {item_path}: {e}")
    
    # If we preserved the debug directory, move it back
    if temp_debug_dir:
        shutil.move(temp_debug_dir, debug_dir)
        logging.info(f"Restored debug directory to {debug_dir}")

def main(input_dir, output_dir=None, debug=False, force_watercolor=True, mode="all", clear_output=False):
    """
    Main function to process all images in a directory.
    
    Args:
        input_dir: Directory containing images to process
        output_dir: Directory to save processed images (defaults to input_dir/processed)
        debug: Whether to save debug information
        force_watercolor: Whether to force watercolor mode (default: True)
        mode: Processing mode - "all", "detect", or "remove" (default: "all")
        clear_output: Whether to clear the output directory before processing (default: False)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    if mode not in ["all", "detect", "remove"]:
        print(f"Error: Invalid mode '{mode}'. Must be 'all', 'detect', or 'remove'.")
        return 1
    
    # Default input directory is current working directory
    directory = input_dir or "."
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return 1  # non-zero exit for error
    
    # Set up output directory
    if output_dir:
        # Use the specified output directory
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Default output directory is input_dir/processed
        output_dir = os.path.join(directory, "processed")
        os.makedirs(output_dir, exist_ok=True)
    
    # Clear the output directory if requested
    if clear_output:
        # Only clear if output_dir is different from input_dir to avoid deleting input files
        if os.path.abspath(output_dir) != os.path.abspath(directory):
            clear_directory(output_dir, preserve_debug=debug)
        else:
            logging.warning("Skipping clear operation: output directory is the same as input directory")
    
    # For "remove" mode, we use process_cropped_directory
    if mode == "remove":
        # In remove mode, the input directory contains the cropped images
        # We should never delete these input files, only create new output files
        process_cropped_directory(directory, output_dir, debug, force_watercolor, clear_output)
        return 0
    
    # For "all" and "detect" modes, we need to find images to process
    images = [f for f in os.listdir(directory) 
              if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
    if not images:
        print("No images found in the specified directory.")
        return 0
    
    logging.info(f"Found {len(images)} images in '{directory}'. Beginning processing...")
    logging.info(f"Mode: {mode}")
    logging.info(f"Debug mode: {'enabled' if debug else 'disabled'}")
    logging.info(f"Watercolor mode: {'enabled' if force_watercolor else 'disabled'}")
    
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
    # Limit to 3 workers as per requirement
    max_workers = 3  # Process only 3 images at a time
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Launch parallel tasks
        future_to_img = {
            executor.submit(process_image, os.path.join(directory, img), output_dir, debug, force_watercolor, mode): img 
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
    if mode == "all":
        logging.info(f"Processing complete. Extracted and processed {total_objects} objects from {len(results)} images.")
    else:  # mode == "detect"
        logging.info(f"Detection complete. Extracted {total_objects} objects from {len(results)} images.")
    
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract objects from images using Google Gemini and remove backgrounds.")
    parser.add_argument("-i", "--input", type=str, default=".", 
                        help="Input directory containing images (PNG/JPG). In 'remove' mode, this should contain the cropped images. Defaults to current directory.")
    parser.add_argument("-o", "--output", type=str, 
                        help="Output directory for processed images. Defaults to input_dir/processed.")
    parser.add_argument("-d", "--debug", action="store_true", 
                        help="Enable debug mode to save additional debug files and visualizations.")
    parser.add_argument("--no-watercolor", action="store_true", 
                        help="Disable watercolor mode (which is enabled by default) and use standard background removal.")
    parser.add_argument("-c", "--clear", action="store_true",
                        help="Clear the output directory before processing.")
    parser.add_argument("-m", "--mode", type=str, choices=["all", "detect", "remove"], default="all", 
                        help="""Processing mode:
                        'all': Detect objects, crop them, and remove backgrounds (default)
                        'detect': Only detect and crop objects without removing backgrounds
                        'remove': Only remove backgrounds from existing cropped images (use with -i pointing to directory with cropped images)""")
    args = parser.parse_args()
    exit_code = main(args.input, args.output, args.debug, not args.no_watercolor, args.mode, args.clear)
    exit(exit_code)