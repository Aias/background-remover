#!/usr/bin/env python3
import os
import json
import logging
import traceback
import re
import concurrent.futures
import subprocess
from PIL import Image, ImageDraw
from rembg import remove

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
    Use Google Gemini 1.5 Pro to detect objects in the image and return a list of 
    objects with names and bounding boxes. Requires a configured API key.
    """
    # Load the image using PIL
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Prepare the prompt for Gemini: ask for JSON output with object names and bboxes
    prompt = [
        f"Return bounding boxes around every distinct object in the image. "
        f"For each object detected, include: "
        f"\"object\": <short descriptive name> and "
        f"\"bbox\": [ymin, xmin, ymax, xmax] as a simple array of 4 integers (not an object). "
        f"The coordinates should be in the range 0-1000, where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner. "
        f"The bounding box should be a bit larger than the object itself, to include some of the surrounding background area."
        f"The descriptive name should be one or two short words that describe the object. "
        f"Example of expected output format: [{{'object': 'cup', 'bbox': [100, 150, 200, 250]}}]. "
        f"Do not use nested objects for the bbox values, just a flat array of 4 integers.",
        img  # the image is provided to the model
    ]
    
    logging.info(f"Sending prompt to Gemini for {os.path.basename(image_path)}")
    
    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response = model.generate_content(prompt)
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
        else:
            # If not in code block, try to find array directly
            try:
                start = content.index('[')
                end = content.rindex(']')
                json_str = content[start:end+1]
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
                
                # Process the raw objects to normalize the bbox format (same as above)
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
            except (ValueError, json.JSONDecodeError) as e:
                logging.error(f"Failed to extract JSON from response for {os.path.basename(image_path)}: {e}")
                logging.debug(f"Response content: {content}")
                return None
                
    except Exception as parse_err:
        logging.error(f"Failed to parse response from model for {os.path.basename(image_path)}: {parse_err}")
        logging.debug(f"Raw response: {response}")
        return None
        
    return objects  # list of dicts like {"object": "...", "bbox": [ymin, xmin, ymax, xmax]}

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
        
        # Save original image for debugging if debug is enabled
        if debug:
            debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_original = os.path.join(debug_dir, f"original_{img_name}")
            image.save(debug_original)
            logging.info(f"Saved original image for debugging to {debug_original}")
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
                
                # Calculate width and height for later use
                width = xmax - xmin
                height = ymax - ymin
                
                # Crop region is (left, top, right, bottom)
                crop_region = (xmin, ymin, xmax, ymax)
                
            except (ValueError, TypeError):
                logging.warning(f"Skipping object with non-numeric bbox values in {img_name}: {obj}")
                continue
                
            logging.info(f"Calculated crop region: {crop_region}")
            
            # Ensure crop region is within image bounds
            if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
                logging.warning(f"Adjusting crop region to fit within image bounds for {name} in {img_name}")
                orig_region = crop_region
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(xmax, img_width)
                ymax = min(ymax, img_height)
                # Recalculate width and height
                width = xmax - xmin
                height = ymax - ymin
                crop_region = (xmin, ymin, xmax, ymax)
                logging.info(f"Adjusted crop region from {orig_region} to {crop_region}")
                
            # Skip if resulting crop region is too small
            if width <= 0 or height <= 0:
                logging.warning(f"Skipping {name} in {img_name} due to invalid crop dimensions: {width}x{height}")
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
                
            cropped_img = image.crop(crop_region)
            
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
            
            # Remove background using rembg
            try:
                cropped_img_rgba = Image.open(cropped_path)  # reopen to ensure compatibility with rembg
                output_img = remove(cropped_img_rgba)       # remove background
            except Exception as re:
                logging.error(f"rembg failed for {obj_filename}: {re}")
                # Continue without background removal (skip final image)
                output_img = None
            
            # Save background-removed image if successful
            if output_img:
                final_name = obj_base + "-final.png"
                final_path = os.path.join(output_dir, final_name)
                output_img.save(final_path, format="PNG")
            
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