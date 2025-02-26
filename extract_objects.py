#!/usr/bin/env python3
import os
import json
import logging
import traceback
import concurrent.futures
from PIL import Image
from rembg import remove

# Import Google Gemini generative AI SDK
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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

def detect_objects_with_gemini(image_path):
    """
    Use Google Gemini 1.5 Pro to detect objects in the image and return a list of 
    objects with names and bounding boxes. Requires a configured API key.
    """
    # Load the image using PIL
    img = Image.open(image_path)
    # Prepare the prompt for Gemini: ask for JSON output with object names and bboxes
    prompt = [
        "Identify all distinct objects in the image and provide a JSON list of objects, where each entry has: "
        "\"object\": <short name> and \"bbox\": [x, y, width, height] in pixels.",
        img  # the image is provided to the model
    ]
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")  # use Gemini 1.5 Pro model
        response = model.generate_content(prompt)
    except Exception as e:
        logging.error(f"Gemini API call failed for image {os.path.basename(image_path)}: {e}")
        return None  # indicating failure
    
    # The response from generate_content could be a JSON string or contain text. 
    # Attempt to parse it as JSON.
    content = str(response)  # ensure we have a string
    objects = []
    try:
        objects = json.loads(content)
    except json.JSONDecodeError:
        # If the model didn't return pure JSON, try to extract JSON substring or eval (as fallback)
        try:
            # Find the first and last brace to isolate JSON
            start = content.index('[')
            end = content.rindex(']')
            json_str = content[start:end+1]
            objects = json.loads(json_str)
        except Exception as parse_err:
            logging.error(f"Failed to parse JSON from model response for {os.path.basename(image_path)}: {parse_err}")
            return None
    return objects  # list of dicts like {"object": "...", "bbox": [x,y,w,h]}

def process_image(image_path, output_dir):
    """Process a single image: detect objects, crop them, remove background, and save results."""
    img_name = os.path.basename(image_path)
    logging.info(f"Processing {img_name}...")
    result_entry = []  # to store objects info for JSON output
    
    # Detect objects using Gemini
    detections = detect_objects_with_gemini(image_path)
    if detections is None:
        # An error occurred (already logged), skip this image
        return None
    
    # Open the image once for cropping (do in RGB mode)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Failed to open image {img_name}: {e}")
        return None
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each detected object
    for obj in detections:
        try:
            name = obj.get("object", "object")
            bbox = obj.get("bbox")
            if not bbox or len(bbox) != 4:
                logging.warning(f"Skipping object with invalid bbox in {img_name}: {obj}")
                continue
            x, y, w, h = bbox
            # Crop the object region (assuming [x,y,w,h] with origin at top-left)
            crop_region = (int(x), int(y), int(x) + int(w), int(y) + int(h))
            cropped_img = image.crop(crop_region)
            
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
                "bbox": bbox
            })
            logging.info(f" - Saved {obj_filename} (and background-removed version).")
        except Exception as e:
            logging.error(f"Error processing object in {img_name}: {e}")
            # Log stack trace for debugging
            logging.debug(traceback.format_exc())
            continue  # skip this object and move to next
    
    return (img_name, result_entry)

def main(input_dir):
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
    
    results = {}
    # Use ThreadPoolExecutor for parallel processing of images
    max_workers = os.cpu_count() or 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Launch parallel tasks
        future_to_img = {
            executor.submit(process_image, os.path.join(directory, img), output_dir): img 
            for img in images
        }
        for future in concurrent.futures.as_completed(future_to_img):
            img = future_to_img[future]
            try:
                res = future.result()
            except Exception as exc:
                logging.error(f"Unhandled exception processing {img}: {exc}")
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
    
    logging.info("Processing complete.")
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract objects from images using Google Gemini and remove backgrounds.")
    parser.add_argument("-i", "--input", type=str, default=".", help="Input directory containing images (PNG/JPG). Defaults to current directory.")
    args = parser.parse_args()
    exit_code = main(args.input)
    exit(exit_code)