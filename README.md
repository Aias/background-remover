# AI-Powered Object Extractor and Background Remover

(Attempts to) use Google's Gemini AI to detect objects in images, extract them, and remove their backgrounds automatically. Works okay, but not entirely reliable.

**Prior Art**: Simon Willison, [Building a tool showing how Gemini Pro can return bounding boxes for objects in images](https://simonwillison.net/2024/Aug/26/gemini-bounding-box-visualization/).

## Overview

This project combines the object detection capabilities of Google's Gemini AI with the background removal functionality of the `rembg` library to create an image processing pipeline. The tool:

1. Detects objects in images using Google Gemini AI
2. Extracts each object by cropping it from the original image
3. Removes the background from each extracted object
4. Saves both the cropped objects and their background-removed versions

## Features

-   **AI-Powered Object Detection**: Uses Google Gemini 2.0 Flash to identify and locate objects in images
-   **Automatic Background Removal**: Leverages the `rembg` library to create transparent backgrounds
-   **JSON Output**: Creates a JSON file with object metadata including bounding box coordinates
-   **Optional Debug Mode**: Generates detailed debug visualizations when enabled

## Requirements

-   Python 3.6+
-   Google API Key for Gemini AI
-   Required Python packages (see `requirements.txt`):
    -   `google-generativeai`: Google Gemini API SDK
    -   `Pillow`: Image processing library
    -   `rembg`: Background removal tool

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/background-remover.git
cd background-remover
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your Google API Key:
    - Option 1: Set it as an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

    - Option 2: Add it to your `.zshrc` file (the script will automatically detect it):

```bash
echo 'export GOOGLE_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## Usage

### Basic Usage

Process all images in the current directory:

```bash
python extract_objects.py
```

Process images in a specific directory:

```bash
python extract_objects.py -i /path/to/images
```

Enable debug mode to generate additional debug files and visualizations:

```bash
python extract_objects.py -d
```

Or combine both options:

```bash
python extract_objects.py -i /path/to/images -d
```

### Command-line Options

| Option             | Description                                                                 |
| ------------------ | --------------------------------------------------------------------------- |
| `-i, --input PATH` | Input directory containing images (PNG/JPG). Defaults to current directory. |
| `-d, --debug`      | Enable debug mode to save additional debug files and visualizations.        |

### Output Structure

The script creates a `processed` directory with the following structure:

```
processed/
├── object1.png                # Cropped object with background
├── object1-final.png          # Object with background removed
├── object2.png                # Another cropped object
├── object2-final.png          # Another object with background removed
├── bounding_boxes.json        # JSON file with object metadata
└── debug/                     # Debug visualizations (only when debug mode is enabled)
    ├── original_image.jpg     # Copy of original image
    ├── image_0_object_bbox.png  # Visualization of bounding box
    ├── image_0_object_crop.png  # Debug crop image
    └── image_gemini_response.txt # Raw Gemini API response
```

Note: The `debug/` directory is only created when the `-d` flag is used.

### JSON Output Format

The `bounding_boxes.json` file contains metadata about all detected objects:

```json
{
	"image1.jpg": [
		{
			"object": "cup",
			"bbox": [100, 150, 200, 250]
		},
		{
			"object": "keyboard",
			"bbox": [300, 400, 350, 550]
		}
	],
	"image2.jpg": [
		{
			"object": "chair",
			"bbox": [50, 100, 300, 400]
		}
	]
}
```

## How It Works

1. **Object Detection**: The script sends each image to Google's Gemini AI with a prompt asking it to identify objects and return their bounding boxes.

2. **Parsing and Normalization**: The JSON response from Gemini is parsed and normalized to handle various response formats.

3. **Object Extraction**: Each detected object is cropped from the original image using the bounding box coordinates.

4. **Background Removal**: The `rembg` library processes each cropped object to remove its background.

5. **Output Generation**: Both versions of each object (with and without background) are saved, along with debug visualizations and metadata.

## Troubleshooting

### API Key Issues

If you encounter API key errors:

-   Ensure your Google API key is correctly set in your environment or `.zshrc` file
-   Verify that your API key has access to the Gemini API
-   Check for any quotas or rate limits on your API key

### Image Processing Issues

If objects aren't being detected properly:

-   Try using images with clearer, more distinct objects
-   Ensure images aren't too large or too small (optimal size is around 1024x1024 pixels)
-   Check the debug visualizations to see how Gemini is interpreting your images (enable debug mode with `-d`)

### Background Removal Issues

If background removal isn't working well:

-   The `rembg` library works best with clear subjects against contrasting backgrounds
-   Complex backgrounds or objects with fuzzy edges may not get perfect results
-   Try pre-processing your images to increase contrast between subject and background

## License

[MIT License](LICENSE)

## Acknowledgments

-   [Google Gemini AI](https://ai.google.dev/) for object detection
-   [rembg](https://github.com/danielgatis/rembg) for background removal
-   [Pillow](https://python-pillow.org/) for image processing
