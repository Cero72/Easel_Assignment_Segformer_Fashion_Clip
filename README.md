# Garment-based Segmentation Application

## Problem Statement

Fashion image analysis often requires identifying and segmenting specific garments in images of people. Traditional approaches segment all clothing items in an image, but many applications need to focus on a specific garment type. This project addresses the challenge of targeted garment segmentation by matching a reference garment image to clothing worn by a person in another image.

## Approaches Explored

During the development of this application, we explored several approaches:

1. **General Segmentation with SegFormer**:
   - Initially, we used SegFormer to segment all clothing items in an image
   - This worked well for general segmentation but lacked specificity for targeting particular garments

2. **Classification with ResNet50**:
   - We tried using a pre-trained ResNet50 model to classify garment types
   - While effective for basic classification, it lacked the semantic understanding needed for matching garments

3. **Fashion-CLIP for Garment Matching**:
   - We implemented Fashion-CLIP, a specialized version of CLIP (Contrastive Language-Image Pre-training) for fashion items
   - This provided better semantic understanding of garment types and styles

4. **Combined Approach**:
   - We found that combining SegFormer for segmentation with Fashion-CLIP for classification yielded the best results
   - This allowed us to first identify the garment type from a reference image, then segment only that type in the person image

## Final Implementation

Our final solution implements a modular architecture with the following components:

### Core Components

1. **Segmentation Module** (`segmentation.py`):
   - Uses SegFormer to identify and segment clothing items in images
   - Provides functions to filter segmentation by specific garment classes

2. **Classification Module** (`classification.py`):
   - Uses Fashion-CLIP to identify garment types from reference images
   - Matches garment semantics between reference and target images

3. **Model Loading** (`models.py`):
   - Handles loading and initialization of SegFormer and Fashion-CLIP models
   - Provides a consistent interface for model access

4. **Constants** (`constants.py`):
   - Defines class names, color mappings, and other constants used throughout the application

5. **Utilities** (`utils.py`):
   - Provides helper functions for image processing, URL handling, and combining segmentation with classification

6. **User Interface** (`app.py`):
   - Implements a Gradio-based interface for user interaction
   - Allows users to upload person and garment images for targeted segmentation

### Key Features

- **Targeted Garment Segmentation**: Segment only the specific garment type that matches a reference image
- **Consistent Image Sizing**: All output images maintain a consistent size for better visualization
- **Interactive Interface**: User-friendly interface with clear sections for person and garment images
- **Visualization Options**: Users can choose to view original images, segmentation maps, and overlays

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/garment-segmentation.git
   cd garment-segmentation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open the provided URL in your browser (typically http://127.0.0.1:7860)

3. Use the interface:
   - Upload an image of a person wearing clothes
   - Upload a reference image of a garment (e.g., a t-shirt, pants, dress)
   - Click "Process Images" to generate the targeted segmentation
   - View the results in the gallery, including original images, segmentation maps, and overlays

## Project Structure

```
├── app.py                 # Main application with Gradio interface
├── models.py              # Model loading functions
├── constants.py           # Class names and constants
├── segmentation.py        # Image segmentation functions
├── classification.py      # Garment classification functions
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Acknowledgements

- [SegFormer](https://huggingface.co/mattmdjaga/segformer_b2_clothes) for semantic segmentation
- [Fashion-CLIP](https://huggingface.co/patrickjohncyh/fashion-clip) for garment classification
- [Gradio](https://gradio.app/) for the web interface
