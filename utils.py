import requests
from PIL import Image
import gradio as gr

from segmentation import segment_image
from classification import get_segments_for_garment

def process_url(url, selected_classes, show_original, show_segmentation, show_overlay, fixed_size=(400, 400)):
    """Process an image from a URL"""
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        return segment_image(image, selected_classes, show_original, show_segmentation, show_overlay, fixed_size)
    except Exception as e:
        return [gr.update(value=None)] * 4, f"Error: {str(e)}"

def process_person_and_garment(person_image, garment_image, show_original, show_segmentation, show_overlay, fixed_size=(400, 400)):
    """Process person and garment images for targeted segmentation"""
    if person_image is None or garment_image is None:
        return [gr.update(value=None)] * 4, "Please provide both person and garment images"
    
    try:
        # Get segments that should be included based on the garment
        selected_class, segformer_idx, result_text = get_segments_for_garment(garment_image)
        
        if selected_class is None:
            return [gr.update(value=None)] * 4, result_text
        
        # Process the person image with the selected garment classes
        result_images = segment_image(person_image, selected_class, show_original, show_segmentation, show_overlay, fixed_size)
        
        return result_images, result_text
    
    except Exception as e:
        return [gr.update(value=None)] * 4, f"Error: {str(e)}"
