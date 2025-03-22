import torch
from models import clip_model, clip_processor, segformer_model, segformer_processor
from constants import fashion_categories, fashion_clip_to_segformer, class_names, category_to_segment_mapping, garment_to_segments
from segmentation import identify_garment_segformer

def identify_garment_clip(image):
    """Identify the garment type using Fashion-CLIP model"""
    # Prepare text prompts
    texts = [f"a photo of a {category}" for category in fashion_categories]
    
    # Process inputs
    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    # Get the top prediction
    top_idx = torch.argmax(probs[0]).item()
    top_category = fashion_categories[top_idx]
    confidence = probs[0][top_idx].item() * 100
    
    # Map to SegFormer class if possible
    if top_category in fashion_clip_to_segformer:
        segformer_idx = fashion_clip_to_segformer[top_category]
        segformer_class = class_names[segformer_idx]
        return top_category, segformer_idx, confidence
    else:
        # Fallback to using SegFormer directly
        return top_category, None, confidence

def get_segments_for_garment(garment_image):
    """Get the segments that should be included for a given garment image"""
    # First try to identify the garment using Fashion-CLIP
    clip_category, segformer_idx, confidence = identify_garment_clip(garment_image)
    
    # If CLIP couldn't map to a SegFormer class, fall back to SegFormer
    if segformer_idx is None:
        garment_name, segformer_idx = identify_garment_segformer(garment_image)
        method = "SegFormer (fallback)"
        confidence_text = ""
    else:
        garment_name = class_names[segformer_idx]
        method = "Fashion-CLIP"
        confidence_text = f" with {confidence:.2f}% confidence"
    
    if segformer_idx is None:
        return None, None, "No clear garment detected in the garment image"
    
    # Get all segments that should be included based on the detected garment
    if method == "Fashion-CLIP" and clip_category in category_to_segment_mapping:
        # Use the detailed mapping for CLIP categories
        selected_class = category_to_segment_mapping[clip_category]
    elif segformer_idx in garment_to_segments:
        # Fall back to the SegFormer class-based mapping
        segment_indices = garment_to_segments[segformer_idx]
        selected_class = [class_names[idx] for idx in segment_indices]
    else:
        # Fallback to just the detected garment if no mapping exists
        selected_class = [class_names[segformer_idx]]
    
    # Prepare a more descriptive result text
    included_segments = ", ".join(selected_class)
    if method == "Fashion-CLIP":
        result_text = f"Detected garment: {clip_category} (mapped to {garment_name})\nUsing {method}{confidence_text}\nSegmented parts: {included_segments}"
    else:
        result_text = f"Detected garment: {garment_name}\nUsing {method}\nSegmented parts: {included_segments}"
    
    return selected_class, segformer_idx, result_text
