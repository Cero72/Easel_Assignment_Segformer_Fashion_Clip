import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from collections import Counter
import gradio as gr

from models import segformer_model, segformer_processor
from constants import class_names, color_map

def segment_image(image, selected_classes=None, show_original=True, show_segmentation=True, show_overlay=True, fixed_size=(400, 400)):
    """Segment the image based on selected classes with consistent output sizes"""
    # Process the image
    inputs = segformer_processor(images=image, return_tensors="pt")
    
    # Get model predictions
    outputs = segformer_model(**inputs)
    logits = outputs.logits.cpu()
    
    # Upsample the logits to match the original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )
    
    # Get the predicted segmentation map
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # Filter classes if specified
    if selected_classes and len(selected_classes) > 0:
        # Create a mask for selected classes
        mask = np.zeros_like(pred_seg, dtype=bool)
        for class_name in selected_classes:
            if class_name in class_names:
                class_idx = class_names.index(class_name)
                mask = np.logical_or(mask, pred_seg == class_idx)
        
        # Apply the mask to keep only selected classes, set others to background (0)
        filtered_seg = np.zeros_like(pred_seg)
        filtered_seg[mask] = pred_seg[mask]
        pred_seg = filtered_seg
    
    # Create a colored segmentation map
    colored_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3))
    for class_idx in range(len(class_names)):
        mask = pred_seg == class_idx
        if mask.any():
            colored_seg[mask] = color_map(class_idx)[:3]
    
    # Create an overlay of the segmentation on the original image
    image_array = np.array(image)
    overlay = image_array.copy()
    alpha = 0.5  # Transparency factor
    mask = pred_seg > 0  # Exclude background
    overlay[mask] = overlay[mask] * (1 - alpha) + colored_seg[mask] * 255 * alpha
    
    # Prepare output images based on user selection
    outputs = []
    
    if show_original:
        # Resize original image to ensure consistent size
        resized_original = image.resize(fixed_size)
        outputs.append(resized_original)
    
    if show_segmentation:
        seg_image = Image.fromarray((colored_seg * 255).astype('uint8'))
        # Ensure segmentation has consistent size
        seg_image = seg_image.resize(fixed_size)
        outputs.append(seg_image)
    
    if show_overlay:
        overlay_image = Image.fromarray(overlay.astype('uint8'))
        # Ensure overlay has consistent size
        overlay_image = overlay_image.resize(fixed_size)
        outputs.append(overlay_image)
    
    # Create a legend for the segmentation classes
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_alpha(0.0)
    ax.axis('off')
    
    # Create legend patches
    legend_elements = []
    for i, class_name in enumerate(class_names):
        if i == 0 and selected_classes:  # Skip background in legend if filtering
            continue
        if not selected_classes or class_name in selected_classes:
            color = color_map(i)[:3]
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color))
    
    # Only add legend if there are elements to show
    if legend_elements:
        legend_class_names = [name for name in class_names if name != "Background" and (not selected_classes or name in selected_classes)]
        ax.legend(legend_elements, legend_class_names, loc='center', ncol=6)
    
    # Save the legend to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    legend_img = Image.open(buf)
    
    plt.close(fig)
    
    outputs.append(legend_img)
    
    return outputs

def identify_garment_segformer(image):
    """Identify the dominant garment type using SegFormer"""
    # Process the image
    inputs = segformer_processor(images=image, return_tensors="pt")
    
    # Get model predictions
    outputs = segformer_model(**inputs)
    logits = outputs.logits.cpu()
    
    # Upsample the logits to match the original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )
    
    # Get the predicted segmentation map
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # Count the pixels for each class (excluding background)
    class_counts = Counter(pred_seg.flatten())
    if 0 in class_counts:  # Remove background
        del class_counts[0]
    
    # Find the most common clothing item
    clothing_classes = [4, 5, 6, 7]  # Upper-clothes, Skirt, Pants, Dress
    
    # Filter to only include clothing items
    clothing_counts = {k: v for k, v in class_counts.items() if k in clothing_classes}
    
    if not clothing_counts:
        return "No garment detected", None
    
    # Get the most common clothing item
    dominant_class = max(clothing_counts.items(), key=lambda x: x[1])[0]
    dominant_class_name = class_names[dominant_class]
    
    return dominant_class_name, dominant_class
