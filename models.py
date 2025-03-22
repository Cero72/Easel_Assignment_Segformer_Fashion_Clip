import torch
import torch.nn as nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation, CLIPProcessor, CLIPModel

# Load the SegFormer model and processor for segmentation
def load_segformer_model():
    """Load and return the SegFormer model and processor"""
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    return processor, model

# Load Fashion-CLIP model for garment classification
def load_clip_model():
    """Load and return the Fashion-CLIP model and processor"""
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    return model, processor

# Initialize models
segformer_processor, segformer_model = load_segformer_model()
clip_model, clip_processor = load_clip_model()
