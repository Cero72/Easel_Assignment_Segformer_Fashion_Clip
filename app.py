import gradio as gr
import os

# Import from our modules
from constants import class_names
from segmentation import segment_image
from utils import process_url, process_person_and_garment

# Define fixed size for consistent image display
FIXED_IMAGE_SIZE = (400, 400)

def create_interface():
    """Create the Gradio interface with improved image consistency"""
    with gr.Blocks(title="Garment-based Segmentation") as demo:
        gr.Markdown("""
        # Garment-based Segmentation with SegFormer and Fashion-CLIP
        
        This application uses AI models to segment specific clothing items in images by matching a garment to a person.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Person image section
                gr.Markdown("### Person Image")
                person_image = gr.Image(
                    type="pil", 
                    label="Upload a person wearing clothes", 
                    height=300,
                    sources=["upload", "webcam", "clipboard"],
                    elem_id="person-image-upload"
                )
                
                # Garment image section
                gr.Markdown("### Garment Image")
                garment_image = gr.Image(
                    type="pil", 
                    label="Upload a garment to detect", 
                    height=300,
                    sources=["upload", "webcam", "clipboard"],
                    elem_id="garment-image-upload"
                )
                
                with gr.Row():
                    show_original_dual = gr.Checkbox(label="Show Original", value=True)
                    show_segmentation_dual = gr.Checkbox(label="Show Segmentation", value=True)
                    show_overlay_dual = gr.Checkbox(label="Show Overlay", value=True)
                
                process_button = gr.Button(
                    "Process Images", 
                    variant="primary",
                    size="lg",
                    elem_id="process-button"
                )
            
            with gr.Column(scale=2):
                dual_output_images = gr.Gallery(
                    label="Results", 
                    columns=3, 
                    height=450,
                    object_fit="contain",
                    elem_id="dual_gallery"
                )
                result_text = gr.Textbox(label="Result", interactive=False, lines=4)
        
        # Set up event handler for dual image processing
        process_button.click(
            fn=lambda p_img, g_img, orig, seg, over: process_person_and_garment(p_img, g_img, orig, seg, over, FIXED_IMAGE_SIZE),
            inputs=[person_image, garment_image, show_original_dual, show_segmentation_dual, show_overlay_dual],
            outputs=[dual_output_images, result_text]
        )
        
        # Add custom CSS for consistent image sizes and improved UI
        gr.HTML("""
        <style>
        .gradio-container img {
            max-height: 400px !important;
            object-fit: contain !important;
        }
        #dual_gallery {
            min-height: 450px;
        }
        /* Larger upload buttons */
        #person-image-upload .upload-button, 
        #garment-image-upload .upload-button {
            font-size: 1.2em !important;
            padding: 12px 20px !important;
            border-radius: 8px !important;
            margin: 10px auto !important;
            display: block !important;
            width: 80% !important;
            text-align: center !important;
            background-color: #4CAF50 !important;
            color: white !important;
            border: none !important;
            cursor: pointer !important;
            transition: background-color 0.3s ease !important;
        }
        #person-image-upload .upload-button:hover, 
        #garment-image-upload .upload-button:hover {
            background-color: #45a049 !important;
        }
        /* Larger process button */
        #process-button {
            font-size: 1.3em !important;
            padding: 15px 25px !important;
            margin: 15px auto !important;
            display: block !important;
            width: 90% !important;
        }
        /* Better section headers */
        h3 {
            font-size: 1.5em !important;
            margin-top: 20px !important;
            margin-bottom: 15px !important;
            color: #2c3e50 !important;
            border-bottom: 2px solid #3498db !important;
            padding-bottom: 8px !important;
        }
        /* Better main heading */
        h1 {
            color: #2c3e50 !important;
            text-align: center !important;
            margin-bottom: 30px !important;
            font-size: 2.5em !important;
        }
        /* Better checkbox layout */
        .gradio-checkbox {
            margin: 10px 5px !important;
        }
        </style>
        """)
    
    return demo

# Main application entry point
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
