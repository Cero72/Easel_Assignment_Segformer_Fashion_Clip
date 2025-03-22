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
                person_image = gr.Image(type="pil", label="Upload a person wearing clothes", height=300)
                
                # Garment image section
                gr.Markdown("### Garment Image")
                garment_image = gr.Image(type="pil", label="Upload a garment to detect", height=300)
                
                with gr.Row():
                    show_original_dual = gr.Checkbox(label="Show Original", value=True)
                    show_segmentation_dual = gr.Checkbox(label="Show Segmentation", value=True)
                    show_overlay_dual = gr.Checkbox(label="Show Overlay", value=True)
                
                process_button = gr.Button("Process Images", variant="primary")
            
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
        
        # Add custom CSS for consistent image sizes
        gr.HTML("""
        <style>
        .gradio-container img {
            max-height: 400px !important;
            object-fit: contain !important;
        }
        #dual_gallery {
            min-height: 450px;
        }
        </style>
        """)
    
    return demo

# Main application entry point
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
