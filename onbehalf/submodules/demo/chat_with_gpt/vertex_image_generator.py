#!/usr/bin/env python3
"""
Vertex AI Image Generator module for generating images using Google's Vertex AI.
"""

import os
import time
import base64
from typing import Optional, Dict, Any, Union, List
import json
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Import Vertex AI libraries
try:
    from vertexai.preview.vision_models import ImageGenerationModel
    import vertexai
    VERTEX_SDK_AVAILABLE = True
except ImportError:
    print("Warning: Vertex AI libraries not installed.")
    print("Please install with: pip install google-cloud-aiplatform vertexai")
    VERTEX_SDK_AVAILABLE = False

class VertexImageGenerator:
    """Class to generate images using Google's Vertex AI Imagen model."""
    
    def __init__(self):
        """Initialize the Vertex AI image generator."""
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "black-radius-453008-r7")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.output_dir = os.getenv("IMAGE_OUTPUT_DIR", "generated_images")
        self.model_name = "imagen-3.0-generate-002"
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if credentials file exists and is valid
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            print("WARNING: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
            print("Using default application credentials.")
        elif not os.path.exists(credentials_path):
            print(f"WARNING: Credentials file not found: {credentials_path}")
            print("Using default application credentials.")
        else:
            print(f"Using credentials from: {credentials_path}")
            # Check if the file is a valid JSON
            try:
                with open(credentials_path, 'r') as f:
                    json.load(f)
                print("Credentials file is valid JSON format.")
            except json.JSONDecodeError:
                print(f"ERROR: Credentials file is not valid JSON: {credentials_path}")
        
        # Initialize Vertex AI if SDK is available
        if VERTEX_SDK_AVAILABLE:
            self._init_vertex_ai()
        else:
            print("Vertex AI SDK not available. Please install required packages.")
    
    def _init_vertex_ai(self):
        """Initialize Vertex AI client."""
        if not VERTEX_SDK_AVAILABLE:
            return False
            
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.generation_model = ImageGenerationModel.from_pretrained(self.model_name)
            print(f"Successfully initialized Vertex AI with project {self.project_id} in {self.location}")
            print(f"Using model: {self.model_name}")
            return True
        except Exception as e:
            print(f"Error initializing Vertex AI: {e}")
            return False
    
    def generate_image(
        self, 
        prompt: str, 
        negative_prompt: str = "", 
        width: int = 1024, 
        height: int = 1024,
        samples: int = 1,
        add_watermark: bool = True
    ) -> Union[str, List[str]]:
        """
        Generate images using Vertex AI Imagen model.
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: Elements to avoid in the image
            width: Width of the generated image
            height: Height of the generated image
            samples: Number of images to generate
            add_watermark: Whether to add a watermark to the generated images
            
        Returns:
            Path to the generated image file(s)
        """
        print(f"Generating image with prompt: {prompt}")
        
        if not VERTEX_SDK_AVAILABLE:
            return self._create_error_image("Vertex AI SDK not available. Please install required packages.")
        
        try:
            # Determine aspect ratio based on width and height
            if width == height:
                aspect_ratio = "1:1"
            elif width > height:
                aspect_ratio = "16:9" if width / height >= 1.7 else "4:3"
            else:
                aspect_ratio = "9:16" if height / width >= 1.7 else "3:4"
            
            print(f"Using aspect ratio: {aspect_ratio}")
            
            # Generate images using the Vertex AI model
            images = self.generation_model.generate_images(
                prompt=prompt,
                number_of_images=samples,
                aspect_ratio=aspect_ratio,
                negative_prompt=negative_prompt,
                add_watermark=add_watermark
            )
            
            # Save all generated images
            saved_paths = []
            for i, image in enumerate(images):
                # Create a unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                short_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c.isspace()).strip().replace(" ", "_")
                filename = f"{timestamp}_{short_prompt}_{i+1}.png"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save the image
                image_bytes = image._image_bytes
                with open(filepath, "wb") as f:
                    f.write(image_bytes)
                
                print(f"Image {i+1} saved to {filepath}")
                saved_paths.append(filepath)
            
            # Return a single path if only one image was generated, otherwise return the list of paths
            return saved_paths[0] if len(saved_paths) == 1 else saved_paths
            
        except Exception as e:
            print(f"Error generating image: {e}")
            # Create a placeholder error image with the error message
            return self._create_error_image(str(e))
    
    def _create_error_image(self, error_message: str) -> str:
        """Create a simple error image with the error message."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a blank image
            img = Image.new('RGB', (800, 400), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            
            # Add error message
            d.text((20, 20), f"Error generating image:\n{error_message}", fill=(255, 0, 0))
            
            # Save the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"{timestamp}_error.png")
            img.save(filepath)
            
            print(f"Error image saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error creating error image: {e}")
            return "Error: Failed to generate image and error image"

def display_image(image):
    """
    Display an image. This function is a utility for notebooks.
    In a regular application, you would typically return the file path.
    
    Args:
        image: The image object from the Vertex AI model or a file path
    """
    try:
        if isinstance(image, str):
            # If image is a file path
            img = Image.open(image)
            img.show()
        else:
            # If image is from the Vertex AI model
            image_bytes = image._image_bytes
            img = Image.open(io.BytesIO(image_bytes))
            img.show()
    except Exception as e:
        print(f"Error displaying image: {e}") 