"""
Sample Images Utility
Downloads and manages sample images for DETR demonstration.
"""

import requests
import os
from PIL import Image
import io
from typing import List, Dict, Optional
import numpy as np


class SampleImageManager:
    """Manager for sample images used in DETR demonstration"""
    
    def __init__(self, samples_dir: str = "samples"):
        """Initialize sample image manager
        
        Args:
            samples_dir: Directory to store sample images
        """
        self.samples_dir = samples_dir
        os.makedirs(samples_dir, exist_ok=True)
        
        # Sample image URLs (COCO dataset examples)
        self.sample_urls = {
            "people": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "animals": "http://images.cocodataset.org/val2017/000000000285.jpg",
            "vehicles": "http://images.cocodataset.org/val2017/000000000724.jpg",
            "objects": "http://images.cocodataset.org/val2017/000000000139.jpg",
            "food": "http://images.cocodataset.org/val2017/000000000632.jpg",
        }
        
        # Sample image descriptions
        self.sample_descriptions = {
            "people": "People in a room with furniture",
            "animals": "Animals including cats and dogs",
            "vehicles": "Various vehicles on the road",
            "objects": "Everyday objects and furniture",
            "food": "Food items and kitchen objects",
        }
    
    def download_sample_images(self) -> Dict[str, str]:
        """Download sample images
        
        Returns:
            Dict[str, str]: Dictionary mapping image names to file paths
        """
        downloaded_images = {}
        
        for name, url in self.sample_urls.items():
            try:
                print(f"Downloading {name} image...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Save image
                image_path = os.path.join(self.samples_dir, f"{name}.jpg")
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded_images[name] = image_path
                print(f"✅ Downloaded {name} image to {image_path}")
                
            except Exception as e:
                print(f"❌ Failed to download {name} image: {str(e)}")
        
        return downloaded_images
    
    def get_sample_images(self) -> Dict[str, Dict]:
        """Get available sample images
        
        Returns:
            Dict[str, Dict]: Dictionary with image information
        """
        images = {}
        
        for name, url in self.sample_urls.items():
            image_path = os.path.join(self.samples_dir, f"{name}.jpg")
            
            if os.path.exists(image_path):
                try:
                    # Load image to get size
                    with Image.open(image_path) as img:
                        size = img.size
                    
                    images[name] = {
                        "path": image_path,
                        "url": url,
                        "description": self.sample_descriptions[name],
                        "size": size,
                        "exists": True
                    }
                except Exception as e:
                    print(f"Error loading {name} image: {str(e)}")
                    images[name] = {
                        "path": image_path,
                        "url": url,
                        "description": self.sample_descriptions[name],
                        "size": None,
                        "exists": False
                    }
            else:
                images[name] = {
                    "path": image_path,
                    "url": url,
                    "description": self.sample_descriptions[name],
                    "size": None,
                    "exists": False
                }
        
        return images
    
    def load_sample_image(self, name: str) -> Optional[Image.Image]:
        """Load a specific sample image
        
        Args:
            name: Name of the sample image
            
        Returns:
            Optional[Image.Image]: Loaded image or None if not found
        """
        image_path = os.path.join(self.samples_dir, f"{name}.jpg")
        
        if os.path.exists(image_path):
            try:
                return Image.open(image_path)
            except Exception as e:
                print(f"Error loading image {name}: {str(e)}")
                return None
        else:
            print(f"Image {name} not found at {image_path}")
            return None
    
    def create_synthetic_image(self, width: int = 800, height: int = 600) -> Image.Image:
        """Create a synthetic image for testing
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Image.Image: Synthetic test image
        """
        # Create a synthetic image with simple shapes
        image = Image.new('RGB', (width, height), color='white')
        
        # Add some colored rectangles to simulate objects
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        
        for i, color in enumerate(colors):
            # Create a simple colored rectangle
            x1 = 50 + i * 100
            y1 = 50 + i * 50
            x2 = x1 + 80
            y2 = y1 + 80
            
            # Create a colored rectangle
            rect_image = Image.new('RGB', (80, 80), color=color)
            image.paste(rect_image, (x1, y1))
        
        return image
    
    def get_image_info(self, image_path: str) -> Dict:
        """Get information about an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dict: Image information
        """
        try:
            with Image.open(image_path) as img:
                return {
                    "size": img.size,
                    "mode": img.mode,
                    "format": img.format,
                    "width": img.width,
                    "height": img.height,
                    "aspect_ratio": img.width / img.height,
                }
        except Exception as e:
            return {"error": str(e)} 