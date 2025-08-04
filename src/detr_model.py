"""
DETR Model Manager
Handles DETR model loading, inference, and visualization for object detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import requests
import io
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
    from transformers import DetrForSegmentation
except ImportError:
    print("Warning: transformers library not found. Please install it for DETR functionality.")


class DETRModelManager:
    """DETR Model Manager for object detection and segmentation"""
    
    def __init__(self, model_name: str = "facebook/detr-resnet-50", 
                 task: str = "object-detection", device: str = None):
        """Initialize DETR model manager
        
        Args:
            model_name: Name of the DETR model
            task: Task type ('object-detection' or 'segmentation')
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.processor = None
        self.config = None
        
        # Model configuration
        self.model_config = {
            "model_name": model_name,
            "task": task,
            "device": self.device,
            "confidence_threshold": 0.5,
            "num_queries": 100,
            "hidden_size": 256,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "num_attention_heads": 8,
            "intermediate_size": 2048,
            "dropout": 0.1,
            "activation_dropout": 0.0,
            "attention_dropout": 0.0,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "is_encoder_decoder": True,
            "image_size": 800,
            "num_channels": 3,
            "num_labels": 91,  # COCO classes
        }
    
    def load_model(self) -> bool:
        """Load DETR model and processor
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading DETR model: {self.model_name}")
            print(f"Task: {self.task}")
            print(f"Device: {self.device}")
            
            # Load processor
            print("Loading processor...")
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            print("Processor loaded successfully")
            
            # Load model based on task
            print(f"Loading model for task: {self.task}")
            if self.task == "object-detection":
                self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            elif self.task == "segmentation":
                self.model = DetrForSegmentation.from_pretrained(self.model_name)
            else:
                raise ValueError(f"Unsupported task: {self.task}")
            
            print("Model loaded successfully")
            
            # Move to device
            print(f"Moving model to device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            
            # Load config
            self.config = self.model.config
            
            print(f"DETR model loaded successfully on {self.device}")
            print(f"Model type: {type(self.model)}")
            print(f"Model is None: {self.model is None}")
            return True
            
        except Exception as e:
            print(f"Error loading DETR model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information
        
        Returns:
            Dict[str, Any]: Model information
        """
        print(f"get_model_info called")
        print(f"self.model is None: {self.model is None}")
        print(f"self.model type: {type(self.model)}")
        print(f"self.config is None: {self.config is None}")
        
        if self.model is None:
            print("Model is None, returning error")
            return {"error": "Model not loaded"}
        
        if self.config is None:
            print("Config is None, returning error")
            return {"error": "Model config not loaded"}
        
        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "model_name": self.model_name,
                "task": self.task,
                "device": self.device,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "config": self.model_config,
                "model_config": {
                    "num_queries": self.config.num_queries,
                    "hidden_size": self.config.hidden_size,
                    "num_encoder_layers": self.config.encoder_layers,
                    "num_decoder_layers": self.config.decoder_layers,
                    "num_attention_heads": self.config.num_attention_heads,
                    "intermediate_size": self.config.encoder_ffn_dim,
                    "dropout": self.config.dropout,
                    "num_labels": self.config.num_labels,
                }
            }
        except Exception as e:
            print(f"Error in get_model_info: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": f"Error getting model info: {str(e)}"}
    
    def detect_objects(self, image: Image.Image, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Detect objects in image
        
        Args:
            image: Input image
            confidence_threshold: Confidence threshold for detections
            
        Returns:
            Dict[str, Any]: Detection results
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded")
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )[0]
        
        # Extract results
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()
        
        # Get COCO class names
        coco_names = self._get_coco_class_names()
        
        detections = []
        for score, label, box in zip(scores, labels, boxes):
            detections.append({
                "score": float(score),
                "label": int(label),
                "label_name": coco_names[label],
                "box": box.tolist(),  # [x1, y1, x2, y2]
                "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                "width": box[2] - box[0],
                "height": box[3] - box[1],
            })
        
        return {
            "detections": detections,
            "num_detections": len(detections),
            "image_size": image.size,
            "confidence_threshold": confidence_threshold,
            "model_outputs": outputs
        }
    
    def get_attention_maps(self, image: Image.Image) -> Dict[str, Any]:
        """Get attention maps from DETR model
        
        Args:
            image: Input image
            
        Returns:
            Dict[str, Any]: Attention maps and model states
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded")
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference with attention outputs
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        
        # Extract attention maps
        attention_data = {
            "encoder_attentions": [],
            "decoder_attentions": [],
            "cross_attentions": [],
            "encoder_hidden_states": [],
            "decoder_hidden_states": [],
            "last_hidden_state": outputs.last_hidden_state.cpu().numpy(),
        }
        
        # Process encoder attentions
        if outputs.encoder_attentions:
            for layer_idx, layer_attentions in enumerate(outputs.encoder_attentions):
                # Average across heads
                avg_attention = layer_attentions.mean(dim=1).cpu().numpy()  # [batch, seq_len, seq_len]
                attention_data["encoder_attentions"].append({
                    "layer": layer_idx,
                    "attention": avg_attention[0],  # Remove batch dimension
                })
        
        # Process decoder attentions
        if outputs.decoder_attentions:
            for layer_idx, layer_attentions in enumerate(outputs.decoder_attentions):
                avg_attention = layer_attentions.mean(dim=1).cpu().numpy()
                attention_data["decoder_attentions"].append({
                    "layer": layer_idx,
                    "attention": avg_attention[0],
                })
        
        # Process cross attentions
        if outputs.cross_attentions:
            for layer_idx, layer_attentions in enumerate(outputs.cross_attentions):
                avg_attention = layer_attentions.mean(dim=1).cpu().numpy()
                attention_data["cross_attentions"].append({
                    "layer": layer_idx,
                    "attention": avg_attention[0],
                })
        
        # Process hidden states
        if outputs.encoder_hidden_states:
            attention_data["encoder_hidden_states"] = [
                hidden.cpu().numpy() for hidden in outputs.encoder_hidden_states
            ]
        
        if outputs.decoder_hidden_states:
            attention_data["decoder_hidden_states"] = [
                hidden.cpu().numpy() for hidden in outputs.decoder_hidden_states
            ]
        
        return attention_data
    
    def get_object_queries_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze object queries behavior
        
        Args:
            image: Input image
            
        Returns:
            Dict[str, Any]: Object queries analysis
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded")
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Analyze object queries
        queries_analysis = {
            "num_queries": self.config.num_queries,
            "query_embeddings": [],
            "query_predictions": [],
            "query_attention": [],
        }
        
        # Get query embeddings from decoder
        if outputs.decoder_hidden_states:
            # Get embeddings from each decoder layer
            for layer_idx, hidden_state in enumerate(outputs.decoder_hidden_states):
                # hidden_state shape: [batch, num_queries, hidden_size]
                query_emb = hidden_state[0].cpu().numpy()  # Remove batch dimension
                queries_analysis["query_embeddings"].append({
                    "layer": layer_idx,
                    "embeddings": query_emb,
                    "mean_activation": np.mean(query_emb, axis=1),
                    "std_activation": np.std(query_emb, axis=1),
                })
        
        # Get predictions for each query
        logits = outputs.logits[0].cpu().numpy()  # [num_queries, num_classes + 1]
        boxes = outputs.pred_boxes[0].cpu().numpy()  # [num_queries, 4]
        
        for query_idx in range(self.config.num_queries):
            query_logits = logits[query_idx]
            query_box = boxes[query_idx]
            
            # Get class probabilities
            probs = F.softmax(torch.tensor(query_logits), dim=0).numpy()
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]
            
            queries_analysis["query_predictions"].append({
                "query_idx": query_idx,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "box": query_box.tolist(),
                "class_probabilities": probs.tolist(),
            })
        
        return queries_analysis
    
    def _get_coco_class_names(self) -> List[str]:
        """Get COCO class names
        
        Returns:
            List[str]: COCO class names
        """
        return [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]


class DETRVisualizer:
    """DETR visualization utilities"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))
    
    def visualize_detections(self, image: Image.Image, detections: List[Dict], 
                           title: str = "DETR Object Detection") -> plt.Figure:
        """Visualize object detections
        
        Args:
            image: Input image
            detections: List of detections
            title: Plot title
            
        Returns:
            plt.Figure: Detection visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display image
        ax.imshow(image)
        
        # Draw bounding boxes
        for i, detection in enumerate(detections):
            box = detection["box"]
            label = detection["label_name"]
            score = detection["score"]
            
            # Create rectangle
            rect = patches.Rectangle(
                (box[0], box[1]), 
                box[2] - box[0], 
                box[3] - box[1],
                linewidth=2, 
                edgecolor=self.colors[i % len(self.colors)], 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                box[0], box[1] - 10, 
                f"{label}: {score:.2f}",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors[i % len(self.colors)], alpha=0.7),
                fontsize=10, 
                color='white'
            )
        
        ax.set_title(title)
        ax.axis('off')
        
        return fig
    
    def visualize_attention_maps(self, attention_maps: Dict[str, Any], 
                                layer_type: str = "encoder", 
                                layer_idx: int = 0) -> plt.Figure:
        """Visualize attention maps
        
        Args:
            attention_maps: Attention maps data
            layer_type: Type of attention ('encoder', 'decoder', 'cross')
            layer_idx: Layer index
            
        Returns:
            plt.Figure: Attention visualization
        """
        if f"{layer_type}_attentions" not in attention_maps:
            return None
        
        attentions = attention_maps[f"{layer_type}_attentions"]
        if layer_idx >= len(attentions):
            return None
        
        attention_matrix = attentions[layer_idx]["attention"]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
        ax.set_title(f'{layer_type.capitalize()} Attention - Layer {layer_idx}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')
        
        return fig
    
    def visualize_query_analysis(self, query_analysis: Dict[str, Any]) -> plt.Figure:
        """Visualize object queries analysis
        
        Args:
            query_analysis: Query analysis data
            
        Returns:
            plt.Figure: Query analysis visualization
        """
        num_queries = query_analysis["num_queries"]
        predictions = query_analysis["query_predictions"]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confidence distribution
        confidences = [pred["confidence"] for pred in predictions]
        axes[0, 0].hist(confidences, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Query Confidence Distribution')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Number of Queries')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Top predictions
        top_predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)[:10]
        query_indices = [pred["query_idx"] for pred in top_predictions]
        top_confidences = [pred["confidence"] for pred in top_predictions]
        
        axes[0, 1].bar(range(len(top_predictions)), top_confidences, color='lightcoral')
        axes[0, 1].set_title('Top 10 Query Confidences')
        axes[0, 1].set_xlabel('Query Index')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].set_xticks(range(len(top_predictions)))
        axes[0, 1].set_xticklabels(query_indices, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Class distribution
        class_counts = {}
        for pred in predictions:
            if pred["confidence"] > 0.1:  # Only count confident predictions
                class_name = pred.get("predicted_class", "unknown")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts:
            classes = list(class_counts.keys())[:10]  # Top 10 classes
            counts = [class_counts[c] for c in classes]
            
            axes[1, 0].bar(range(len(classes)), counts, color='lightgreen')
            axes[1, 0].set_title('Top Predicted Classes')
            axes[1, 0].set_xlabel('Class')
            axes[1, 0].set_ylabel('Number of Queries')
            axes[1, 0].set_xticks(range(len(classes)))
            axes[1, 0].set_xticklabels(classes, rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Query embeddings statistics
        if query_analysis["query_embeddings"]:
            embeddings = query_analysis["query_embeddings"][-1]  # Last layer
            mean_activations = embeddings["mean_activation"]
            
            axes[1, 1].plot(range(num_queries), mean_activations, 'o-', color='purple', alpha=0.7)
            axes[1, 1].set_title('Query Embedding Mean Activations')
            axes[1, 1].set_xlabel('Query Index')
            axes[1, 1].set_ylabel('Mean Activation')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_detr_architecture_diagram(self) -> plt.Figure:
        """Create DETR architecture diagram
        
        Returns:
            plt.Figure: Architecture diagram
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Define components
        components = [
            ("Input Image\n(3×H×W)", (1, 8), "lightblue"),
            ("CNN Backbone\n(ResNet-50)", (3, 8), "lightgreen"),
            ("Feature Map\n(2048×H/32×W/32)", (5, 8), "lightyellow"),
            ("Projection\n(256×H/32×W/32)", (7, 8), "lightcoral"),
            ("Flatten\n(HW/1024×256)", (9, 8), "lightgray"),
            ("Transformer\nEncoder", (11, 8), "lightpink"),
            ("Object Queries\n(100×256)", (9, 4), "lightcyan"),
            ("Transformer\nDecoder", (11, 4), "lightyellow"),
            ("Classification\nHead", (13, 6), "lightgreen"),
            ("Bounding Box\nHead", (13, 10), "lightgreen"),
        ]
        
        # Draw components
        for name, pos, color in components:
            rect = patches.FancyBboxPatch(
                (pos[0]-0.5, pos[1]-0.5), 1, 1,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], name, ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Draw arrows
        arrows = [
            ((1, 8), (3, 8)),
            ((3, 8), (5, 8)),
            ((5, 8), (7, 8)),
            ((7, 8), (9, 8)),
            ((9, 8), (11, 8)),
            ((9, 4), (11, 4)),
            ((11, 8), (11, 4)),
            ((11, 4), (13, 6)),
            ((11, 4), (13, 10)),
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 12)
        ax.set_title('DETR Architecture', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        return fig 