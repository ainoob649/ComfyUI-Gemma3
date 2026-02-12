import os
import sys
import torch
import numpy as np
import logging
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from huggingface_hub import snapshot_download
import folder_paths
from comfy import model_management

# Setup model paths
MODELS_DIR = os.path.join(folder_paths.models_dir, "LLM", "gemma")
os.makedirs(MODELS_DIR, exist_ok=True)

# Flash Attention Availability Logic
FLASH_ATTENTION_IS_AVAILABLE = False
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_IS_AVAILABLE = True
except ImportError:
    pass
    
    
def tensor_to_pil(image):
    # ComfyUI images are [B, H, W, C]
    # We take the first image in the batch [0]
    i = 255. * image[0].cpu().numpy()
    return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    

class Gemma3ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"], {"default": "google/gemma-3-4b-it"}),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "compile_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("GEMMA3_MODEL", "GEMMA3_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "Gemma3"

    def load_model(self, model_name, quantization, compile_model):
        device = model_management.get_torch_device()
        torch_dtype = getattr(torch, "bfloat16")
        
        local_path = os.path.join(MODELS_DIR, model_name.split("/")[-1])
        
        if not os.path.exists(local_path):
            logging.info(f"Gemma3: Model not found locally. Downloading {model_name} to {local_path}...")
            snapshot_download(repo_id=model_name, local_dir=local_path, local_dir_use_symlinks=False)

        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Apply requested attention logic
        if FLASH_ATTENTION_IS_AVAILABLE and model_management.flash_attention_enabled():
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"

        logging.info(f"Gemma3: Loading model with {attn_impl} attention...")
        
        processor = AutoProcessor.from_pretrained(local_path)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            local_path,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            device_map="auto" if bnb_config else None,
            attn_implementation=attn_impl
        )
        
        # Optimization: Move to Eval mode
        model.eval()
                
        if compile_model:
            try:
                # Use 'reduce-overhead' to maximize GPU utilization
                model = torch.compile(model, mode="reduce-overhead")
                logging.info("Gemma3: Model compiled successfully.")
            except Exception as e:
                logging.warning(f"Gemma3: Could not compile model: {e}")
        
        if not bnb_config:
            model.to(device)

        return (model, processor)

class ApplyGemma3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("GEMMA3_MODEL",),
                "processor": ("GEMMA3_PROCESSOR",),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_image"
    CATEGORY = "Gemma3"

    def analyze_image(self, model, processor, max_new_tokens, prompt, image=None):
        # Ensure model is on the right device
        device = model_management.get_torch_device()
    
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            }
        ]
        
        if image is not None:
            image_pil = tensor_to_pil(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": prompt}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            })

        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device)  # dtype is handled by processor
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Optimized Inference Mode
        with torch.inference_mode():
            # Move model to device if it's not already there (for non-quantized)
            if hasattr(model, "device") and model.device != device:
                model.to(device)
            
            generation = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                use_cache=True  # Faster inference
            )
            # Slice only the new tokens
            generated_tokens = generation[0][input_len:]

        output_text = processor.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return (output_text,)

NODE_CLASS_MAPPINGS = {
    "Gemma3ModelLoader": Gemma3ModelLoader,
    "ApplyGemma3": ApplyGemma3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemma3ModelLoader": "Gemma 3 Model Loader",
    "ApplyGemma3": "Apply Gemma 3"
}
