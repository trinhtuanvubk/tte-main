# import torch
# from FlagEmbedding import BGEM3FlagModel

# class TextEmbedder:
#     def __init__(self, model_name="BAAI/bge-m3"):
#         """
#         Initialize the text embedder with the BGE M3 model
        
#         Args:
#             model_name: Name of the pre-trained model to use (default: BAAI/bge-m3)
#         """
#         self.model_name = model_name
#         self.model = BGEM3FlagModel(model_name, use_fp16=True)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     def get_embedding(self, text, max_length=8192):
#         """
#         Get text embedding for a single text
        
#         Args:
#             text: Input text to embed
#             max_length: Maximum sequence length for encoding
            
#         Returns:
#             Tensor with text embedding
#         """
#         if not text or text == "":
#             # Return zero embedding for empty text 
#             # Let's assume 1024 is the embedding dimension, adjust as needed
#             return torch.zeros(1, 1024)
        
#         # Encode the text using BGE M3 model
#         embeddings = self.model.encode(
#             [text], 
#             batch_size=1,
#             max_length=max_length
#         )['dense_vecs']
        
#         # Convert to PyTorch tensor
#         embedding_tensor = torch.tensor(embeddings)
        
#         return embedding_tensor
    
#     def batch_get_embeddings(self, texts, max_length=8192, batch_size=12):
#         """
#         Get text embeddings for a batch of texts
        
#         Args:
#             texts: List of input texts to embed
#             max_length: Maximum sequence length for encoding
#             batch_size: Batch size for encoding
            
#         Returns:
#             Tensor with batch of text embeddings
#         """
#         if not texts:
#             # Return empty tensor with the right dimension
#             return torch.zeros(0, 1024)
        
#         # Filter out empty texts
#         non_empty_texts = [text for text in texts if text and text != ""]
        
#         if not non_empty_texts:
#             return torch.zeros(len(texts), 1024)
        
#         # Encode the texts using BGE M3 model
#         embeddings = self.model.encode(
#             non_empty_texts, 
#             batch_size=batch_size,
#             max_length=max_length
#         )['dense_vecs']
        
#         # Convert to PyTorch tensor
#         embedding_tensor = torch.tensor(embeddings)
        
#         # If there were empty texts, we need to reconstruct the full tensor
#         if len(non_empty_texts) < len(texts):
#             result = torch.zeros(len(texts), embedding_tensor.shape[1])
#             non_empty_idx = 0
#             for i, text in enumerate(texts):
#                 if text and text != "":
#                     result[i] = embedding_tensor[non_empty_idx]
#                     non_empty_idx += 1
#             return result
        
#         return embedding_tensor

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import re
import pandas as pd


class TextEmbedder:
    def __init__(self, model_name="vinai/phobert-base", max_length=128):
        """
        Initialize the PhoBERT model for Vietnamese text embedding
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        
        # Put model in evaluation mode and move to GPU if available
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def preprocess_text(self, text):
        """
        Preprocess text specifically for PhoBERT (Vietnamese text)
        """
        if not text or pd.isna(text) or text == "Unknown":
            return "không rõ"
        
        # Simple preprocessing for Vietnamese text
        text = text.lower()
        # Remove extra spaces and special characters
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_embedding(self, text):
        """
        Get embedding for a text string using PhoBERT
        """
        text = self.preprocess_text(text)
        
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use average of the last hidden state as embedding
            embeddings = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            
            # Apply mask and average (ignoring padding tokens)
            sum_embeddings = torch.sum(embeddings * mask, dim=1)
            sum_mask = torch.sum(mask, dim=1)
            mean_embeddings = sum_embeddings / sum_mask
            
            return mean_embeddings.cpu()