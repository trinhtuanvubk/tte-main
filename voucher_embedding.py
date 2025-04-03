import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple

# Configuration
class Config:
    # PhoBERT model for Vietnamese text embedding
    BERT_MODEL_NAME = "vinai/phobert-base"  # Vietnamese BERT model
    MAX_TEXT_LENGTH = 128
    EMBEDDING_DIM = 768  # PhoBERT base embedding dimension
    CATEGORICAL_EMBEDDING_DIM = 32
    FINAL_EMBEDDING_DIM = 256
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Voucher Tower Model
class VoucherTowerModel(nn.Module):
    def __init__(self, 
                variant_name_embedding_dim: int = 768,
                category_embedding_dim: int = 768,
                out_project_embedding_dim: int = 256,
                final_embedding: int = 128

                ):
        # Category: 'category_brand_name', 'category_partner_new', 'sub_category_partner_new'

        super(VoucherTowerModel, self).__init__()
        
        
        self.variant_name_projection = nn.Sequential(
                        nn.Linear(variant_name_embedding_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, out_project_embedding_dim))
        
        self.category_embedding_projection = nn.Sequential(
                        nn.Linear(category_embedding_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, out_project_embedding_dim))
        
        total_concat_dim = out_project_embedding_dim + out_project_embedding_dim

        self.embedding_network = nn.Sequential(
            nn.Linear(total_concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, final_embedding),
            nn.LayerNorm(final_embedding))

        
    def forward(self, 
                variant_name_embedding: torch.Tensor,
                category_embedding: torch.Tensor):
        
        variant_name_emb = self.variant_name_projection(variant_name_embedding)

        category_emb = self.category_embedding_projection(category_embedding)

        all_embeddings = [variant_name_emb, category_emb]

        concat_emb = torch.cat(all_embeddings, dim=1)

        voucher_emb = self.embedding_network(concat_emb)
        
        return voucher_emb
