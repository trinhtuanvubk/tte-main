import torch
import torch.nn as nn
from user_embedding import UserTowerModel
from voucher_embedding import VoucherTowerModel

# Enhanced Two-Tower Model
class EnhancedTwoTowerRecommender(nn.Module):
    def __init__(
        self, 
        user_tower: nn.Module,
        voucher_tower: nn.Module,
        temperature: float = 0.1
    ):
        """
        Enhanced two-tower recommendation model
        
        Args:
            user_tower: Enhanced user tower model
            voucher_tower: Voucher tower model
            temperature: Temperature parameter for softmax
        """
        super(EnhancedTwoTowerRecommender, self).__init__()
        self.user_tower = user_tower
        self.voucher_tower = voucher_tower
        self.temperature = temperature
    
    def forward(
        self,
        user_features,
        positive_voucher_features,
        negative_voucher_features=None
    ):
        """
        Forward pass of the enhanced two-tower model
        
        Args:
            user_features: Batch of user features
            positive_voucher_features: Batch of positive voucher features
            negative_voucher_features: List of batches of negative voucher features
        
        Returns:
            User embeddings, voucher embeddings, and similarity scores
        """
        # Get user embeddings
        user_embeddings = self.user_tower(
            user_features['gender_idx'],
            user_features['age'],
            user_features['vinid_tier_idx'],
            user_features['tcbr_tier_idx'],
            user_features['province_embedding'],
            user_features['job_embedding'],
            user_features['search_embedding'],
            user_features['claim_embedding']
        )
        
        # Get positive voucher embeddings
        pos_voucher_embeddings = self.voucher_tower(
            positive_voucher_features['variant_name_embedding'],
            positive_voucher_features['category_embedding']
        )
        
        # Normalize embeddings
        user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=1)
        pos_voucher_embeddings = nn.functional.normalize(pos_voucher_embeddings, p=2, dim=1)
        
        # Compute similarity for positive pairs
        pos_similarity = torch.sum(user_embeddings * pos_voucher_embeddings, dim=1) / self.temperature
        
        # If negative samples are provided, compute similarity for negative pairs
        if negative_voucher_features is not None:
            neg_similarities = []
            neg_voucher_embeddings_list = []
            
            for neg_features in negative_voucher_features:
                neg_voucher_embeddings = self.voucher_tower(
                    neg_features['variant_name_embedding'],
                    neg_features['category_embedding']
                )
                neg_voucher_embeddings = nn.functional.normalize(neg_voucher_embeddings, p=2, dim=1)
                neg_voucher_embeddings_list.append(neg_voucher_embeddings)
                
                neg_similarity = torch.sum(user_embeddings * neg_voucher_embeddings, dim=1) / self.temperature
                neg_similarities.append(neg_similarity)
            
            return user_embeddings, pos_voucher_embeddings, pos_similarity, neg_similarities, neg_voucher_embeddings_list
        
        return user_embeddings, pos_voucher_embeddings, pos_similarity