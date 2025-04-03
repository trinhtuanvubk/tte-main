
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random

class VoucherDataset(Dataset):
    def __init__(self, voucher_df, training_df, text_embedder):
        """
        Initialize the VoucherDataset with voucher data and text embedder
        
        Args:
            voucher_df: DataFrame containing voucher information
            training_df: DataFrame containing training data with user interactions
            text_embedder: Text embedder to generate embeddings for text fields
        """
        self.voucher_df = voucher_df
        self.training_df = training_df
        self.text_embedder = text_embedder

        # Create mapping from variant_id to voucher information
        self.variant_id_to_voucher = {}
        for idx, row in voucher_df.iterrows():
            variant_id = row['variant_id']
            if pd.notna(variant_id):
                self.variant_id_to_voucher[variant_id] = row
        
        # Create list of variant_ids from training data
        self.variant_ids = []
        for _, row in training_df.iterrows():
            # Check for claim variant columns (there might be multiple per row)
            for col in row.index:
                if 'claim' in col and 'variant_id' in col and pd.notna(row[col]):
                    variant_id = row[col]
                    if variant_id in self.variant_id_to_voucher and variant_id not in self.variant_ids:
                        self.variant_ids.append(variant_id)
        
        # Create index mapping
        self.variant_id_to_idx = {variant_id: idx for idx, variant_id in enumerate(self.variant_ids)}

        # Initialize dictionaries to store embeddings
        self.variant_name_embeddings = {}
        self.category_embeddings = {}
        
        # Pre-compute embeddings for all relevant vouchers
        for variant_id in self.variant_ids:
            # Get voucher information
            voucher_info = self.variant_id_to_voucher.get(variant_id)
            if voucher_info is None:
                continue
                
            # Generate variant name embedding
            variant_name = voucher_info.get('variant_name', '')
            if pd.notna(variant_name) and variant_name:
                self.variant_name_embeddings[variant_id] = text_embedder.get_embedding(variant_name).squeeze(0)
            else:
                # Use a zero tensor for empty variant names
                self.variant_name_embeddings[variant_id] = torch.zeros(768)  # Default size for PhoBERT
            
            # Generate category embedding by concatenating category fields
            category_name = voucher_info.get('category_partner_new', '')
            brand_name = voucher_info.get('category_brand_name', '')
            sub_category_name = voucher_info.get('sub_category_partner_new', '')
            
            # Concatenate the category fields
            category_text = " ".join([
                str(field) for field in [category_name, brand_name, sub_category_name] 
                if pd.notna(field) and field
            ])
            
            if category_text:
                self.category_embeddings[variant_id] = text_embedder.get_embedding(category_text).squeeze(0)
            else:
                # Use a zero tensor for empty category text
                self.category_embeddings[variant_id] = torch.zeros(768)  # Default size for PhoBERT
    
    def __len__(self):
        return len(self.voucher_df)
    
    def __getitem__(self, idx):
        """
        Get item by index
        
        Args:
            idx: Index of the voucher
            
        Returns:
            Dictionary with voucher features including embeddings
        """
        row = self.voucher_df.iloc[idx]
        variant_id = row['variant_id']
        
        # Get pre-computed embeddings
        variant_name_embedding = self.variant_name_embeddings.get(variant_id, torch.zeros(768))
        category_embedding = self.category_embeddings.get(variant_id, torch.zeros(768))
        
        # Create categorical features tensor if available
        categorical_features = []
        if 'voucher_discount_type_encoded' in row:
            categorical_features.append(row['voucher_discount_type_encoded'])
        if 'category_brand_name_encoded' in row:
            categorical_features.append(row['category_brand_name_encoded'])
        if 'category_partner_new_encoded' in row:
            categorical_features.append(row['category_partner_new_encoded'])
        if 'sub_category_partner_new_encoded' in row:
            categorical_features.append(row['sub_category_partner_new_encoded'])
        
        if categorical_features:
            categorical_features = torch.tensor(categorical_features, dtype=torch.long)
        else:
            categorical_features = None
        
        return {
            'variant_id': variant_id,
            'variant_name_embedding': variant_name_embedding,
            'category_embedding': category_embedding,
            'categorical_features': categorical_features
        }
    
    def get_embeddings_by_variant_id(self, variant_id):
        """
        Get embeddings for a specific variant_id
        
        Args:
            variant_id: ID of the voucher variant
            
        Returns:
            Dictionary with embeddings for the specified variant
        """
        if variant_id not in self.variant_id_to_idx:
            return None
        
        idx = self.variant_id_to_idx[variant_id]
        return self.__getitem__(idx)
    

class UserDataset(Dataset):
    def __init__(self, user_df, training_df, text_embedder):
        """
        Dataset for user data with search and claim history
        
        Args:
            user_df: DataFrame with user demographic data
            training_df: DataFrame with search and claim data
            text_embedder: TextEmbedder for embedding text
        """
        self.user_df = user_df
        self.training_df = training_df
        self.text_embedder = text_embedder
        
        # Create user_id to index mapping
        self.user_ids = user_df['user_id'].tolist()
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        
        # Create mappings for categorical features
        self.gender_to_idx = self._create_mapping(user_df['gender'].fillna('Unknown'))
        # self.province_to_idx = self._create_mapping(user_df['province_name'].fillna('Unknown'))
        self.vinid_tier_to_idx = self._create_mapping(user_df['VinID_tier_name_en'].fillna('Unknown'))
        self.tcbr_tier_to_idx = self._create_mapping(user_df['TCBR_tier_name_en'].fillna('Unknown'))
        
        # Pre-compute job embeddings
        self.job_embeddings = {}
        unique_jobs = user_df['job'].fillna('Unknown').unique()
        for job in unique_jobs:
            self.job_embeddings[job] = text_embedder.get_embedding(job).squeeze(0)

        # Pre-compute province embeddings
        self.province_embeddings = {}
        unique_province = user_df['job'].fillna('Unknown').unique()
        for province in unique_province:
            self.province_embeddings[province] = text_embedder.get_embedding(province).squeeze(0)
        
        
        # Pre-compute search embeddings
        self.search_embeddings = {}
        for _, row in training_df.iterrows():
            user_id = row['user_id']
            
            # Skip if user doesn't exist in user_df
            if user_id not in self.user_id_to_idx:
                continue
                
            # Combine all search data (keywords, categories, subcategories)
            search_text = ""
            for i in range(1, 4):  # For the 3 search fields
                search_key = f'search_{i}'
                category_key = f'search_{i}_category'
                subcategory_key = f'search_{i}_subcategory'
                
                if search_key in row and pd.notna(row[search_key]) and row[search_key]:
                    search_text += f" {row[search_key]}"
                if category_key in row and pd.notna(row[category_key]) and row[category_key]:
                    search_text += f" {row[category_key]}"
                if subcategory_key in row and pd.notna(row[subcategory_key]) and row[subcategory_key]:
                    search_text += f" {row[subcategory_key]}"
            
            search_text = search_text.strip()
            
            if search_text:
                self.search_embeddings[user_id] = text_embedder.get_embedding(search_text).squeeze(0)
            else:
                self.search_embeddings[user_id] = torch.zeros(768)  # Default embedding size
        
        # Pre-compute claim history embeddings (excluding the latest claim)
        self.claim_embeddings = {}
        self.latest_claims = {}
        
        for _, row in training_df.iterrows():
            user_id = row['user_id']
            
            # Skip if user doesn't exist in user_df
            if user_id not in self.user_id_to_idx:
                continue
            
            # Find the latest claim (highest number with non-null value)
            latest_claim_idx = None
            for i in range(4, 0, -1):  # Check from claim_4 down to claim_1
                variant_key = f'claim_{i}_variant_id'
                if variant_key in row and pd.notna(row[variant_key]):
                    latest_claim_idx = i
                    break
            
            if latest_claim_idx is not None:
                # Store the latest claim variant_id
                self.latest_claims[user_id] = row[f'claim_{latest_claim_idx}_variant_id']
                
                # Combine all other claims into history text
                claim_history_text = ""
                for i in range(1, 5):  # Check all claims
                    if i == latest_claim_idx:
                        continue  # Skip the latest claim
                    
                    variant_name_key = f'claim_{i}_variant_name'
                    merchant_key = f'claim_{i}_merchant_code'
                    
                    if variant_name_key in row and pd.notna(row[variant_name_key]) and row[variant_name_key]:
                        claim_history_text += f" {row[variant_name_key]}"
                    if merchant_key in row and pd.notna(row[merchant_key]) and row[merchant_key]:
                        claim_history_text += f" {row[merchant_key]}"
                
                claim_history_text = claim_history_text.strip()
                
                if claim_history_text:
                    self.claim_embeddings[user_id] = text_embedder.get_embedding(claim_history_text).squeeze(0)
                else:
                    self.claim_embeddings[user_id] = torch.zeros(768)  # Default embedding size
            else:
                # No claims found
                self.claim_embeddings[user_id] = torch.zeros(768)  # Default embedding size
    
    def _create_mapping(self, series):
        """Create a mapping from unique values to indices"""
        unique_values = sorted(series.unique())
        return {value: idx for idx, value in enumerate(unique_values)}
    
    def __len__(self):
        return len(self.user_df)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        row = self.user_df.iloc[idx]
        
        # Process gender
        gender = row['gender'] if pd.notna(row['gender']) else 'Unknown'
        gender_idx = self.gender_to_idx[gender]
        
        
        # Process age
        current_year = 2025
        yob = int(row['yob']) if pd.notna(row['yob']) and row['yob'] != '' else current_year - 25  # Default age
        age = current_year - yob
        
        # Process job
        job = row['job'] if pd.notna(row['job']) else 'Unknown'
        job_embedding = self.job_embeddings[job]
        
        # Process tiers
        vinid_tier = row['VinID_tier_name_en'] if pd.notna(row['VinID_tier_name_en']) else 'Unknown'
        vinid_tier_idx = self.vinid_tier_to_idx[vinid_tier]
        
        tcbr_tier = row['TCBR_tier_name_en'] if pd.notna(row['TCBR_tier_name_en']) else 'Unknown'
        tcbr_tier_idx = self.tcbr_tier_to_idx[tcbr_tier]
        

        province_embedding = self.province_embeddings.get(user_id, torch.zeros(768))

        # Get search embedding
        search_embedding = self.search_embeddings.get(user_id, torch.zeros(768))
        
        # Get claim history embedding
        claim_embedding = self.claim_embeddings.get(user_id, torch.zeros(768))
        
        # Get latest claim (if any)
        latest_claim = self.latest_claims.get(user_id, None)
        
        return {
            'user_id': user_id,
            'gender_idx': torch.tensor(gender_idx, dtype=torch.long),
            'age': torch.tensor(age, dtype=torch.float32),
            'vinid_tier_idx': torch.tensor(vinid_tier_idx, dtype=torch.long),
            'tcbr_tier_idx': torch.tensor(tcbr_tier_idx, dtype=torch.long),
            'job_embedding': job_embedding,
            'province_embedding': province_embedding,
            'search_embedding': search_embedding,
            'claim_embedding': claim_embedding,
            'latest_claim': latest_claim
        }
    
    def get_vocab_sizes(self):
        """Get the vocabulary sizes for all categorical features"""
        return {
            'gender_vocab_size': len(self.gender_to_idx),
            # 'province_vocab_size': len(self.province_to_idx),
            'vinid_tier_vocab_size': len(self.vinid_tier_to_idx),
            'tcbr_tier_vocab_size': len(self.tcbr_tier_to_idx)
        }



class EnhancedTwoTowerDataset(Dataset):
    def __init__(
        self, 
        user_dataset: UserDataset,
        voucher_df: pd.DataFrame,
        voucher_dataset,
        neg_samples_per_pos: int = 4
    ):
        """
        Dataset for training a two-tower model with positive and negative pairs
        where negatives can't be from the same sub-category as the positive sample
        
        Args:
            user_dataset: Enhanced user dataset with search and claim history
            voucher_df: DataFrame with voucher information
            voucher_dataset: Voucher dataset with processed voucher features
            neg_samples_per_pos: Number of negative samples for each positive pair
        """
        self.user_dataset = user_dataset
        self.voucher_df = voucher_df
        self.voucher_dataset = voucher_dataset
        self.neg_samples_per_pos = neg_samples_per_pos
        
        # Create variant_id to index mapping for vouchers
        self.variant_id_to_idx = {}
        for idx, row in enumerate(self.voucher_dataset.df.iterrows()):
            variant_id = row[1].get('variant_id')
            if variant_id is not None:
                self.variant_id_to_idx[variant_id] = idx
        
        # Create mapping from variant_id to subcategory
        self.variant_to_subcategory = {}
        for _, row in voucher_df.iterrows():
            variant_id = row.get('variant_id')
            subcategory = row.get('sub_category_partner_new')
            
            if pd.notna(subcategory) and variant_id is not None:
                self.variant_to_subcategory[variant_id] = subcategory
        
        # Create subcategory to variants mapping for negative sampling
        self.subcategory_to_variants = {}
        for variant_id, subcategory in self.variant_to_subcategory.items():
            if subcategory not in self.subcategory_to_variants:
                self.subcategory_to_variants[subcategory] = []
            self.subcategory_to_variants[subcategory].append(variant_id)
        
        # Create mapping of variant_ids not in the same subcategory for efficient negative sampling
        self.variant_to_diff_subcategory_variants = {}
        all_variant_ids = list(self.variant_id_to_idx.keys())
        
        for variant_id, subcategory in self.variant_to_subcategory.items():
            # Get all variants not in this subcategory
            self.variant_to_diff_subcategory_variants[variant_id] = [
                vid for vid in all_variant_ids
                if vid != variant_id and self.variant_to_subcategory.get(vid) != subcategory
            ]
        
        # Create valid training pairs (user_id, latest_claim_variant_id)
        self.valid_pairs = []
        
        for idx in range(len(user_dataset)):
            user_data = user_dataset[idx]
            user_id = user_data['user_id']
            latest_claim = user_data['latest_claim']
            
            if latest_claim is not None and latest_claim in self.variant_id_to_idx:
                self.valid_pairs.append((idx, latest_claim))
        
        print(f"Created dataset with {len(self.valid_pairs)} valid positive pairs")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        user_idx, variant_id = self.valid_pairs[idx]
        
        # Get user features
        user_features = self.user_dataset[user_idx]
        
        # Get voucher features
        voucher_idx = self.variant_id_to_idx[variant_id]
        voucher_features = self.voucher_dataset[voucher_idx]
        
        # Get the subcategory of the positive variant
        pos_subcategory = self.variant_to_subcategory.get(variant_id)
        
        # Sample negative vouchers from different subcategories
        neg_voucher_features = []
        
        for _ in range(self.neg_samples_per_pos):
            # Try to sample from different subcategory first
            if variant_id in self.variant_to_diff_subcategory_variants and self.variant_to_diff_subcategory_variants[variant_id]:
                # Choose a random variant from a different subcategory
                neg_variant_ids = self.variant_to_diff_subcategory_variants[variant_id]
                neg_variant_id = random.choice(neg_variant_ids)
                
                if neg_variant_id in self.variant_id_to_idx:
                    neg_idx = self.variant_id_to_idx[neg_variant_id]
                    neg_features = self.voucher_dataset[neg_idx]
                    neg_voucher_features.append(neg_features)
                    continue
            
            # Fallback: try to find any variant with a different subcategory
            attempts = 0
            while attempts < 10:  # Limit attempts to avoid infinite loop
                attempts += 1
                # Sample a random voucher
                neg_idx = random.randint(0, len(self.voucher_dataset) - 1)
                neg_variant_id = self.voucher_dataset.df.iloc[neg_idx].get('variant_id')
                
                # Check if it's a different variant and from a different subcategory
                neg_subcategory = self.variant_to_subcategory.get(neg_variant_id)
                if neg_variant_id != variant_id and neg_subcategory != pos_subcategory:
                    neg_features = self.voucher_dataset[neg_idx]
                    neg_voucher_features.append(neg_features)
                    break
            
            # If we couldn't find a good negative after max attempts, use any different variant
            if len(neg_voucher_features) <= _:
                while True:
                    neg_idx = random.randint(0, len(self.voucher_dataset) - 1)
                    neg_variant_id = self.voucher_dataset.df.iloc[neg_idx].get('variant_id')
                    
                    if neg_variant_id != variant_id:
                        neg_features = self.voucher_dataset[neg_idx]
                        neg_voucher_features.append(neg_features)
                        break
        
        return {
            'user_features': user_features,
            'positive_voucher_features': voucher_features,
            'negative_voucher_features': neg_voucher_features
        }

