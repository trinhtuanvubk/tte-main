
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Dataset
class VoucherDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, config: Config):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize text (variant_name)
        text_encoding = self.tokenizer(
            row['variant_name'],
            max_length=self.config.MAX_TEXT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get categorical features
        categorical_features = torch.tensor([
            row['voucher_discount_type_encoded'],
            row['category_brand_name_encoded'],
            row['category_partner_new_encoded'],
            row['sub_category_partner_new_encoded']
        ], dtype=torch.long)
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'categorical_features': categorical_features,
            'voucher_id': idx  # Using index as voucher ID for simplicity
        }




class EnhancedUserDataset(Dataset):
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
        self.province_to_idx = self._create_mapping(user_df['province_name'].fillna('Unknown'))
        self.vinid_tier_to_idx = self._create_mapping(user_df['VinID_tier_name_en'].fillna('Unknown'))
        self.tcbr_tier_to_idx = self._create_mapping(user_df['TCBR_tier_name_en'].fillna('Unknown'))
        
        # Pre-compute job embeddings
        self.job_embeddings = {}
        unique_jobs = user_df['job'].fillna('Unknown').unique()
        for job in unique_jobs:
            self.job_embeddings[job] = text_embedder.get_embedding(job).squeeze(0)
        
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
        
        # Process province
        province = row['province_name'] if pd.notna(row['province_name']) else 'Unknown'
        province_idx = self.province_to_idx[province]
        
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
        
        # Get search embedding
        search_embedding = self.search_embeddings.get(user_id, torch.zeros(768))
        
        # Get claim history embedding
        claim_embedding = self.claim_embeddings.get(user_id, torch.zeros(768))
        
        # Get latest claim (if any)
        latest_claim = self.latest_claims.get(user_id, None)
        
        return {
            'user_id': user_id,
            'gender_idx': torch.tensor(gender_idx, dtype=torch.long),
            'province_idx': torch.tensor(province_idx, dtype=torch.long),
            'age': torch.tensor(age, dtype=torch.float32),
            'job_embedding': job_embedding,
            'vinid_tier_idx': torch.tensor(vinid_tier_idx, dtype=torch.long),
            'tcbr_tier_idx': torch.tensor(tcbr_tier_idx, dtype=torch.long),
            'search_embedding': search_embedding,
            'claim_embedding': claim_embedding,
            'latest_claim': latest_claim
        }
    
    def get_vocab_sizes(self):
        """Get the vocabulary sizes for all categorical features"""
        return {
            'gender_vocab_size': len(self.gender_to_idx),
            'province_vocab_size': len(self.province_to_idx),
            'vinid_tier_vocab_size': len(self.vinid_tier_to_idx),
            'tcbr_tier_vocab_size': len(self.tcbr_tier_to_idx)
        }



class EnhancedTwoTowerDataset(Dataset):
    def __init__(
        self, 
        user_dataset: EnhancedUserDataset,
        voucher_df: pd.DataFrame,
        voucher_dataset,
        neg_samples_per_pos: int = 4
    ):
        """
        Dataset for training a two-tower model with positive and negative pairs
        
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
        
        # Create category to variants mapping
        self.category_to_variants = {}
        for _, row in voucher_df.iterrows():
            variant_id = row.get('variant_id')
            category = row.get('category_partner_new')
            
            if pd.notna(category) and variant_id is not None:
                if category not in self.category_to_variants:
                    self.category_to_variants[category] = []
                self.category_to_variants[category].append(variant_id)
        
        # Create valid training pairs (user_id, latest_claim_variant_id)
        self.valid_pairs = []
        
        for idx in range(len(user_dataset)):
            user_data = user_dataset[idx]
            user_id = user_data['user_id']
            latest_claim = user_data['latest_claim']
            
            if latest_claim is not None and latest_claim in self.variant_id_to_idx:
                self.valid_pairs.append((idx, latest_claim))
        
        logger.info(f"Created dataset with {len(self.valid_pairs)} valid positive pairs")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        user_idx, variant_id = self.valid_pairs[idx]
        
        # Get user features
        user_features = self.user_dataset[user_idx]
        
        # Get voucher features
        voucher_idx = self.variant_id_to_idx[variant_id]
        voucher_features = self.voucher_dataset[voucher_idx]
        
        # Sample negative vouchers
        neg_voucher_features = []
        
        for _ in range(self.neg_samples_per_pos):
            # Sample a random voucher that's not the positive example
            while True:
                neg_idx = np.random.randint(0, len(self.voucher_dataset))
                neg_variant_id = self.voucher_dataset.df.iloc[neg_idx].get('variant_id')
                
                if neg_variant_id != variant_id:
                    break
            
            neg_features = self.voucher_dataset[neg_idx]
            neg_voucher_features.append(neg_features)
        
        return {
            'user_features': user_features,
            'positive_voucher_features': voucher_features,
            'negative_voucher_features': neg_voucher_features
        }


