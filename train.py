import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Optional
import pandas as pd

from dataset import UserDataset, VoucherDataset, EnhancedTwoTowerDataset
from user_embedding import UserTowerModel
from voucher_embedding import VoucherTowerModel
from two_tower import EnhancedTwoTowerRecommender
from loss import contrastive_loss_fn
from utils import TextEmbedder

logger = logging.getLogger(__name__)

# Training function
def train_enhanced_two_tower_model(
    user_df: pd.DataFrame,
    voucher_df: pd.DataFrame,
    training_df: pd.DataFrame,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    neg_samples_per_pos: int = 4,
    user_embedding_dim: int = 256,
    voucher_embedding_dim: int = 256,
    save_dir: str = "./models",
    device: Optional[torch.device] = None
):
    """
    Train the enhanced two-tower recommendation model
    
    Args:
        user_df: DataFrame with user information
        voucher_df: DataFrame with voucher information
        training_df: DataFrame with training data (searches and claims)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        neg_samples_per_pos: Number of negative samples for each positive pair
        user_embedding_dim: Dimension of user embeddings
        voucher_embedding_dim: Dimension of voucher embeddings
        save_dir: Directory to save the model
        device: Device to train on (GPU or CPU)
    
    Returns:
        Trained model, user_dataset, voucher_dataset, text_embedder, tokenizer, encoders
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on device: {device}")
    
    # Initialize text embedder
    text_embedder = TextEmbedder()
    
    # Create enhanced user dataset
    user_dataset = UserDataset(user_df, training_df, text_embedder)
    
    # Get vocabulary sizes for user tower
    vocab_sizes = user_dataset.get_vocab_sizes()
    
    # Initialize enhanced user tower model
    user_tower = UserTowerModel(
        gender_vocab_size=vocab_sizes['gender_vocab_size'],
        province_text_embedding_dim=768,  # Default for PhoBERT
        job_text_embedding_dim=768,  # Default for PhoBERT
        vinid_tier_vocab_size=vocab_sizes['vinid_tier_vocab_size'],
        tcbr_tier_vocab_size=vocab_sizes['tcbr_tier_vocab_size'],
        search_embedding_dim=768,  # Default for PhoBERT
        claim_embedding_dim=768,  # Default for PhoBERT
        embedding_dim=10,
        out_project_embedding_dim=256,
        final_embedding_dim=user_embedding_dim
    )
    
    # Preprocess voucher data
    from sklearn.preprocessing import LabelEncoder
    
    # Select relevant columns
    relevant_columns = [
        'variant_id',
        'variant_name',
        'voucher_discount_type',
        'category_brand_name',
        'category_partner_new',
        'sub_category_partner_new'
    ]
    
    processed_voucher_df = voucher_df[relevant_columns].copy()
    
    # Handle missing values
    processed_voucher_df['variant_name'] = processed_voucher_df['variant_name'].fillna('')
    categorical_columns = [
        'voucher_discount_type',
        'category_brand_name', 
        'category_partner_new',
        'sub_category_partner_new'
    ]
    
    for col in categorical_columns:
        processed_voucher_df[col] = processed_voucher_df[col].fillna('unknown')
    
    # Encode categorical features
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        processed_voucher_df[f"{col}_encoded"] = le.fit_transform(processed_voucher_df[col])
        encoders[col] = le
    
    # Create voucher dataset
    voucher_dataset = VoucherDataset(processed_voucher_df, training_df, text_embedder)
    
    # Initialize voucher tower model
    voucher_tower = VoucherTowerModel(
        variant_name_embedding_dim=768,  # Default for PhoBERT
        category_embedding_dim=768,  # Default for PhoBERT
        out_project_embedding_dim=256,
        final_embedding=voucher_embedding_dim
    )
    
    # Create enhanced two-tower model
    model = EnhancedTwoTowerRecommender(user_tower, voucher_tower)
    model = model.to(device)
    
    # Create enhanced two-tower dataset
    dataset = EnhancedTwoTowerDataset(
        user_dataset=user_dataset,
        voucher_df=voucher_df,
        voucher_dataset=voucher_dataset,
        neg_samples_per_pos=neg_samples_per_pos
    )
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Check if we have enough data for validation
    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )
    else:
        # If not enough data, use all for training
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = None
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    if val_loader:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch data to device
            user_features = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch['user_features'].items()}
            
            pos_voucher_features = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in batch['positive_voucher_features'].items()}
            
            neg_voucher_features_list = []
            for neg_features in batch['negative_voucher_features']:
                neg_features_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                      for k, v in neg_features.items()}
                neg_voucher_features_list.append(neg_features_device)
            
            # Forward pass
            _, _, pos_similarity, neg_similarities, _ = model(
                user_features, 
                pos_voucher_features,
                neg_voucher_features_list
            )
            
            # Compute loss
            loss = contrastive_loss_fn(pos_similarity, neg_similarities)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update loss
            train_loss += loss.item()
            
            # Log progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                logger.info(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Move batch data to device
                    user_features = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                    for k, v in batch['user_features'].items()}
                    
                    pos_voucher_features = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                           for k, v in batch['positive_voucher_features'].items()}
                    
                    neg_voucher_features_list = []
                    for neg_features in batch['negative_voucher_features']:
                        neg_features_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                              for k, v in neg_features.items()}
                        neg_voucher_features_list.append(neg_features_device)
                    
                    # Forward pass
                    _, _, pos_similarity, neg_similarities, _ = model(
                        user_features, 
                        pos_voucher_features,
                        neg_voucher_features_list
                    )
                    
                    # Compute loss
                    loss = contrastive_loss_fn(pos_similarity, neg_similarities)
                    
                    # Update loss
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
                # Save the entire model
                # Save the entire two-tower model
                model_path = os.path.join(save_dir, "enhanced_two_tower_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'user_embedding_dim': user_embedding_dim,
                    'voucher_embedding_dim': voucher_embedding_dim,
                    'max_searches': 3  # Fixed value for the number of max searches considered
                }, model_path)
                
                # Save the user model components
                user_model_path = os.path.join(save_dir, "enhanced_user_tower_model.pth")
                torch.save({
                    'model_state_dict': user_tower.state_dict(),
                    'gender_to_idx': user_dataset.gender_to_idx,
                    # 'province_to_idx': user_dataset.province_to_idx,
                    'vinid_tier_to_idx': user_dataset.vinid_tier_to_idx,
                    'tcbr_tier_to_idx': user_dataset.tcbr_tier_to_idx
                }, user_model_path)
                
                # Save the voucher model components
                voucher_model_path = os.path.join(save_dir, "voucher_tower_model.pth")
                torch.save({
                    'model_state_dict': voucher_tower.state_dict(),
                    'encoders': encoders
                }, voucher_model_path)
                
                print(f"Saved best model at epoch {epoch+1}")
        else:
            # Save model every few epochs if no validation set
            if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                model_path = os.path.join(save_dir, f"enhanced_two_tower_model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'user_embedding_dim': user_embedding_dim,
                    'voucher_embedding_dim': voucher_embedding_dim
                }, model_path)
                
                print(f"Saved model checkpoint at epoch {epoch+1}")
    
    # No need for tokenizer now, so we'll return None
    tokenizer = None
    
    return model, user_dataset, voucher_dataset, text_embedder, tokenizer, encoders