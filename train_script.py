import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
import pickle
import argparse
from datetime import datetime
import random
from tqdm import tqdm

# Import the enhanced functions from the new modules
from train import train_enhanced_two_tower_model
from utils import TextEmbedder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_v5.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train an enhanced two-tower recommendation model with search, category, and recent voucher data')
    
    # Data paths
    parser.add_argument('--user_data', type=str, default='/home/user/tte-test/training_data_user.csv', help='Path to user data CSV')
    parser.add_argument('--voucher_data', type=str, default='/home/user/tte-test/r_d_voucher.csv', help='Path to voucher data CSV')
    parser.add_argument('--training_data', type=str, default='/home/user/tte-test/training_data_label.csv', help='Path to enhanced mapping data with searches, categories, and recent vouchers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--neg_samples', type=int, default=4, help='Number of negative samples per positive pair')
    
    # Balancing parameters
    parser.add_argument('--samples_per_category', type=int, default=None, help='Number of samples per category for balanced dataset')
    parser.add_argument('--weight_alpha', type=float, default=0.5, help='Alpha for category weight smoothing (0-1)')
    
    # Model parameters
    parser.add_argument('--user_dim', type=int, default=128, help='Dimension of user embeddings')
    parser.add_argument('--voucher_dim', type=int, default=128, help='Dimension of voucher embeddings')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./models_v6', help='Directory to save the model and embeddings')
    parser.add_argument('--model_name', type=str, default="20epochs", help='Custom name for the model (default: timestamp)')
    
    # Other parameters
    parser.add_argument('--gpu', action='store_false', help='Use GPU for training if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up model name with timestamp if not provided
    if args.model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_name = f"enhanced_two_tower_{timestamp}"
    
    # Create output directory
    model_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device for training
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    
    user_df = pd.read_csv(args.user_data)
    voucher_df = pd.read_csv(args.voucher_data)
    training_df = pd.read_csv(args.training_data)
    
    # Basic data analysis
    logger.info(f"User data shape: {user_df.shape}")
    logger.info(f"Voucher data shape: {voucher_df.shape}")
    logger.info(f"Mapping data shape: {training_df.shape}")
    
    # Check search and category columns
    search_columns = [f'search_{i}' for i in range(1, 4)]
    category_columns = [f'search_{i}_category' for i in range(1, 4)]
    subcategory_columns = [f'search_{i}_subcategory' for i in range(1, 4)]
    claim_columns = [f'claim_{i}_variant_id' for i in range(1, 5)]
    
    has_search_columns = all(col in training_df.columns for col in search_columns)
    has_category_columns = all(col in training_df.columns for col in category_columns)
    has_subcategory_columns = all(col in training_df.columns for col in subcategory_columns)
    has_claim_columns = all(col in training_df.columns for col in claim_columns)
    
    logger.info(f"Has search columns: {has_search_columns}")
    logger.info(f"Has category columns: {has_category_columns}")
    logger.info(f"Has subcategory columns: {has_subcategory_columns}")
    logger.info(f"Has claim columns: {has_claim_columns}")
    
    # Basic preprocessing
    logger.info("Preprocessing data...")
    
    user_df['gender'] = user_df['gender'].fillna('Unknown')
    user_df['province_name'] = user_df['province_name'].fillna('Unknown')
    user_df['job'] = user_df['job'].fillna('Unknown')
    user_df['VinID_tier_name_en'] = user_df['VinID_tier_name_en'].fillna('Unknown')
    user_df['TCBR_tier_name_en'] = user_df['TCBR_tier_name_en'].fillna('Unknown')
    
    # Save arguments to config file
    config_path = os.path.join(model_dir, "config.txt")
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Train model with enhanced features
    logger.info(f"Training enhanced model with search, category, and recent voucher data (output: {model_dir})...")
    
    model, user_dataset, voucher_dataset, text_embedder, tokenizer, encoders = train_enhanced_two_tower_model(
        user_df=user_df,
        voucher_df=voucher_df,
        training_df=training_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        neg_samples_per_pos=args.neg_samples,
        user_embedding_dim=args.user_dim,
        voucher_embedding_dim=args.voucher_dim,
        save_dir=model_dir,
        device=device
    )
    
    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()