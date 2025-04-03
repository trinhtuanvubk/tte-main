import pandas as pd
import numpy as np
from datetime import datetime

def create_category_mapping(keyword_mapping_path):
    """
    Create a dictionary mapping keywords to categories and subcategories
    """
    # Read the keyword mapping file
    keyword_df = pd.read_csv(keyword_mapping_path)
    
    # Create a dictionary for faster lookups
    keyword_dict = {}
    for _, row in keyword_df.iterrows():
        if pd.notna(row['keyword']):
            keyword_dict[row['keyword'].lower()] = {
                'category': row['category'] if pd.notna(row['category']) else '',
                'subcategory': row['subcategory'] if pd.notna(row['subcategory']) else ''
            }
    
    return keyword_dict

def get_category_mapping(keyword, keyword_dict):
    """
    Get category and subcategory for a given keyword using the mapping dictionary
    Uses multiple approaches for matching:
    1. Direct match
    2. Keyword contains known term
    3. Known term contains keyword
    """
    if not isinstance(keyword, str) or not keyword:
        return {'category': '', 'subcategory': ''}
    
    # Convert to lowercase for case-insensitive matching
    lower_keyword = keyword.lower()
    
    # Direct lookup (exact match)
    if lower_keyword in keyword_dict:
        return keyword_dict[lower_keyword]
    
    # Try partial matches - first check if the keyword contains any of our known keywords
    for dict_keyword, mapping in keyword_dict.items():
        if lower_keyword in dict_keyword:
            return mapping
    
    # If no match found, try if any dictionary keyword contains our search term
    for dict_keyword, mapping in keyword_dict.items():
        if dict_keyword in lower_keyword:
            return mapping
    
    return {'category': '', 'subcategory': ''}

def process_search_data(search_data_path, keyword_dict):
    """
    Process search data to get the 3 most recent searches for each user
    with category mappings
    """
    # Read the search data
    search_df = pd.read_csv(search_data_path)
    
    # Create timestamp column for sorting
    search_df['date'] = search_df['event_date__partition'].str.split('T').str[0]
    search_df['timestamp'] = pd.to_datetime(
        search_df['date'] + ' ' + search_df['search_hour'].astype(str) + ':00:00', 
        errors='coerce'
    )
    
    # Sort by user_id and timestamp (most recent first)
    search_df = search_df.sort_values(['user_id', 'timestamp'], ascending=[True, False])
    
    # Group by user_id and get the 3 most recent searches
    user_searches = {}
    for user_id, group in search_df.groupby('user_id'):
        recent_searches = []
        for _, row in group.head(3).iterrows():
            if pd.notna(row['keyword_value']):
                keyword = row['keyword_value']
                mapping = get_category_mapping(keyword, keyword_dict)
                recent_searches.append({
                    'keyword': keyword,
                    'category': mapping['category'],
                    'subcategory': mapping['subcategory'],
                    'timestamp': row['timestamp']
                })
        user_searches[user_id] = recent_searches
    
    return user_searches

def process_claim_data(claim_data_path):
    """
    Process claim data to get the 4 most recent claims for each user
    """
    # Read the claim data
    claim_df = pd.read_csv(claim_data_path, low_memory=False)

    
    # Filter for claimed tickets
    claim_df = claim_df[
            (claim_df['distribution_type'] == 'CLAIMED') & 
            (claim_df['claim_date'].notna())
            ]
    # Convert claim_date to datetime for sorting
    claim_df['claim_timestamp'] = pd.to_datetime(claim_df['claim_date'], errors='coerce')
    
    # Sort by user_id and claim_timestamp (most recent first)
    claim_df = claim_df.sort_values(['user_id', 'claim_timestamp'], ascending=[True, False])
    
    # Group by user_id and get the 4 most recent claims
    user_claims = {}
    for user_id, group in claim_df.groupby('user_id'):
        recent_claims = []
        for _, row in group.head(4).iterrows():
            recent_claims.append({
                'variant_id': row['variant_id'],
                'variant_name': row['variant_name'],
                'merchant_voucher_code': row['merchant_voucher_code'],
                'timestamp': row['claim_timestamp']
            })
        user_claims[user_id] = recent_claims
    
    return user_claims

def create_training_data(user_searches, user_claims):
    """
    Create training data by combining search and claim data
    """
    training_data = []
    
    # Process all users with search data
    for user_id, searches in user_searches.items():
        user_data = {'user_id': user_id}
        
        # Add search data and their categories
        for i in range(3):
            if i < len(searches):
                user_data[f'search_{i+1}'] = searches[i]['keyword']
                user_data[f'search_{i+1}_category'] = searches[i]['category']
                user_data[f'search_{i+1}_subcategory'] = searches[i]['subcategory']
            else:
                user_data[f'search_{i+1}'] = ''
                user_data[f'search_{i+1}_category'] = ''
                user_data[f'search_{i+1}_subcategory'] = ''
        
        # Add claim data if available
        claims = user_claims.get(user_id, [])
        for i in range(4):
            if i < len(claims):
                user_data[f'claim_{i+1}_variant_id'] = claims[i]['variant_id']
                user_data[f'claim_{i+1}_variant_name'] = claims[i]['variant_name']
                user_data[f'claim_{i+1}_merchant_code'] = claims[i]['merchant_voucher_code']
            else:
                user_data[f'claim_{i+1}_variant_id'] = ''
                user_data[f'claim_{i+1}_variant_name'] = ''
                user_data[f'claim_{i+1}_merchant_code'] = ''
        
        training_data.append(user_data)
    
    return pd.DataFrame(training_data)

def main():
    # File paths
    keyword_mapping_path = 'keyword_mapping.csv'
    search_data_path = 'r_f_user_oneu_app_search_lnd.csv'
    claim_data_path = 'r_f_ticket_non_tcb.csv'
    output_path = 'training_data.csv'
    
    # Create keyword mapping dictionary
    keyword_dict = create_category_mapping(keyword_mapping_path)
    
    # Process search data
    user_searches = process_search_data(search_data_path, keyword_dict)
    print(f"Processed search data for {len(user_searches)} users")
    
    # Process claim data
    user_claims = process_claim_data(claim_data_path)
    print(f"Processed claim data for {len(user_claims)} users")
    
    # Create training data
    training_df = create_training_data(user_searches, user_claims)
    print(f"Created training data with {len(training_df)} entries")
    
    # Save to CSV
    training_df.to_csv(output_path, index=False)
    print(f"Training data saved to {output_path}")
    
    # Print some stats
    users_with_both = sum(1 for user_id in user_searches if user_id in user_claims)
    print(f"Number of users with both search and claim data: {users_with_both}")
    
    # Preview the first few rows
    print("\nPreview of training data:")
    print(training_df.head().to_string())

if __name__ == "__main__":
    main()
