import pandas as pd

def basic_cleaning(input_artifact: str, 
                   output_artifact: str, 
                   output_type: str, 
                   output_description: str, 
                   min_price: float, 
                   max_price: float):
    """
    This function processes the input data and applies cleaning steps to create a new artifact with cleaned data.
    
    Args:
    - input_artifact (str): Path to the input file (CSV).
    - output_artifact (str): Path where the cleaned data will be saved.
    - output_type (str): Type of the output artifact (e.g., 'csv').
    - output_description (str): Description of the output artifact.
    - min_price (float): Minimum price to consider.
    - max_price (float): Maximum price to consider.
    """
    # Step 2: Load the dataset
    df = pd.read_csv(input_artifact)
    
    # Step 3: Data cleaning operations (example steps)
    # - Removing rows with missing values (you can modify this as per your requirement)
    df.dropna(inplace=True)
    
    # - Filter based on price range
    df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
    
    # Step 4: Save the cleaned dataset to the specified output path
    if output_type == 'csv':
        df.to_csv(output_artifact, index=False)
    
    # Step 5: Return the cleaned dataset (optional for further use)
    return df
