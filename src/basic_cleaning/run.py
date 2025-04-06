#!/usr/bin/env python
"""
An example of a step using MLflow and Weights & Biases: Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact.
parameters [parameter1, parameter2]: input_artifact, output_artifact, output_type, output_description, min_price, max_price
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    # Start a new W&B run
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact from W&B
    logger.info("Downloading input artifact: %s", args.input_artifact)
    artifact = run.use_artifact(f"{args.input_artifact}:latest")
    artifact_dir = artifact.download()

    # Load the dataset into a pandas DataFrame
    logger.info("Loading data from %s", artifact_dir)
    df = pd.read_csv(f"{artifact_dir}/sample.csv")
    
    # Data cleaning process
    logger.info("Starting data cleaning...")
    
    # Drop rows with missing values (example cleaning)
    df.dropna(inplace=True)
    
    # Filter rows based on price range
    df = df[(df['price'] >= args.min_price) & (df['price'] <= args.max_price)]
    
    # Save cleaned data to CSV
    df.to_csv("clean_sample.csv", index=False)
    logger.info("Cleaned data saved to clean_sample.csv")

    # Upload the cleaned data as a new artifact to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    logger.info("Uploaded cleaned artifact to W&B")

    # Finish the W&B run
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact on W&B (e.g., sample.csv)",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact to upload to W&B",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact (e.g., clean_sample)",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to filter the dataset",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to filter the dataset",
        required=True
    )

    args = parser.parse_args()

    # Run the data cleaning process
    go(args)
