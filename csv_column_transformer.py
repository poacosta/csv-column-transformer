"""
CSV Column Transformer - High-performance utility for transforming columns in large CSV files.

This module provides functions to efficiently process specific columns in CSV files while
preserving the original structure. It is optimized for large datasets with chunked processing.
"""

import argparse
import os
import time
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import pandas as pd


def transform_chunk(
        chunk_data: pd.DataFrame,
        column: str,
        transform_type: str,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Transform a specific column in a DataFrame chunk based on the specified transform type.

    Args:
        chunk_data: DataFrame chunk to transform
        column: Column name to transform
        transform_type: Type of transformation to apply
        params: Additional parameters for the transformation

    Returns:
        Transformed DataFrame chunk
    """
    if column not in chunk_data.columns:
        raise ValueError(f"Column '{column}' not in DataFrame")

    # Apply the appropriate transformation
    if transform_type == "remove_first_segment":
        def process_path(path):
            if not isinstance(path, str):
                return path

            # Try to process as URL first
            if path.startswith(('http://', 'https://', 'ftp://')):
                try:
                    parsed = urlparse(path)
                    path_part = parsed.path
                    query = f"?{parsed.query}" if parsed.query else ""
                    return f"{path_part}{query}"
                except (ValueError, AttributeError):
                    # Fall back to path processing if URL parsing fails
                    pass

            # Process as path (split by / and remove first segment)
            segments = path.split('/')

            # If there's only one segment or empty, return as is
            if len(segments) <= 1:
                return path

            # Join all segments except the first one
            return '/'.join(segments[1:])

        chunk_data[column] = chunk_data[column].apply(process_path)

    elif transform_type == "add_prefix":
        prefix = params.get("text", "")
        chunk_data[column] = chunk_data[column].apply(
            lambda x: f"{prefix}{x}" if isinstance(x, str) else x
        )

    elif transform_type == "add_suffix":
        suffix = params.get("text", "")
        chunk_data[column] = chunk_data[column].apply(
            lambda x: f"{x}{suffix}" if isinstance(x, str) else x
        )

    elif transform_type == "to_string":
        # Convert numbers to strings
        mask = chunk_data[column].notna()
        chunk_data.loc[mask, column] = chunk_data.loc[mask, column].astype(str)

    elif transform_type == "to_numeric":
        # Convert strings to numbers
        errors = params.get("errors", "coerce")
        chunk_data[column] = pd.to_numeric(chunk_data[column], errors=errors)

    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    return chunk_data


def process_csv(
        input_file: str,
        output_file: str,
        column: str,
        transform_type: str,
        params: Optional[Dict[str, Any]] = None,
        chunk_size: int = 100000
) -> None:
    """
    Process a large CSV file in chunks, transforming a specific column.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        column: Column name to transform
        transform_type: Type of transformation to apply
        params: Additional parameters for the transformation
        chunk_size: Size of chunks to process at a time
    """
    if params is None:
        params = {}

    # Get total rows (to calculate progress)
    print(f"Counting rows in {input_file}...")
    total_rows = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        # Skip header row
        total_rows = sum(1 for _ in f) - 1

    # Process in chunks
    start_time = time.time()
    print(f"Processing {total_rows:,} rows in chunks of {chunk_size:,}...")

    # Create reader and writer
    reader = pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)

    # Process first chunk to get headers and dtypes
    first_chunk = next(reader)
    transformed_chunk = transform_chunk(first_chunk, column, transform_type, params)

    # Write header
    transformed_chunk.to_csv(output_file, index=False)

    # Process remaining chunks
    mode = 'a'  # Append mode
    header = False  # Don't write header again

    rows_processed = len(first_chunk)
    chunks_processed = 1

    print(f"Processed chunk {chunks_processed}, "
          f"{rows_processed:,}/{total_rows:,} rows ({rows_processed / total_rows:.1%})")

    # Process remaining chunks
    for chunk in reader:
        transformed_chunk = transform_chunk(chunk, column, transform_type, params)
        transformed_chunk.to_csv(output_file, mode=mode, header=header, index=False)

        rows_processed += len(chunk)
        chunks_processed += 1

        # Show progress
        if chunks_processed % 5 == 0:
            elapsed = time.time() - start_time
            rows_per_second = rows_processed / elapsed
            print(f"Processed chunk {chunks_processed}, "
                  f"{rows_processed:,}/{total_rows:,} rows "
                  f"({rows_processed / total_rows:.1%}) - {rows_per_second:.1f} rows/sec")

    # Show final stats
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Processed {rows_processed:,} rows at {rows_processed / elapsed:.1f} rows/second")
    print(f"Output saved to: {output_file}")


def main():
    """Command line interface for the CSV transformer."""
    parser = argparse.ArgumentParser(description="Transform a specific column in a large CSV file")

    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_file", help="Path to output CSV file")
    parser.add_argument("column", help="Column name to transform")
    parser.add_argument(
        "transform_type",
        choices=[
            "remove_first_segment",
            "add_prefix",
            "add_suffix",
            "to_string",
            "to_numeric"
        ],
        help="Type of transformation to apply"
    )

    parser.add_argument("--prefix", help="Prefix to add (for add_prefix transform)")
    parser.add_argument("--suffix", help="Suffix to add (for add_suffix transform)")
    parser.add_argument(
        "--errors",
        choices=["ignore", "raise", "coerce"],
        default="coerce",
        help="How to handle errors in to_numeric transform"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Size of chunks to process at a time"
    )

    args = parser.parse_args()

    # Validate file paths
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return

    # Set up parameters based on transform type
    params = {}
    if args.transform_type == "add_prefix" and args.prefix:
        params["text"] = args.prefix
    elif args.transform_type == "add_suffix" and args.suffix:
        params["text"] = args.suffix
    elif args.transform_type == "to_numeric":
        params["errors"] = args.errors

    # Process the file
    try:
        process_csv(
            args.input_file,
            args.output_file,
            args.column,
            args.transform_type,
            params,
            args.chunk_size
        )
    except (ValueError, IOError, pd.errors.EmptyDataError) as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
