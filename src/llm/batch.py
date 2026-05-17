"""
Author: Zhengyuan Dong
Date: 2025-04-01
Description: This script demonstrates how to use OpenAI Batch API for submitting multiple chat completions asynchronously.
"""

import argparse
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

def setup_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your .env file")
    client = OpenAI(api_key=api_key)
    return client

def upload_batch_file(client, file_path: str) -> str:
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="batch")
    return response.id

def create_batch_job(client, file_id: str) -> str:
    response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return response.id

def wait_for_completion(client, batch_id: str, poll_interval: int = 10) -> str:
    print(f"Waiting for batch {batch_id} to complete...")
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"📡 Current status: {status}")
        if status in ["completed", "failed", "expired", "cancelled"]:
            if status != "completed":
                error_info = batch.errors
                if error_info:
                    print("❌ Batch failed with error:")
                    for i, err in enumerate(error_info):
                        print('?'*10)
                        print("Error in batch querying openai")
                        print(f"  🚨 Error {i+1}:")
                        print(err)
                else:
                    print("⚠️ Batch failed but no error info provided.")
            return status
        time.sleep(poll_interval)

def download_batch_result(client, batch_id: str, output_path):
    batch = client.batches.retrieve(batch_id)
    output_file_id = batch.output_file_id
    if output_file_id:
        file_content = client.files.content(output_file_id).read()
        with open(output_path, "wb") as f:
            f.write(file_content)
        print(f"✅ Batch output saved to {output_path}")
    else:
        print("⚠️ No output file found for this batch.")

def main_batch_query(input_path: str, output_path: str):
    # ---------- Load API Key and Create Client ----------
    client = setup_client()
    if not os.path.exists(input_path):
        print("❌ batch_input.jsonl not found. Please prepare it first.")
        exit(1)
    # test time
    t1 = time.time()
    # Step 1: Upload
    file_id = upload_batch_file(client, input_path)
    print(f"📤 Uploaded input file: {file_id}")
    # Step 2: Create batch
    batch_id = create_batch_job(client, file_id)
    print(f"📦 Created batch: {batch_id}")
    # Step 3: Wait for completion
    status = wait_for_completion(client, batch_id)
    print(f"✅ Batch finished with status: {status}")
    # Step 4: Download output
    download_batch_result(client, batch_id, output_path)
    print(f"🕒 Total time: {time.time() - t1:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit an OpenAI Batch input file and download the output.")
    parser.add_argument("input_path", nargs="?", default="batch_input_eg.jsonl", help="Batch input JSONL file")
    parser.add_argument("output_path", nargs="?", default="batch_output_eg.jsonl", help="Batch output JSONL file")
    args = parser.parse_args()
    main_batch_query(args.input_path, args.output_path)
