# do pinecone complete activity here

from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone

# Import from our custom modules
import config
import data_loader
import pinecone_utils

def main():
    """Main function to orchestrate the data setup and upload process."""
    
    # --- 1. INITIALIZE CLIENTS ---
    print("Initializing API clients...")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    print(" Clients initialized.")

    # --- 2. GET OR CREATE PINECONE INDEX ---
    index = pinecone_utils.get_or_create_index(
        pc_client=pc,
        index_name=config.PINECONE_INDEX_NAME,
        dimension=config.PINECONE_DIMENSION,
        metric=config.PINECONE_METRIC,
        cloud=config.PINECONE_CLOUD,
        region=config.PINECONE_REGION
    )

    # --- 3. LOAD AND PREPARE DATA ---
    fashion_data = data_loader.load_fashion_data(config.DATA_FILE_PATH)
    
    # --- 4. EMBED AND UPLOAD IN BATCHES ---
    print(f"Embedding and uploading {len(fashion_data)} entries to Pinecone...")
    batch_size = 100
    for i in tqdm(range(0, len(fashion_data), batch_size)):
        batch_data = fashion_data[i:i + batch_size]
        
        # Format prompts for embedding
        prompts = [data_loader.format_prompt_for_embedding(entry) for entry in batch_data]
        
        # Create embeddings
        response = openai_client.embeddings.create(input=prompts, model=config.EMBEDDING_MODEL)
        
        # Prepare vectors for upsert
        vectors = []
        for j, embedding in enumerate(response.data):
            original_entry = batch_data[j]
            vectors.append({
                "id": f"jeans-{i + j}", # Consider a more robust ID generation
                "values": embedding.embedding,
                "metadata": {
                    "item": original_entry.get("item"),
                    "shirt_colors": original_entry.get("shirt_colors"),
                    "style": original_entry.get("style"),
                }
            })
            
        # Upsert batch to Pinecone
        index.upsert(vectors=vectors)

    print("\n All data has been successfully uploaded to your Pinecone index!")

if __name__ == "__main__":
    main()