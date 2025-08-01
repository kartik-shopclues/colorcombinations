import os
import json
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# ===================================================================
# ⚙️ 1. SETUP YOUR CLIENTS
# ===================================================================
# Add your secret API keys here.
# For better security, consider using environment variables.
PINECONE_API_KEY = "pcsk_4nFuNi_M1wuGssZwd9TirQ6u8CdkgsbkZ84kQNtuCWBMsrZ33q2jigzpbfqtpARiKzjewp"
OPENAI_API_KEY = "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"

# Initialize the clients
pc = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


# ===================================================================
# 📇 2. DEFINE INDEX PARAMETERS
# ===================================================================
index_name = "fashion-combos2"
dimension = 3072  # Dimension for OpenAI's text-embedding-3-small


# ===================================================================
# 🏗️ 3. CREATE THE INDEX
# ===================================================================
if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' not found. Creating a new one...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1' # Feel free to change this region
        )
    )
    print(f"✅ Index '{index_name}' created successfully.")
else:
    print(f"✅ Index '{index_name}' already exists.")

# Get a handle to the index
index = pc.Index(index_name)


# ===================================================================
# 💾 4. LOAD, EMBED, AND UPLOAD YOUR DATA
# ===================================================================
print("Loading data from JSON file...")
try:
    with open("jeans_shirt_contrast_1000.json") as f:
        data = json.load(f)
except FileNotFoundError:
    print("❌ Error: 'jeans_shirt_contrast_1000.json' not found.")
    print("Please make sure the JSON file is in the same directory as this script.")
    exit()

# This formats your data into prompts for better embedding
def format_entry(entry):
    return f"Fashion combination: The item is {entry['item']}. It can be worn in a {entry['style']} style. Good color pairings include: {', '.join(entry['shirt_colors'])}."

rag_prompts = [format_entry(d) for d in data]

print(f"Embedding and uploading {len(data)} entries to Pinecone...")
batch_size = 100
for i in tqdm(range(0, len(rag_prompts), batch_size)):
    batch_prompts = rag_prompts[i:i + batch_size]
    batch_meta = data[i:i + batch_size]

    response = client.embeddings.create(
        input=batch_prompts,
        model="text-embedding-3-small"
    )

    vectors = []
    for j, embedding in enumerate(response.data):
        vectors.append({
            "id": f"jeans-{i + j}",
            "values": embedding.embedding,
            "metadata": {
                "item": batch_meta[j]["item"],
                "shirt_colors": batch_meta[j]["shirt_colors"], # This key name is from your JSON
                "style": batch_meta[j]["style"],
            }
        })

    index.upsert(vectors=vectors)

print("✅✅✅ All data has been successfully uploaded to your Pinecone index!")