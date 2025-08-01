# import openai
# import pinecone
# import json
# from tqdm import tqdm
import os
import json
from tqdm import tqdm
from pinecone import Pinecone
from openai import OpenAI

from pinecone import ServerlessSpec
# ==== 1. SETUP OPENAI + PINECONE KEYS ====
# openai.api_key = "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"
# pinecone.init(api_key="pcsk_4nFuNi_M1wuGssZwd9TirQ6u8CdkgsbkZ84kQNtuCWBMsrZ33q2jigzpbfqtpARiKzjewp", environment="colorcombination")
pc = Pinecone(api_key="pcsk_4nFuNi_M1wuGssZwd9TirQ6u8CdkgsbkZ84kQNtuCWBMsrZ33q2jigzpbfqtpARiKzjewp")
client = OpenAI(api_key="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")
# index_name = "fashion-combos"
# dimension = 1536  # text-embedding-3-small

# # ==== 2. CREATE OR CONNECT TO INDEX ====
# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(index_name, dimension=dimension, metric="cosine")
# index = pinecone.Index(index_name)

# # ==== 3. LOAD YOUR JEANS-SHIRT DATA ====
# with open("jeans_shirt_contrast_1000.json") as f:
#     data = json.load(f)

# def format_entry(entry):
#     return f"What shirt can I wear with {entry['item']}? Suggested: {', '.join(entry['shirt_colors'])}. Style: {entry['style']}."

# rag_prompts = [format_entry(d) for d in data]

# # ==== 4. EMBED + UPSERT TO PINECONE ====
# batch_size = 100
# for i in tqdm(range(0, len(rag_prompts), batch_size)):
#     batch_prompts = rag_prompts[i:i + batch_size]
#     batch_meta = data[i:i + batch_size]

#     response = openai.Embedding.create(
#         input=batch_prompts,
#         model="text-embedding-3-small"
#     )

#     vectors = []
#     for j, embedding in enumerate(response["data"]):
#         vectors.append({
#             "id": f"jeans-{i + j}",
#             "values": embedding["embedding"],
#             "metadata": {
#                 "item": batch_meta[j]["item"],
#                 "shirt_colors": batch_meta[j]["shirt_colors"],
#                 "style": batch_meta[j]["style"],
#                 "type": "jeans-shirt"
#             }
#         })

#     index.upsert(vectors=vectors)

# print("✅ All 1000 entries embedded and uploaded to Pinecone!")

# # ==== 5. RUN RAG QUERY ====
# def rag_ask(user_query):
#     # Embed query
#     query_embed = openai.Embedding.create(
#         input=[user_query],
#         model="text-embedding-3-small"
#     )["data"][0]["embedding"]

#     # Pinecone semantic search
#     results = index.query(
#         vector=query_embed,
#         top_k=5,
#         include_metadata=True
#     )

#     # Build context from results
#     context = "\n".join([
#         f"{r['metadata']['item']}: Suggested shirts - {', '.join(r['metadata']['shirt_colors'])}, Style: {r['metadata']['style']}"
#         for r in results["matches"]
#     ])

#     # Ask GPT-4o using context
#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are a fashion stylist assistant."},
#             {"role": "user", "content": f"{context}\n\nUser question: {user_query}"}
#         ]
#     )

#     return response["choices"][0]["message"]["content"]

# # ==== 6. TEST IT ====
# user_input = "What shirt should I wear with white distressed jeans?"
# print("🧠 GPT-4o Suggestion:")
# print(rag_ask(user_input))

# index_name = "fashion-combos"
# dimension = 1536  # text-embedding-3-small

# # ==== 2. CREATE OR CONNECT TO INDEX ====
# if index_name not in pc.list_indexes().names():
#     print(f"Creating index '{index_name}'...")
#     pc.create_index(
#         name=index_name,
#         dimension=dimension,
#         metric="cosine"
#         # If using a serverless index, you would add a 'spec' here
#         # spec=ServerlessSpec(cloud='aws', region='us-west-2')
#     )
# index = pc.Index(index_name)

# # ==== 3. LOAD YOUR JEANS-SHIRT DATA ====
# with open("jeans_shirt_contrast_1000.json") as f:
#     data = json.load(f)

# def format_entry(entry):
#     return f"What shirt can I wear with {entry['item']}? Suggested: {', '.join(entry['shirt_colors'])}. Style: {entry['style']}."

# rag_prompts = [format_entry(d) for d in data]

# # ==== 4. EMBED + UPSERT TO PINECONE ====
# print("Embedding and upserting data to Pinecone...")
# batch_size = 100
# for i in tqdm(range(0, len(rag_prompts), batch_size)):
#     batch_prompts = rag_prompts[i:i + batch_size]
#     batch_meta = data[i:i + batch_size]

#     # Use the new client.embeddings.create method
#     response = client.embeddings.create(
#         input=batch_prompts,
#         model="text-embedding-3-small"
#     )

#     vectors = []
#     for j, embedding in enumerate(response.data):
#         vectors.append({
#             "id": f"jeans-{i + j}",
#             "values": embedding.embedding, # Note: access embedding with .embedding
#             "metadata": {
#                 "item": batch_meta[j]["item"],
#                 "shirt_colors": batch_meta[j]["shirt_colors"],
#                 "style": batch_meta[j]["style"],
#                 "type": "jeans-shirt"
#             }
#         })

#     index.upsert(vectors=vectors)

# print("✅ All 1000 entries embedded and uploaded to Pinecone!")

# # ==== 5. RUN RAG QUERY ====
# def rag_ask(user_query):
#     # Embed query using the new client syntax
#     query_embed = client.embeddings.create(
#         input=[user_query],
#         model="text-embedding-3-small"
#     ).data[0].embedding

#     # Pinecone semantic search
#     results = index.query(
#         vector=query_embed,
#         top_k=5,
#         include_metadata=True
#     )

#     # Build context from results
#     context = "\n".join([
#         f"{r.metadata['item']}: Suggested shirts - {', '.join(r.metadata['shirt_colors'])}, Style: {r.metadata['style']}"
#         for r in results.matches # Note: access metadata with .metadata and matches with .matches
#     ])

#     # Ask GPT-4o using context with the new client syntax
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are a fashion stylist assistant. Provide a clear, concise recommendation based on the user's question and the context provided."},
#             {"role": "user", "content": f"Context:\n{context}\n\nUser question: {user_query}"}
#         ]
#     )

#     return response.choices[0].message.content

# # ==== 6. TEST IT ====
# user_input = "What shirt should I wear with white distressed jeans?"
# print("\n🧠 GPT-4o Suggestion:")
# print(rag_ask(user_input))





index_name = "fashion-combos"
dimension = 1536  # Dimension for text-embedding-3-small

# ==== CREATE OR CONNECT TO INDEX ====
if index_name not in pc.list_indexes().names():
    print(f"Creating index '{index_name}'...")
    # 👇 2. Add the 'spec' argument here
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1' # You can change this to your preferred region
        )
    )

# Get a handle to the index
index = pc.Index(index_name)

# ==== LOAD YOUR DATA ====
with open("jeans_shirt_contrast_1000.json") as f:
    data = json.load(f)

def format_entry(entry):
    return f"What shirt can I wear with {entry['item']}? Suggested: {', '.join(entry['shirt_colors'])}. Style: {entry['style']}."

rag_prompts = [format_entry(d) for d in data]

# ==== EMBED + UPSERT TO PINECONE ====
print("Embedding and upserting data to Pinecone...")
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
                "shirt_colors": batch_meta[j]["shirt_colors"],
                "style": batch_meta[j]["style"],
                "type": "jeans-shirt"
            }
        })

    index.upsert(vectors=vectors)

print("✅ All entries are embedded and uploaded to Pinecone!")

# ==== RUN RAG QUERY ====
def rag_ask(user_query):
    query_embed = client.embeddings.create(
        input=[user_query],
        model="text-embedding-3-small"
    ).data[0].embedding

    results = index.query(
        vector=query_embed,
        top_k=5,
        include_metadata=True
    )

    context = "\n".join([
        f"{r.metadata['item']}: Suggested shirts - {', '.join(r.metadata['shirt_colors'])}, Style: {r.metadata['style']}"
        for r in results.matches
    ])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a fashion stylist assistant. Provide a clear, concise recommendation based on the user's question and the context provided."},
            {"role": "user", "content": f"Context:\n{context}\n\nUser question: {user_query}"}
        ]
    )

    return response.choices[0].message.content

# ==== TEST IT ====
user_input = "What shirt should I wear with white distressed jeans?"
print("\n🧠 GPT-4o Suggestion:")
print(rag_ask(user_input))


