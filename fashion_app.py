import os
from pinecone import Pinecone
from openai import OpenAI

# ===================================================================
# ⚙️ 1. SETUP YOUR CLIENTS
# ===================================================================
# Add your secret API keys here.
PINECONE_API_KEY = "pcsk_4nFuNi_M1wuGssZwd9TirQ6u8CdkgsbkZ84kQNtuCWBMsrZ33q2jigzpbfqtpARiKzjewp"
OPENAI_API_KEY = "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"

# Initialize the clients
pc = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


# ===================================================================
# 📇 2. CONNECT TO YOUR EXISTING INDEX
# ===================================================================
index_name = "fashion-combos2"

try:
    index = pc.Index(index_name)
    print(f"✅ Successfully connected to Pinecone index '{index_name}'.")
except Exception as e:
    print(f"❌ Error connecting to Pinecone index: {e}")
    print("Please make sure you have run the 'setup_pinecone.py' script first.")
    exit()


# ===================================================================
# 🧠 3. DEFINE THE CORE FUNCTIONS
# ===================================================================

def classify_intent(user_query):
    """Classifies the user's query to see if it's about fashion."""
    system_prompt = """
    You are an expert classifier. Your task is to determine if the user's question is about fashion,
    clothing, style advice, or color combinations.
    Respond with a single word: 'FASHION' if it is a fashion-related question,
    and 'OTHER' if it is not.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        max_tokens=5,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def rag_ask(user_query):
    """Finds relevant context in Pinecone and asks the final question."""
    # Embed the user's query
    query_embed = client.embeddings.create(
        input=[user_query],
        model="text-embedding-3-small"
    ).data[0].embedding

    # Query Pinecone
    results = index.query(vector=query_embed, top_k=5, include_metadata=True)

    # Build a generic context
    context = "\n".join([
        f"Item: {r.metadata['item']}, Pairs well with colors: {', '.join(r.metadata['shirt_colors'])}, Style: {r.metadata['style']}"
        for r in results.matches
    ])

    # Ask the final question
    system_prompt = "You are a fashion stylist assistant. Provide a clear, concise recommendation based on the user's question and the context provided."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context facts:\n{context}\n\nUser question: {user_query}"}
        ]
    )
    return response.choices[0].message.content


# ===================================================================
# ▶️ 4. RUN THE APPLICATION
# ===================================================================
def run_fashion_app():
    # --- Example 1: Fashion Question ---
    user_input_fashion = "what kind of t-shirt should i wear with light blue ripped jeans?"
    print("-" * 50)
    print(f"👤 User asks: {user_input_fashion}")

    intent = classify_intent(user_input_fashion)

    if intent == 'FASHION':
        print("🧠 Intent: Fashion. Generating suggestion...")
        suggestion = rag_ask(user_input_fashion)
        print(f"🤖 App says: {suggestion}")
    else:
        print("🧠 Intent: Other.")
        print("🤖 App says: Sorry, I do not have expertise in this domain.")

    

# Run the main function when the script is executed
if __name__ == "__main__":
    run_fashion_app()