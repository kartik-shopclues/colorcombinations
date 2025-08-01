

def classify_intent(client, query: str, model: str) -> str:
    """
    Classifies the user's query to see if it's about fashion.

    Returns:
        str: 'FASHION' or 'OTHER'.
    """
    system_prompt = """
    You are a classifier. Your task is to determine if a user's question
    is about fashion, clothing, style advice, or color combinations.
    Respond with a single word: 'FASHION' or 'OTHER'.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=5,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during intent classification: {e}")
        return "OTHER"

def get_embedding(client, text: str, model: str) -> list[float]:
    """
    Creates an embedding vector for a given text.
    
    Returns:
        list[float]: The embedding vector.
    """
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return [] 