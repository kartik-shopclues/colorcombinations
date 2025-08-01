# rag_core.py

import llm_utils # Import our other module

def get_fashion_suggestion(query: str, index, llm_client, config) -> str:
    """
    Performs the full RAG process to get a fashion suggestion.

    Args:
        query (str): The user's fashion question.
        index: An initialized Pinecone index object.
        llm_client: An initialized OpenAI client object.
        config: The configuration module.

    Returns:
        str: The generated fashion suggestion.
    """
  
    query_vector = llm_utils.get_embedding(
        client=llm_client,
        text=query,
        model=config.EMBEDDING_MODEL
    )
    if not query_vector:
        return "Sorry, I couldn't process your request right now."

   
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    
    context = "\n".join([
        f"Item: {r.metadata['item']}, Pairs well with colors: {', '.join(r.metadata['shirt_colors'])}, Style: {r.metadata['style']}"
        for r in results.matches
    ])

  
    system_prompt = "You are a fashion stylist assistant. Provide a clear, concise recommendation based on the user's question and the context provided."
    
    try:
        response = llm_client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context facts:\n{context}\n\nUser question: {query}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during final response generation: {e}")
        return "I found some ideas, but I'm having trouble phrasing my suggestion."