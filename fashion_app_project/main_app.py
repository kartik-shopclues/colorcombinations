

from openai import OpenAI
from pinecone import Pinecone


import config
import llm_utils
import rag_core

def run_app():
    """Initializes clients and runs the main application logic."""
    
    
    print("Initializing clients and connecting to Pinecone...")
    try:
        openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        index = pc.Index(config.PINECONE_INDEX_NAME)
        print(f" Successfully connected to index '{config.PINECONE_INDEX_NAME}'.")
    except Exception as e:
        print(f"Initialization Error: {e}")
        print("Please check your API keys and index name in config.py")
        return 

    
    queries = [
        "what kind of t-shirt should i wear with light blue ripped jeans?",
        "I have a formal event, what can I wear with black trousers i am a men?"
    ]

   
    for query in queries:
        print("-" * 50)
        print(f"👤 User asks: {query}")

     
        intent = llm_utils.classify_intent(
            client=openai_client,
            query=query,
            model=config.CLASSIFIER_MODEL
        )

        if intent == 'FASHION':
            print("Intent: Fashion. Generating suggestion...")
            suggestion = rag_core.get_fashion_suggestion(
                query=query,
                index=index,
                llm_client=openai_client,
                config=config
            )
            print(f" App says: {suggestion}")
        else:
            print(" Intent: Other.")
            print(" App says: Sorry, I do not have expertise in this domain.")
    
    print("-" * 50)

if __name__ == "__main__":
    run_app()