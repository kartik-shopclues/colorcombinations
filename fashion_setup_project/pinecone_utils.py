from pinecone import Pinecone, ServerlessSpec


# create index


def get_or_create_index(pc_client: Pinecone, index_name: str, dimension: int, metric: str, cloud: str, region: str):
    """
    Checks if a Pinecone index exists. If not, it creates one.
    
    Args:
        pc_client (Pinecone): An initialized Pinecone client.
        index_name (str): The name of the index.
        dimension (int): The dimension of the vectors.
        metric (str): The distance metric for the index.
        cloud (str): The cloud provider for the serverless spec.
        region (str): The region for the serverless spec.
        
    Returns:
        pinecone.Index: A handle to the Pinecone index.
    """
    if index_name not in pc_client.list_indexes().names():
        print(f"Index '{index_name}' not found. Creating a new one...")
        pc_client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
        print(f" Index '{index_name}' created successfully.")
    else:
        print(f" Index '{index_name}' already exists.")
    
    return pc_client.Index(index_name)