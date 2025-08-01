import json



# Again ensure that you have json dataset and giving check again 


def load_fashion_data(file_path: str) -> list:
    """
    Loads fashion data from a JSON file.
    
    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: A list of dictionaries containing the data.
    """
    print(f"Loading data from '{file_path}'...")
    try:
        with open(file_path) as f:
            data = json.load(f)
        print(f"✅ Successfully loaded {len(data)} entries.")
        return data
    except FileNotFoundError:
        print(f" Error: '{file_path}' not found.")
        print("Please make sure the data file is in the correct directory.")
        exit()
    except json.JSONDecodeError:
        print(f" Error: Could not decode JSON from '{file_path}'. Please check the file format.")
        exit()

def format_prompt_for_embedding(entry: dict) -> str:
    """
    Formats a single data entry into a descriptive string for better embedding.

    Args:
        entry (dict): A dictionary representing one fashion combination.

    Returns:
        str: A formatted string.
    """
    item = entry.get('item', 'unknown item')
    style = entry.get('style', 'general')
    colors = entry.get('shirt_colors', [])
    return f"Fashion combination: The item is {item}. It can be worn in a {style} style. Good color pairings include: {', '.join(colors)}."