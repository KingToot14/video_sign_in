'''A collection of useful utility functions
'''

def get(dict, key, default):
    '''Attempts to retrieve a value from a dictionary given a key, resorts to a default value if not found
    Args:
        dict (Dictionary): the dictionary to search
        key: the desired key to search for
        default: the value to return if the key is not found
    
    Returns:
        value: dict[key] if key is valid, else default
    '''
    if key in dict:
        return dict[key]
    return default