REGISTRY = {}

def register(name):
    global REGISTRY
    
    def wrapper(fn):
        REGISTRY[name] = fn
        return fn
    
    return wrapper

def get_model(name):
    return REGISTRY[name]()

def has_model(name):
    return name in REGISTRY

def list_model():
    return list(REGISTRY.keys())