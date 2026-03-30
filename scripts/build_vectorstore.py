from src.utils.config import load_config
from src.data.vectorstore import build_vectorstore

if __name__ == '__main__':
    config = load_config()
    build_vectorstore(config)
