import os
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
from dotenv import load_dotenv
load_dotenv(dotenv_path)
API_KEY = os.getenv('API_KEY1')

print(API_KEY)
