from dataclasses import dataclass , field
from dotenv import load_dotenv
import os


load_dotenv()

@dataclass
class settings:

    llm_name:str = 'gpt4o'
    embedding_name:str = 'test'


    data_base_url:str = field(default_factory = lambda : os.getenv('DATABASE_URL') )






s = settings(llm_name='cloud')
print(s.data_base_url)
print(s.llm_name)