import os
from typing import List
import dotenv
from llama_index.readers.web import WholeSiteReader
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext, Settings
from llama_index.core.llms.utils import LLMType
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel, Field
from llama_index.core.prompts import PromptTemplate

dotenv.load_dotenv()

local_llm = os.getenv("LOCAL_LLM") == "True"
INDEX_PATH = os.getenv("INDEX_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

class HardwareItem(BaseModel):
    title: str = Field(..., description="Item title")
    description: str = Field(..., description="Item description")
    price: str = Field(..., description="Item price")
    fit: float = Field(..., description="The fit score based on the prompt")
    url: str = Field(..., description="The URL of the item")
    
def get_llm_model() -> LLMType:
    return Gemini(model="models/gemini-1.5-pro") if not local_llm else Ollama(model="llama3.1:latest",  request_timeout=360.0)

def configure_settings():
    Settings.chunk_size = 2048
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm = get_llm_model()

def run_crawler(prompt: str | None):
    index = VectorStoreIndex([])
    crawl_sites(index, prompt=prompt)
    index.storage_context.persist(persist_dir=INDEX_PATH)
    return index

def get_or_create_index(prompt: str) -> VectorStoreIndex:
    is_data_folder_empty = len(os.listdir(INDEX_PATH)) == 0
    run = os.getenv("RERUN_ON_PROMPT") == "True"
    if os.path.exists(INDEX_PATH) and not is_data_folder_empty and not run:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
        return load_index_from_storage(storage_context)

    return run_crawler(prompt)

def select_best_url(prompt: str | None) -> List[str]:
    if prompt is None:
        return [
            ("https://www.kabum.com.br/hardware", "https://www.kabum.com.br/"),
            ("https://www.pichau.com.br/hardware", "https://www.pichau.com.br/"),
            ("https://www.terabyteshop.com.br/hardware", "https://www.terabyteshop.com.br/") 
        ]
    
    sites = [
        ("https://www.kabum.com.br/busca/{value}", "https://www.kabum.com.br/busca/{value}"),
        ("https://www.pichau.com.br/search?q={value}", "https://www.pichau.com.br/search?q={value}"),
        ("https://www.terabyteshop.com.br/busca?str={value}", "https://www.terabyteshop.com.br/busca?str={value}"),
    ]
    
    selection_prompt = f"""
    Given the following text: "{prompt}"
    Select a keyword in brazilian portuguese to use as a search in a site. (return only the keyword) without any special characters
    """
     
    result = get_llm_model().complete(selection_prompt)
    result = result.text.replace('"', "")

    urls = [
        (url.
         format(value=result).
         replace("\n","").
         replace(" ", "-" if base == "https://www.kabum.com.br/busca/{value}" or base == "https://www.pichau.com.br/search?q={value}" else "+"),
         base.
         format(value=result).
         replace("\n","").
         replace(" ", "-" if base == "https://www.kabum.com.br/busca/{value}" or base == "https://www.pichau.com.br/search?q={value}" else "+")
        )
        for base, url in sites
    ]
    
    return urls
    
def crawl_sites(index: VectorStoreIndex, prompt: str | None, pages: int = 1 ):    
    urls = select_best_url(prompt) 
    for prefix, base_url in urls:
        crawl_site(index, prefix, base_url, pages)
            
def crawl_site(index: VectorStoreIndex, prefix: str, base_url: str, pages: int, max_depth: int = 3):
    scraper = WholeSiteReader(prefix=prefix, max_depth=max_depth)
    
    for page in range(pages):
        documents = scraper.load_data(base_url=base_url)
        for document in documents:
            index.insert(document)

def query_hardware(index: VectorStoreIndex, prompt: str) -> str:
    query_engine = index.as_query_engine(
        response_mode='tree_summarize',
        llm=get_llm_model()
    )

    selection_prompt = f"""
    Given the following text: "{prompt}"
    Select a keyword in brazilian portuguese to use as a search in a site. (return only the keyword) without any special characters
    """
     
    result = get_llm_model().complete(selection_prompt)
    
    result = query_engine.query(f"List all items with price that match the keyword: {result}")

    analysis_prompt = f"""
    Given the following context about available hardware in the market:
    {result}
    Just consider the item and its price above.
    And considering the user's request:
    {prompt}
    
    Please answer the user's request considering the points below:
    - You should allways return the name with the price of the item
    Just list the best items that are the best fit for the answer, desconsider anything else
    """
    
    analysis_result = get_llm_model().complete(analysis_prompt)
    
    return analysis_result

def run_prompt(prompt: str):
    configure_settings()
    index = get_or_create_index(prompt)   
    result = query_hardware(index, prompt)
    return result 

