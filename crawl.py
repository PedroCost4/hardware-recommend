import os
from typing import List, Callable
from functools import partial
import dotenv
from llama_index.readers.web import WholeSiteReader
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext, Settings
from llama_index.core.llms.utils import LLMType
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel, Field
dotenv.load_dotenv()

local_llm = False
INDEX_PATH = "data"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

class HardwareItem(BaseModel):
    title: str = Field(..., description="Item title")
    description: str = Field(..., description="Item description")
    price: str = Field(..., description="Item price")
    fit: float = Field(..., description="The fit score based on the prompt")
    
class URLSelection(BaseModel):
    url: str = Field(..., description="The selected URL")
    reason: str = Field(..., description="Reason for selecting this URL")

def get_llm_model() -> LLMType:
    return Gemini(model="models/gemini-1.5-pro") if not local_llm else Ollama(model="ollama/llama3.1", request_timeout=360.0)

def configure_settings():
    Settings.chunk_size = 2048
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm = get_llm_model()

def get_or_create_index(prompt) -> VectorStoreIndex:
    is_data_folder_empty = len(os.listdir(INDEX_PATH)) == 0
  
    if os.path.exists(INDEX_PATH) and not is_data_folder_empty:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
        return load_index_from_storage(storage_context)

    index = VectorStoreIndex([])
    crawl_sites(index, prompt)
    index.storage_context.persist(persist_dir=INDEX_PATH)
    return index

def select_best_url(prompt: str, available_urls: List[tuple[str, str]]) -> str:
    sllm = Settings.llm.as_structured_llm(output_cls=URLSelection)
    
    url_descriptions = "\n".join([f"- {prefix}: {base_url}" for prefix, base_url in available_urls])
    
    selection_prompt = f"""
    Given the following prompt: "{prompt}"
    
    And the following available URLs:
    {url_descriptions}
    
    Select the best URL to use as a prefix for crawling, considering which one is most likely to contain relevant information for the prompt. Provide the URL and a brief reason for your selection.
    """
    
    result = sllm.predict(selection_prompt)
    return result.url

def crawl_sites(index: VectorStoreIndex, prompt: str, pages: int = 1):
    available_urls = [
        ("https://www.terabyteshop.com.br/hardware", "https://www.terabyteshop.com.br/"),
        ("https://www.kabum.com.br/hardware", "https://www.kabum.com.br/hardware?page_number={}&page_size=100&facet_filters=&sort=most_searched"),
        ("https://www.pichau.com.br/hardware", "https://www.pichau.com.br/"),
        ("https://www.pichau.com.br/hardware/processadores", "https://www.pichau.com.br/")
    ]
    
    best_url = select_best_url(prompt, available_urls)
    print(f"Selected URL for crawling: {best_url}")
    
    for prefix, base_url in available_urls:
        if prefix == best_url:
            crawl_site(index, prefix, base_url, pages)
            break

def crawl_site(index: VectorStoreIndex, prefix: str, base_url: str, pages: int, max_depth: int = 10):
    scraper = WholeSiteReader(prefix=prefix, max_depth=max_depth)
    
    for page in range(pages):
        url = base_url.format(page * 10) if '{}' in base_url else base_url
        documents = scraper.load_data(base_url=url)
        for document in documents:
            index.insert(document)

def query_hardware(index: VectorStoreIndex, prompt: str) -> List[HardwareItem]:
    # sllm = Settings.llm.as_structured_llm(output_cls=HardwareItem)
    query_engine = index.as_query_engine(
        # llm=sllm,
        similarity_top_k=10,
        response_mode="tree_summarize",
    )
    result = query_engine.query(prompt)
    return result

def run_prompt(prompt: str):
    configure_settings()
    index = get_or_create_index(prompt)   
    result = query_hardware(index, prompt)
    print(result)

