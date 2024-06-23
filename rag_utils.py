import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS # depricated
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFacePipeline
from PyPDF2 import PdfFileReader
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import dill, os, yaml
from ragatouille import RAGPretrainedModel


class Vectorizer():
    """
    The Vectorizer class is designed to handle the loading of documents, splitting them into chunks, and preparing a vector database for these chunks.

    Attributes:
        cache_dir (str): The directory where the cache files are stored.
        cache_docs (bool): A flag indicating whether to cache the loaded documents.
        cache_chunks (bool): A flag indicating whether to cache the split chunks.
        cache_vector_db (bool): A flag indicating whether to cache the prepared vector database.

    Methods:
        __init__(self, cache_dir, cache_docs=True, cache_chunks=True, cache_vector_db=True): Initializes a new instance of the Vectorizer class.
        load_docs_from_hf(self, dataset_id, split, column): Loads a dataset from Hugging Face and converts it into a list of LangChain Documents.
        load_docs_from_pdf_dir(self, dir_path): Loads all PDF files from a specified directory and converts each page into a separate LangChain Document.
        load_docs_from_file_lines(self, file_path): Loads a text file and converts each line into a separate LangChain Document.
        load_embedding_model_from_hf(self, model_id, device='cuda:0', max_seq_len=None): Loads an embedding model and its tokenizer from Hugging Face.
        split_docs(self, percent_overlap=0.1): Splits the loaded documents into chunks based on the 'chunk_size' attribute of the class instance.
        prepare_vector_db(self): Prepares a vector database (FAISS) from the chunks of documents.
        from_yaml_config(config): Creates a Vectorizer instance from a configuration dictionary.
    """
    
    def __init__(self, cache_dir: str, cache_docs: bool =True, cache_chunks: bool =True, cache_vector_db: bool =True):
        """
        Initializes a new instance of the Vectorizer class.

        Args:
            cache_dir (str): The directory where the cache files are stored.
            cache_docs (bool, optional): A flag indicating whether to cache the loaded documents. Defaults to True.
            cache_chunks (bool, optional): A flag indicating whether to cache the split chunks. Defaults to True.
            cache_vector_db (bool, optional): A flag indicating whether to cache the prepared vector database. Defaults to True.

        Returns:
            None.
        """
        
        self.cache_dir = cache_dir
        self.cache_docs = cache_docs
        self.cache_chunks = cache_chunks
        self.cache_vector_db = cache_vector_db
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    # Save Cache Methods
    def save_docs_to_cache(self):
        """
        This method saves the documents to the cache. The documents are stored in a pickle file.
        
        The path of the cache directory is defined by `self.cache_dir`. The documents are stored in a file named 'cached_docs.pkl' within this directory.
        
        The documents to be cached are contained in `self.ls_docs`.
        """
        cache_path = f'{self.cache_dir}/cached_docs.pkl'
        with open(cache_path, 'wb') as f:
            print(f'- Caching Documents at "{cache_path}"')
            dill.dump(self.ls_docs, f)
            
            
    def save_chunks_to_cache(self):
        """
        This method saves the chunks to the cache. The chunks are stored in a pickle file.
        
        The path of the cache directory is defined by `self.cache_dir`. The chunks are stored in a file named 'cached_chunks.pkl' within this directory.
        
        The chunks to be cached are contained in `self.ls_chunks`.
        """
        cache_path = f'{self.cache_dir}/cached_chunks.pkl'
        with open(cache_path, 'wb') as f:
            print(f'- Caching Chunks at "{cache_path}"')
            dill.dump(self.ls_chunks, f)
            
            
    def save_vector_db_to_cache(self):
        """
        This method saves the vector database to the cache.
        
        The path of the cache directory is defined by `self.cache_dir`. The vector database is stored in a file named 'cached_vector_db' within this directory.
        
        The vector database to be cached is contained in `self.vector_db`.
        """
        cache_path = f'{self.cache_dir}/cached_vector_db'
        print(f'- Caching Vector DB at "{cache_path}"')
        self.vector_db.save_local(cache_path)

    # Load Cache Methods
    def load_docs_from_cache(self):
        """
        This method loads the documents from the cache if they exist. The documents are loaded from a pickle file.
        
        The path of the cache directory is defined by `self.cache_dir`. The documents are loaded from a file named 'cached_docs.pkl' within this directory.
        
        If the file exists, the documents are loaded into `self.ls_docs` and the method returns True. If the file does not exist, the method returns False.
        """
        cache_path = f'{self.cache_dir}/cached_docs.pkl'
        if Path(cache_path).is_file():
            with open(cache_path, 'rb') as f:
                print(f'- Loading Documents from Cache: "{cache_path}"')
                self.ls_docs = dill.load(f)
            return True
        return False
            
            
    def load_chunks_from_cache(self):
        """
        This method loads the chunks from the cache if they exist. The chunks are loaded from a pickle file.
        
        The path of the cache directory is defined by `self.cache_dir`. The chunks are loaded from a file named 'cached_chunks.pkl' within this directory.
        
        If the file exists, the chunks are loaded into `self.ls_chunks` and the method returns True. If the file does not exist, the method returns False.
        """
        cache_path = f'{self.cache_dir}/cached_chunks.pkl'
        if Path(cache_path).is_file():
            with open(cache_path, 'rb') as f:
                print(f'- Loading Chunks from Cache: "{cache_path}"')
                self.ls_chunks = dill.load(f)
            return True
        return False
            
            
    def load_vector_db_from_cache(self):
        """
        This method loads the vector database from the cache if it exists.
        
        The path of the cache directory is defined by `self.cache_dir`. The vector database is loaded from a directory named 'cached_vector_db' within this directory.
        
        If the directory exists, the vector database is loaded into `self.vector_db` using the `FAISS.load_local` method and the method returns True. If the directory does not exist, the method returns False.
        """
        cache_path = f'{self.cache_dir}/cached_vector_db'
        if Path(cache_path).is_dir():
            print(f'- Loading Vector DB from Cache: "{cache_path}"')
            self.vector_db = FAISS.load_local(
                cache_path,
                self.embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )
            return True
        return False
    
    
    # Load Docs Methods
    def load_docs_from_hf(self, dataset_id: str, split: str, column: str):
        """
        This method loads a dataset from Hugging Face (HF) and converts it into a list of LangChain Documents.

        Args:
            dataset_id (str): The identifier of the dataset to be loaded from HF.
            split (str): The specific split of the dataset to be loaded (e.g., 'train', 'test').
            column (str): The specific column in the dataset that contains the document content.

        Returns:
            None. The method updates the 'ls_docs' attribute of the class instance with the loaded documents.
        """
        
        self.hf_dataset_id = dataset_id
        self.hf_dataset_split = split
        self.hf_dataset_column = column
        
        print(f'- Loading Dataset from HF: {self.hf_dataset_id}')
        ds = load_dataset(dataset_id, split=split)
        self.ls_docs = [
            Document(page_content=row[column], metadata={'hf_ds_id': dataset_id, 'hf_ds_split': split, 'hf_ds_col': column})
            for row in tqdm(ds, desc='- Converting to LangChain Document')
        ]
        
        if self.cache_docs:
            self.save_docs_to_cache()


    def load_docs_from_pdf_dir(self, dir_path: str):
        """
        This method loads all PDF files from a specified directory and converts each page into a separate LangChain Document.

        Args:
            dir_path (str): The path to the directory containing the PDF files.

        Returns:
            None. The method updates the 'ls_docs' attribute of the class instance with the loaded documents.
        """
        
        ls_docs_temp = []
        ls_pdf_paths = [os.path.join(dir_path, fn) for fn in os.listdir(dir_path) if fn.endswith('.pdf')]
        
        for i, pdf_path in enumerate(ls_pdf_paths):
            with open(pdf_path, 'rb') as pdffile:
                pdf_reader = PdfFileReader(pdffile)
                pdf_num_pages = pdf_reader.numPages
                for pdf_page_num in tqdm(range(pdf_num_pages), desc=f'{i+1}/{len(ls_pdf_paths)} - Converting "{pdf_path}" to LangChain Document', leave=False):
                    pdf_page = pdf_reader.getPage(pdf_page_num)
                    pdf_page_text = pdf_page.extractText()
                    langchain_doc = Document(page_content=pdf_page_text, metadata={'pdf_path': pdf_path, 'page_num': pdf_page_num})
                    ls_docs_temp.append(langchain_doc)
        
        self.ls_docs = ls_docs_temp
        
        if self.cache_docs:
            self.save_docs_to_cache()
            
            
    def load_docs_from_file_lines(self, file_path: str):
        """
        This method loads a text file and converts each line into a separate LangChain Document.

        Args:
            file_path (str): The path to the text file.

        Returns:
            None. The method updates the 'ls_docs' attribute of the class instance with the loaded documents.
        """
        
        self.corpus_file_path = file_path
        with open(file_path, 'r') as file:
            print(f'- Reading Lines: {file_path}')
            lines = file.readlines()
            print(f'Read {len(lines):,} Lines.')
        
        self.ls_docs = [
            Document(page_content=line, metadata={'corpus_file_path': self.corpus_file_path, 'line_id': i})
            for i, line in tqdm(enumerate(lines), desc='- Converting Lines to LangChain Document', total=len(lines))
        ]
        
        if self.cache_docs:
            self.save_docs_to_cache()
                    

    # Load Embedding Model Methods
    def load_embedding_model_from_hf(self, model_id: str, device: str ='cuda:0', max_seq_len: int =None):
        """
        This method loads an embedding model and its tokenizer from Hugging Face (HF).

        Args:
            model_id (str): The identifier of the model to be loaded from HF.
            device (str, optional): The device where the model will be loaded. Defaults to 'cuda:0'.
            max_seq_len (int, optional): The maximum sequence length for the model. If not provided, it will use the maximum sequence length of the model.

        Returns:
            None. The method updates the 'embedding_model', 'embedding_tokenizer', and 'chunk_size' attributes of the class instance.
        """
        
        self.hf_embedding_model_id = model_id

        print(f'- Loading Embedding Model & Tokenizer: {self.hf_embedding_model_id}')
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_id,
            multi_process=True, # multi GPU
            model_kwargs={'device': device},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        )

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_id)

        if max_seq_len is None:
            self.chunk_size = self.embedding_model.client.max_seq_length
        else:
            self.chunk_size = max_seq_len
        print(f'- Chunk Size (#Tokens): {self.chunk_size}') # todo: make this a parameter too


    def split_docs(self, percent_overlap: float =0.1):
        """
        This method splits the loaded documents into chunks based on the 'chunk_size' attribute of the class instance.

        Args:
            percent_overlap (float, optional): The percentage of overlap between consecutive chunks. Defaults to 0.1.

        Returns:
            None. The method updates the 'ls_chunks' attribute of the class instance with the split chunks.
        """
        
        self.ls_text_seps = [
            "\n#{1,6} ",
            "\n\n",
            "\n",
            " ",
            "",
        ]

        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.embedding_tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size * percent_overlap),
            add_start_index=True,
            strip_whitespace=True,
            separators=self.ls_text_seps,
        )
        
        ls_chunks_temp= []
        for doc in tqdm(self.ls_docs, desc='- Splitting Documents to Chunks'):
            ls_chunks_temp += self.text_splitter.split_documents([doc])
            
        set_unique_texts = set()
        ls_unique_chunks = []
        for chunk in tqdm(ls_chunks_temp, desc='- Removing Duplicated Chunks'):
            if chunk.page_content not in set_unique_texts:
                set_unique_texts.add(chunk.page_content)
                ls_unique_chunks.append(chunk)
                
        self.ls_chunks = ls_unique_chunks
        
        if self.cache_chunks:
            self.save_chunks_to_cache()

        print(f'- {len(self.ls_docs):,} Documents splitted into {len(self.ls_chunks):,} Chunks')


    def prepare_vector_db(self):
        """
        This method prepares a vector database (FAISS) from the chunks of documents.

        Args:
            None.

        Returns:
            None. The method updates the 'vector_db' attribute of the class instance with the prepared vector database.
        """
        
        def get_cur_time():
            return datetime.now().isoformat()

        time_start = get_cur_time()
        print(f'- Vector DB: Start Embedding at {time_start}')
        
        self.vector_db = FAISS.from_documents(
            self.ls_chunks,
            self.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )
        
        time_end = get_cur_time()
        print(f'- Vector DB: Finished Embedding at {time_end}')
        
        if self.cache_vector_db:
            self.save_vector_db_to_cache()


    @staticmethod
    def from_yaml_config(config: dict):
        """
        This static method creates a Vectorizer instance from a configuration dictionary.

        Args:
            config (dict): A dictionary containing the configuration parameters. The dictionary keys should include:
                - 'cache_dir': The directory where the cache files are stored.
                - 'cache_docs': A boolean indicating whether to cache the loaded documents.
                - 'cache_chunks': A boolean indicating whether to cache the split chunks.
                - 'cache_vector_db': A boolean indicating whether to cache the prepared vector database.
                - 'docs_from_hf': A boolean indicating whether to load documents from Hugging Face.
                - 'docs_dataset_id': The identifier of the dataset to be loaded from Hugging Face.
                - 'docs_dataset_split': The specific split of the dataset to be loaded.
                - 'docs_dataset_column': The specific column in the dataset that contains the document content.
                - 'docs_from_pdf_dir': A boolean indicating whether to load documents from a directory of PDF files.
                - 'docs_pdf_dir_path': The path to the directory containing the PDF files.
                - 'docs_from_file_lines': A boolean indicating whether to load documents from a text file.
                - 'docs_file_lines_path': The path to the text file.
                - 'embedding_model_from_hf': A boolean indicating whether to load the embedding model from Hugging Face.
                - 'embedding_model_id': The identifier of the model to be loaded from Hugging Face.
                - 'embedding_model_device': The device where the model will be loaded.
                - 'embedding_model_max_seq_len': The maximum sequence length for the model.
                - 'split_overlap': The percentage of overlap between consecutive chunks.

        Returns:
            vec (Vectorizer): A Vectorizer instance initialized with the provided configuration.
        """

        # The order of the operations are important.
        vec = Vectorizer(config['cache_dir'], config['cache_docs'], config['cache_chunks'], config['cache_vector_db'])
        vec.config = config
        
        if not vec.load_docs_from_cache():
            if config['docs_from_hf']:
                vec.load_docs_from_hf(config['docs_dataset_id'], config['docs_dataset_split'], config['docs_dataset_column'])
            elif config['docs_from_pdf_dir']:
                vec.load_docs_from_pdf_dir(config['docs_pdf_dir_path'])
            elif config['docs_from_file_lines']:
                vec.load_docs_from_file_lines(config['docs_file_lines_path'])

        if config['embedding_model_from_hf']:
            vec.load_embedding_model_from_hf(config['embedding_model_id'], config['embedding_model_device'], config['embedding_model_max_seq_len'])

        if not vec.load_chunks_from_cache():
            vec.split_docs(config['split_overlap'])
            
        if not vec.load_vector_db_from_cache():
            vec.prepare_vector_db()
            
        return vec
    






class SimpleRAG():
    """
    The SimpleRAG class is a simple implementation of the RAG (Retrieval-Augmented Generation) model.
    
    It uses a vectorizer to retrieve similar documents from a database, and optionally a reranker to rerank the retrieved documents. It then generates an answer using the retrieved (and possibly reranked) documents.
    
    Attributes:
        reranker (Reranker, optional): The reranker model to use. Defaults to None.
        vectorizer (Vectorizer): The vectorizer to use for retrieving similar documents.
        num_retrievals (int): The number of documents to retrieve.
        num_selections (int): The number of documents to select after reranking.
        
    Methods:
        set_prompts(system_prompt, context_prompt): Sets the system and context prompts for the model.
        load_generator_model_from_hf(model_id, device, torch_dtype, max_new_tokens, trust_remote_code): Loads the generator model and tokenizer from Hugging Face.
        load_reranker(model_id): Loads the reranker model from the given model ID.
        generate_answer(question, context): Generates an answer to a given question, optionally using a context.
        rag_invoke(question): Retrieves similar documents from the vector database, reranks them if a reranker is available, and generates an answer using the retrieved documents.
        rag_pipeline(ls_questions): Applies the RAG pipeline to a list of questions. Generates answers and retrieves documents for each question.
        from_yaml_config(config): Creates an instance of the SimpleRAG class from a YAML configuration file.
    """
    
    def __init__(self, vectorizer: Vectorizer, num_retrievals: int, num_selections: int):
        """
        Initializes the SimpleRAG class.
        
        Args:
            vectorizer (Vectorizer): The vectorizer to use for retrieving similar documents.
            num_retrievals (int): The number of documents to retrieve.
            num_selections (int): The number of documents to select after reranking.
        """
        
        self.reranker = None
        self.vectorizer = vectorizer
        self.num_retrievals = num_retrievals
        self.num_selections = num_selections


    def set_prompts(self, system_prompt: str, context_prompt: str):
        """
        This function sets the system and context prompts for the model.
        
        Args:
            system_prompt (str): The system prompt to be set.
            context_prompt (str): The context prompt to be set.
        """
        self.system_prompt = system_prompt
        self.context_prompt = context_prompt


    def load_generator_model_from_hf(self, model_id, device='cuda:0', torch_dtype=torch.bfloat16, max_new_tokens=1024, trust_remote_code=True):
        """
        This function loads the generator model and tokenizer from Hugging Face.
        
        Args:
            model_id (str): The ID of the model to be loaded.
            device (str, optional): The device to load the model on. Defaults to 'cuda:0'.
            torch_dtype (torch.dtype, optional): The data type to use. Defaults to torch.bfloat16.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 1024.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to True.
        """
        
        self.hf_generator_model_id = model_id

        print(f'- Loading Generator Model & Tokenizer: "{self.hf_generator_model_id}"')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens
        )
        # this is bug: https://github.com/langchain-ai/langchain/issues/22776#issue-2346538588
        self.generator_model = HuggingFacePipeline(pipeline=pipe)
        self.generator_model_output_parser = StrOutputParser()


    def load_reranker(self, model_id):
        """
        This function loads the reranker model from the given model ID.
        
        Args:
            model_id (str): The ID of the model to be loaded.
        """
        self.reranker = RAGPretrainedModel.from_pretrained(model_id)


    def generate_answer(self, question: str, context: Optional[str] = None) -> str:
        """
        This function generates an answer to a given question, optionally using a context.
        
        Args:
            question (str): The question to generate an answer for.
            context (str, optional): The context to use for generating the answer. Defaults to None.
        
        Returns:
            str: The generated answer.
        """
        
        if 'phi' in self.hf_generator_model_id.lower():
            # phi3 does not follow system prompt instructions.
            rag_prompt = ChatPromptTemplate.from_messages([
                ("user", self.system_prompt + '\n' + self.context_prompt)
            ])
        else:
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("user", self.context_prompt)
            ])
        chain = rag_prompt | self.generator_model #| self.generator_model_output_parser
        answer = chain.invoke({'context': context, 'question': question})
        return answer


    def rag_invoke(self, question):
        """
        This function retrieves similar documents from the vector database, reranks them if a reranker is available, and generates an answer using the retrieved documents.
        
        Args:
            question (str): The question to generate an answer for.
        
        Returns:
            tuple: The generated answer and the retrieved documents.
        """
        
        print(f'- Vector DB: Retrieving {self.num_retrievals} Similar Documents')
        retrieved_docs = self.vectorizer.vector_db.similarity_search(
            query=question,
            k=self.num_retrievals,
        )

        print(f'- Vector DB: Retrieved {len(retrieved_docs):,} Documents')

        if self.reranker:
            print(f"- Reranking Documents: {self.num_retrievals} -> {self.num_selections}")
            relevant_docs_reranked = self.reranker.rerank(question,
                                                          [doc.page_content for doc in retrieved_docs],
                                                           k=self.num_selections)

            print('- Generating Answer')
            prompt_context = '\nExtracted documents:\n' + '\n'.join(f"Document {str(i)}:::\n{doc['content']}" for i, doc in enumerate(relevant_docs_reranked))
            rag_answer = self.generate_answer(question, prompt_context)
            return (rag_answer, relevant_docs_reranked)
        else:
            print('- Generating Answer')
            prompt_context = '\nExtracted documents:\n' + '\n'.join(f"Document {str(i)}:::\n{doc.page_content}" for i, doc in enumerate(retrieved_docs))
            rag_answer = self.generate_answer(question, prompt_context)
            return (rag_answer, retrieved_docs)
        
        
    def rag_pipeline(self, ls_questions):
        """
        This function applies the RAG pipeline to a list of questions. It generates answers and retrieves documents for each question.
        
        Args:
            ls_questions (list): The list of questions to generate answers for.
        
        Returns:
            dict: A dictionary containing the questions, generated answers, and retrieved documents.
        """
        
        data = {
            'questions': [],
            'rag_answers': [],
            'retrieved_docs': [],
        }
        
        for i, question in tqdm(enumerate(ls_questions), total=len(ls_questions)):
            rag_answer, retrieved_docs = self.rag_invoke(question)
            data['questions'].append(question)
            data['rag_answers'].append(rag_answer)
            data['retrieved_docs'].append(retrieved_docs)
            
        return data


    @staticmethod
    def from_yaml_config(config):
        """
        This static method creates an instance of the SimpleRAG class from a YAML configuration file.
        
        The configuration file should contain the following keys:
        - 'num_retrievals': The number of documents to retrieve.
        - 'num_selections': The number of documents to select after reranking.
        - 'system_prompt': The system prompt to be set.
        - 'context_prompt': The context prompt to be set.
        - 'generator_model_from_hf': A boolean indicating whether to load the generator model from Hugging Face.
        - 'generator_model_id': The ID of the generator model to be loaded.
        - 'generator_model_device': The device to load the generator model on.
        - 'generator_model_torch_dtype': The data type to use for the generator model.
        - 'generator_model_max_new_tokens': The maximum number of new tokens to generate.
        - 'generator_model_trust_remote_code': Whether to trust remote code.
        - 'reranker_model_from_hf': A boolean indicating whether to load the reranker model from Hugging Face.
        - 'reranker_model_id': The ID of the reranker model to be loaded.
        
        Args:
            config (dict): The configuration dictionary.
        
        Returns:
            SimpleRAG: An instance of the SimpleRAG class.
        """

        vectorizer = Vectorizer.from_yaml_config(config)
        rag = SimpleRAG(vectorizer, config['num_retrievals'], config['num_selections'])
        rag.config = config
        
        rag.set_prompts(config['system_prompt'], config['context_prompt'])
        
        if config['generator_model_from_hf']:
            rag.load_generator_model_from_hf(config['generator_model_id'], config['generator_model_device'], config['generator_model_torch_dtype'], config['generator_model_max_new_tokens'], config['generator_model_trust_remote_code'])
        
        if config['reranker_model_from_hf']:
            rag.load_reranker(config['reranker_model_id'])
        
        return rag




