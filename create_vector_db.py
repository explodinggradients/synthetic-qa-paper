# %%
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
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



class SimpleRAG():
    def __init__(self, cache_dir, cache_docs=True, cache_chunks=True, cache_vector_db=True):
        self.cache_dir = cache_dir
        self.cache_docs = cache_docs
        self.cache_chunks = cache_chunks
        self.cache_vector_db = cache_vector_db
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


    @staticmethod
    def load_cached_docs(cache_path):
        cache_file = Path(cache_path)
        if cache_file.is_file():
            with open(cache_path, 'rb') as f:
                print(f'- Loading Documents from Cache: "{cache_path}"')
                ls_docs = dill.load(f)
            
            return ls_docs
        else:
            return None


    def load_docs_from_hf(self, dataset_id, split, column):
        cache_path = f'{self.cache_dir}/cached_docs.pkl'
        cached_ls_docs = SimpleRAG.load_cached_docs(cache_path)
        if cached_ls_docs is not None:
            self.ls_docs = cached_ls_docs
        else:
            self.hf_dataset_id = dataset_id
            self.hf_dataset_split = split
            self.hf_dataset_column = column

            print('- Loading HF Dataset...')
            ds = load_dataset(dataset_id, split=split)

            self.ls_docs = [
                Document(page_content=row[column], metadata={'hf_ds_id': dataset_id, 'hf_ds_split': split, 'hf_ds_col': column})
                for row in tqdm(ds, desc='Converting to LangChain Document')
            ]

            if self.cache_docs:
                with open(cache_path, 'wb') as f:
                    print(f'- Caching Documents at "{cache_path}"')
                    dill.dump(self.ls_docs, f)


    def load_docs_from_pdf_dir(self, dir_path):
        cache_path = f'{self.cache_dir}/cached_docs.pkl'
        cached_ls_docs = SimpleRAG.load_cached_docs(cache_path)
        if cached_ls_docs is not None:
            self.ls_docs = cached_ls_docs
        else:
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
                with open(cache_path, 'wb') as f:
                    print(f'- Caching Documents at "{cache_path}"')
                    dill.dump(self.ls_docs, f)

    def load_docs_from_file_lines(self, file_path):
        cache_path = f'{self.cache_dir}/cached_docs.pkl'
        cached_ls_docs = SimpleRAG.load_cached_docs(cache_path)
        if cached_ls_docs is not None:
            self.ls_docs = cached_ls_docs
        else:
            self.corpus_file_path = file_path
            with open(file_path, 'r') as file:
                print(f'- Reading Lines: {file_path}')
                lines = file.readlines()
                print(f'Read {len(lines):,} Lines.')


            self.ls_docs = [
                Document(page_content=line, metadata={'corpus_file_path': self.corpus_file_path, 'line_id': i})
                for i, line in tqdm(enumerate(lines), desc='Converting Lines to LangChain Document', total=len(lines))
            ]

            if self.cache_docs:
                with open(cache_path, 'wb') as f:
                    print(f'- Caching Documents at "{cache_path}"')
                    dill.dump(self.ls_docs, f)


    def load_embedding_model_from_hf(self, model_id, device='cuda:0'):
        self.hf_embedding_model_id = model_id

        print(f'- Loading Embedding Model & Tokenizer: {self.hf_embedding_model_id}')
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_id,
            multi_process=True, # multi GPU
            model_kwargs={'device': device},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        )

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.chunk_size = self.embedding_model.client.max_seq_length
        print(f'Max Seq Length: {self.chunk_size}') # todo: make this a parameter too


    def split_docs(self, percent_overlap=0.1):
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

        cache_path = f'{self.cache_dir}/cached_chunks.pkl'
        try:
            with open(cache_path, 'rb') as f:
                print(f'- Loading Chunks from Cache: "{cache_path}"')
                self.ls_chunks = dill.load(f)
        except FileNotFoundError as e:
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
                with open(cache_path, 'wb') as f:
                    print(f'- Caching Chunks at "{cache_path}"')
                    dill.dump(self.ls_chunks, f)

        print(f'{len(self.ls_docs):,} Documents splitted into {len(self.ls_chunks):,} Chunks')


    def prepare_vector_db(self, num_retrievals=30):
        self.num_retrievals = num_retrievals
        def get_cur_time():
            return datetime.now().isoformat()


        #cache_path = f'{self.cache_dir}/cached_vector_db.pkl'
        cache_path = f'{self.cache_dir}/cached_vector_db'
        try:
            # with open(cache_path, 'rb') as f:
            #     print(f'Loading Vector DB from Cache: "{cache_path}"')
            #     self.vector_db = dill.load(f)
            print(f'- Loading Vector DB from Cache: "{cache_path}"')
            self.vector_db = FAISS.load_local(
                cache_path,
                self.embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
                allow_dangerous_deserialization=True,
            )
        except RuntimeError as e:
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
                # with open(cache_path, 'wb') as f:
                #     print(f'Caching Vector DB at "{cache_path}"')
                #     dill.dump(self.vector_db, f)
                print(f'- Caching Vector DB at "{cache_path}"')
                self.vector_db.save_local(cache_path)


    @staticmethod
    def from_yaml_config(yaml_config_path):
        with open(yaml_config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)

        # The order of the operations are important.
        cache = config['cache']
        rag = SimpleRAG(cache['cache_dir'], cache['cache_docs'], cache['cache_chunks'], cache['cache_vector_db'])

        corpus = config['corpus']
        if 'from_file_lines' in corpus.keys():
            rag.load_docs_from_file_lines(corpus['from_file_lines']['file_path'])
        elif 'from_hf' in corpus.keys():
            rag.load_docs_from_hf(corpus['from_hf']['dataset_id'], corpus['from_hf']['split'], corpus['from_hf']['column'])

        embedding_model = config['embedding_model']
        if 'hf' in embedding_model.keys():
            rag.load_embedding_model_from_hf(embedding_model['hf']['model_id'], embedding_model['hf']['device'])

        rag.split_docs(corpus['split_overlap'])
        rag.prepare_vector_db(corpus['num_retrievals'])

        rag.config = config
        return config, rag






# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml-config', dest='yaml_config', default='')
    args = parser.parse_args()
    
    rag = SimpleRAG.from_yaml_config(args.yaml_config)


