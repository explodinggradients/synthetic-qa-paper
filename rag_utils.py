import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS # deprecated
# from langchain_community.vectorstores import FAISS # use custom FAISS lib
from customized_faiss import FAISS
from langchain_core.documents import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings # deprecated
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import dill
from ragatouille import RAGPretrainedModel



def cache_object(cache_dir, obj, name, verbose=True):
    if cache_dir is None:
        return False
    
    cache_path = f'{cache_dir}/cached_{name}.pkl'
    with open(cache_path, 'wb') as f:
        if verbose: print(f'- Caching {name} @ "{cache_path}"')
        dill.dump(obj, f)
    return True



class VectorBase():
    def __init__(self, cache_dir: str, verbose=True):
        self.cache_dir = cache_dir
        if cache_dir is not None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.verbose = verbose


    def cache_vector_db(self):
        cache_path = f'{self.cache_dir}/cached_vector_db'
        print(f'- Caching Vector DB at "{cache_path}"')
        self.vector_db.save_local(cache_path)


    # Load Cached Objects
    def load_obj_from_cache(self, name):
        cache_path = f'{self.cache_dir}/cached_{name}.pkl'
        if Path(cache_path).is_file():
            with open(cache_path, 'rb') as f:
                print(f'- Loading {name} from Cache: "{cache_path}"')
                loaded_obj = dill.load(f)
                match name:
                    case 'docs':
                        self.ls_docs = loaded_obj
                    case 'chunks':
                        self.ls_chunks = loaded_obj
                    case _:
                        raise NotImplementedError()
            return True
        return False 
            
            
    def load_vector_db_from_cache(self):
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
    def load_docs_from_hf(self, dataset_id: str, subset: str, split: str, column: str):
        self.hf_dataset_id = dataset_id
        self.hf_dataset_subset = subset
        self.hf_dataset_split = split
        self.hf_dataset_column = column
        
        print(f'- Loading Corpus Dataset from HF: {self.hf_dataset_id} ({self.hf_dataset_subset})')
        ds = load_dataset(dataset_id, subset, split=split)
        self.ls_docs = []
        for row in tqdm(ds, desc='- Converting to LangChain Document'):
            row = row.copy()
            content = row[column]
            del row[column]
            metadata = row | {'hf_ds_id': dataset_id, 'hf_ds_subset': subset, 'hf_ds_split': split, 'hf_ds_col': column}
            self.ls_docs.append(Document(page_content=content, metadata=metadata))

        cache_object(self.cache_dir, self.ls_docs, 'docs')


    # Load Embedding Model Methods
    def load_embedding_model_from_hf(self, model_id: str, device: str ='cuda:0', max_seq_len: int =None):
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
        self.ls_text_seps = ["\n#{1,6} ", "\n\n", "\n", " ", ""]

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
        cache_object(self.cache_dir, self.ls_chunks, 'chunks')

        print(f'- {len(self.ls_docs):,} Documents splitted into {len(self.ls_chunks):,} Chunks')
        

    def prepare_vector_db(self):
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
        
        if self.cache_dir is not None:
            self.cache_vector_db()


    @staticmethod
    def from_yaml_config(config: dict):
        # The order of the operations are important.
        vec = VectorBase(config['vb_cache_dir'], config['vb_verbose'])
        vec.config = config
        
        if not vec.load_obj_from_cache('docs'):
            vec.load_docs_from_hf(config['docs_dataset_id'], config['docs_dataset_subset'], config['docs_dataset_split'], config['docs_dataset_column'])

        vec.load_embedding_model_from_hf(config['embedding_model_id'], config['embedding_model_device'], config['chunk_size'])

        if not vec.load_obj_from_cache('chunks'):
            vec.split_docs(config['split_overlap'])
            
        if not vec.load_vector_db_from_cache():
            vec.prepare_vector_db()
            
        return vec
    


class Retriever():
    def __init__(self, vb: VectorBase, cache_dir: str, num_retrievals: int =10, verbose: bool =True):
        self.vb = vb
        self.cache_dir = cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.num_retrievals = num_retrievals
        self.verbose = verbose


    def load_queries_from_cache(self):
        cache_path = f'{self.cache_dir}/cached_queries.pkl'
        if Path(cache_path).is_file():
            with open(cache_path, 'rb') as f:
                print(f'- Loading Queries from Cache: "{cache_path}"')
                self.ls_queries = dill.load(f)
            return True
        return False 
    

    def load_queries_from_hf(self, dataset_id: str, subset: str, split: str, column: str):
        self.hf_dataset_id = dataset_id
        self.hf_dataset_subset = subset
        self.hf_dataset_split = split
        self.hf_dataset_column = column
        
        print(f'- Loading Queries Dataset from HF: {self.hf_dataset_id} ({self.hf_dataset_subset})')
        ds = load_dataset(dataset_id, subset, split=split)
        self.ls_queries = ds[column]

        cache_object(self.cache_dir, self.ls_queries, 'queries')


    def load_rets_from_cache(self):
        cache_path = f'{self.cache_dir}/cached_rets.pkl'
        if Path(cache_path).is_file():
            with open(cache_path, 'rb') as f:
                print(f'- Loading Rets from Cache: "{cache_path}"')
                self.ls_rets = dill.load(f)
            return True
        return False         
    

    def retrieve(self) -> Dict[str, List[Document]]:
        ls_ls_docs = self.vb.vector_db.batch_similarity_search(self.ls_queries, self.num_retrievals, self.verbose)
        self.ls_rets = {k:v for k, v in zip(self.ls_queries, ls_ls_docs)}
        cache_object(self.cache_dir, self.ls_rets, 'rets')
    
    
    @staticmethod
    def from_yaml_config(vb: VectorBase, config: dict):
        ret = Retriever(vb, config['rets_cache_dir'], config['num_retrievals'], config['rets_verbose'])
        # The order of the operations are important.
        
        if not ret.load_queries_from_cache():
            ret.load_queries_from_hf(config['queries_dataset_id'], config['queries_dataset_subset'], config['queries_dataset_split'], config['queries_dataset_column'])
        
        if not ret.load_rets_from_cache():
            ret.retrieve()
                  
        return ret
    

    
class Generator():
    def __init__(self, ret: Retriever, cache_dir: str, num_selections: int, system_prompt: str, context_prompt: str):
        self.ret = ret
        self.cache_dir = cache_dir
        self.num_selections = num_selections
        self.system_prompt = system_prompt
        self.context_prompt = context_prompt
        
        
    # Load Cached Objects
    def load_obj_from_cache(self, name):
        cache_path = f'{self.cache_dir}/cached_{name}.pkl'
        if Path(cache_path).is_file():
            with open(cache_path, 'rb') as f:
                print(f'- Loading {name} from Cache: "{cache_path}"')
                loaded_obj = dill.load(f)
                match name:
                    case 'reranked_rets':
                        self.ls_reranked_rets = loaded_obj
                    case 'augmented_generations':
                        self.ls_augmented_generations = loaded_obj
                    case _:
                        raise NotImplementedError()
            return True
        return False 


    def load_generator_model_from_hf(self, model_id, generation_config, device='cuda:0', torch_dtype=torch.bfloat16, trust_remote_code=True):
        self.hf_generator_model_id = model_id
        self.generation_config = generation_config
        print(f'- Loading Generator Model & Tokenizer: "{self.hf_generator_model_id}"')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code)
        self.pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer
        )
        # # this is bug: https://github.com/langchain-ai/langchain/issues/22776#issue-2346538588
        # self.generator_model = HuggingFacePipeline(pipeline=pipe)
        # self.generator_model_output_parser = StrOutputParser()


    def load_reranker(self, model_id):
        self.reranker = RAGPretrainedModel.from_pretrained(model_id)

    
    def augmented_generate(self, query: str, context: str) -> str:
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': self.context_prompt.format(CONTEXT=context, QUERY=query)},
        ]

        # generation_args = {
        #     "max_new_tokens": 8,
        #     "return_full_text": False,
        #     "temperature": 0.0,
        #     "do_sample": False,
        # }

        output = self.pipe(messages, **self.generation_config)
        return output[0]['generated_text']


    def rerank_rets(self):
        ls_rets = self.ret.ls_rets
        self.ls_reranked_rets = {}

        for query, rets in tqdm(ls_rets.items(), total=len(ls_rets), desc='Reranking Retrievals'):
            if self.reranker is None:
                self.ls_reranked_rets[query] = [doc.page_content for doc in rets]
            else:
                relevant_docs_reranked = self.reranker.rerank(query,
                                                        [doc.page_content for doc in rets],
                                                        k=self.num_selections)
                self.ls_reranked_rets[query] = [doc['content'] for doc in relevant_docs_reranked]

        cache_object(self.cache_dir, self.ls_reranked_rets, 'reranked_rets')
        
        
    def generate_responses(self):
        self.ls_augmented_generations = {}
        
        for query, relevant_docs in tqdm(self.ls_reranked_rets.items(), total=len(self.ls_reranked_rets)):
            docs_prompt = [f"Document {str(i)}:::\n{doc}" for i, doc in enumerate(relevant_docs)]
            prompt_context = '\nExtracted documents:\n' + '\n'.join(docs_prompt)
            
            answer = self.augmented_generate(query, prompt_context)
            self.ls_augmented_generations[query] = answer
        

        cache_object(self.cache_dir, self.ls_augmented_generations, 'augmented_generations')


    @staticmethod
    def from_yaml_config(ret: Retriever, config: dict):
        gen = Generator(ret, config['gens_cache_dir'], config['num_selections'], config['system_prompt'], config['context_prompt'])
        gen.load_generator_model_from_hf(config['generator_model_id'], config['generator_model_config'], config['generator_model_device'], config['generator_model_torch_dtype'], config['generator_model_trust_remote_code'])
        gen.load_reranker(config['reranker_model_id'])
        
        if not gen.load_obj_from_cache('reranked_rets'):
            gen.rerank_rets()

        if not gen.load_obj_from_cache('augmented_generations'):
            gen.generate_responses()
            
        return gen



