import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
# from langchain.vectorstores import FAISS # deprecated
# from langchain_community.vectorstores import FAISS # use custom FAISS lib
from .customized_faiss import FAISS
from langchain_core.documents import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings # deprecated
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import dill, json, yaml
from ragatouille import RAGPretrainedModel
from together import Together
from openai import OpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ExactMatch, StringPresence, NonLLMStringSimilarity, RougeScore, BleuScore, SemanticSimilarity


class RAGEval():
    @staticmethod
    def mrr_at(k, ls_rets, ls_golds):
        ls_rets = [ls_rets[q] for q in ls_rets.keys() if len(ls_golds[q]) > 0]
        ls_golds = [ls_golds[q] for q in ls_golds.keys() if len(ls_golds[q]) > 0]
        ls_mrr = []

        for i, rets, golds in (pbar := tqdm(zip(range(len(ls_golds)), ls_rets, ls_golds))):
            first_relevant_rank = None
            
            golds_stripped = [''.join(gold.split()) for gold in golds]
            rets_stripped = [''.join(ret.split()) for ret in rets]
            
            for r, ret_item in enumerate(rets_stripped):
                if any(gold_item in ret_item for gold_item in golds_stripped):
                    if r < k:
                        if first_relevant_rank is None:
                            first_relevant_rank = r + 1
                        
            ls_mrr.append(1 / first_relevant_rank if first_relevant_rank else 0)
            pbar.set_description(f"MRR@{k} {sum(ls_mrr) / len(ls_golds):.4f}")
            
        return sum(ls_mrr) / len(ls_golds)


    @staticmethod
    def map_at(k, ls_rets, ls_golds):
        ls_rets = [ls_rets[q] for q in ls_rets.keys() if len(ls_golds[q]) > 0]
        ls_golds = [ls_golds[q] for q in ls_golds.keys() if len(ls_golds[q]) > 0]
        ls_apk = []
        for i, rets, golds in (pbar := tqdm(zip(range(len(ls_golds)), ls_rets, ls_golds))):
            ap_sum = 0
            found_golds = []

            golds_stripped = [''.join(gold.split()) for gold in golds]
            rets_stripped = [''.join(ret.split()) for ret in rets]
            
            for r, ret_item in enumerate(rets_stripped):
                if any(gold_item in ret_item for gold_item in golds_stripped):
                    if r < k:
                        # Compute precision at this rank for this query
                        count = 0
                        for gold_item in golds_stripped:
                            if gold_item in ret_item and not gold_item in found_golds:
                                count =  count + 1
                                found_golds.append(gold_item)
                        p_at_r = count / (r+1)
                        ap_sum += p_at_r

            # Calculate metrics for this query
            ls_apk.append(ap_sum / min(len(golds_stripped), k))
            pbar.set_description(f"MAP@{k} {sum(ls_apk) / len(ls_golds):.4f}")
            
        return sum(ls_apk) / len(ls_golds)


    @staticmethod
    def hits_at(k, ls_rets, ls_golds):
        ls_rets = [ls_rets[q] for q in ls_rets.keys() if len(ls_golds[q]) > 0]
        ls_golds = [ls_golds[q] for q in ls_golds.keys() if len(ls_golds[q]) > 0]
        hits = 0
        for i, rets, golds in (pbar := tqdm(zip(range(len(ls_golds)), ls_rets, ls_golds))):
            is_hit = False
            golds_stripped = [''.join(gold.split()) for gold in golds]
            rets_stripped = [''.join(ret.split()) for ret in rets]
            
            for ret_item in rets_stripped[:k]:
                if any(gold_item in ret_item for gold_item in golds_stripped):
                    is_hit = True
                        
            hits += int(is_hit)
            pbar.set_description(f"Hits@{k} {hits/(i+1):.4f}")
            
        return hits / len(ls_golds)
        
        
    @staticmethod
    def retrieval_metrics(ls_rets, ls_golds):
        rets, golds = ls_rets, ls_golds
        
        eval_dict = {}
        eval_dict['hit10'] = RAGEval.hits_at(10, rets, golds)
        eval_dict['hit4'] = RAGEval.hits_at(4, rets, golds)
        eval_dict['map10'] = RAGEval.map_at(10, rets, golds)
        eval_dict['mrr10'] = RAGEval.mrr_at(10, rets, golds)
        
        return eval_dict
    
    
    @staticmethod
    def generation_metrics(ls_preds, ls_golds, embedding_model):
        gen_eval = {}
        for query in (pbar := tqdm(ls_preds.keys())):
            sample = SingleTurnSample(
                response=ls_preds[query],
                reference=ls_golds[query]
            )
            
            eval_dict = {}

            pbar.set_description(f"ExactMatch")
            eval_dict['exact'] = ExactMatch().single_turn_score(sample)
            pbar.set_description(f"StringPresence")
            eval_dict['presence'] = StringPresence().single_turn_score(sample)
            pbar.set_description(f"NonLLMStringSimilarity")
            eval_dict['string_sim'] = NonLLMStringSimilarity().single_turn_score(sample)
            pbar.set_description(f"RougeScore")
            eval_dict['rouge'] =  RougeScore().single_turn_score(sample)
            
            pbar.set_description(f"SemanticSimilarity")
            # Semantic Sim
            embedding_1 = np.array(embedding_model.embed_query(ls_preds[query]))
            embedding_2 = np.array(embedding_model.embed_query(ls_golds[query]))
            # Normalization factors of the above embeddings
            norms_1 = np.linalg.norm(embedding_1, keepdims=True)
            norms_2 = np.linalg.norm(embedding_2, keepdims=True)
            embedding_1_normalized = embedding_1 / norms_1
            embedding_2_normalized = embedding_2 / norms_2
            similarity = embedding_1_normalized @ embedding_2_normalized.T
            eval_dict['semantic_sim'] = similarity.flatten().item()
            
            gen_eval[query] = eval_dict
            
            # pbar.set_description(f"Accuracy: {num_corrects/len(ls_preds):.4f}")

        return gen_eval


TEXT_SEPS = ["\n#{1,6} ", "\n\n", "\n", " ", ""]
class RAGTools():
    @staticmethod
    def load_docs_from_hf(dataset_id: str, subset: str, split: str, column: str) -> List[str]:        
        print(f'- Loading Corpus Dataset from HF: {dataset_id} ({subset})')
        ds = load_dataset(dataset_id, subset, split=split)
        ls_corpus = []
        for row in tqdm(ds, desc='- Converting to LangChain Document'):
            row = row.copy()
            content = row[column]
            del row[column]
            metadata = row | {'hf_ds_id': dataset_id, 'hf_ds_subset': subset, 'hf_ds_split': split, 'hf_ds_col': column}
            ls_corpus.append(Document(page_content=content, metadata=metadata))

        return ls_corpus
    
    
    @staticmethod
    def load_queries_from_hf(dataset_id: str, subset: str, split: str, column: str):
        print(f'- Loading Queries Dataset from HF: {dataset_id} ({subset})')
        ds = load_dataset(dataset_id, subset, split=split)
        ls_queries = ds[column]
        return ls_queries
    
    
    
    @staticmethod
    def load_text_splitter_char(chunk_size, percent_overlap):
        return CharacterTextSplitter(
            # separators=TEXT_SEPS,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * percent_overlap),
            length_function=len,
            is_separator_regex=False,
        )
    
    def load_text_splitter_recursive_char(chunk_size, percent_overlap):
        return RecursiveCharacterTextSplitter(
            separators=TEXT_SEPS,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * percent_overlap),
            length_function=len,
            is_separator_regex=False,
        )
        
    def load_text_splitter_hf_tokenizer(tokenizer, chunk_size, percent_overlap):
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * percent_overlap),
            add_start_index=True,
            strip_whitespace=True,
            separators=TEXT_SEPS,
        )
    
    def load_text_splitter_tiktoken(model_id, chunk_size, percent_overlap):
        return CharacterTextSplitter.from_tiktoken_encoder(
            encoding = chunk_size, chunk_size=chunk_size, chunk_overlap=percent_overlap
        )

        

class RAGPipeline():
    def __init__(self, cache_dir, system_prompt, context_prompt):
        self.cache_dir = cache_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.system_prompt = system_prompt
        self.context_prompt = context_prompt
        
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.openai_client = None
        self.together_client = None
        self.pipe = None
        
    
    def load_embedding_model_from_hf(self, model_id: str):
        self.hf_embedding_model_id = model_id

        print(f'- Loading Embedding Model & Tokenizer: {self.hf_embedding_model_id}')
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_id,
            multi_process=True, # multi GPU
            model_kwargs={'device': self.device},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        )

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_id)

        # print(f'- Chunk Size (#Tokens): {self.chunk_size}')
    
    
    def load_reranker_model_from_hf(self, model_id):
        if model_id is not None:
            self.reranker = RAGPretrainedModel.from_pretrained(model_id)
        else:
            self.reranker = None
            
    
    def load_generator_model_from_hf(self, model_id, generation_config):
        self.hf_generator_model_id = model_id
        self.generation_config = generation_config
        print(f'- Loading Generator Model & Tokenizer: "{self.hf_generator_model_id}"')
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, trust_remote_code=True)
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
    
    def load_generator_model_from_together(self, model_id, api_key):
        self.together_api_key = api_key
        self.together_model_id = model_id
        self.together_client = Together(api_key=api_key)
        
    
    def load_generator_model_from_openai(self, model_id, api_key):
        self.openai_api_key = api_key
        self.openai_model_id = model_id
        self.openai_client = OpenAI(api_key=api_key)
        
    
    def split_docs(self, ls_docs, text_splitter) -> List[str]:
        # print(f'- Chunk Size (#Tokens): {self.chunk_size}')
        ls_chunks_temp= []
        for doc in tqdm([Document(page_content=doc) for doc in ls_docs], desc='- Splitting Documents to Chunks'):
            ls_chunks_temp += text_splitter.split_documents([doc])
            
        set_unique_texts = set()
        ls_unique_chunks = []
        for chunk in tqdm(ls_chunks_temp, desc='- Removing Duplicated Chunks'):
            if chunk.page_content not in set_unique_texts:
                set_unique_texts.add(chunk.page_content)
                ls_unique_chunks.append(chunk)
                
        ls_chunks = ls_unique_chunks 
        # cache_object(self.cache_dir, self.ls_chunks, 'chunks')

        print(f'- {len(ls_docs):,} Documents splitted into {len(ls_chunks):,} Chunks')
        
        self.ls_docs = ls_docs
        self.ls_chunks = ls_chunks
        return ls_chunks
    
    
    def prepare_vector_db(self, ls_chunks):
        def get_cur_time():
            return datetime.now().isoformat()

        time_start = get_cur_time()
        print(f'- Vector DB ({self.device}): Start Embedding at {time_start}')
        
        self.vector_db = FAISS.from_documents(
            ls_chunks,
            self.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )
        
        time_end = get_cur_time()
        print(f'- Vector DB: Finished Embedding at {time_end}')
        
        # if self.cache_dir is not None:
        #     self.cache_vector_db()
        
    
    def retrieve(self, ls_queries: List[str], num_retrievals: int) -> Dict[str, List[str]]:
        ls_ls_docs = self.vector_db.batch_similarity_search(ls_queries, num_retrievals, True)
        ls_rets = {k:[item.page_content for item in v] for k, v in zip(ls_queries, ls_ls_docs)}
        return ls_rets
    
    
    def rerank(self, ls_rets: Dict[str, List[str]], num_selections: int) -> Dict[str, List[str]]:
        # ls_reranked_rets = {}
        # for query, rets in tqdm(ls_rets.items(), total=len(ls_rets), desc='Reranking Retrievals'):
        #         relevant_docs_reranked = self.reranker.rerank(query, rets, k=num_selections)
        #         ls_reranked_rets[query] = [doc['content'] for doc in relevant_docs_reranked]

        # return ls_reranked_rets
        
        all_rets = list(set([item for sublist in list(ls_rets.values()) for item in sublist]))
        ls_queries = list(ls_rets.keys())
        output = self.reranker.rerank(ls_queries, all_rets, k=num_selections)
        ls_reranked_rets = {query: [item['content'] for item in output[ls_queries.index(query)]] for query in ls_queries}
        torch.cuda.empty_cache()
        return ls_reranked_rets
    
    def augmented_generate(self, query: str, context: str) -> str:
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': self.context_prompt.format(CONTEXT=context, QUERY=query)},
        ]

        if self.pipe:
            print(self.generation_config)
            output = self.pipe(messages, **self.generation_config)
            response = output[0]['generated_text']
            
        if self.together_client:
            output = self.together_client.chat.completions.create(model=self.together_model_id, messages=messages)
            response = output.choices[0].message.content      
                  
        if self.openai_client:
            output = self.openai_client.chat.completions.create(model=self.openai_model_id, messages=messages)
            response = output.choices[0].message.content
            
        return response

    
    def create_prompts(self, ls_rets: Dict[str, List[str]]) -> Dict[str, List[str]]:
        ls_generation_prompts = {}
        for query, relevant_docs in tqdm(ls_rets.items(), total=len(ls_rets), desc='Creating Generation Prompts'):
            docs_prompt = [f"Document {str(i)}:::\n{doc}" for i, doc in enumerate(relevant_docs)]
            prompt_context = '\nExtracted documents:\n' + '\n'.join(docs_prompt)
            ls_generation_prompts[query] = prompt_context
        
        return ls_generation_prompts
        
        
    def generate_responses(self, ls_prompts: Dict[str, List[str]]) -> Dict[str, List[str]]:
        ls_responses = {}
        
        for query, generation_prompt in tqdm(ls_prompts.items(), total=len(ls_prompts), desc='Generating Responses'):
            answer = self.augmented_generate(query, generation_prompt)
            ls_responses[query] = answer
        
        return ls_responses

    
    @staticmethod
    def run_pipeline_from_yaml(yaml_path):
        with open(yaml_path, 'r') as file:
            config = yaml.load(file, yaml.FullLoader)
            
        with open(config['dataset_path'], 'r') as file:
            ds = json.load(file)
            
        rag_pipe = RAGPipeline(config['cache_dir'], config['system_prompt'], config['context_prompt'])
        ls_docs = ds['corpus']
        ls_queries = list(ds['gold_answers'].keys())
        
        match config['text_splitter']:
            case 'char':
                text_splitter = RAGTools.load_text_splitter_char(config['chunk_size'], config['split_overlap'])
            case'recursive_char':
                text_splitter = RAGTools.load_text_splitter_recursive_char(config['chunk_size'], config['split_overlap'])
            case 'hf_tokenizer':
                tokenizer = AutoTokenizer.from_pretrained(config['text_splitter_model_id'])
                text_splitter = RAGTools.load_text_splitter_hf_tokenizer(tokenizer, config['chunk_size'], config['split_overlap'])
            case 'tiktoken':
                text_splitter = RAGTools.load_text_splitter_tiktoken(config['text_splitter_model_id'], config['chunk_size'], config['split_overlap'])
                pass
        
        rag_pipe.load_embedding_model_from_hf('thenlper/gte-small')
        ls_chunks = rag_pipe.split_docs(ls_docs, text_splitter)
        rag_pipe.prepare_vector_db(ls_chunks)
        
        ls_rets = rag_pipe.retrieve(ls_queries, config['num_retrievals'])
        ds['pred_retrieves'] = ls_rets
        ds['retrieval_eval']= RAGEval.retrieval_metrics(ls_rets, ds['gold_retrieves'])
        
        
        if config['reranker_model_id']:
            rag_pipe.load_reranker_model_from_hf(config['reranker_model_id'])
            ls_reranked_rets = rag_pipe.rerank(ls_rets, config['num_selections'])
        else:
            ls_reranked_rets = ls_rets
        
        torch.cuda.empty_cache()
        ds['reranker_eval']= RAGEval.retrieval_metrics(ls_reranked_rets, ds['gold_retrieves'])
        
        match config['generator']:
            case 'hf':
                rag_pipe.load_generator_model_from_hf(config['generator_model_id'], config['generator_model_config'])
            case 'together':
                rag_pipe.load_generator_model_from_together(config['generator_model_id'], config['api_key'])
            case 'openai':
                rag_pipe.load_generator_model_from_openai(config['generator_model_id'], config['api_key'])
        
        ls_prompts = rag_pipe.create_prompts(ls_reranked_rets)
        # ls_responses = rag_pipe.generate_responses(ls_prompts)
        # DEBUG
        random_queries = list(ls_prompts.keys())[:10]
        random_prompts = {k:ls_prompts[k] for k in random_queries}
        ls_responses = rag_pipe.generate_responses(random_prompts)
        # END DEBUG
        ds['pred_answers'] = ls_responses
        ds['generation_eval'] = RAGEval.generation_metrics(ls_responses, ds['gold_answers'], rag_pipe.embedding_model)
        
        print(f'- Saving Results @ {config['results_path']}')
        with open(config['results_path'], 'w') as file:
            json.dump(ds, file, indent=4, ensure_ascii=False)