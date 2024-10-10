import os
from langchain_text_splitters.base import TextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
import logging


from rag.config import Config, instantiate_component
from rag.prompt import RAGInput, RAGPrompt
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import EvaluationDataset
from ragas import evaluate
from ragas.metrics.base import Metric


from typing import List

VECTOR_STORE_PATH = "../vector_stores"
RESULTS_PATH = "../results"

logger = logging.getLogger(__name__)


class RAGBuilder:
    def __init__(
        self,
        config: Config,
    ) -> None:
        self.config = config
        llm = instantiate_component(config.llm) if config.llm else None
        assert isinstance(llm, BaseLanguageModel)
        text_splitter = (
            instantiate_component(config.text_splitter)
            if config.text_splitter
            else None
        )
        assert isinstance(text_splitter, TextSplitter)
        embedding_model = (
            instantiate_component(config.embedding_model)
            if config.embedding_model
            else None
        )
        assert isinstance(embedding_model, Embeddings)
        index_name = hash(
            tuple(
                [
                    self.config.directory.model_dump_json(),
                    self.config.text_splitter.model_dump_json(),
                    self.config.embedding_model.model_dump_json(),
                ]
            )
        )
        self.index_name = f"index_{index_name}"

        self.db = None
        self._directory = config.directory
        self.llm = llm
        self.text_splitter = text_splitter
        self.embedding_model = embedding_model
        self.db = self.build_index()
        self.retriever_config = config.retriever
        self.prompt = config.model_prompt

        if not os.path.exists(self.config.experiment_name):
            os.makedirs(self.config.experiment_name)

        self.config.save(os.path.join(self.config.experiment_name, "config.json"))

    @classmethod
    def from_saved_config(cls, config_name_or_path: str):
        config = Config.load(config_name_or_path)
        return cls(config)

    @property
    def prompt(self):
        return self._prompt.instruction

    @prompt.setter
    def prompt(self, prompt):
        self._prompt = RAGPrompt()
        self._prompt.instruction = prompt

    def build_index(self):
        if os.path.exists(os.path.join(VECTOR_STORE_PATH, self.index_name)):
            return FAISS.load_local(
                VECTOR_STORE_PATH,
                self.embedding_model,
                index_name=self.index_name,
                allow_dangerous_deserialization=True,
            )
        loader = DirectoryLoader(self._directory.name, **self._directory.params)
        documents = loader.load()
        nodes = self._text_splitter.split_documents(documents)
        db = FAISS.from_documents(nodes, self.embedding_model)
        db.save_local(VECTOR_STORE_PATH, self.index_name)
        return db

    @property
    def embedding_model(self):
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, embedding_model):
        self._embedding_model = embedding_model
        if self.db is not None:
            logger.info("Embedding model set. Reinitalizing index.")
            self.db = self.build_index()

    @property
    def text_splitter(self):
        return self._text_splitter

    @text_splitter.setter
    def text_splitter(self, text_splitter):
        self._text_splitter = text_splitter
        if self.db is not None:
            logger.info("Text splitter set. Reinitalizing index.")
            self.db = self.build_index()

    @property
    def retriever_config(self):
        return self._retriever_config

    @retriever_config.setter
    def retriever_config(self, retriever_config):
        assert (
            self.db is not None
        ), "Index must be built before retriever can be loaded."

        params = retriever_config.params
        name = retriever_config.type
        if name == "vanilla":
            self._retriever = self.db.as_retriever(**params)

        elif name == "multi_query":
            from langchain.retrievers.multi_query import MultiQueryRetriever

            self._retriever = MultiQueryRetriever.from_llm(
                retriever=self.db.as_retriever(**params), llm=self.llm
            )
        self._retriever_config = retriever_config

    async def invoke(self, dataset: EvaluationDataset):
        assert self._retriever is not None, "Retriever must be set before invoking."
        llm = LangchainLLMWrapper(self.llm)

        for sample in dataset:
            documents = self._retriever.invoke(sample.user_input)
            context = [document.page_content for document in documents]
            prompt_input = RAGInput(query=sample.user_input, context="\n".join(context))
            response = await self._prompt.generate(data=prompt_input, llm=llm)
            sample.retrieved_contexts = context
            sample.response = response.response

        return dataset

    async def benchmark(
        self,
        dataset: EvaluationDataset,
        metrics: List[Metric],
        eval_llm: BaseLanguageModel,
        **kwargs,
    ):
        dataset = await self.invoke(dataset)
        results = evaluate(dataset, metrics=metrics, llm=eval_llm, **kwargs)
        results.to_pandas().to_csv(os.path.join(RESULTS_PATH, self.config.experiment_name, "results.csv"))
        return results
