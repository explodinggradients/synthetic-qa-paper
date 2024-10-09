from langchain_text_splitters.base import TextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
import logging


from rag.config import Config, RetrieverConfig, instantiate_component
from rag.prompt import RAGInput, RAGPrompt
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample


logger = logging.getLogger(__name__)


class RAGBuilder:
    def __init__(
        self,
        directory: str,
        llm: BaseLanguageModel,
        text_splitter: TextSplitter,
        embedding_model: Embeddings,
        retriever_config: RetrieverConfig,
        prompt: str,
    ) -> None:
        self._directory = directory
        self.llm = llm
        self._text_splitter = text_splitter
        self._embedding_model = embedding_model

        self.db = self.build_index()
        self._retriever = None
        self._retriever_config = retriever_config

        self._prompt = RAGPrompt()
        self._prompt.instruction = prompt

    def build_index(self):
        documents = self.load_documents(self._directory)
        nodes = self._text_splitter.split_documents(documents)
        db = FAISS.from_documents(nodes, self.embedding_model)
        return db

    @property
    def embedding_model(self):
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, embedding_model):
        self._embedding_model = embedding_model
        logger.info("Embedding model set. Reinitalizing index.")
        self.db = self.build_index()

    @property
    def text_splitter(self):
        return self._text_splitter

    @text_splitter.setter
    def text_splitter(self, text_splitter):
        self._text_splitter = text_splitter
        logger.info("Text splitter set. Reinitalizing index.")
        self.db = self.build_index()

    def load_documents(self, directory: str):
        """Load documents from a directory."""
        loader = DirectoryLoader(directory, show_progress=True)
        documents = loader.load()
        return documents

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

    @classmethod
    def from_config(cls, config: Config):
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

        return cls(
            directory=config.directory,
            llm=llm,
            text_splitter=text_splitter,
            embedding_model=embedding_model,
            retriever_config=config.retriever,
            prompt=config.model_prompt,
        )

    async def invoke(self, dataset: EvaluationDataset):
        assert self._retriever is not None, "Retriever must be set before invoking."
        llm = LangchainLLMWrapper(self.llm)

        for sample in dataset:
            documents = self._retriever.invoke(sample.user_input)
            context = [document.page_content for document in documents]
            prompt_input = RAGInput(query=sample.user_input, context="\n".join(context))
            response = await self._prompt.generate(data=prompt_input, llm=llm)
            sample.retrieved_context = context
            sample.response = response

        return dataset
