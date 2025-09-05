"""
RAG (Retrieval-Augmented Generation) System for Project Maestro

This module implements a comprehensive RAG system using modern LangChain patterns
with LCEL and Runnable interfaces for enhanced AI capabilities.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.language_models import BaseLanguageModel
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

from .config import settings
from .logging import LoggerMixin


class RAGSystem(LoggerMixin):
    """
    Modern RAG system implementation using LCEL patterns.
    
    Features:
    - Vector store with semantic search
    - Document chunking and embedding
    - Streaming responses
    - Context-aware retrieval
    - Memory management
    """
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        embeddings: Optional[Embeddings] = None,
        vector_store_path: Optional[str] = None
    ):
        super().__init__()
        self.llm = llm or self._create_default_llm()
        self.embeddings = embeddings or self._create_default_embeddings()
        self.vector_store_path = vector_store_path or "data/vectorstore"
        
        # Initialize components
        self.vector_store = None
        self.retriever = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # RAG chains
        self._qa_chain = None
        self._contextual_compression_chain = None
        self._multi_query_chain = None
        
    def _create_default_llm(self) -> BaseLanguageModel:
        """Create default LLM for RAG operations."""
        if settings.openai_api_key:
            return ChatOpenAI(
                api_key=settings.openai_api_key,
                model="gpt-4o",
                temperature=0.1,
                streaming=True
            )
        else:
            raise ValueError("OpenAI API key required for RAG system")
            
    def _create_default_embeddings(self) -> Embeddings:
        """Create default embeddings model."""
        if settings.openai_api_key:
            return OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model="text-embedding-3-large"
            )
        else:
            # Fallback to local embeddings
            return SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            
    async def initialize_vector_store(self, documents: Optional[List[Document]] = None):
        """Initialize vector store with documents."""
        try:
            # Try to load existing store
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
            self.info("Loaded existing vector store", store_path=self.vector_store_path)
        except Exception:
            # Create new store
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
            self.info("Created new vector store", store_path=self.vector_store_path)
            
        if documents:
            await self.add_documents(documents)
            
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
    async def add_documents(self, documents: List[Document]):
        """Add documents to vector store."""
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        await self.vector_store.aadd_documents(chunks)
        self.info("Added documents to vector store", doc_count=len(documents), chunk_count=len(chunks))
        
    async def load_directory(self, directory_path: str, pattern: str = "**/*.md"):
        """Load documents from directory."""
        loader = DirectoryLoader(
            directory_path,
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()
        await self.add_documents(documents)
        return documents
        
    @property
    def qa_chain(self):
        """Get or create Q&A chain using LCEL."""
        if not self._qa_chain:
            self._qa_chain = self._create_qa_chain()
        return self._qa_chain
        
    def _create_qa_chain(self):
        """Create modern Q&A chain using LCEL patterns."""
        # RAG prompt template
        template = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Provide a comprehensive answer based on the context. If the context doesn't contain 
enough information, say so explicitly.
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Document formatting function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        # LCEL chain composition
        qa_chain = (
            RunnableParallel({
                "context": self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return qa_chain
        
    @property
    def contextual_compression_chain(self):
        """Get or create contextual compression chain."""
        if not self._contextual_compression_chain:
            self._contextual_compression_chain = self._create_contextual_compression_chain()
        return self._contextual_compression_chain
        
    def _create_contextual_compression_chain(self):
        """Create contextual compression retrieval chain."""
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor
        
        # Compression prompt
        compression_template = """Given the following question and context, extract only the most relevant information that helps answer the question.

Question: {question}
Context: {context}

Relevant information:"""
        
        compression_prompt = ChatPromptTemplate.from_template(compression_template)
        
        # Create compression chain using LCEL
        compression_chain = (
            compression_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Contextual compression retriever
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )
        
        # Enhanced RAG chain with compression
        enhanced_chain = (
            RunnableParallel({
                "context": compression_retriever | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough(),
            })
            | self._create_enhanced_prompt()
            | self.llm
            | StrOutputParser()
        )
        
        return enhanced_chain
        
    def _create_enhanced_prompt(self):
        """Create enhanced prompt for better responses."""
        template = """You are an AI assistant with access to relevant context. Use the following context to provide accurate, comprehensive answers.

Context:
{context}

Question: {question}

Instructions:
1. Base your answer primarily on the provided context
2. If context is insufficient, explicitly state what information is missing
3. Provide specific details and examples when available
4. Structure your response clearly and logically

Answer:"""
        return ChatPromptTemplate.from_template(template)
        
    async def query(self, question: str, use_compression: bool = False) -> str:
        """Query the RAG system."""
        chain = self.contextual_compression_chain if use_compression else self.qa_chain
        result = await chain.ainvoke(question)
        return result
        
    async def stream_query(self, question: str, use_compression: bool = False) -> AsyncGenerator[str, None]:
        """Stream query responses for real-time feedback."""
        chain = self.contextual_compression_chain if use_compression else self.qa_chain
        async for chunk in chain.astream(question):
            yield chunk
            
    async def batch_query(self, questions: List[str]) -> List[str]:
        """Process multiple queries in batch."""
        results = await self.qa_chain.abatch(questions)
        return results
        
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search in vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search(query, k=k)
        
    async def asimilarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Async similarity search."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return await self.vector_store.asimilarity_search(query, k=k)
        
    def _format_docs(self, docs):
        """Format documents for context."""
        return "\n\n".join(doc.page_content for doc in docs)
        
    async def get_relevant_context(self, query: str, max_docs: int = 5) -> List[Document]:
        """Get relevant context documents for a query."""
        return await self.asimilarity_search(query, k=max_docs)


class SentenceTransformerEmbeddings(Embeddings):
    """Local sentence transformer embeddings for fallback."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents."""
        return self.model.encode(texts).tolist()
        
    def embed_query(self, text: str) -> List[float]:
        """Embed query."""
        return self.model.encode([text])[0].tolist()
        
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed documents."""
        return await asyncio.to_thread(self.embed_documents, texts)
        
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed query."""
        return await asyncio.to_thread(self.embed_query, text)


class EnhancedRAGSystem(RAGSystem):
    """
    Enhanced RAG system with advanced features:
    - Multi-query generation
    - Hypothetical document embedding
    - Cross-encoder reranking
    - Dynamic retrieval strategy
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._multi_query_chain = None
        self._hyde_chain = None
        
    @property
    def multi_query_chain(self):
        """Multi-query generation chain."""
        if not self._multi_query_chain:
            self._multi_query_chain = self._create_multi_query_chain()
        return self._multi_query_chain
        
    def _create_multi_query_chain(self):
        """Create multi-query generation chain for diverse retrieval."""
        template = """You are an AI language model assistant. Your task is to generate 3 
different versions of the given user question to retrieve relevant documents from 
a vector database. By generating multiple perspectives on the user question, your 
goal is to help the user overcome some of the limitations of the distance-based 
similarity search.

Provide these alternative questions separated by newlines.

Original question: {question}
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Multi-query generation chain
        multi_query_chain = (
            prompt
            | self.llm
            | StrOutputParser()
            | RunnableLambda(lambda x: x.strip().split("\n"))
        )
        
        return multi_query_chain
        
    @property 
    def hyde_chain(self):
        """Hypothetical Document Embedding chain."""
        if not self._hyde_chain:
            self._hyde_chain = self._create_hyde_chain()
        return self._hyde_chain
        
    def _create_hyde_chain(self):
        """Create HyDE (Hypothetical Document Embedding) chain."""
        template = """Please write a passage to answer the question:
Question: {question}
Passage:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # HyDE chain
        hyde_chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )
        
        return hyde_chain
        
    async def enhanced_retrieval(self, question: str, strategy: str = "multi_query") -> List[Document]:
        """Enhanced retrieval using different strategies."""
        if strategy == "multi_query":
            # Generate multiple query variations
            queries = await self.multi_query_chain.ainvoke({"question": question})
            
            # Retrieve for each query
            all_docs = []
            for query in queries:
                docs = await self.asimilarity_search(query, k=3)
                all_docs.extend(docs)
                
            # Remove duplicates and return top results
            unique_docs = self._remove_duplicate_docs(all_docs)
            return unique_docs[:5]
            
        elif strategy == "hyde":
            # Generate hypothetical document
            hypothetical_doc = await self.hyde_chain.ainvoke({"question": question})
            
            # Use hypothetical doc for retrieval
            docs = await self.asimilarity_search(hypothetical_doc, k=5)
            return docs
            
        else:
            # Standard retrieval
            return await self.asimilarity_search(question, k=5)
            
    def _remove_duplicate_docs(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content."""
        seen_content = set()
        unique_docs = []
        
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
                
        return unique_docs
        
    async def conversational_rag(
        self, 
        question: str, 
        chat_history: List[tuple] = None
    ) -> str:
        """Conversational RAG with chat history."""
        chat_history = chat_history or []
        
        # Condense question with chat history
        condense_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
        
        condense_prompt = ChatPromptTemplate.from_template(condense_template)
        
        # Format chat history
        def format_chat_history(history):
            formatted = []
            for human, ai in history:
                formatted.extend([f"Human: {human}", f"Assistant: {ai}"])
            return "\n".join(formatted)
        
        # Conversational chain using LCEL
        conversational_chain = (
            RunnableParallel({
                "standalone_question": (
                    RunnablePassthrough.assign(
                        chat_history=lambda x: format_chat_history(x.get("chat_history", []))
                    )
                    | condense_prompt
                    | self.llm
                    | StrOutputParser()
                ),
                "chat_history": lambda x: x.get("chat_history", [])
            })
            | RunnablePassthrough.assign(
                context=lambda x: self._format_docs(
                    self.similarity_search(x["standalone_question"])
                )
            )
            | self._create_conversational_prompt()
            | self.llm
            | StrOutputParser()
        )
        
        result = await conversational_chain.ainvoke({
            "question": question,
            "chat_history": chat_history
        })
        
        return result
        
    def _create_conversational_prompt(self):
        """Create conversational RAG prompt."""
        template = """Answer the question based on the following context and chat history:

Context:
{context}

Chat History:
{chat_history}

Current Question: {question}

Answer:"""
        return ChatPromptTemplate.from_template(template)


# Global RAG system instance
rag_system = None

def get_rag_system() -> RAGSystem:
    """Get global RAG system instance."""
    global rag_system
    if not rag_system:
        rag_system = RAGSystem()
    return rag_system

async def initialize_rag_system(documents_path: str = "docs/") -> RAGSystem:
    """Initialize RAG system with project documents."""
    rag = get_rag_system()
    
    # Load project documents
    if Path(documents_path).exists():
        documents = await rag.load_directory(documents_path)
        await rag.initialize_vector_store(documents)
    else:
        await rag.initialize_vector_store()
        
    return rag