"""적응형 RAG 시스템 구현"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
from abc import ABC, abstractmethod

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


class RetrievalStrategy(Enum):
    """검색 전략"""
    SEMANTIC = "semantic"           # 의미론적 검색
    KEYWORD = "keyword"            # 키워드 검색
    HYBRID = "hybrid"              # 하이브리드 검색
    RERANK = "rerank"              # 재순위 검색
    CONTEXTUAL = "contextual"      # 맥락 기반 검색
    TEMPORAL = "temporal"          # 시간 기반 검색


class QueryComplexity(Enum):
    """쿼리 복잡도"""
    SIMPLE = "simple"              # 단순 질문
    MODERATE = "moderate"          # 보통 복잡도
    COMPLEX = "complex"            # 복잡한 질문
    MULTI_HOP = "multi_hop"        # 다단계 추론 필요


@dataclass
class RetrievalConfig:
    """검색 설정"""
    
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_threshold: float = 0.7
    max_documents: int = 10
    diversity_threshold: float = 0.8
    temporal_decay_factor: float = 0.1
    context_window: int = 3


@dataclass
class QueryAnalysis:
    """쿼리 분석 결과"""
    
    complexity: QueryComplexity
    intent: str
    entities: List[str]
    keywords: List[str]
    temporal_references: List[str]
    domain: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class RetrievalResult:
    """검색 결과"""
    
    documents: List[Document]
    scores: List[float]
    strategy_used: RetrievalStrategy
    query_analysis: QueryAnalysis
    retrieval_time: float
    total_candidates: int
    reranked: bool = False


@dataclass
class UserFeedback:
    """사용자 피드백"""
    
    query: str
    retrieved_docs: List[Document]
    satisfaction: float  # 0.0 - 1.0
    relevance_scores: List[float]  # 문서별 관련성 점수
    strategy_used: RetrievalStrategy
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """기본 검색기 인터페이스"""
    
    @abstractmethod
    async def retrieve(self, query: str, config: RetrievalConfig) -> List[Document]:
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        pass


class SemanticRetriever(BaseRetriever):
    """의미론적 검색기"""
    
    def __init__(self, vectorstore: FAISS, embeddings: OpenAIEmbeddings):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.performance_history = deque(maxlen=1000)
    
    async def retrieve(self, query: str, config: RetrievalConfig) -> List[Document]:
        start_time = time.perf_counter()
        
        # 의미론적 유사도 검색
        docs = self.vectorstore.similarity_search_with_score(
            query,
            k=config.max_documents,
            score_threshold=config.similarity_threshold
        )
        
        # 점수와 문서 분리
        documents = [doc for doc, score in docs]
        scores = [score for doc, score in docs]
        
        # 성능 기록
        retrieval_time = time.perf_counter() - start_time
        self.performance_history.append({
            'time': retrieval_time,
            'num_results': len(documents),
            'avg_score': np.mean(scores) if scores else 0.0
        })
        
        return documents
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        if not self.performance_history:
            return {}
        
        times = [h['time'] for h in self.performance_history]
        scores = [h['avg_score'] for h in self.performance_history]
        
        return {
            'avg_retrieval_time': np.mean(times),
            'avg_relevance_score': np.mean(scores),
            'total_retrievals': len(self.performance_history)
        }


class KeywordRetriever(BaseRetriever):
    """키워드 검색기"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.performance_history = deque(maxlen=1000)
        
        # TF-IDF 매트릭스 구축
        texts = [doc.page_content for doc in documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    async def retrieve(self, query: str, config: RetrievalConfig) -> List[Document]:
        start_time = time.perf_counter()
        
        # 쿼리를 TF-IDF 벡터로 변환
        query_vector = self.vectorizer.transform([query])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # 상위 문서 선택
        top_indices = np.argsort(similarities)[::-1][:config.max_documents]
        top_similarities = similarities[top_indices]
        
        # 임계값 필터링
        filtered_docs = []
        for idx, sim in zip(top_indices, top_similarities):
            if sim >= config.similarity_threshold:
                filtered_docs.append(self.documents[idx])
        
        # 성능 기록
        retrieval_time = time.perf_counter() - start_time
        self.performance_history.append({
            'time': retrieval_time,
            'num_results': len(filtered_docs),
            'avg_score': np.mean(top_similarities) if len(top_similarities) > 0 else 0.0
        })
        
        return filtered_docs
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        if not self.performance_history:
            return {}
        
        times = [h['time'] for h in self.performance_history]
        scores = [h['avg_score'] for h in self.performance_history]
        
        return {
            'avg_retrieval_time': np.mean(times),
            'avg_relevance_score': np.mean(scores),
            'total_retrievals': len(self.performance_history)
        }


class HybridRetriever(BaseRetriever):
    """하이브리드 검색기"""
    
    def __init__(
        self, 
        semantic_retriever: SemanticRetriever,
        keyword_retriever: KeywordRetriever,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4
    ):
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.performance_history = deque(maxlen=1000)
    
    async def retrieve(self, query: str, config: RetrievalConfig) -> List[Document]:
        start_time = time.perf_counter()
        
        # 병렬로 두 검색기 실행
        semantic_docs, keyword_docs = await asyncio.gather(
            self.semantic_retriever.retrieve(query, config),
            self.keyword_retriever.retrieve(query, config)
        )
        
        # 결과 융합
        fused_docs = self._fuse_results(semantic_docs, keyword_docs, config)
        
        # 성능 기록
        retrieval_time = time.perf_counter() - start_time
        self.performance_history.append({
            'time': retrieval_time,
            'num_results': len(fused_docs),
            'semantic_results': len(semantic_docs),
            'keyword_results': len(keyword_docs)
        })
        
        return fused_docs
    
    def _fuse_results(
        self, 
        semantic_docs: List[Document],
        keyword_docs: List[Document], 
        config: RetrievalConfig
    ) -> List[Document]:
        """검색 결과 융합"""
        
        # 문서별 점수 계산
        doc_scores = defaultdict(float)
        seen_docs = {}
        
        # 의미론적 검색 결과 처리
        for i, doc in enumerate(semantic_docs):
            doc_id = hash(doc.page_content)
            semantic_score = (len(semantic_docs) - i) / len(semantic_docs)
            doc_scores[doc_id] += semantic_score * self.semantic_weight
            seen_docs[doc_id] = doc
        
        # 키워드 검색 결과 처리
        for i, doc in enumerate(keyword_docs):
            doc_id = hash(doc.page_content)
            keyword_score = (len(keyword_docs) - i) / len(keyword_docs)
            doc_scores[doc_id] += keyword_score * self.keyword_weight
            seen_docs[doc_id] = doc
        
        # 점수순 정렬
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 상위 문서 반환
        result_docs = []
        for doc_id, score in sorted_docs[:config.max_documents]:
            if score >= config.similarity_threshold:
                result_docs.append(seen_docs[doc_id])
        
        return result_docs
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        if not self.performance_history:
            return {}
        
        times = [h['time'] for h in self.performance_history]
        
        return {
            'avg_retrieval_time': np.mean(times),
            'total_retrievals': len(self.performance_history),
            'semantic_metrics': self.semantic_retriever.get_performance_metrics(),
            'keyword_metrics': self.keyword_retriever.get_performance_metrics()
        }


class RerankRetriever(BaseRetriever):
    """재순위 검색기"""
    
    def __init__(self, base_retriever: BaseRetriever):
        self.base_retriever = base_retriever
        self.performance_history = deque(maxlen=1000)
        
        # 재순위를 위한 교차 인코더 (실제로는 sentence-transformers 사용)
        # 여기서는 간단한 휴리스틱 사용
        
    async def retrieve(self, query: str, config: RetrievalConfig) -> List[Document]:
        start_time = time.perf_counter()
        
        # 기본 검색기로 후보 문서 검색 (더 많이)
        expanded_config = RetrievalConfig(**config.__dict__)
        expanded_config.max_documents = config.max_documents * 2
        
        candidate_docs = await self.base_retriever.retrieve(query, expanded_config)
        
        # 재순위
        reranked_docs = await self._rerank_documents(query, candidate_docs, config)
        
        # 성능 기록
        retrieval_time = time.perf_counter() - start_time
        self.performance_history.append({
            'time': retrieval_time,
            'candidates': len(candidate_docs),
            'final_results': len(reranked_docs)
        })
        
        return reranked_docs
    
    async def _rerank_documents(
        self, 
        query: str, 
        documents: List[Document],
        config: RetrievalConfig
    ) -> List[Document]:
        """문서 재순위"""
        
        # 간단한 재순위 로직 (실제로는 더 정교한 모델 사용)
        scored_docs = []
        
        query_tokens = set(query.lower().split())
        
        for doc in documents:
            # 여러 요소를 고려한 점수 계산
            content = doc.page_content.lower()
            content_tokens = set(content.split())
            
            # 토큰 겹침 비율
            token_overlap = len(query_tokens & content_tokens) / len(query_tokens | content_tokens)
            
            # 문서 길이 정규화
            length_penalty = min(1.0, len(doc.page_content) / 2000)
            
            # 위치 기반 점수 (쿼리 토큰이 문서 앞쪽에 있을수록 높은 점수)
            position_scores = []
            for token in query_tokens:
                if token in content:
                    pos = content.find(token) / len(content)
                    position_scores.append(1 - pos)
            position_score = np.mean(position_scores) if position_scores else 0.0
            
            # 종합 점수
            final_score = (
                token_overlap * 0.5 +
                length_penalty * 0.2 + 
                position_score * 0.3
            )
            
            scored_docs.append((doc, final_score))
        
        # 점수순 정렬 후 상위 문서 반환
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:config.max_documents] 
                if score >= config.similarity_threshold]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        if not self.performance_history:
            return {}
        
        times = [h['time'] for h in self.performance_history]
        
        return {
            'avg_retrieval_time': np.mean(times),
            'avg_candidates': np.mean([h['candidates'] for h in self.performance_history]),
            'avg_final_results': np.mean([h['final_results'] for h in self.performance_history]),
            'base_retriever_metrics': self.base_retriever.get_performance_metrics()
        }


class QueryAnalyzer:
    """쿼리 분석기"""
    
    def __init__(self):
        # NLTK 데이터 다운로드 (필요시)
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        
        # 복잡도 패턴
        self.complexity_indicators = {
            QueryComplexity.SIMPLE: [
                'what is', 'who is', 'when is', 'where is',
                'define', 'explain simply'
            ],
            QueryComplexity.MODERATE: [
                'how does', 'why does', 'compare', 'difference',
                'advantages', 'disadvantages'
            ],
            QueryComplexity.COMPLEX: [
                'analyze', 'evaluate', 'relationship between',
                'implications', 'consequences'
            ],
            QueryComplexity.MULTI_HOP: [
                'first', 'then', 'after that', 'following',
                'step by step', 'process'
            ]
        }
        
        # 도메인 키워드
        self.domain_keywords = {
            'technology': ['software', 'programming', 'algorithm', 'database'],
            'science': ['research', 'experiment', 'hypothesis', 'theory'],
            'business': ['strategy', 'market', 'revenue', 'profit'],
            'health': ['medical', 'treatment', 'diagnosis', 'patient'],
            'education': ['learning', 'teaching', 'curriculum', 'student']
        }
    
    async def analyze_query(self, query: str, context: Dict[str, Any] = None) -> QueryAnalysis:
        """쿼리 종합 분석"""
        
        # 기본 전처리
        query_lower = query.lower()
        tokens = word_tokenize(query_lower)
        
        # 복잡도 분석
        complexity = self._analyze_complexity(query_lower)
        
        # 의도 분석
        intent = self._analyze_intent(query_lower, tokens)
        
        # 엔티티 추출
        entities = self._extract_entities(query)
        
        # 키워드 추출
        keywords = self._extract_keywords(tokens)
        
        # 시간 참조 추출
        temporal_refs = self._extract_temporal_references(query)
        
        # 도메인 분류
        domain = self._classify_domain(query_lower, keywords)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(query, complexity, intent, entities)
        
        return QueryAnalysis(
            complexity=complexity,
            intent=intent,
            entities=entities,
            keywords=keywords,
            temporal_references=temporal_refs,
            domain=domain,
            confidence=confidence,
            metadata={
                'query_length': len(query),
                'token_count': len(tokens),
                'context': context or {}
            }
        )
    
    def _analyze_complexity(self, query: str) -> QueryComplexity:
        """복잡도 분석"""
        
        # 패턴 매칭
        for complexity, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                if pattern in query:
                    return complexity
        
        # 길이와 구조 기반 휴리스틱
        if len(query.split()) < 5:
            return QueryComplexity.SIMPLE
        elif len(query.split()) < 15:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX
    
    def _analyze_intent(self, query: str, tokens: List[str]) -> str:
        """의도 분석"""
        
        intent_patterns = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who'],
            'comparison': ['compare', 'difference', 'versus', 'vs', 'better'],
            'definition': ['define', 'meaning', 'what is', 'explain'],
            'instruction': ['how to', 'step', 'guide', 'tutorial'],
            'analysis': ['analyze', 'evaluate', 'assess', 'review']
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in query for pattern in patterns):
                return intent
        
        return 'general'
    
    def _extract_entities(self, query: str) -> List[str]:
        """간단한 엔티티 추출"""
        
        # 대문자로 시작하는 단어들 (고유명사 추정)
        tokens = word_tokenize(query)
        entities = []
        
        for token in tokens:
            if token[0].isupper() and len(token) > 1 and token.isalpha():
                entities.append(token)
        
        # 숫자, 날짜 패턴 등도 추출 가능
        import re
        
        # 날짜 패턴
        date_pattern = r'\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{1,2}-\d{1,2}'
        dates = re.findall(date_pattern, query)
        entities.extend(dates)
        
        # 숫자 패턴
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, query)
        entities.extend(numbers)
        
        return list(set(entities))
    
    def _extract_keywords(self, tokens: List[str]) -> List[str]:
        """키워드 추출"""
        
        # 불용어 제거 및 필터링
        keywords = []
        
        for token in tokens:
            if (token.lower() not in self.stop_words and 
                len(token) > 2 and 
                token.isalpha()):
                keywords.append(token.lower())
        
        return keywords
    
    def _extract_temporal_references(self, query: str) -> List[str]:
        """시간 참조 추출"""
        
        temporal_patterns = [
            'today', 'yesterday', 'tomorrow', 'now',
            'recent', 'latest', 'current', 'past',
            'future', 'next', 'previous', 'last',
            'this year', 'last year', 'next year'
        ]
        
        temporal_refs = []
        query_lower = query.lower()
        
        for pattern in temporal_patterns:
            if pattern in query_lower:
                temporal_refs.append(pattern)
        
        return temporal_refs
    
    def _classify_domain(self, query: str, keywords: List[str]) -> str:
        """도메인 분류"""
        
        domain_scores = defaultdict(float)
        
        for domain, domain_words in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in domain_words:
                    domain_scores[domain] += 1
            
            # 쿼리 전체에서 도메인 키워드 검색
            for domain_word in domain_words:
                if domain_word in query:
                    domain_scores[domain] += 0.5
        
        if not domain_scores:
            return 'general'
        
        return max(domain_scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_confidence(
        self, 
        query: str, 
        complexity: QueryComplexity,
        intent: str,
        entities: List[str]
    ) -> float:
        """분석 신뢰도 계산"""
        
        confidence_factors = []
        
        # 쿼리 길이 기반
        if 5 <= len(query.split()) <= 20:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        # 엔티티 존재
        if entities:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)
        
        # 의도 명확성
        if intent != 'general':
            confidence_factors.append(0.85)
        else:
            confidence_factors.append(0.6)
        
        # 복잡도 일치성
        query_length = len(query.split())
        if complexity == QueryComplexity.SIMPLE and query_length < 8:
            confidence_factors.append(0.9)
        elif complexity == QueryComplexity.COMPLEX and query_length > 15:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.75)
        
        return np.mean(confidence_factors)


class AdaptiveRAGSystem:
    """사용자 피드백 기반 적응형 RAG"""
    
    def __init__(
        self, 
        vectorstore: FAISS,
        embeddings: OpenAIEmbeddings,
        documents: List[Document]
    ):
        # 기본 검색기들
        self.semantic_retriever = SemanticRetriever(vectorstore, embeddings)
        self.keyword_retriever = KeywordRetriever(documents)
        self.hybrid_retriever = HybridRetriever(
            self.semantic_retriever, 
            self.keyword_retriever
        )
        self.rerank_retriever = RerankRetriever(self.hybrid_retriever)
        
        # 전략별 검색기 매핑
        self.retrieval_strategies = {
            RetrievalStrategy.SEMANTIC: self.semantic_retriever,
            RetrievalStrategy.KEYWORD: self.keyword_retriever,
            RetrievalStrategy.HYBRID: self.hybrid_retriever,
            RetrievalStrategy.RERANK: self.rerank_retriever
        }
        
        # 전략 가중치 (초기값)
        self.strategy_weights = {
            RetrievalStrategy.SEMANTIC: 0.25,
            RetrievalStrategy.KEYWORD: 0.20,
            RetrievalStrategy.HYBRID: 0.35,
            RetrievalStrategy.RERANK: 0.20
        }
        
        # 피드백 히스토리
        self.feedback_history: deque = deque(maxlen=1000)
        
        # 쿼리 분석기
        self.query_analyzer = QueryAnalyzer()
        
        # 성능 추적
        self.performance_tracker = {
            'total_queries': 0,
            'avg_satisfaction': 0.0,
            'strategy_usage': defaultdict(int),
            'query_complexity_distribution': defaultdict(int)
        }
    
    async def adaptive_retrieve(
        self, 
        query: str, 
        user_context: Dict[str, Any] = None,
        config: RetrievalConfig = None
    ) -> RetrievalResult:
        """적응형 검색"""
        
        start_time = time.perf_counter()
        
        # 기본 설정
        if config is None:
            config = RetrievalConfig()
        
        # 쿼리 분석
        query_analysis = await self.query_analyzer.analyze_query(query, user_context)
        
        # 사용자 컨텍스트 기반 전략 선택
        best_strategy = await self._select_optimal_strategy(query_analysis, user_context)
        
        # 쿼리 복잡도에 따른 설정 조정
        adapted_config = await self._adapt_config_to_complexity(config, query_analysis)
        
        # 다중 전략 앙상블 (상위 전략들 사용)
        results = await self._ensemble_retrieve(query, query_analysis, adapted_config)
        
        # 다양성 확보
        final_docs = await self._ensure_diversity(results, adapted_config)
        
        # 동적 청킹 적용 (필요시)
        optimized_docs = await self._dynamic_chunking(final_docs, query, query_analysis)
        
        retrieval_time = time.perf_counter() - start_time
        
        # 성능 추적 업데이트
        self.performance_tracker['total_queries'] += 1
        self.performance_tracker['strategy_usage'][best_strategy] += 1
        self.performance_tracker['query_complexity_distribution'][query_analysis.complexity] += 1
        
        return RetrievalResult(
            documents=optimized_docs,
            scores=[],  # 앙상블에서는 개별 점수 추적 복잡
            strategy_used=best_strategy,
            query_analysis=query_analysis,
            retrieval_time=retrieval_time,
            total_candidates=len(results),
            reranked=best_strategy == RetrievalStrategy.RERANK
        )
    
    async def _select_optimal_strategy(
        self, 
        query_analysis: QueryAnalysis,
        user_context: Dict[str, Any] = None
    ) -> RetrievalStrategy:
        """최적 전략 선택"""
        
        strategy_scores = {}
        
        for strategy, weight in self.strategy_weights.items():
            score = weight
            
            # 쿼리 복잡도 기반 조정
            if query_analysis.complexity == QueryComplexity.SIMPLE:
                if strategy == RetrievalStrategy.KEYWORD:
                    score *= 1.3
                elif strategy == RetrievalStrategy.RERANK:
                    score *= 0.7
            elif query_analysis.complexity == QueryComplexity.COMPLEX:
                if strategy == RetrievalStrategy.RERANK:
                    score *= 1.4
                elif strategy == RetrievalStrategy.SEMANTIC:
                    score *= 1.2
            
            # 의도 기반 조정
            if query_analysis.intent == 'definition':
                if strategy == RetrievalStrategy.KEYWORD:
                    score *= 1.2
            elif query_analysis.intent == 'analysis':
                if strategy == RetrievalStrategy.HYBRID:
                    score *= 1.3
            
            # 사용자 컨텍스트 기반 조정
            if user_context:
                preferred_strategy = user_context.get('preferred_strategy')
                if preferred_strategy and strategy.value == preferred_strategy:
                    score *= 1.5
            
            strategy_scores[strategy] = score
        
        # 가장 높은 점수의 전략 선택
        return max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    async def _adapt_config_to_complexity(
        self, 
        base_config: RetrievalConfig, 
        query_analysis: QueryAnalysis
    ) -> RetrievalConfig:
        """복잡도에 따른 설정 적응"""
        
        adapted_config = RetrievalConfig(**base_config.__dict__)
        
        if query_analysis.complexity == QueryComplexity.SIMPLE:
            adapted_config.chunk_size = 500
            adapted_config.max_documents = 5
            adapted_config.similarity_threshold = 0.8
        elif query_analysis.complexity == QueryComplexity.COMPLEX:
            adapted_config.chunk_size = 1500
            adapted_config.max_documents = 15
            adapted_config.similarity_threshold = 0.6
        elif query_analysis.complexity == QueryComplexity.MULTI_HOP:
            adapted_config.chunk_size = 2000
            adapted_config.max_documents = 20
            adapted_config.similarity_threshold = 0.5
            adapted_config.context_window = 5  # 더 많은 컨텍스트
        
        return adapted_config
    
    async def _ensemble_retrieve(
        self,
        query: str,
        query_analysis: QueryAnalysis, 
        config: RetrievalConfig
    ) -> List[Document]:
        """앙상블 검색 (상위 전략들 조합)"""
        
        # 가중치 상위 전략들 선택 (상위 2-3개)
        sorted_strategies = sorted(
            self.strategy_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_strategies = sorted_strategies[:3]
        
        # 병렬 검색 실행
        retrieval_tasks = []
        for strategy, weight in top_strategies:
            if weight > 0.1:  # 임계값 이상만 사용
                retriever = self.retrieval_strategies[strategy]
                task = retriever.retrieve(query, config)
                retrieval_tasks.append((strategy, weight, task))
        
        # 결과 수집
        results = []
        for strategy, weight, task in retrieval_tasks:
            try:
                docs = await task
                for doc in docs:
                    # 전략별 가중치 메타데이터 추가
                    doc.metadata['strategy'] = strategy.value
                    doc.metadata['weight'] = weight
                results.extend(docs)
            except Exception as e:
                print(f"Error in {strategy.value} retrieval: {e}")
        
        return results
    
    async def _ensure_diversity(
        self, 
        documents: List[Document], 
        config: RetrievalConfig
    ) -> List[Document]:
        """결과 다양성 확보"""
        
        if not documents:
            return []
        
        # 간단한 다양성 알고리즘 (MMR과 유사)
        selected_docs = [documents[0]]  # 첫 번째 문서 선택
        
        for doc in documents[1:]:
            if len(selected_docs) >= config.max_documents:
                break
            
            # 기존 선택된 문서들과의 유사도 계산
            max_similarity = 0.0
            doc_content = doc.page_content.lower()
            
            for selected_doc in selected_docs:
                selected_content = selected_doc.page_content.lower()
                
                # 간단한 Jaccard 유사도
                doc_tokens = set(doc_content.split())
                selected_tokens = set(selected_content.split())
                
                if doc_tokens | selected_tokens:
                    similarity = len(doc_tokens & selected_tokens) / len(doc_tokens | selected_tokens)
                    max_similarity = max(max_similarity, similarity)
            
            # 다양성 임계값 확인
            if max_similarity < config.diversity_threshold:
                selected_docs.append(doc)
        
        return selected_docs
    
    async def _dynamic_chunking(
        self, 
        docs: List[Document], 
        query: str,
        query_analysis: QueryAnalysis
    ) -> List[Document]:
        """쿼리 기반 동적 청킹"""
        
        # 쿼리 복잡도에 따른 청크 크기 결정
        if query_analysis.complexity == QueryComplexity.SIMPLE:
            chunk_size = 800
            overlap = 100
        elif query_analysis.complexity == QueryComplexity.MODERATE:
            chunk_size = 1200
            overlap = 200
        elif query_analysis.complexity == QueryComplexity.COMPLEX:
            chunk_size = 1800
            overlap = 300
        else:  # MULTI_HOP
            chunk_size = 2500
            overlap = 400
        
        # 문서별 재청킹
        rechunked_docs = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        for doc in docs:
            if len(doc.page_content) > chunk_size * 1.5:  # 재청킹이 필요한 경우만
                chunks = text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(chunks):
                    new_doc = Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_index': i,
                            'parent_doc_id': doc.metadata.get('doc_id', 'unknown'),
                            'rechunked': True
                        }
                    )
                    rechunked_docs.append(new_doc)
            else:
                rechunked_docs.append(doc)
        
        return rechunked_docs
    
    def update_strategy_weights(self, feedback: UserFeedback):
        """피드백 기반 전략 가중치 업데이트"""
        
        self.feedback_history.append(feedback)
        
        strategy_used = feedback.strategy_used
        satisfaction = feedback.satisfaction
        
        # 지수 이동 평균으로 가중치 업데이트
        alpha = 0.1  # 학습률
        
        for strategy in self.strategy_weights:
            if strategy == strategy_used:
                # 사용된 전략의 성과에 따라 조정
                delta = (satisfaction - 0.5) * alpha
            else:
                # 다른 전략들은 약간 감소
                delta = -0.005 * alpha
            
            self.strategy_weights[strategy] += delta
            
            # 가중치 범위 제한
            self.strategy_weights[strategy] = max(0.05, min(0.8, self.strategy_weights[strategy]))
        
        # 정규화
        total_weight = sum(self.strategy_weights.values())
        for strategy in self.strategy_weights:
            self.strategy_weights[strategy] /= total_weight
        
        # 성능 통계 업데이트
        if self.feedback_history:
            recent_satisfactions = [f.satisfaction for f in self.feedback_history]
            self.performance_tracker['avg_satisfaction'] = np.mean(recent_satisfactions)
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """성능 분석 정보"""
        
        analytics = {
            'system_performance': self.performance_tracker.copy(),
            'strategy_weights': self.strategy_weights.copy(),
            'feedback_summary': self._analyze_feedback_trends(),
            'strategy_performance': {}
        }
        
        # 전략별 성능 메트릭
        for strategy, retriever in self.retrieval_strategies.items():
            analytics['strategy_performance'][strategy.value] = retriever.get_performance_metrics()
        
        return analytics
    
    def _analyze_feedback_trends(self) -> Dict[str, Any]:
        """피드백 트렌드 분석"""
        
        if not self.feedback_history:
            return {"message": "No feedback available"}
        
        # 최근 피드백들
        recent_feedback = list(self.feedback_history)[-50:]
        
        # 전략별 만족도
        strategy_satisfaction = defaultdict(list)
        for feedback in recent_feedback:
            strategy_satisfaction[feedback.strategy_used].append(feedback.satisfaction)
        
        strategy_avg_satisfaction = {}
        for strategy, satisfactions in strategy_satisfaction.items():
            strategy_avg_satisfaction[strategy.value] = np.mean(satisfactions)
        
        # 시간대별 트렌드
        current_time = time.time()
        time_buckets = defaultdict(list)
        
        for feedback in recent_feedback:
            time_diff = current_time - feedback.timestamp
            if time_diff < 3600:  # 1시간 이내
                bucket = "recent"
            elif time_diff < 86400:  # 24시간 이내
                bucket = "today"
            else:
                bucket = "older"
            
            time_buckets[bucket].append(feedback.satisfaction)
        
        time_trends = {}
        for bucket, satisfactions in time_buckets.items():
            time_trends[bucket] = np.mean(satisfactions)
        
        return {
            'total_feedback': len(self.feedback_history),
            'recent_feedback': len(recent_feedback),
            'overall_satisfaction': np.mean([f.satisfaction for f in recent_feedback]),
            'strategy_satisfaction': strategy_avg_satisfaction,
            'time_trends': time_trends
        }