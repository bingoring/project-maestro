"""스트리밍 응답 최적화 핸들러"""

import asyncio
import time
from typing import AsyncIterator, List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
from collections import deque
import threading
import queue


class BufferStrategy(Enum):
    """버퍼링 전략"""
    SIZE_BASED = "size_based"          # 크기 기반 버퍼링
    TIME_BASED = "time_based"          # 시간 기반 버퍼링
    ADAPTIVE = "adaptive"              # 적응형 버퍼링
    PRIORITY_BASED = "priority_based"  # 우선순위 기반


@dataclass
class StreamChunk:
    """스트림 청크 데이터"""
    content: str
    agent_name: str
    timestamp: float
    chunk_type: str = "content"  # content, metadata, error, complete
    priority: int = 5  # 1(highest) - 10(lowest)
    metadata: Dict[str, Any] = None


@dataclass
class StreamConfig:
    """스트리밍 설정"""
    buffer_size: int = 10
    buffer_timeout: float = 0.1  # 100ms
    max_chunk_size: int = 1024
    enable_compression: bool = True
    priority_threshold: int = 3
    adaptive_buffer: bool = True


class AdaptiveBuffer:
    """적응형 버퍼"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.buffer: deque = deque()
        self.last_flush_time = time.time()
        self.performance_stats = {
            'avg_latency': 0.1,
            'throughput': 0,
            'buffer_efficiency': 1.0
        }
        self._lock = threading.Lock()
    
    def should_flush(self) -> bool:
        """버퍼 플러시 여부 결정"""
        with self._lock:
            current_time = time.time()
            time_since_flush = current_time - self.last_flush_time
            
            # 크기 기반 체크
            if len(self.buffer) >= self.config.buffer_size:
                return True
            
            # 시간 기반 체크
            if time_since_flush >= self.config.buffer_timeout:
                return True
            
            # 우선순위 기반 체크
            if self.buffer and min(chunk.priority for chunk in self.buffer) <= self.config.priority_threshold:
                return True
            
            # 적응형 체크
            if self.config.adaptive_buffer:
                optimal_buffer_size = self._calculate_optimal_buffer_size()
                if len(self.buffer) >= optimal_buffer_size:
                    return True
            
            return False
    
    def add_chunk(self, chunk: StreamChunk):
        """청크 추가"""
        with self._lock:
            self.buffer.append(chunk)
    
    def flush(self) -> List[StreamChunk]:
        """버퍼 플러시"""
        with self._lock:
            chunks = list(self.buffer)
            self.buffer.clear()
            self.last_flush_time = time.time()
            return chunks
    
    def _calculate_optimal_buffer_size(self) -> int:
        """적응형 최적 버퍼 크기 계산"""
        # 성능 통계 기반 계산
        latency_factor = min(self.performance_stats['avg_latency'] * 10, 2.0)
        throughput_factor = max(self.performance_stats['throughput'] / 100, 0.5)
        
        optimal_size = int(self.config.buffer_size * latency_factor * throughput_factor)
        return max(1, min(optimal_size, self.config.buffer_size * 2))
    
    def update_stats(self, latency: float, throughput: float):
        """성능 통계 업데이트"""
        alpha = 0.1  # 지수 이동 평균
        self.performance_stats['avg_latency'] = (
            (1 - alpha) * self.performance_stats['avg_latency'] + 
            alpha * latency
        )
        self.performance_stats['throughput'] = (
            (1 - alpha) * self.performance_stats['throughput'] + 
            alpha * throughput
        )


class StreamingResponseHandler:
    """스트리밍 응답 최적화 핸들러"""
    
    def __init__(self, config: StreamConfig = None):
        self.config = config or StreamConfig()
        self.active_streams: Dict[str, AdaptiveBuffer] = {}
        self.compression_enabled = self.config.enable_compression
        
    async def stream_with_buffering(
        self,
        stream_id: str,
        agent_response: AsyncIterator[Union[str, StreamChunk]],
        strategy: BufferStrategy = BufferStrategy.ADAPTIVE
    ) -> AsyncIterator[str]:
        """버퍼링을 통한 스트리밍 최적화"""
        
        buffer = AdaptiveBuffer(self.config)
        self.active_streams[stream_id] = buffer
        
        try:
            # 버퍼링 및 플러시 태스크
            flush_task = asyncio.create_task(
                self._buffer_flush_loop(stream_id, buffer, strategy)
            )
            
            # 스트림 소비 태스크
            consume_task = asyncio.create_task(
                self._consume_stream(agent_response, buffer)
            )
            
            # 플러시 이벤트 스트리밍
            async for flushed_content in self._stream_flushed_content(stream_id):
                yield flushed_content
                
        finally:
            # 정리
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            
            if not flush_task.done():
                flush_task.cancel()
            if not consume_task.done():
                consume_task.cancel()
    
    async def _consume_stream(
        self,
        agent_response: AsyncIterator[Union[str, StreamChunk]],
        buffer: AdaptiveBuffer
    ):
        """스트림 소비 및 버퍼링"""
        
        async for item in agent_response:
            if isinstance(item, str):
                # 문자열을 StreamChunk로 변환
                chunk = StreamChunk(
                    content=item,
                    agent_name="unknown",
                    timestamp=time.time()
                )
            else:
                chunk = item
            
            # 청크 크기 제한
            if len(chunk.content) > self.config.max_chunk_size:
                # 큰 청크를 여러 개로 분할
                for i in range(0, len(chunk.content), self.config.max_chunk_size):
                    sub_chunk = StreamChunk(
                        content=chunk.content[i:i + self.config.max_chunk_size],
                        agent_name=chunk.agent_name,
                        timestamp=chunk.timestamp,
                        chunk_type=chunk.chunk_type,
                        priority=chunk.priority
                    )
                    buffer.add_chunk(sub_chunk)
            else:
                buffer.add_chunk(chunk)
    
    async def _buffer_flush_loop(
        self,
        stream_id: str,
        buffer: AdaptiveBuffer,
        strategy: BufferStrategy
    ):
        """버퍼 플러시 루프"""
        
        while stream_id in self.active_streams:
            if buffer.should_flush():
                chunks = buffer.flush()
                
                if chunks:
                    # 전략에 따른 처리
                    processed_content = await self._process_chunks(chunks, strategy)
                    
                    # 스트림에 결과 추가
                    self.active_streams[stream_id]._flushed_queue = getattr(
                        self.active_streams[stream_id], '_flushed_queue', asyncio.Queue()
                    )
                    await self.active_streams[stream_id]._flushed_queue.put(processed_content)
            
            await asyncio.sleep(0.01)  # 10ms 체크 주기
    
    async def _stream_flushed_content(self, stream_id: str) -> AsyncIterator[str]:
        """플러시된 컨텐츠 스트리밍"""
        
        buffer = self.active_streams.get(stream_id)
        if not buffer:
            return
        
        # 플러시 큐 초기화
        if not hasattr(buffer, '_flushed_queue'):
            buffer._flushed_queue = asyncio.Queue()
        
        while stream_id in self.active_streams:
            try:
                # 큐에서 컨텐츠 가져오기 (타임아웃 설정)
                content = await asyncio.wait_for(
                    buffer._flushed_queue.get(),
                    timeout=0.1
                )
                yield content
            except asyncio.TimeoutError:
                # 타임아웃 시 계속 대기
                continue
            except Exception as e:
                print(f"Error streaming content: {e}")
                break
    
    async def _process_chunks(
        self,
        chunks: List[StreamChunk],
        strategy: BufferStrategy
    ) -> str:
        """청크 처리 전략 적용"""
        
        if strategy == BufferStrategy.PRIORITY_BASED:
            # 우선순위별 정렬
            chunks.sort(key=lambda x: x.priority)
        elif strategy == BufferStrategy.TIME_BASED:
            # 시간순 정렬
            chunks.sort(key=lambda x: x.timestamp)
        
        # 컨텐츠 결합
        combined_content = ''.join(chunk.content for chunk in chunks)
        
        # 압축 적용
        if self.compression_enabled and len(combined_content) > 100:
            combined_content = await self._compress_content(combined_content)
        
        return combined_content
    
    async def _compress_content(self, content: str) -> str:
        """컨텐츠 압축"""
        # 간단한 압축 로직 (실제로는 gzip 등 사용)
        # 여기서는 반복되는 공백 제거
        import re
        compressed = re.sub(r'\s+', ' ', content.strip())
        return compressed
    
    async def parallel_stream_merge(
        self,
        streams: List[AsyncIterator[StreamChunk]],
        merge_strategy: str = "round_robin"
    ) -> AsyncIterator[str]:
        """여러 스트림 병합 처리"""
        
        async def consume_stream(stream: AsyncIterator, stream_id: str, queue: asyncio.Queue):
            """개별 스트림 소비"""
            try:
                async for chunk in stream:
                    await queue.put((stream_id, chunk))
            except Exception as e:
                await queue.put((stream_id, StreamChunk(
                    content=f"Error in stream {stream_id}: {e}",
                    agent_name=f"stream_{stream_id}",
                    timestamp=time.time(),
                    chunk_type="error"
                )))
            finally:
                await queue.put((stream_id, None))  # 스트림 종료 신호
        
        # 공유 큐 생성
        merged_queue = asyncio.Queue()
        
        # 각 스트림을 병렬로 소비
        tasks = []
        for i, stream in enumerate(streams):
            task = asyncio.create_task(
                consume_stream(stream, str(i), merged_queue)
            )
            tasks.append(task)
        
        # 병합된 스트림 출력
        active_streams = set(str(i) for i in range(len(streams)))
        
        while active_streams:
            try:
                stream_id, chunk = await merged_queue.get()
                
                if chunk is None:
                    # 스트림 종료
                    active_streams.discard(stream_id)
                    continue
                
                # 전략에 따른 처리
                if merge_strategy == "priority":
                    # 우선순위 기반 (우선순위 높은 것부터)
                    if chunk.priority <= 3:
                        yield chunk.content
                elif merge_strategy == "round_robin":
                    # 라운드 로빈 (모든 스트림 균등하게)
                    yield chunk.content
                elif merge_strategy == "fastest_first":
                    # 가장 빠른 스트림 우선
                    yield chunk.content
                    
            except Exception as e:
                print(f"Error merging streams: {e}")
                break
        
        # 모든 태스크 정리
        for task in tasks:
            if not task.done():
                task.cancel()
    
    async def stream_with_backpressure(
        self,
        agent_response: AsyncIterator[str],
        max_queue_size: int = 100
    ) -> AsyncIterator[str]:
        """백프레셔 제어를 포함한 스트리밍"""
        
        output_queue = asyncio.Queue(maxsize=max_queue_size)
        
        async def producer():
            """프로듀서: 에이전트 응답을 큐에 추가"""
            try:
                async for chunk in agent_response:
                    await output_queue.put(chunk)
            except Exception as e:
                await output_queue.put(f"Error: {e}")
            finally:
                await output_queue.put(None)  # 종료 신호
        
        # 프로듀서 태스크 시작
        producer_task = asyncio.create_task(producer())
        
        try:
            # 컨슈머: 큐에서 데이터를 소비하여 출력
            while True:
                chunk = await output_queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            if not producer_task.done():
                producer_task.cancel()
    
    def get_stream_stats(self, stream_id: str) -> Dict[str, Any]:
        """스트림 통계 조회"""
        
        if stream_id not in self.active_streams:
            return {}
        
        buffer = self.active_streams[stream_id]
        return {
            'buffer_size': len(buffer.buffer),
            'performance_stats': buffer.performance_stats,
            'last_flush_time': buffer.last_flush_time,
            'config': {
                'buffer_size': self.config.buffer_size,
                'buffer_timeout': self.config.buffer_timeout,
                'compression_enabled': self.compression_enabled
            }
        }
    
    async def stream_with_error_recovery(
        self,
        agent_response: AsyncIterator[str],
        retry_attempts: int = 3,
        fallback_content: str = "[Connection restored]"
    ) -> AsyncIterator[str]:
        """에러 복구 기능을 포함한 스트리밍"""
        
        attempt = 0
        
        while attempt < retry_attempts:
            try:
                async for chunk in agent_response:
                    yield chunk
                break  # 성공적으로 완료
                
            except Exception as e:
                attempt += 1
                
                if attempt >= retry_attempts:
                    yield f"[Error after {retry_attempts} attempts: {e}]"
                    break
                else:
                    yield f"[Connection issue, retrying... ({attempt}/{retry_attempts})]"
                    await asyncio.sleep(1)  # 재시도 전 대기