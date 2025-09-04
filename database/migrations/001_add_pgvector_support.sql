-- 대화 메모리 시스템을 위한 pgvector 확장 및 임베딩 컬럼 추가
-- Migration: 001_add_pgvector_support.sql
-- Created: 2024-01-01
-- Description: pgvector 확장 설치 및 대화 메모리 테이블에 벡터 임베딩 컬럼 추가

-- ========================================
-- pgvector 확장 설치
-- ========================================

-- pgvector 확장 설치 (관리자 권한 필요)
CREATE EXTENSION IF NOT EXISTS vector;

-- ========================================
-- 기존 테이블에 임베딩 컬럼 추가
-- ========================================

-- conversation_messages 테이블에 임베딩 벡터 컬럼 추가
-- 384차원: sentence-transformers/all-MiniLM-L6-v2 모델용
ALTER TABLE conversation_messages 
ADD COLUMN IF NOT EXISTS embedding_vector vector(384);

-- conversation_summaries 테이블에 임베딩 벡터 컬럼 추가
ALTER TABLE conversation_summaries 
ADD COLUMN IF NOT EXISTS embedding_vector vector(384);

-- ========================================
-- 벡터 유사도 검색을 위한 인덱스 생성
-- ========================================

-- 메시지 임베딩 벡터용 IVFFLAT 인덱스
-- 코사인 거리 기반 유사도 검색용
CREATE INDEX IF NOT EXISTS idx_conversation_messages_embedding_cosine 
ON conversation_messages 
USING ivfflat (embedding_vector vector_cosine_ops) 
WITH (lists = 100);

-- 내적 기반 유사도 검색용 인덱스
CREATE INDEX IF NOT EXISTS idx_conversation_messages_embedding_ip 
ON conversation_messages 
USING ivfflat (embedding_vector vector_ip_ops) 
WITH (lists = 100);

-- L2 거리 기반 검색용 인덱스  
CREATE INDEX IF NOT EXISTS idx_conversation_messages_embedding_l2 
ON conversation_messages 
USING ivfflat (embedding_vector vector_l2_ops) 
WITH (lists = 100);

-- 요약 임베딩 벡터용 인덱스
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_embedding_cosine 
ON conversation_summaries 
USING ivfflat (embedding_vector vector_cosine_ops) 
WITH (lists = 50);

-- ========================================
-- 벡터 검색을 위한 헬퍼 함수 생성
-- ========================================

-- 코사인 유사도 계산 함수 (1에 가까울수록 유사)
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector) 
RETURNS float AS $$
BEGIN
    RETURN 1 - (a <=> b);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- 메시지 유사도 검색 함수
CREATE OR REPLACE FUNCTION find_similar_messages(
    query_vector vector(384),
    user_id_param text,
    conversation_id_param uuid DEFAULT NULL,
    similarity_threshold float DEFAULT 0.7,
    result_limit int DEFAULT 10
) 
RETURNS TABLE (
    message_id uuid,
    conversation_id uuid,
    content text,
    similarity_score float,
    created_at timestamp,
    message_type text
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cm.id as message_id,
        cm.conversation_id,
        cm.content,
        cosine_similarity(cm.embedding_vector, query_vector) as similarity_score,
        cm.created_at,
        cm.message_type
    FROM conversation_messages cm
    JOIN conversations c ON c.id = cm.conversation_id
    WHERE 
        c.user_id = user_id_param
        AND c.is_active = true
        AND cm.embedding_vector IS NOT NULL
        AND (conversation_id_param IS NULL OR cm.conversation_id = conversation_id_param)
        AND cosine_similarity(cm.embedding_vector, query_vector) >= similarity_threshold
    ORDER BY cosine_similarity(cm.embedding_vector, query_vector) DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- 관련 대화 검색 함수  
CREATE OR REPLACE FUNCTION find_related_conversations(
    query_vector vector(384),
    user_id_param text,
    project_id_param uuid DEFAULT NULL,
    similarity_threshold float DEFAULT 0.6,
    result_limit int DEFAULT 5
)
RETURNS TABLE (
    conversation_id uuid,
    title text,
    similarity_score float,
    created_at timestamp,
    message_count bigint
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id as conversation_id,
        c.title,
        AVG(cosine_similarity(cm.embedding_vector, query_vector)) as similarity_score,
        c.created_at,
        COUNT(cm.id) as message_count
    FROM conversations c
    JOIN conversation_messages cm ON cm.conversation_id = c.id
    WHERE 
        c.user_id = user_id_param
        AND c.is_active = true
        AND cm.embedding_vector IS NOT NULL
        AND (project_id_param IS NULL OR c.project_id = project_id_param)
    GROUP BY c.id, c.title, c.created_at
    HAVING AVG(cosine_similarity(cm.embedding_vector, query_vector)) >= similarity_threshold
    ORDER BY AVG(cosine_similarity(cm.embedding_vector, query_vector)) DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- 임베딩 품질 관리를 위한 함수들
-- ========================================

-- 임베딩이 없는 메시지 조회 함수
CREATE OR REPLACE FUNCTION get_messages_without_embeddings(batch_size int DEFAULT 100)
RETURNS TABLE (
    message_id uuid,
    conversation_id uuid,
    content text,
    created_at timestamp
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cm.id as message_id,
        cm.conversation_id,
        cm.content,
        cm.created_at
    FROM conversation_messages cm
    JOIN conversations c ON c.id = cm.conversation_id
    WHERE 
        c.is_active = true
        AND cm.embedding_vector IS NULL
        AND LENGTH(cm.content) > 10  -- 최소 길이 필터
    ORDER BY cm.created_at ASC
    LIMIT batch_size;
END;
$$ LANGUAGE plpgsql;

-- 임베딩 벡터 업데이트 함수
CREATE OR REPLACE FUNCTION update_message_embedding(
    message_id_param uuid,
    embedding_param vector(384)
)
RETURNS boolean AS $$
DECLARE
    updated_count int;
BEGIN
    UPDATE conversation_messages 
    SET embedding_vector = embedding_param
    WHERE id = message_id_param;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count > 0;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- 통계 및 모니터링 함수들
-- ========================================

-- 임베딩 통계 조회 함수
CREATE OR REPLACE FUNCTION get_embedding_statistics()
RETURNS TABLE (
    total_messages bigint,
    messages_with_embeddings bigint,
    embedding_coverage_percent numeric,
    total_conversations bigint,
    conversations_with_embeddings bigint
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_messages,
        COUNT(cm.embedding_vector) as messages_with_embeddings,
        ROUND((COUNT(cm.embedding_vector)::numeric / COUNT(*)::numeric * 100), 2) as embedding_coverage_percent,
        COUNT(DISTINCT cm.conversation_id) as total_conversations,
        COUNT(DISTINCT CASE WHEN cm.embedding_vector IS NOT NULL THEN cm.conversation_id END) as conversations_with_embeddings
    FROM conversation_messages cm
    JOIN conversations c ON c.id = cm.conversation_id
    WHERE c.is_active = true;
END;
$$ LANGUAGE plpgsql;

-- 사용자별 임베딩 통계
CREATE OR REPLACE FUNCTION get_user_embedding_stats(user_id_param text)
RETURNS TABLE (
    user_id text,
    total_messages bigint,
    messages_with_embeddings bigint,
    embedding_coverage_percent numeric,
    last_embedding_update timestamp
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        user_id_param as user_id,
        COUNT(cm.*) as total_messages,
        COUNT(cm.embedding_vector) as messages_with_embeddings,
        ROUND((COUNT(cm.embedding_vector)::numeric / COUNT(cm.*)::numeric * 100), 2) as embedding_coverage_percent,
        MAX(cm.created_at) FILTER (WHERE cm.embedding_vector IS NOT NULL) as last_embedding_update
    FROM conversation_messages cm
    JOIN conversations c ON c.id = cm.conversation_id
    WHERE c.user_id = user_id_param AND c.is_active = true;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- 벤치마크 및 성능 테스트 함수
-- ========================================

-- 벡터 검색 성능 테스트 함수
CREATE OR REPLACE FUNCTION benchmark_vector_search(
    test_vector vector(384),
    test_iterations int DEFAULT 10
)
RETURNS TABLE (
    iteration int,
    execution_time_ms numeric,
    result_count bigint
) AS $$
DECLARE
    i int;
    start_time timestamp;
    end_time timestamp;
    exec_time numeric;
    res_count bigint;
BEGIN
    FOR i IN 1..test_iterations LOOP
        start_time := clock_timestamp();
        
        SELECT COUNT(*) INTO res_count
        FROM conversation_messages cm
        WHERE cm.embedding_vector IS NOT NULL
          AND cosine_similarity(cm.embedding_vector, test_vector) >= 0.5
        LIMIT 20;
        
        end_time := clock_timestamp();
        exec_time := EXTRACT(MILLISECONDS FROM (end_time - start_time));
        
        RETURN QUERY SELECT i, exec_time, res_count;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- 권한 및 보안 설정
-- ========================================

-- 임베딩 관련 함수들의 실행 권한을 애플리케이션 사용자에게 부여
-- (실제 배포시 적절한 사용자명으로 변경 필요)
GRANT EXECUTE ON FUNCTION cosine_similarity(vector, vector) TO maestro_app;
GRANT EXECUTE ON FUNCTION find_similar_messages(vector(384), text, uuid, float, int) TO maestro_app;
GRANT EXECUTE ON FUNCTION find_related_conversations(vector(384), text, uuid, float, int) TO maestro_app;
GRANT EXECUTE ON FUNCTION get_messages_without_embeddings(int) TO maestro_app;
GRANT EXECUTE ON FUNCTION update_message_embedding(uuid, vector(384)) TO maestro_app;
GRANT EXECUTE ON FUNCTION get_embedding_statistics() TO maestro_app;
GRANT EXECUTE ON FUNCTION get_user_embedding_stats(text) TO maestro_app;
GRANT EXECUTE ON FUNCTION benchmark_vector_search(vector(384), int) TO maestro_app;

-- ========================================
-- 마이그레이션 완료 로그
-- ========================================

-- 마이그레이션 이력 테이블 생성 (없는 경우)
CREATE TABLE IF NOT EXISTS migration_history (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(255) NOT NULL UNIQUE,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- 마이그레이션 완료 기록
INSERT INTO migration_history (migration_name, description) 
VALUES (
    '001_add_pgvector_support',
    'pgvector 확장 설치 및 대화 메모리 시스템 임베딩 컬럼 추가'
) ON CONFLICT (migration_name) DO NOTHING;

-- 완료 메시지
DO $$
BEGIN
    RAISE NOTICE '=== 마이그레이션 001_add_pgvector_support 완료 ===';
    RAISE NOTICE '- pgvector 확장 설치됨';
    RAISE NOTICE '- conversation_messages.embedding_vector 컬럼 추가됨 (384차원)';
    RAISE NOTICE '- conversation_summaries.embedding_vector 컬럼 추가됨 (384차원)';
    RAISE NOTICE '- 벡터 유사도 검색용 인덱스 생성됨';
    RAISE NOTICE '- 유사도 검색 헬퍼 함수들 생성됨';
    RAISE NOTICE '- 임베딩 관리 및 통계 함수들 생성됨';
    RAISE NOTICE '================================================';
END $$;