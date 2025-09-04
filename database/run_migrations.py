#!/usr/bin/env python3
"""
데이터베이스 마이그레이션 실행 스크립트
Project Maestro 대화 메모리 시스템 마이그레이션 관리
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import argparse
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.project_maestro.core.config import settings


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("migration")


class MigrationManager:
    """데이터베이스 마이그레이션 관리자"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.migrations_dir = Path(__file__).parent / "migrations"
        self.connection = None
        
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.connection = psycopg2.connect(self.database_url)
            self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            logger.info("데이터베이스 연결 성공")
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            raise
            
    def disconnect(self):
        """데이터베이스 연결 해제"""
        if self.connection:
            self.connection.close()
            logger.info("데이터베이스 연결 해제")
            
    def create_migration_table(self):
        """마이그레이션 이력 테이블 생성"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS migration_history (
                    id SERIAL PRIMARY KEY,
                    migration_name VARCHAR(255) NOT NULL UNIQUE,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    checksum VARCHAR(64)
                );
                
                CREATE INDEX IF NOT EXISTS idx_migration_history_name 
                ON migration_history(migration_name);
            """)
            cursor.close()
            logger.info("마이그레이션 이력 테이블 준비 완료")
        except Exception as e:
            logger.error(f"마이그레이션 테이블 생성 실패: {e}")
            raise
            
    def get_applied_migrations(self) -> List[str]:
        """적용된 마이그레이션 목록 조회"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT migration_name 
                FROM migration_history 
                ORDER BY applied_at
            """)
            
            applied_migrations = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
            logger.info(f"적용된 마이그레이션: {len(applied_migrations)}개")
            return applied_migrations
            
        except Exception as e:
            logger.error(f"적용된 마이그레이션 조회 실패: {e}")
            return []
            
    def get_pending_migrations(self) -> List[Dict[str, str]]:
        """대기 중인 마이그레이션 목록 조회"""
        applied_migrations = self.get_applied_migrations()
        
        # migrations 디렉토리의 .sql 파일들 조회
        migration_files = []
        for sql_file in sorted(self.migrations_dir.glob("*.sql")):
            migration_name = sql_file.stem
            if migration_name not in applied_migrations:
                migration_files.append({
                    "name": migration_name,
                    "path": str(sql_file),
                    "content": sql_file.read_text(encoding='utf-8')
                })
                
        logger.info(f"대기 중인 마이그레이션: {len(migration_files)}개")
        return migration_files
        
    def calculate_checksum(self, content: str) -> str:
        """마이그레이션 파일의 체크섬 계산"""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()
        
    def execute_migration(self, migration: Dict[str, str]) -> bool:
        """단일 마이그레이션 실행"""
        migration_name = migration["name"]
        migration_content = migration["content"]
        checksum = self.calculate_checksum(migration_content)
        
        logger.info(f"마이그레이션 실행 중: {migration_name}")
        
        try:
            cursor = self.connection.cursor()
            
            # 마이그레이션 SQL 실행
            cursor.execute(migration_content)
            
            # 마이그레이션 이력에 기록 (중복 방지)
            cursor.execute("""
                INSERT INTO migration_history (migration_name, description, checksum)
                VALUES (%s, %s, %s)
                ON CONFLICT (migration_name) DO NOTHING
            """, (
                migration_name,
                f"Applied migration {migration_name}",
                checksum
            ))
            
            cursor.close()
            
            logger.info(f"마이그레이션 완료: {migration_name}")
            return True
            
        except Exception as e:
            logger.error(f"마이그레이션 실행 실패 [{migration_name}]: {e}")
            return False
            
    def run_migrations(self, dry_run: bool = False) -> bool:
        """모든 대기 중인 마이그레이션 실행"""
        if not self.migrations_dir.exists():
            logger.error(f"마이그레이션 디렉토리가 존재하지 않습니다: {self.migrations_dir}")
            return False
            
        try:
            self.connect()
            self.create_migration_table()
            
            pending_migrations = self.get_pending_migrations()
            
            if not pending_migrations:
                logger.info("실행할 마이그레이션이 없습니다.")
                return True
                
            if dry_run:
                logger.info("=== DRY RUN 모드 ===")
                for migration in pending_migrations:
                    logger.info(f"실행 예정: {migration['name']}")
                return True
                
            # 마이그레이션 실행
            success_count = 0
            for migration in pending_migrations:
                if self.execute_migration(migration):
                    success_count += 1
                else:
                    logger.error(f"마이그레이션 실패로 인해 중단: {migration['name']}")
                    break
                    
            logger.info(f"마이그레이션 완료: {success_count}/{len(pending_migrations)}")
            return success_count == len(pending_migrations)
            
        except Exception as e:
            logger.error(f"마이그레이션 실행 중 오류: {e}")
            return False
        finally:
            self.disconnect()
            
    def rollback_migration(self, migration_name: str) -> bool:
        """특정 마이그레이션 롤백 (기본적으로는 지원하지 않음)"""
        logger.warning("마이그레이션 롤백은 수동으로 처리해야 합니다.")
        logger.warning(f"롤백 대상: {migration_name}")
        
        # 실제 롤백은 별도의 rollback SQL 파일을 만들어서 처리
        rollback_file = self.migrations_dir / f"rollback_{migration_name}.sql"
        if rollback_file.exists():
            logger.info(f"롤백 스크립트 발견: {rollback_file}")
            # 롤백 실행 로직을 여기에 추가
            return True
        else:
            logger.error(f"롤백 스크립트가 없습니다: {rollback_file}")
            return False
            
    def check_pgvector_extension(self) -> bool:
        """pgvector 확장 설치 상태 확인"""
        try:
            self.connect()
            cursor = self.connection.cursor()
            
            # 확장 설치 상태 확인
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension 
                    WHERE extname = 'vector'
                );
            """)
            
            is_installed = cursor.fetchone()[0]
            cursor.close()
            
            if is_installed:
                logger.info("pgvector 확장이 설치되어 있습니다.")
                
                # 벡터 함수 테스트
                cursor = self.connection.cursor()
                cursor.execute("SELECT '[1,2,3]'::vector(3) <=> '[1,2,4]'::vector(3);")
                distance = cursor.fetchone()[0]
                cursor.close()
                
                logger.info(f"pgvector 테스트 성공 (거리: {distance})")
                return True
            else:
                logger.warning("pgvector 확장이 설치되지 않았습니다.")
                logger.warning("다음 명령으로 설치하세요:")
                logger.warning("1. PostgreSQL에서 CREATE EXTENSION vector;")
                logger.warning("2. 또는 pgvector를 시스템에 먼저 설치")
                return False
                
        except Exception as e:
            logger.error(f"pgvector 확인 실패: {e}")
            return False
        finally:
            self.disconnect()
            
    def generate_migration_template(self, name: str):
        """새 마이그레이션 템플릿 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{name}.sql"
        filepath = self.migrations_dir / filename
        
        template = f"""-- 마이그레이션: {name}
-- 생성일: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
-- 설명: {name} 관련 데이터베이스 변경사항

-- ========================================
-- 변경사항 시작
-- ========================================

-- 여기에 SQL 명령어들을 작성하세요

-- ========================================
-- 마이그레이션 기록
-- ========================================

INSERT INTO migration_history (migration_name, description) 
VALUES (
    '{timestamp}_{name}',
    '{name} 관련 마이그레이션'
) ON CONFLICT (migration_name) DO NOTHING;

-- 완료 메시지
DO $$
BEGIN
    RAISE NOTICE '=== 마이그레이션 {timestamp}_{name} 완료 ===';
    RAISE NOTICE '설명: {name}';
    RAISE NOTICE '================================================';
END $$;
"""
        
        self.migrations_dir.mkdir(exist_ok=True)
        filepath.write_text(template, encoding='utf-8')
        
        logger.info(f"마이그레이션 템플릿 생성: {filepath}")
        return str(filepath)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Project Maestro 데이터베이스 마이그레이션")
    
    parser.add_argument(
        "command",
        choices=["migrate", "status", "check", "generate"],
        help="실행할 명령어"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 실행 없이 계획만 표시"
    )
    
    parser.add_argument(
        "--name",
        help="생성할 마이그레이션 이름 (generate 명령어용)"
    )
    
    parser.add_argument(
        "--database-url",
        default=settings.database_url,
        help="데이터베이스 연결 URL"
    )
    
    args = parser.parse_args()
    
    manager = MigrationManager(args.database_url)
    
    try:
        if args.command == "migrate":
            logger.info("=== 마이그레이션 실행 ===")
            success = manager.run_migrations(dry_run=args.dry_run)
            sys.exit(0 if success else 1)
            
        elif args.command == "status":
            logger.info("=== 마이그레이션 상태 확인 ===")
            manager.connect()
            manager.create_migration_table()
            
            applied = manager.get_applied_migrations()
            pending = manager.get_pending_migrations()
            
            print(f"\n적용된 마이그레이션: {len(applied)}개")
            for migration in applied:
                print(f"  ✅ {migration}")
                
            print(f"\n대기 중인 마이그레이션: {len(pending)}개")
            for migration in pending:
                print(f"  ⏳ {migration['name']}")
                
            manager.disconnect()
            
        elif args.command == "check":
            logger.info("=== 시스템 상태 확인 ===")
            
            # 데이터베이스 연결 테스트
            try:
                manager.connect()
                logger.info("✅ 데이터베이스 연결 성공")
                manager.disconnect()
            except Exception as e:
                logger.error(f"❌ 데이터베이스 연결 실패: {e}")
                sys.exit(1)
                
            # pgvector 확장 확인
            if manager.check_pgvector_extension():
                logger.info("✅ pgvector 확장 사용 가능")
            else:
                logger.warning("⚠️ pgvector 확장 설치 필요")
                
        elif args.command == "generate":
            if not args.name:
                logger.error("마이그레이션 이름을 지정해주세요: --name <이름>")
                sys.exit(1)
                
            logger.info("=== 마이그레이션 템플릿 생성 ===")
            filepath = manager.generate_migration_template(args.name)
            print(f"템플릿이 생성되었습니다: {filepath}")
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"실행 중 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()