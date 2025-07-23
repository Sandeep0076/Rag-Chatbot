#!/usr/bin/env python3
"""
Simple Database Test Script
===========================

This script provides basic database functionality to:
1. View FileInfo table data
2. Insert new FileInfo records

Database Connection Details:
- PostgreSQL database for metadata storage
- Tables: FileInfo, Conversation
- Connection: Uses SQLAlchemy ORM
"""

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import the database models
from rtl_rag_chatbot_api.common.db import FileInfo

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleDatabaseTester:
    """Simple database tester for FileInfo operations."""

    def __init__(self):
        """Initialize database connection."""
        self.db_username = os.getenv("DB_USERNAME")
        self.db_password = os.getenv("DB_PASSWORD")
        self.db_instance = os.getenv("DB_INSTANCE")

        if not self.db_instance:
            logger.warning("DB_INSTANCE not set. Database operations will fail.")
            self.engine = None
            self.SessionLocal = None
        else:
            self.database_url = f"postgresql://{self.db_username}:{self.db_password}@127.0.0.1:5432/chatbot_ui"
            self.engine = create_engine(self.database_url)
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info("Database connection initialized")

    @contextmanager
    def get_db_session(self):
        """Context manager for database sessions."""
        if not self.engine:
            raise Exception(
                "Database not initialized. Set DB_INSTANCE environment variable."
            )

        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_db_session() as session:
                session.execute(text("SELECT 1"))
                logger.info("Database connection successful")
                return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def view_file_info(self, limit: int = 10):
        """View FileInfo table data."""
        try:
            with self.get_db_session() as session:
                file_infos = session.query(FileInfo).limit(limit).all()

                print(f"\n📋 FileInfo Table Data (showing {len(file_infos)} records):")
                print("-" * 80)

                if not file_infos:
                    print("No data found in FileInfo table")
                    return

                for i, file_info in enumerate(file_infos, 1):
                    print(f"Record {i}:")
                    print(f"  ID: {file_info.id}")
                    print(f"  File ID: {file_info.file_id}")
                    print(f"  File Hash: {file_info.file_hash}")
                    print(f"  Created At: {file_info.createdAt}")
                    print("-" * 40)

                logger.info(f"Retrieved {len(file_infos)} FileInfo records")

        except Exception as e:
            logger.error(f"Error viewing FileInfo: {e}")
            print(f"❌ Error: {e}")

    def insert_file_info(self, file_id: str = None, file_hash: str = None):
        """Insert a new FileInfo record."""
        if not file_id:
            file_id = f"test_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not file_hash:
            file_hash = f"test_hash_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            with self.get_db_session() as session:
                file_info = FileInfo(
                    id=f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    file_id=file_id,
                    file_hash=file_hash,
                    createdAt=datetime.now(),
                )
                session.add(file_info)
                session.commit()

                print("\n✅ Successfully inserted FileInfo record:")
                print(f"  ID: {file_info.id}")
                print(f"  File ID: {file_info.file_id}")
                print(f"  File Hash: {file_info.file_hash}")
                print(f"  Created At: {file_info.createdAt}")

                logger.info(f"Inserted FileInfo record: {file_info.id}")
                return True

        except Exception as e:
            logger.error(f"Error inserting FileInfo: {e}")
            print(f"❌ Error inserting FileInfo: {e}")
            return False

    def get_file_info_count(self):
        """Get the total number of FileInfo records."""
        try:
            with self.get_db_session() as session:
                count = session.query(FileInfo).count()
                print(f"\n📊 Total FileInfo records: {count}")
                return count
        except Exception as e:
            logger.error(f"Error getting count: {e}")
            print(f"❌ Error getting count: {e}")
            return 0


def main():
    """Main function to run the database tests."""
    print("=" * 60)
    print("SIMPLE DATABASE TEST - FileInfo Operations")
    print("=" * 60)

    tester = SimpleDatabaseTester()

    # Test connection
    print("\n1. Testing Database Connection...")
    if not tester.test_connection():
        print("❌ Cannot proceed without database connection")
        print("\n🔧 Setup Instructions:")
        print("1. Set environment variables:")
        print("   export DB_USERNAME=your_username")
        print("   export DB_PASSWORD=your_password")
        print("   export DB_INSTANCE=true")
        print("2. Ensure PostgreSQL is running on localhost:5432")
        print("3. Create database 'chatbot_ui' if it doesn't exist")
        return

    print("✅ Database connection successful!")

    # Get current count
    print("\n2. Current FileInfo Count...")
    tester.get_file_info_count()

    # View existing data
    print("\n3. Viewing Existing FileInfo Data...")
    tester.view_file_info(limit=5)

    # Insert test data
    print("\n4. Inserting Test FileInfo Record...")
    tester.insert_file_info()

    # View updated data
    print("\n5. Viewing Updated FileInfo Data...")
    tester.view_file_info(limit=5)

    # Get updated count
    print("\n6. Updated FileInfo Count...")
    tester.get_file_info_count()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
