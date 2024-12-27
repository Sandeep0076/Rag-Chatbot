import logging
import os
import shutil
import sqlite3
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine, inspect


class PrepareSQLFromTabularData:
    """
    A class that prepares a SQL database from CSV or XLSX files within a specified directory.

    This class reads each file, converts the data to a DataFrame, and then
    stores it as a table in a SQLite database, which is specified by the application configuration.
    """

    def __init__(self, file_path, output_dir) -> None:
        """
        Initialize an instance of PrepareSQLFromTabularData.

        Args:
            file_path (str): Path to the input file (CSV, XLSX, or SQLite DB)
            output_dir (str): Directory where the output database will be stored
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.file_name = os.path.basename(file_path)
        self.file_extension = os.path.splitext(self.file_name)[1].lower()

        db_name = "tabular_data.db"
        self.db_path = os.path.join(output_dir, db_name)
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self.db_url)

    def _is_valid_sqlite_db(self, db_path: str) -> bool:
        """
        Check if the given file is a valid SQLite database.

        Args:
            db_path (str): Path to the database file

        Returns:
            bool: True if valid SQLite database, False otherwise
        """
        try:
            if not os.path.exists(db_path):
                logging.error(f"Database file not found: {db_path}")
                return False

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check if it's a valid SQLite database
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            if not tables:
                logging.warning(f"No tables found in database: {db_path}")
                conn.close()
                return False

            cursor.close()
            conn.close()
            return True
        except sqlite3.Error as e:
            logging.error(f"SQLite error while validating database {db_path}: {str(e)}")
            return False
        except Exception as e:
            logging.error(
                f"Unexpected error while validating database {db_path}: {str(e)}"
            )
            return False

    def _copy_db_file(self) -> bool:
        """
        Copy the source database file to the output directory.

        Returns:
            bool: True if copy successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
                logging.info(f"Created output directory: {self.output_dir}")

            # Validate source database
            if not os.path.exists(self.file_path):
                logging.error(f"Source database file not found: {self.file_path}")
                return False

            if not self._is_valid_sqlite_db(self.file_path):
                logging.error(f"Invalid source SQLite database: {self.file_path}")
                return False

            # Remove existing database if it exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                logging.info(f"Removed existing database: {self.db_path}")

            # Copy the database file
            shutil.copy2(self.file_path, self.db_path)
            logging.info(f"Database file copied successfully to {self.db_path}")

            # Verify the copied file
            if not self._is_valid_sqlite_db(self.db_path):
                logging.error(f"Copied database file is invalid: {self.db_path}")
                if os.path.exists(self.db_path):
                    os.remove(self.db_path)
                return False

            return True
        except Exception as e:
            logging.error(f"Error copying database file: {str(e)}")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            return False

    def _get_table_info(self, table_name: str) -> Optional[List[tuple]]:
        """
        Get information about table columns and their types.

        Args:
            table_name (str): Name of the table to inspect

        Returns:
            Optional[List[tuple]]: List of column information tuples or None if error
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            conn.close()
            return columns
        except sqlite3.Error as e:
            logging.error(f"Error getting table info: {str(e)}")
            return None

    def _prepare_db(self):
        """
        Private method to handle different types of input files (CSV, XLSX, SQLite).

        For CSV/XLSX files, converts them into SQL tables.
        For SQLite files, copies them to the target location.
        """
        if self.file_extension in [".csv"]:
            df = pd.read_csv(self.file_path)
            table_name = os.path.splitext(self.file_name)[0]
            self._save_dataframe_to_sql(df, table_name)
        elif self.file_extension in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(self.file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                self._save_dataframe_to_sql(df, sheet_name)
        elif self.file_extension in [".db", ".sqlite", ".sqlite3"]:
            if not self._is_valid_sqlite_db(self.file_path):
                raise ValueError(f"Invalid SQLite database file: {self.file_path}")
            if not self._copy_db_file():
                raise ValueError(f"Failed to copy database file to {self.db_path}")
            logging.info(f"SQLite database copied successfully to {self.db_path}")
        else:
            raise ValueError(f"Unsupported file type: {self.file_extension}")

        logging.info("File has been processed and saved to the SQL database.")

    def _save_dataframe_to_sql(self, df, table_name):
        # Skip empty DataFrames
        if df.empty:
            logging.info(f"Skipping empty sheet '{table_name}'")
            return

        # Clean up table name - remove file ID if present
        if "_" in table_name:
            # Split by underscore and take everything after the last UUID part (5 groups of hex numbers)
            parts = table_name.split("_")
            for i in range(len(parts)):
                if len(parts[i]) == 36 and "-" in parts[i]:  # UUID format check
                    table_name = "_".join(parts[i + 1 :])
                    break

        inspector = inspect(self.engine)
        if not inspector.has_table(table_name):
            df.to_sql(table_name, self.engine, index=False)
            logging.info(f"Table '{table_name}' created and data inserted.")
        else:
            logging.info(f"Table '{table_name}' already exists. Skipping.")

    def _validate_db(self):
        """
        Validate and inspect the database structure.
        Provides information about tables, columns, and data types.
        """
        try:
            insp = inspect(self.engine)
            table_names = insp.get_table_names()

            if not table_names:
                logging.warning("Warning: No tables found in the database")
                return

            logging.info("\nDatabase Inspection Results:")
            logging.info("-" * 50)
            logging.info(f"Database Location: {self.db_path}")
            logging.info(f"Number of Tables: {len(table_names)}")
            logging.info("\nTable Details:")

            # Use SQLite connection directly for better compatibility
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for table_name in table_names:
                logging.info(f"\nTable: {table_name}")
                # Escape table name by wrapping in quotes
                escaped_table_name = f'"{table_name}"'
                try:
                    cursor.execute(f"PRAGMA table_info({escaped_table_name})")
                    columns = cursor.fetchall()
                    if columns:
                        logging.info("Columns:")
                        for col in columns:
                            # col format: (cid, name, type, notnull, dflt_value, pk)
                            logging.info(f"  - {col[1]} ({col[2]})")
                            if col[5]:  # is primary key
                                logging.info("    Primary Key: Yes")

                    # Get row count using escaped table name
                    cursor.execute(f"SELECT COUNT(*) FROM {escaped_table_name}")
                    row_count = cursor.fetchone()[0]
                    logging.info(f"Row Count: {row_count}")
                except sqlite3.Error as e:
                    logging.error(f"Error accessing table {table_name}: {str(e)}")

            conn.close()
            logging.info("\nValidation complete.")

        except Exception as e:
            logging.error(f"Error during database validation: {str(e)}")

    def run_pipeline(self):
        """
        Public method to run the data import pipeline, which includes preparing the database
        and validating the created tables. It is the main entry point for converting files
        to SQL tables and confirming their creation.

        Returns:
            bool: True if pipeline completed successfully, False otherwise
        """
        try:
            self._prepare_db()
            self._validate_db()
            return True
        except Exception as e:
            logging.error(f"Error in pipeline execution: {str(e)}")
            return False


if __name__ == "__main__":
    # # Test with a CSV file
    # csv_test = PrepareSQLFromTabularData(
    #     file_path="path/to/your/test.csv",
    #     output_dir="./test_output"
    # )
    # print("\nTesting CSV file processing:")
    # csv_test.run_pipeline()

    # # Test with an Excel file
    # excel_test = PrepareSQLFromTabularData(
    #     file_path="path/to/your/test.xlsx",
    #     output_dir="./test_output"
    # )
    # print("\nTesting Excel file processing:")
    # excel_test.run_pipeline()

    # Test with a SQLite database file
    db_test = PrepareSQLFromTabularData(
        file_path="local_data/Car_Database.db", output_dir="processed_data"
    )
    print("\nTesting Database file processing:")
    db_test.run_pipeline()
