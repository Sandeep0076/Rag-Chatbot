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
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            cursor.fetchall()
            conn.close()
            return True
        except sqlite3.Error:
            return False

    def _copy_db_file(self) -> bool:
        """
        Copy the source database file to the output directory.

        Returns:
            bool: True if copy successful, False otherwise
        """
        try:
            if not self._is_valid_sqlite_db(self.file_path):
                print(f"Error: {self.file_path} is not a valid SQLite database")
                return False

            shutil.copy2(self.file_path, self.db_path)
            print(f"Database file copied successfully to {self.db_path}")
            return True
        except Exception as e:
            print(f"Error copying database file: {str(e)}")
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
            print(f"Error getting table info: {str(e)}")
            return None

    def _prepare_db(self):
        """
        Private method to convert CSV/XLSX files from the specified directory into SQL tables.

        Each file's name (excluding the extension) is used as the table name.
        The data is saved into the SQLite database referenced by the engine attribute.
        """
        if self.file_extension == ".csv":
            df = pd.read_csv(self.file_path)
            self._save_dataframe_to_sql(df, self.file_name)
        elif self.file_extension in [".xlsx", ".xls"]:
            excel_file = pd.ExcelFile(self.file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                # Use just the sheet name as the table name
                self._save_dataframe_to_sql(df, sheet_name)
        elif self.file_extension == ".db":
            self._copy_db_file()
        else:
            raise ValueError("The selected file type is not supported")
        print("File has been processed and saved to the SQL database.")

    def _save_dataframe_to_sql(self, df, table_name):
        # Skip empty DataFrames
        if df.empty:
            print(f"Skipping empty sheet '{table_name}'")
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
            print(f"Table '{table_name}' created and data inserted.")
        else:
            print(f"Table '{table_name}' already exists. Skipping.")

    def _validate_db(self):
        """
        Validate and inspect the database structure.
        Provides information about tables, columns, and data types.
        """
        try:
            insp = inspect(self.engine)
            table_names = insp.get_table_names()

            if not table_names:
                print("Warning: No tables found in the database")
                return

            print("\nDatabase Inspection Results:")
            print("-" * 50)
            print(f"Database Location: {self.db_path}")
            print(f"Number of Tables: {len(table_names)}")
            print("\nTable Details:")

            # Use SQLite connection directly for better compatibility
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for table_name in table_names:
                print(f"\nTable: {table_name}")
                # Escape table name by wrapping in quotes
                escaped_table_name = f'"{table_name}"'
                try:
                    cursor.execute(f"PRAGMA table_info({escaped_table_name})")
                    columns = cursor.fetchall()
                    if columns:
                        print("Columns:")
                        for col in columns:
                            # col format: (cid, name, type, notnull, dflt_value, pk)
                            print(f"  - {col[1]} ({col[2]})")
                            if col[5]:  # is primary key
                                print("    Primary Key: Yes")

                    # Get row count using escaped table name
                    cursor.execute(f"SELECT COUNT(*) FROM {escaped_table_name}")
                    row_count = cursor.fetchone()[0]
                    print(f"Row Count: {row_count}")
                except sqlite3.Error as e:
                    print(f"Error accessing table {table_name}: {str(e)}")

            conn.close()
            print("\nValidation complete.")

        except Exception as e:
            print(f"Error during database validation: {str(e)}")

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
            print(f"Error in pipeline execution: {str(e)}")
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
