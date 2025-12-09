import csv
import logging
import os
import shutil
import sqlite3
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine, inspect

from rtl_rag_chatbot_api.common.errors import (
    CsvAllTablesEmptyError,
    CsvInvalidOrEmptyError,
    CsvNoTablesError,
    TabularInvalidDataError,
)


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

    def _sniff_csv_dialect(self, encoding: str):
        """
        Use Python's csv.Sniffer to detect delimiter and header presence.

        This makes CSV ingestion robust for different delimiters like ',', ';', tab, '|'.
        """
        try:
            with open(
                self.file_path, "r", encoding=encoding, errors="ignore"
            ) as file_obj:
                sample = file_obj.read(4096)
                if not sample:
                    return None, None

                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
                has_header = sniffer.has_header(sample)
                logging.info(
                    "Detected CSV dialect: delimiter=%r, has_header=%s "
                    "for encoding=%s",
                    dialect.delimiter,
                    has_header,
                    encoding,
                )
                return dialect, has_header
        except Exception as e:
            logging.debug(
                "Failed to sniff CSV dialect for encoding %s: %s",
                encoding,
                str(e),
            )
            return None, None

    def _infer_header_from_sample(
        self, delimiter: Optional[str], encoding: str
    ) -> Optional[bool]:
        """
        Heuristic header detection when csv.Sniffer is inconclusive or wrong.

        If the first row is all non-numeric tokens and the second row has at least
        one numeric-looking token, we treat the first row as header.
        """
        try:
            if not delimiter:
                delimiter = ","

            with open(self.file_path, "r", encoding=encoding, errors="ignore") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = []
                for _ in range(2):  # first two rows are enough
                    try:
                        rows.append(next(reader))
                    except StopIteration:
                        break

            if len(rows) < 2:
                return None

            def is_numeric_like(token: str) -> bool:
                t = token.strip()
                if not t:
                    return False
                # remove common thousands/decimal separators
                t = t.replace(".", "").replace(",", "")
                return t.replace("-", "", 1).isdigit()

            first_row = rows[0]
            second_row = rows[1]

            # All tokens non-numeric in first row?
            first_non_numeric = all(not is_numeric_like(tok) for tok in first_row)
            # Any numeric tokens in second row?
            second_has_numeric = any(is_numeric_like(tok) for tok in second_row)

            if first_non_numeric and second_has_numeric:
                logging.info(
                    "Header heuristic: treating first row as header "
                    "(delimiter=%r, encoding=%s)",
                    delimiter,
                    encoding,
                )
                return True
            return None
        except Exception as e:
            logging.debug(
                "Header heuristic failed for delimiter %r, encoding %s: %s",
                delimiter,
                encoding,
                str(e),
            )
            return None

    def _read_csv_with_encoding(self, encodings: List[str]):
        """
        Try reading CSV file with different encodings.

        Args:
            encodings (List[str]): List of encodings to try

        Returns:
            tuple: (DataFrame, successful_encoding) or (None, None) if failed
        """
        for encoding in encodings:
            try:
                logging.info(
                    "Attempting to read CSV with encoding=%s using robust "
                    "delimiter detection",
                    encoding,
                )

                # First try to sniff dialect (delimiter + header)
                dialect, has_header = self._sniff_csv_dialect(encoding)

                read_kwargs = {
                    "filepath_or_buffer": self.file_path,
                    "encoding": encoding,
                    "on_bad_lines": "skip",
                    "engine": "python",
                }

                # If we detected a delimiter, use it. Otherwise, let pandas infer (sep=None).
                if dialect is not None:
                    read_kwargs["sep"] = dialect.delimiter
                else:
                    read_kwargs["sep"] = None

                # Respect detected header if available; otherwise, apply heuristic
                header_inferred = has_header
                if header_inferred is None or header_inferred is False:
                    heuristic_header = self._infer_header_from_sample(
                        dialect.delimiter if dialect else None, encoding
                    )
                    if heuristic_header is True:
                        header_inferred = True

                if header_inferred is not None and not header_inferred:
                    # Explicitly no header
                    read_kwargs["header"] = None

                df = pd.read_csv(**read_kwargs)

                if df is not None and not df.empty:
                    logging.info(
                        "Successfully read %d rows and %d columns from CSV "
                        "with encoding=%s",
                        len(df),
                        len(df.columns),
                        encoding,
                    )
                    return df, encoding
                logging.warning(
                    "DataFrame is empty after reading CSV with encoding=%s",
                    encoding,
                )
            except UnicodeDecodeError:
                logging.debug("Failed to read CSV with encoding=%s", encoding)
            except Exception as e:
                logging.error(
                    "Error reading CSV with encoding=%s: %s", encoding, str(e)
                )
        return None, None

    def _handle_csv_file(self):
        """Handle CSV file processing."""
        try:
            # Try different encodings
            encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
            df, successful_encoding = self._read_csv_with_encoding(encodings)

            if df is None or df.empty:
                raise CsvInvalidOrEmptyError(
                    f"Failed to read CSV file or file is empty: {self.file_path}",
                    details={"file_path": self.file_path},
                )

            # Clean column names and remove any completely empty columns
            df.columns = [str(col).strip() for col in df.columns]
            df = df.dropna(axis=1, how="all")

            # Skip if all data was removed
            if df.empty:
                raise CsvAllTablesEmptyError(
                    "CSV file has no valid data after cleaning",
                    details={"file_path": self.file_path},
                )

            # Validate that we have some data
            if len(df.columns) == 0:
                raise CsvInvalidOrEmptyError(
                    "CSV file has no valid columns",
                    details={"file_path": self.file_path},
                )

            # Get table name from file name
            base_name = os.path.splitext(self.file_name)[0]
            table_name = base_name.split("_", 1)[-1] if "_" in base_name else base_name

            # Save to database
            logging.info(
                f"Successfully read CSV file with {successful_encoding} encoding"
            )
            logging.info(f"Processing {len(df)} rows with {len(df.columns)} columns")
            self._save_dataframe_to_sql(df, table_name)
            logging.info(f"Successfully processed CSV file into table: {table_name}")

        except Exception as e:
            logging.error(f"Error processing CSV file: {str(e)}")
            raise CsvInvalidOrEmptyError(
                f"Failed to process CSV file: {str(e)}",
                details={"file_path": self.file_path},
            )

    def _handle_excel_file(self):
        """Handle Excel file processing."""
        try:
            excel_file = pd.ExcelFile(self.file_path)
            if not excel_file.sheet_names:
                raise CsvNoTablesError(
                    "Excel file contains no sheets",
                    details={"file_path": self.file_path},
                )

            processed_sheets = 0
            for sheet_name in excel_file.sheet_names:
                try:
                    logging.info(f"Processing sheet: {sheet_name}")
                    df = pd.read_excel(self.file_path, sheet_name=sheet_name)

                    # Skip empty sheets
                    if df.empty:
                        logging.warning(f"Skipping empty sheet: {sheet_name}")
                        continue

                    # Skip sheets with no valid columns
                    if len(df.columns) == 0:
                        logging.warning(f"Skipping sheet with no columns: {sheet_name}")
                        continue

                    # Clean column names and remove any completely empty columns
                    df.columns = [str(col).strip() for col in df.columns]
                    df = df.dropna(axis=1, how="all")

                    # Skip if all data was removed
                    if df.empty:
                        logging.warning(
                            f"Sheet {sheet_name} has no valid data after cleaning"
                        )
                        continue

                    # Save to database
                    self._save_dataframe_to_sql(df, sheet_name)
                    processed_sheets += 1
                    logging.info(f"Successfully processed sheet: {sheet_name}")

                except Exception as sheet_error:
                    logging.error(
                        f"Error processing sheet {sheet_name}: {str(sheet_error)}"
                    )
                    continue

            if processed_sheets == 0:
                raise CsvAllTablesEmptyError(
                    "No valid data found in any sheet of the Excel file",
                    details={"file_path": self.file_path},
                )

            logging.info(f"Successfully processed {processed_sheets} sheets")

        except Exception as e:
            logging.error(f"Error processing Excel file: {str(e)}")
            raise TabularInvalidDataError(
                f"Failed to process Excel file: {str(e)}",
                details={"file_path": self.file_path},
            )

    def _handle_sqlite_file(self):
        """Handle SQLite file processing."""
        if not self._is_valid_sqlite_db(self.file_path):
            raise TabularInvalidDataError(
                f"Invalid SQLite database file: {self.file_path}",
                details={"file_path": self.file_path},
            )
        if not self._copy_db_file():
            raise TabularInvalidDataError(
                f"Failed to copy database file to {self.db_path}",
                details={"file_path": self.file_path, "target": self.db_path},
            )
        logging.info(f"SQLite database copied successfully to {self.db_path}")

    def _validate_tables(self):
        """Validate that tables were created successfully."""
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        if not tables:
            raise CsvNoTablesError(
                "No tables were created in the database",
                details={"file_path": self.file_path},
            )
        logging.info(
            f"File processed successfully. Created tables: {', '.join(tables)}"
        )

    def _cleanup_on_error(self):
        """Clean up database file on error."""
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
                logging.info(f"Cleaned up partial database file: {self.db_path}")
            except Exception as cleanup_error:
                logging.error(f"Error cleaning up database file: {str(cleanup_error)}")

    def _prepare_db(self):
        """
        Handle different types of input files (CSV, XLSX, SQLite).

        For CSV/XLSX files, converts them into SQL tables.
        For SQLite files, copies them to the target location.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(f"Ensuring output directory exists: {self.output_dir}")

            if self.file_extension in [".csv"]:
                self._handle_csv_file()
            elif self.file_extension in [".xlsx", ".xls"]:
                self._handle_excel_file()
            elif self.file_extension in [".db", ".sqlite", ".sqlite3"]:
                self._handle_sqlite_file()
            else:
                from rtl_rag_chatbot_api.common.errors import (
                    BaseAppError,
                    ErrorRegistry,
                )

                raise BaseAppError(
                    ErrorRegistry.ERROR_FILE_TYPE_UNSUPPORTED,
                    f"Unsupported file type: {self.file_extension}",
                    details={
                        "file_path": self.file_path,
                        "extension": self.file_extension,
                    },
                )

            self._validate_tables()

        except Exception as e:
            logging.error(f"Error in _prepare_db: {str(e)}")
            self._cleanup_on_error()
            raise

    def _save_dataframe_to_sql(self, df, table_name):
        """
        Save a DataFrame to SQL database with proper error handling and validation.

        Args:
            df (pd.DataFrame): DataFrame to save
            table_name (str): Name of the table to create
        """
        try:
            # Skip empty DataFrames
            if df.empty:
                logging.warning(f"Skipping empty DataFrame for table '{table_name}'")
                return

            # Clean and validate column names
            df.columns = [
                str(col).strip().replace(" ", "_").lower() for col in df.columns
            ]

            # Remove any invalid characters from table name
            table_name = "".join(
                c if c.isalnum() or c == "_" else "_" for c in table_name
            )

            # Ensure table name doesn't start with a number
            if table_name[0].isdigit():
                table_name = "table_" + table_name

            # Always append '_table' suffix to avoid reserved keywords and ambiguity
            if not table_name.endswith("_table"):
                table_name = f"{table_name}_table"

            inspector = inspect(self.engine)
            if not inspector.has_table(table_name):
                logging.info(f"Creating table '{table_name}' with {len(df)} rows")
                df.to_sql(table_name, self.engine, index=False)
                logging.info(
                    f"Successfully created table '{table_name}' and inserted {len(df)} rows"
                )
            else:
                logging.info(f"Table '{table_name}' already exists. Skipping.")

        except Exception as e:
            logging.error(f"Error saving DataFrame to SQL: {str(e)}")
            raise TabularInvalidDataError(
                f"Failed to save data to table '{table_name}': {str(e)}",
                details={"table_name": table_name},
            )

    def _validate_db(self):
        """
        Validate and inspect the database structure.
        Provides information about tables, columns, and data types.
        """
        try:
            if not os.path.exists(self.db_path):
                from rtl_rag_chatbot_api.common.errors import (
                    BaseAppError,
                    ErrorRegistry,
                )

                raise BaseAppError(
                    ErrorRegistry.ERROR_FILE_NOT_FOUND,
                    f"Database file does not exist at {self.db_path}",
                    details={"db_path": self.db_path},
                )

            insp = inspect(self.engine)
            table_names = insp.get_table_names()

            if not table_names:
                raise CsvNoTablesError(
                    "No tables found in the database. This could be because:\n"
                    "1. The input file was empty or contained no valid data\n"
                    "2. All sheets/tables were skipped due to data validation\n"
                    "3. There was an error during data import\n"
                    "Please check the logs above for specific warnings or errors."
                )

            logging.info("\nDatabase Inspection Results:")
            logging.info("-" * 50)
            logging.info(f"Database Location: {self.db_path}")
            logging.info(f"Number of Tables: {len(table_names)}")
            logging.info("\nTable Details:")

            # Use SQLite connection directly for better compatibility
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            total_rows = 0
            for table_name in table_names:
                logging.info(f"\nTable: {table_name}")
                # Escape table name by wrapping in quotes
                escaped_table_name = f'"{table_name}"'
                try:
                    cursor.execute(f"PRAGMA table_info({escaped_table_name})")
                    columns = cursor.fetchall()
                    if not columns:
                        logging.warning(
                            f"Warning: No columns found in table {table_name}"
                        )
                        continue

                    logging.info("Columns:")
                    for col in columns:
                        # col format: (cid, name, type, notnull, dflt_value, pk)
                        logging.info(f"  - {col[1]} ({col[2]})")
                        if col[5]:  # is primary key
                            logging.info("    Primary Key: Yes")

                    # Get row count using escaped table name
                    cursor.execute(f"SELECT COUNT(*) FROM {escaped_table_name}")
                    row_count = cursor.fetchone()[0]
                    total_rows += row_count
                    logging.info(f"Row Count: {row_count}")

                    if row_count == 0:
                        logging.warning(f"Warning: Table {table_name} is empty")

                except sqlite3.Error as e:
                    logging.error(f"Error accessing table {table_name}: {str(e)}")

            conn.close()

            if total_rows == 0:
                raise CsvAllTablesEmptyError(
                    "Database contains tables but no data. This could be because:\n"
                    "1. The input file contained only headers\n"
                    "2. All data was filtered out during cleaning\n"
                    "3. There was an error during data import\n"
                    "Please check the logs above for specific warnings or errors."
                )

            logging.info(f"\nTotal Rows Across All Tables: {total_rows}")
            logging.info("\nValidation complete.")

        except Exception as e:
            logging.error(f"Error during database validation: {str(e)}")
            raise

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
