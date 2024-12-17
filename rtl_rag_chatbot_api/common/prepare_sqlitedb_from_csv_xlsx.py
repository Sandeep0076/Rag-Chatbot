import os

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
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.file_name = os.path.basename(file_path)
        self.file_extension = os.path.splitext(self.file_name)[1].lower()

        db_name = "tabular_data.db"
        self.db_path = os.path.join(output_dir, db_name)
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self.db_url)

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
        insp = inspect(self.engine)
        table_names = insp.get_table_names()
        print("Available table names in created SQL DB:", table_names)

    def run_pipeline(self):
        """
        Public method to run the data import pipeline, which includes preparing the database
        and validating the created tables. It is the main entry point for converting files
        to SQL tables and confirming their creation.
        """
        self._prepare_db()
        self._validate_db()
