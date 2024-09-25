import os

import pandas as pd
from sqlalchemy import create_engine, inspect


class PrepareSQLFromTabularData:
    def __init__(self, files_dir) -> None:
        self.files_directory = files_dir
        self.file_dir_list = [
            f for f in os.listdir(files_dir) if f.endswith((".csv", ".xlsx"))
        ]

        db_name = "tabular_data.db"
        self.db_path = os.path.join(files_dir, db_name)
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self.db_url)

        print("Number of CSV/XLSX files:", len(self.file_dir_list))

    def _prepare_db(self):
        for file in self.file_dir_list:
            full_file_path = os.path.join(self.files_directory, file)
            file_name, file_extension = os.path.splitext(file)
            if file_extension == ".csv":
                df = pd.read_csv(full_file_path)
                self._save_dataframe_to_sql(df, file_name)
            elif file_extension == ".xlsx":
                # Handle multiple sheets in Excel files
                excel_file = pd.ExcelFile(full_file_path)
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(full_file_path, sheet_name=sheet_name)
                    table_name = f"{file_name}_{sheet_name}"
                    self._save_dataframe_to_sql(df, table_name)
            else:
                raise ValueError("The selected file type is not supported")
        print(
            "All CSV and Excel files have been processed and saved to the SQL database."
        )

    def _save_dataframe_to_sql(self, df, table_name):
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
        self._prepare_db()
        self._validate_db()
