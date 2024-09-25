import os
from typing import List, Optional

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from sqlalchemy import create_engine, inspect

from configs.app_config import Config
from rtl_rag_chatbot_api.common.prepare_sqlitedb_from_csv_xlsx import (
    PrepareSQLFromTabularData,
)


class TabularDataHandler:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.db_name = "tabular_data.db"
        self.db_path = os.path.join(data_dir, self.db_name)
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self.db_url)
        self.db = SQLDatabase(engine=self.engine)
        self.config = Config()
        self.llm = self._initialize_llm()
        self.agent = None

    def _initialize_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_endpoint=self.config.azure_llm.azure_llm_endpoint,
            azure_deployment=self.config.azure_llm.azure_llm_deployment,
            openai_api_version=self.config.azure_llm.azure_llm_api_version,
            model_name=self.config.azure_llm.azure_llm_model_name,
            temperature=0.2,
        )

    def prepare_database(self):
        data_preparer = PrepareSQLFromTabularData(self.data_dir)
        data_preparer.run_pipeline()

    def _initialize_agent(self):
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.agent = create_sql_agent(llm=self.llm, toolkit=toolkit, verbose=True)

    def get_table_names(self) -> List[str]:
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def ask_question(self, question: str) -> Optional[str]:
        if not self.agent:
            self._initialize_agent()

        try:
            response = self.agent.invoke({"input": question})
            return response["output"]
        except Exception as e:
            print(f"An error occurred while processing the question: {e}")
            return None

    def interactive_session(self):
        print("Welcome to the interactive SQL query session.")
        print("Type 'exit' to end the session.")

        while True:
            question = input("\nEnter your question: ").strip()

            if question.lower() == "exit":
                print("Exiting the session. Goodbye!")
                break

            answer = self.ask_question(question)

            if answer:
                print(f"\nAnswer: {answer}")
            else:
                print("Sorry, I couldn't find an answer to that question.")


def main(data_dir: str):
    handler = TabularDataHandler(data_dir)

    # Prepare the database from CSV and Excel files
    handler.prepare_database()

    # Get and print available table names
    table_names = handler.get_table_names()
    print("Available tables in the database:", table_names)

    # Start interactive session
    handler.interactive_session()


if __name__ == "__main__":
    data_dir = "rtl_rag_chatbot_api/tabularData/csv_dir"
    main(data_dir)
