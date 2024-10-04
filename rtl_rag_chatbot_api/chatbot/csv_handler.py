import logging
import os
from typing import List, Optional

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from configs.app_config import Config
from rtl_rag_chatbot_api.chatbot.chatbot_creator import get_azure_non_rag_response
from rtl_rag_chatbot_api.chatbot.prompt_handler import format_question
from rtl_rag_chatbot_api.common.prepare_sqlitedb_from_csv_xlsx import (
    PrepareSQLFromTabularData,
)


class TabularDataHandler:
    def __init__(
        self, config: Config, file_id: str = None, model_choice: str = "gpt_4o_mini"
    ):
        self.config = config
        self.file_id = file_id
        self.model_choice = model_choice
        self.data_dir = (
            "rtl_rag_chatbot_api/tabularData/csv_dir"
            if file_id is None
            else f"./chroma_db/{file_id}"
        )
        self.db_name = "tabular_data.db"
        self.db_path = os.path.join(self.data_dir, self.db_name)
        self.db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.db = SQLDatabase(engine=self.engine)
        self.llm = self._initialize_llm()
        self.agent = None

    def _initialize_llm(self) -> AzureChatOpenAI:
        model_config = self.config.azure_llm.models.get(self.model_choice)

        if not model_config:
            raise ValueError(f"Configuration for model {self.model_choice} not found")

        return AzureChatOpenAI(
            azure_endpoint=model_config.endpoint,
            azure_deployment=model_config.deployment,
            api_version=model_config.api_version,
            api_key=model_config.api_key,
            model_name=model_config.model_name,
            temperature=0.2,
        )

    def prepare_database(self):
        data_preparer = PrepareSQLFromTabularData(self.data_dir)
        data_preparer.run_pipeline()

    def _initialize_agent(self):
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.agent = create_sql_agent(
            llm=self.llm, toolkit=toolkit, verbose=True, handle_parsing_errors=True
        )

    def get_table_info(self) -> List[dict]:
        inspector = inspect(self.engine)
        table_info = []
        with self.Session() as session:
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                row_count = session.execute(
                    text(f'SELECT COUNT(*) FROM "{table_name}"')
                ).scalar()
                sample_data = session.execute(
                    text(f'SELECT * FROM "{table_name}" LIMIT 3')
                ).fetchall()

                column_stats = {}
                for column in columns:
                    if column["type"].python_type in (int, float):
                        stats = session.execute(
                            text(
                                f"SELECT MIN(\"{column['name']}\"), MAX(\"{column['name']}\"), "
                                f"AVG(\"{column['name']}\") FROM \"{table_name}\""
                            )
                        ).fetchone()
                        column_stats[column["name"]] = {
                            "min": stats[0],
                            "max": stats[1],
                            "avg": stats[2],
                        }

                table_info.append(
                    {
                        "name": table_name,
                        "columns": [
                            {"name": col["name"], "type": str(col["type"])}
                            for col in columns
                        ],
                        "row_count": row_count,
                        "sample_data": sample_data,
                        "column_stats": column_stats,
                    }
                )
        return table_info

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
                print(
                    "Sorry, I couldn't find an answer to that question.Let me try again"
                )
                return self.get_forced_answer(question, answer)

    def get_forced_answer(self, question: str, answer: str):
        prompt = (
            f"Question: {question}\n\n"
            f"Try to find an answer from the following text:\n{answer}\n\n"
            "If no accurate answer can be found, return 'Cannot find answer'. "
            "Otherwise, return the answer."
        )
        return get_azure_non_rag_response(self.config, prompt)

    def get_answer(self, question: str) -> str:
        try:
            answer = self.ask_question(question)
            if answer:
                return answer
            else:
                return self.get_forced_answer(question, answer)
        except Exception as e:
            logging.error(f"Error in TabularDataHandler get_answer: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"

    def ask_question(self, question: str) -> Optional[str]:
        if not self.agent:
            self._initialize_agent()

        try:
            # Get database info
            db_info = self.get_table_info()

            # Use the format_question function from prompt_handler
            formatted_question = format_question(db_info, question)
            print(f"This is a formatted by GPT: {formatted_question}")

            # Check if the formatted question contains specific keywords
            keywords = ["SELECT", "FIND", "LIST", "SHOW", "CALCULATE"]
            if any(keyword in formatted_question.upper() for keyword in keywords):
                # Only invoke the agent if the formatted question contains specific keywords
                response = self.agent.invoke({"input": formatted_question})
                return response["output"]
            else:
                # If no keywords are found, return the formatted question as is
                return formatted_question

        except Exception as e:
            print(f"An error occurred while processing the question: {e}")
            print(f"Debug: Full exception: {repr(e)}")
            return None


# def main(data_dir: str):
#     handler = TabularDataHandler(data_dir)
#     handler.prepare_database()
#     table_info = handler.get_table_info()
#     handler.interactive_session()


# if __name__ == "__main__":
#     data_dir = "rtl_rag_chatbot_api/tabularData/csv_dir"
#     main(data_dir)
