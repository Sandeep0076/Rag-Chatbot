import os

from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Define the Base class for ORM mapping
Base = declarative_base()

engine = create_engine(os.getenv("DATABASE_URL", ""))
Base.metadata.create_all(engine)

# Create a Session for interacting with the database
Session = sessionmaker(bind=engine)
session = Session()


# Define your table/model
class User(Base):
    __tablename__ = "User"

    id = Column(String, primary_key=True)
    email = Column(String)
    name = Column(String)


def get_user_list():
    """"""

    users = session.query(User).all()
    print(len(users))


if __name__ == "__main__":
    get_user_list()
