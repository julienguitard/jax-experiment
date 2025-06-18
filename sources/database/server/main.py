import argparse
from sources.database.connectors.connexions import DBClient
from sources.database.connectors.connexions import Buffer
from sources.database.connectors.connexions import generate_config
from sources.database.connectors.connexions import generate_token
from sources.common.constants import TOKEN_TEMPLATE


def main(query: str) -> str:
    """
    Executes a given query on the database and returns the result as a string.

    Args:
        query (str): The query to be executed on the database.

    Returns:
        str: The result of the query as a string.
    """
    config = generate_config(TOKEN_TEMPLATE)
    token = generate_token(config, TOKEN_TEMPLATE)
    with DBClient(token) as db:
        with Buffer(db) as b:
            return b.query(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a query on the database")
    _ = parser.add_argument("query", type=str)
    args = parser.parse_args()
    query = args.query
    print(main(query))
