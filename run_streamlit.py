import sys

import streamlit.web.cli as stcli


def run():
    sys.argv = ["streamlit", "run", "streamlit_app.py"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    run()
