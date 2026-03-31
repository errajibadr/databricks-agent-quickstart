# Import agent to register @invoke/@stream functions with the server
import agent_server.agent  # noqa: F401
from mlflow.genai.agent_server import AgentServer
from dotenv import load_dotenv

load_dotenv()

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app


def main():
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()
