import unittest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import Tool

from imc_agents.agents.onboarding_agent import get_onboarding_graph
from imc_agents.costum_llm_model import CustomChatModel


class MyTestCase(unittest.TestCase):

    def test_bind_tools(self):
        llm = CustomChatModel()
        dummy_tools = [Tool(name="ToolA", description="Testtool",func=self.dummy_tool_func)]
        llm.bind_tools(dummy_tools)
        self.assertEqual(llm.tools, dummy_tools)

    def test_graph_with_custom_model(self):
        llm = CustomChatModel()
        graph = get_onboarding_graph(llm)
        result = graph.invoke({"user_input": "Max Mustermann, Vertrieb"})

        self.assertIsInstance(result, dict)
        self.assertIn("llm_output", result)
        self.assertIsInstance(result["llm_output"], str)
        self.assertGreater(len(result["llm_output"]), 0)

        print("Antwort:", result["llm_output"])

    @patch("imc_agents.base_llm_wrapper.call_llm")
    def test_invoke(self, mock_call):
        mock_call.return_value = "Antwort vom LLM"
        llm = CustomChatModel()

        messages = [
            SystemMessage(content="Du bist ein Assistent."),
            HumanMessage(content="Hallo, wer bist du?")
        ]
        result = llm.invoke(messages)

        self.assertIsInstance(result, AIMessage)
        self.assertEqual(result.content, "Antwort vom LLM")

    @patch("imc_agents.base_llm_wrapper.call_llm")
    def test_generate(self, mock_call):
        mock_call.return_value = "Generierte Antwort"
        llm = CustomChatModel()

        messages = [
            HumanMessage(content="Frage 1"),
            HumanMessage(content="Frage 2")
        ]
        result = llm._generate(messages)

        self.assertIsInstance(result, AIMessage)
        self.assertEqual(result.content, "Generierte Antwort")
        mock_call.assert_called_once_with("Frage 2", model="GPT-4o", temperature=0.7)

    @patch("imc_agents.base_llm_wrapper.call_llm")
    def test_invoke_with_tools(self, mock_call):
        mock_call.return_value = "Tool-Antwort"
        llm = CustomChatModel()
        tool = Tool(name="InfoTool", description="Gibt Infos aus", func=self.dummy_tool_func)
        llm.bind_tools([tool])

        messages = [HumanMessage(content="Nutze ein Tool")]
        result = llm.invoke(messages)

        self.assertIn("Tool-Antwort", result.content)
        mock_call.assert_called_once()


    def dummy_tool_func(input: str) -> str:
        return f"Tool called with input: {input}"


if __name__ == '__main__':
    unittest.main()
