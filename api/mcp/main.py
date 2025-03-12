import json

import requests

from api.clients import LlmClient, ChatCompletionResponse
from api.logger import logger
import api.models as models
from api.utils import log_and_raise_for_status


class MCPBridgeClient:
    def __init__(self):
        self.url = "http://127.0.0.1:9092"
        self.refresh()

    def fetch_tools(self) -> dict:
        url = self.url
        path = "/mcp/tools"
        headers = {}
        response = requests.get(
            url + path,
            headers=headers,
        )
        log_and_raise_for_status(response, "MCP bridge client error")
        tools = response.json()
        return tools

    def get_tool(self, tool_name: str) -> list:
        for tool_set_name, tools in self.tools.items():
            if tool_set_name == tool_name:
                return tools["tools"]

            for tool in tools["tools"]:
                if tool_name == tool["name"]:
                    return [tool]

        raise ValueError("MCP tool not found")

    def call_tool(self, tool_name: str, params: str):
        url = self.url
        path = f"/mcp/tools/{tool_name}/call"
        headers = {}

        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            logger.error(f"failed to decode json for {tool_name}")
            return None

        response = requests.post(url + path, headers=headers, json=params)
        log_and_raise_for_status(response, "MCP bridge tools call error")
        response = response.json()
        return response

    def tools2openai(self, tool_names: list[str]):
        tools = []
        for tool_name in tool_names:
            mcp_tools = self.get_tool(tool_name)
            for mcp_tool in mcp_tools:
                tool = {
                    "type": "function",
                    "function": {
                        "name": mcp_tool["name"],
                        "description": mcp_tool["description"],
                        "parameters": mcp_tool["inputSchema"],
                        # "strict": False,
                    },
                }
                tools.append(tool)
        return tools

    def refresh(self):
        self.tools: dict = self.fetch_tools()


def multi_step_generate(
    model: models.Model,
    messages: list,
    sampling_params_plus: dict,
    mcp_bridge: MCPBridgeClient,
    max_step=10,
) -> (ChatCompletionResponse, list[list[dict]]):
    cpt = 0
    steps: list[list[dict]] = []  # list of tools calls
    aiclient = LlmClient(base_url=model.base_url, api_key=model.api_key)
    while cpt < max_step:
        cpt += 1
        result = aiclient.generate(model=model.name, messages=messages, **sampling_params_plus)
        completion = result.choices[0]

        if completion.finish_reason in [None, "", "stop", "length"]:
            break

        # MCP/toolings loop
        # --
        # Add the completion result to messages
        if completion.finish_reason not in ["tool_calls"]:
            logger.warning("Unknown LLM finish reason: %s" % completion.finish_reason)
        messages.append(completion.message.model_dump())
        # If tool_calls, loop over the tool and execute @FUTURE async.gather for concurent execution
        substeps = []
        for tool_call in completion.message.tool_calls or []:
            # Run tool
            tool_call_result = mcp_bridge.call_tool(
                tool_call.function.name, tool_call.function.arguments
            )
            # Format result
            if tool_call_result is None:
                tool_content = []
            else:
                tool_content = [
                    {"type": "text", "text": part["text"]}
                    for part in filter(lambda x: x["type"] == "text", tool_call_result["content"])
                ]

            if len(tool_content) == 0:
                tool_content = [{"type": "text", "text": "the tool call result is empty"}]
            elif len(tool_content) > 1:
                logger.warning(
                    "Tool call content size is greater than 1 for {tool_call.function.name}"
                )
            tool_content = "\n".join([x["text"] for x in tool_content])
            messages.append(
                {
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": tool_call.id,
                }
            )

            # Accumute tool call result to save it in answers result.
            substeps.append(
                {
                    "tool_name": tool_call.function.name,
                    "tool_params": tool_call.function.arguments,
                    "tool_result": tool_content,
                }
            )

        steps.append(substeps)

    return result, steps
