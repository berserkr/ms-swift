# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import TYPE_CHECKING, List, Tuple, Union
import ast
import json

from .base import BaseAgentTemplate

if TYPE_CHECKING:
    from swift.llm.infer import Function
    from swift.llm.template import Prompt

from swift.utils import get_logger

logger = get_logger()

class GraniteThinkingAgentTemplate(BaseAgentTemplate):
    """Agent template for Granite Thinking models using XML tool call format.

    Tool calls use the format:
        <tool_call>
        <function=function_name>
        <parameter=param1>
        value1
        </parameter>
        </function>
        </tool_call>

    Tool responses are wrapped in:
        <tool_response>
        content
        </tool_response>
    """

    def get_toolcall(self, response: str) -> List['Function']:
        from swift.llm.infer import Function
        functions = []
        tool_call_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
        for block in tool_call_blocks:
            func_match = re.search(r'<function=([^>]+)>', block)
            if not func_match:
                continue
            name = func_match.group(1).strip()
            params = {}
            param_matches = re.findall(r'<parameter=([^>]+)>\n?(.*?)\n?</parameter>', block, re.DOTALL)
            for param_name, param_value in param_matches:
                param_value = param_value.strip()
                try:
                    param_value = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    pass
                params[param_name.strip()] = param_value
            functions.append(Function(name=name, arguments=params))
        if not functions:
            return super().get_toolcall(response)
        return functions

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        res = ['<|im_end|>\n', '<|im_start|>user\n']
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res.append(f'<tool_response>\n{tool_content}\n</tool_response>\n')
        res.append('<|im_end|>\n<|im_start|>assistant\n')
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: str, user_message=None) -> str:
        tool_lines = []
        tool_lines.append('# Tools\n\nYou have access to the following functions:\n')
        tool_lines.append('<tools>')
        for tool in tools:
            tool = self.unwrap_tool(tool)
            name = self._get_tool_name(tool)
            tool_lines.append(f'\n<function>\n<name>{name}</name>')
            if tool.get('description'):
                tool_lines.append(f'\n<description>{tool["description"].strip()}</description>')
            tool_lines.append('\n<parameters>')
            params = tool.get('parameters', {})
            if isinstance(params, dict) and 'properties' in params:
                for pname, pfields in params['properties'].items():
                    tool_lines.append(f'\n<parameter>\n<name>{pname}</name>')
                    if 'type' in pfields:
                        tool_lines.append(f'\n<type>{pfields["type"]}</type>')
                    if 'description' in pfields:
                        tool_lines.append(f'\n<description>{pfields["description"].strip()}</description>')
                    if 'enum' in pfields:
                        tool_lines.append(f'\n<enum>{json.dumps(pfields["enum"])}</enum>')
                    tool_lines.append('\n</parameter>')
            if isinstance(params, dict) and 'required' in params:
                tool_lines.append(f'\n<required>{json.dumps(params["required"])}</required>')
            tool_lines.append('\n</parameters>')
            tool_lines.append('\n</function>')
        tool_lines.append('\n</tools>')
        tool_lines.append(
            '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:'
            '\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>'
            '\nvalue_1\n</parameter>\n<parameter=example_parameter_2>'
            '\nThis is the value for the second parameter\nthat can span\nmultiple lines'
            '\n</parameter>\n</function>\n</tool_call>')
        return ''.join(tool_lines)

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            args = tool_call['arguments']
            parts = [f'<tool_call>\n<function={name}>\n']
            if isinstance(args, dict):
                for k, v in args.items():
                    v_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                    parts.append(f'<parameter={k}>\n{v_str}\n</parameter>\n')
            parts.append('</function>\n</tool_call>')
            tool_calls.append(''.join(parts))
        return '\n'.join(tool_calls)


class GraniteAgentTemplate(BaseAgentTemplate):

    def get_toolcall(self, response: str) -> List['Function']:
        from swift.llm.infer import Function
        res_list = re.findall(r'<tool_call>(.+?)</tool_call>', response, re.DOTALL)
        functions = []
        for res in res_list:
            res = self._parse_json(res)
            if isinstance(res, dict) and 'name' in res and 'arguments' in res:
                functions.append(Function(name=res['name'], arguments=res['arguments']))
        if len(functions) == 0:
            # compat react_en
            return super().get_toolcall(response)
        return functions

    def _format_tool_responsesEOT(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        if hasattr(self, 'template_meta'):
            prompt = self.template_meta.prompt
            chat_sep = self.template_meta.chat_sep
        else:
            prompt = ['<|start_of_role|>user<|end_of_role|>{{QUERY}}<|end_of_turn|>\n<|start_of_role|>assistant<|end_of_role|>']
            chat_sep = ['<|end_of_turn|>\n']
        res = chat_sep.copy()
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res_tool.append(f'<tool_response>\n{tool_content}\n</tool_response>')
        total_tool = '\n'.join(res_tool)
        for context in prompt:
            if isinstance(context, str):
                context = context.replace('{{QUERY}}', total_tool)
            res.append(context)
        return assistant_content, res

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages,
    ) -> Tuple[str, 'Prompt']:
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        if hasattr(self, 'template_meta'):
            prompt = self.template_meta.prompt
            chat_sep = self.template_meta.chat_sep
        else:
            prompt = ['<|start_of_role|>user<|end_of_role|>{{QUERY}}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>']
            chat_sep = ['<|end_of_text|>\n']
        res = chat_sep.copy()
        res_tool = []
        for tool_message in tool_messages:
            tool_content = tool_message['content']
            res_tool.append(f'<tool_response>\n{tool_content}\n</tool_response>')
        total_tool = '\n'.join(res_tool)
        for context in prompt:
            if isinstance(context, str):
                context = context.replace('{{QUERY}}', total_tool)
            res.append(context)
        return assistant_content, res

    def _format_tools(self, tools: List[Union[str, dict]], system: str, user_message=None) -> str:
        # edit 1 - added new line as our chat template has \n
        # edit 2 - new line causes too many lines between calls... 
        tool_descs = [json.dumps(self.wrap_tool(tool), ensure_ascii=False) for tool in tools]
        return f"""You are a helpful assistant with access to the following tools. You may call one or more tools to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + '\n'.join(tool_descs) + """
</tools>

For each tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{\"name\": <function-name>, \"arguments\": <args-json-object>}
</tool_call>. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request."""

    def _format_tool_calls(self, tool_call_messages) -> str:
        tool_calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            if isinstance(tool_call, list):
                for tc in tool_call:
                    tool_calls.append(f'<tool_call>\n{json.dumps(tc, ensure_ascii=False)}\n</tool_call>')
            else:
                tool_calls.append(f'<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>')
        return '\n'.join(tool_calls)
    
    
    def _fix_tool_calls(text):

        # do not break things!
        try: 
            pattern = r"<tool_call>\s*([\s\S]*?)\s*</tool_call>"
            matches = re.findall(pattern, text, re.DOTALL)
            fixed_text = ''
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        for item in parsed:
                            fixed_text += f"<tool_call>\n{json.dumps(item).strip()}\n</tool_call>\n"
                    else:
                        fixed_text += f"<tool_call>\n{json.dumps(parsed).strip()}\n</tool_call>\n"
                except:
                    # load it as a dict from string...
                    actual_dict = ast.literal_eval(match)
                    fixed_text += f"<tool_call>\n{json.dumps(actual_dict).strip()}\n</tool_call>\n"
        except:
            return text
        
        return fixed_text

