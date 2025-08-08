# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from typing import TYPE_CHECKING, List, Tuple, Union

import json

from .base import BaseAgentTemplate

if TYPE_CHECKING:
    from swift.llm.infer import Function
    from swift.llm.template import Prompt

from swift.utils import get_logger

logger = get_logger()

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

