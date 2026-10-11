# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared train/evaluation instructions and Qwen3 continuation boundaries."""

from copy import deepcopy

TOOL_INSTRUCTION = "Please call given tools to answer the question. Please note that all your information must be obtained by calling tools and not by answering the question directly.\nIf the call fails, you need to try to correct it and continue until you arrive at an answer."
ANSWER_INSTRUCTION = " Only output the final answer (in words, numbers or phrase) inside the <answer></answer> tag, without any explanations or extra information."
CONCISE_INSTRUCTION = "Before each tool call, briefly reason about the next useful action, then issue the call in the required tool-call format. Keep this reasoning concise and avoid repeating prior analysis."


def student_messages(messages, style="upstream"):
    """Change only the known instruction prefix, preserving the exact question."""
    result = deepcopy(messages)
    if style == "upstream":
        return result
    if style != "concise_shared_v1":
        raise ValueError(f"Unknown FTRL prompt style: {style}")
    if len(result) != 1 or result[0]["role"] != "user":
        raise ValueError("Expected the pinned FTRL single-user task message")
    prefix, separator, question = result[0]["content"].partition("\nQuestion: ")
    if not separator or prefix not in (TOOL_INSTRUCTION, TOOL_INSTRUCTION + ANSWER_INSTRUCTION):
        raise ValueError("Unexpected FTRL instruction prefix; refusing to rewrite the question")
    result[0]["content"] = TOOL_INSTRUCTION + "\n" + CONCISE_INSTRUCTION + "\n" + ANSWER_INSTRUCTION.lstrip() + "\nQuestion: " + question
    return result


def assistant_generation_prefix(enable_thinking):
    """Match the pinned Qwen3 chat template at every fresh assistant turn."""
    suffix = "<|im_start|>assistant\n"
    if not enable_thinking:
        suffix += "<think>\n\n</think>\n\n"
    return suffix


def tool_feedback(outputs, ended_with_eos, enable_thinking):
    """Keep full tool observations and close native thinking on every new turn."""
    feedback = ("\n" if ended_with_eos else "<|im_end|>\n") + "<|im_start|>user\n"
    feedback += "".join("<tool_response>\n" + text + "\n</tool_response>\n" for text in outputs)
    return feedback + "<|im_end|>\n" + assistant_generation_prefix(enable_thinking)
