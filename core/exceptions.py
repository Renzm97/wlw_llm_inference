# -*- coding: utf-8 -*-
"""推理相关异常定义。"""


class LLMInferenceError(Exception):
    pass


class EngineNotInstalledError(LLMInferenceError):
    pass


class EngineNotRunningError(LLMInferenceError):
    pass


class ModelNotFoundError(LLMInferenceError):
    pass


class InvalidParameterError(LLMInferenceError):
    pass


class StructuredOutputNotSupportedError(LLMInferenceError):
    pass
