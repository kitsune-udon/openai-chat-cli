import logging
import os
from typing import Callable, Dict, List, Optional

import openai
import tiktoken

logger = logging.getLogger(__name__)


class Message:
    ROLE_API: Optional[str] = None
    ROLE_MODEL: Optional[str] = None

    def __init__(self, content: str):
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.ROLE_API, "content": self.content}


class SystemMessage(Message):
    ROLE_API = "system"
    ROLE_MODEL = "System"


class UserMessage(Message):
    ROLE_API = "user"
    ROLE_MODEL = "Human"


class AssistantMessage(Message):
    ROLE_API = "assistant"
    ROLE_MODEL = "AI"


class ChatParameters:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: str | List[str] | None = None,
        max_tokens: int | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: float | None = None,
        user: str | None = None,
    ):
        self.params = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1,
            "stream": False,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        if stop is not None:
            self.params["stop"] = stop
        if max_tokens is not None:
            self.params["max_tokens"] = max_tokens
        if logit_bias is not None:
            self.params["logit_bias"] = logit_bias
        if user is not None:
            self.params["user"] = user

    def to_dict(self) -> dict:
        return self.params


class ChatError(Exception):
    pass


class ContentTooLargeError(ChatError):
    pass


class InvalidApiKeyError(ChatError):
    pass


class InvalidCallError(ChatError):
    pass


MessageListFormatterType = Callable[[List[Message]], List[Dict[str, str]]]


class Chat:
    def __init__(
        self,
        messages: List[Message],
        parameters: ChatParameters | None = None,
        messages_formatter: MessageListFormatterType | None = None,
    ):
        self.messages = messages
        self.parameters = parameters
        self._response = None
        self._messages_formatter = (
            self.__class__.api_format_of_messages
            if messages_formatter is None
            else messages_formatter
        )

    def _get_content(self) -> str:
        if self._response is None:
            raise InvalidCallError
        else:
            return self._response["choices"][0]["message"]["content"]

    @staticmethod
    def api_format_of_messages(messages: List[Message]) -> List[Dict[str, str]]:
        return [m.to_dict() for m in messages]

    def execute(self) -> str:
        if self._response is not None:
            return self.get_content()

        params_dict = (
            ChatParameters().to_dict()
            if self.parameters is None
            else self.parameters.to_dict()
        )
        params_dict["messages"] = self._messages_formatter(self.messages)
        openai.api_key = os.environ["OPENAI_API_KEY"]
        try:
            self._response = openai.ChatCompletion.create(**params_dict)
        except openai.error.InvalidRequestError as e:
            if "reduce the length of the messages" in str(e):
                raise ContentTooLargeError from e
            else:
                raise e
        except openai.error.AuthenticationError as e:
            if "Incorrect API key provided" in str(e):
                error_message = "Set the correct API key to the OPENAI_API_KEY environment variable."
                raise InvalidApiKeyError(error_message) from e
            else:
                raise e

        return self._get_content()


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        logger.warning(
            "gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        logger.warning(
            "gpt-4 may change over time. Returning num tokens assuming gpt-4-0314."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
