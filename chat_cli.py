import logging
from prompt_toolkit import PromptSession
from chat import (
    AssistantMessage,
    Chat,
    ChatParameters,
    ContentTooLargeError,
    UserMessage,
)
from argparse import ArgumentParser, Namespace


def main(args: Namespace) -> None:
    messages = []
    prompt_session = PromptSession()

    print("Please ask AI for any questions.")

    while True:
        user_input = prompt_session.prompt("> ", multiline=args.multiline, vi_mode=True)
        messages.append(UserMessage(user_input.strip()))

        while True:
            try:
                response = Chat(
                    messages,
                    ChatParameters(model=args.model, temperature=args.temperature),
                ).execute()
            except ContentTooLargeError:
                logger.info("ContentTooLargeError: messages are going to be reduced.")
                prev_size = len(messages)
                if prev_size > 1:
                    next_size = prev_size // 2
                    messages = messages[0:next_size]
                else:
                    next_size = 0
                    messages = []

                logger.info(f"reduced messages list items: {prev_size} -> {next_size}")
                continue
            break

        print(response)
        messages.append(AssistantMessage(response))


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    parser = ArgumentParser()
    parser.add_argument("--multiline", action="store_true")
    parser.add_argument(
        "--loglevel", choices=["INFO", "WARN", "ERROR"], type=str, default="INFO"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4", "gpt-4-0314"],
        default="gpt-3.5-turbo",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    main(args)
