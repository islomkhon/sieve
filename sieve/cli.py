"""Command-line entry point for the Sieve demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import Interpreter
from .memory import LearningMemory


DEMO_LINES = [
    "put the cup on the shelf",
    "why",
    "the red one",
    "where is the red cup",
    "wht color is the red cup",
    "what is on the table",
    "is the fridge open",
    "open the fridge",
    "is the fridge open",
    "what color is the apple",
    "the red one",
    "put the cup in the box",
    "the red one",
    "the blue one",
    "what is in the blue box",
    "what color is the car",
    "call the red box a car",
    "what color is the car",
    "define serendipity",
    "define elephnat",
    "what is an elephant",
    "what is the red cup",
    "show me everything in the fridge",
    "how many cups are there on the table",
    "what is the size of the red cup",
    "put the red cup on the shelf",
    "where is it",
    "put it on the table",
    "put it back",
    "what did you do",
    "what about the blue cup",
    "where did you put it",
    "where is it",
    "what color is the red cup",
    "wht abut the blu one",
    "the other one",
    "context",
    "do it again",
]


def run_line(interpreter: Interpreter, line: str) -> None:
    result = interpreter.handle(line)
    print(f"you> {line}")
    print(f"sieve> {result.text}")
    print()


def run_demo() -> None:
    print("Sieve demo: meaning is what survives constraints.")
    print("No model, no training, no embeddings.\n")
    interpreter = Interpreter()
    for line in DEMO_LINES:
        run_line(interpreter, line)


def run_repl(interpreter: Interpreter, memory_path: Path | None = None) -> None:
    print("Sieve demo. Type 'help' for examples, 'quit' to exit.")
    print("No model, no training, no embeddings.\n")
    if memory_path is not None:
        print(f"Learning memory: {memory_path}\n")
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in {"quit", "exit"}:
            break
        result = interpreter.handle(line)
        print(f"sieve> {result.text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sieve language demo.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run a scripted transcript instead of an interactive session",
    )
    parser.add_argument(
        "--memory",
        type=Path,
        default=None,
        help="persistent learning-memory file (default: ~/.sieve/memory.json)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="do not load or save learned knowledge",
    )
    parser.add_argument(
        "--forget-learning",
        action="store_true",
        help="erase learned rules before starting",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        if args.no_memory:
            run_repl(Interpreter())
        else:
            path = args.memory or LearningMemory.default_path()
            memory = LearningMemory(path=path)
            memory.load()
            if args.forget_learning:
                memory.clear()
            run_repl(Interpreter(memory=memory), path)


if __name__ == "__main__":
    main()
