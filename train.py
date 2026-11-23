#!/usr/bin/env python3
"""Canonical command-line entry point for OncoBERT pretraining."""

from oncobert.training import build_parser, train


if __name__ == "__main__":
    train(build_parser().parse_args())
