"""
External API should only call preprocess, but preprocess is composed of steps of preprocessing
"""


def preprocess_step_1(): ...


def preprocess_step_2(): ...


def preprocess(): ...
