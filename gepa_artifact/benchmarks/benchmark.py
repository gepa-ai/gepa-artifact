from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import dspy
from typing import Callable, List, Type

dataset_size = {"full": None, "lite": 500, "tiny": 200, "test": 50}

class Benchmark(ABC):
    def __init__(self, dataset_mode="lite"):
        # dataset for training and validation
        self.dataset = None
        # dataset for the actual benchmarking
        self.train_set = None
        self.test_set = None
        self.val_set = None

        self.init_dataset()
        assert self.dataset is not None, "Dataset not initialized"
        self.max_testset_size = dataset_size[dataset_mode]

        # TODO: FIXME: "test" option is for debugging purposes only, should be removed for final release
        if dataset_mode == "test":
            self.dataset = self.trim_dataset(self.dataset, 60)
            self.create_splits()

        if not self.train_set or not self.test_set or not self.val_set:
            self.create_splits()

        self.train_set = self.trim_dataset(self.train_set, 150)
        self.test_set = self.trim_dataset(self.test_set, 300)
        self.val_set = self.trim_dataset(self.val_set, 300)

        assert self.train_set is not None, "Train set not initialized"
        assert self.test_set is not None, "Dev set not initialized"
        assert self.val_set is not None, "Val set not initialized"

    @abstractmethod
    def init_dataset(self) -> None:
        """
        Initializes the dataset for the benchmark, and sets it to self.dataset.
        Each element in the dataset should be an instance of dspy.Example.
        """
        return

    def trim_dataset(self, dataset, size: int) -> None:
        if size is None or size >= len(dataset):
            return dataset
        rng = random.Random()
        rng.seed(1)
        return rng.sample(dataset, size)

    def create_splits(self) -> None:
        """
        Creates the splits for the dataset (not including test).
        Upon completion, self.train_set, self.test_set, and self.val_set should be set.
        """

        total_len = len(self.dataset)
        self.test_set = self.dataset[: int(0.4 * total_len)]
        self.val_set = self.dataset[int(0.4 * total_len) : int(0.8 * total_len)]
        self.train_set = self.dataset[int(0.8 * total_len) :]

    def get_dataset(self):
        return self.dataset

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set


@dataclass
class BenchmarkMeta:
    benchmark: Type[Benchmark]
    program: List[dspy.Module]
    metric: Callable
    dataset_mode: str = "lite"
    # BenchmarkMeta.num_threads has higher priority than run time argument of num_threads
    # use this as an upper bound for the number of threads to use
    num_threads: int = None
    name: str = None
    metric_with_feedback: Callable = None
    feedback_fn_maps: list[dict] = None

@dataclass
class EvaluationResult:
    benchmark: str
    program: str

    score: float = None
    cost: float = None
    input_tokens: int = None
    output_tokens: int = None

    optimizer: str = None
    optimized_program: dspy.Module = None
    optimizer_input_tokens: int = None
    optimizer_output_tokens: int = None
    optimizer_cost: float = None

    # Teacher LM cost (e.g., Claude for prompt generation)
    teacher_lm_cost: float = None
    teacher_lm_input_tokens: int = None
    teacher_lm_output_tokens: int = None

    optimizer_program_scores: list[float] = None
