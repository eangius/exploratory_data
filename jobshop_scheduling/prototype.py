#!usr/bin/env python

# AUTHOR: Elian Angius
# ABOUT: Dynamic programming approach & exploration environment
# to a variant of the np-hard job-cut scheduling problem.

from functools import *
from collections import *
from typing import *
import random
import math


# ABOUT: helper immutable list-like data type to convert & validate
# from string representation
class Job:
    def __init__(self, raw: str):
        self.parts = sorted(int(part) for part in raw.split('-'))  # normalize irrelevant cut order
        assert \
            all(part > 0 for part in self.parts), \
            "Invalid cut lengths"

    def __repr__(self) -> str:
        return '-'.join([str(part) for part in self.parts])

    # largest sized part.
    @cached_property
    def largest(self) -> int:
        return max(self.parts)

    # number of total parts.
    @cached_property
    def size(self) -> int:
        return len(self.parts)

    # total length of parts
    @cached_property
    def length(self) -> int:
        return sum(self.parts)

    # distribution of parts-sizes
    @cached_property
    def frequency(self) -> Counter:
        return Counter(self.parts)


# ABOUT: helper data type to track snapshots of current new_state as distributions of
# remaining, incomplete & completed parts-of-jobs.
class ProcessingState:
    def __init__(
        self,
        remaining=None,              # tracks pending job-parts
        incomplete: Counter = None,  # tracks processed but not yet finished
        completed: Counter = None,   # tracks fully finished job-parts
    ):
        self.remaining = remaining or Counter()
        self.incomplete = incomplete or Counter()
        self.completed = completed or Counter()

    def copy(self):
        return ProcessingState(
            remaining=self.remaining.copy(),
            incomplete=self.incomplete.copy(),
            completed=self.completed.copy()
        )


# ABOUT: implementing some job selection strategies to experiment with.
# Interface to all these functions returns a the relative wight corresponding
# to a job that caused (simulated) processing state change from old to new.

# Dummy baseline selection that randomly picks job ordering.
def strategy_rnd_order(job: Job, old_state: ProcessingState, new_state: ProcessingState) -> float:
    return random.random()


# Prefer jobs that minimize net effect on storage used via combination of:
# [1] maximizing stacking of job parts
# [2] minimizing coverage of large job-parts
# [3] maximizing part-type completion via bundling & freeing up space.
def strategy_min_net_space(job: Job, old_state: ProcessingState, new_state: ProcessingState) -> float:
    old_metric = sum(old_state.incomplete.keys())
    new_metric = sum(new_state.incomplete.keys())
    return 1 / (1 + new_metric - old_metric)


# Prefer jobs who's largest part is relatively smaller than the global workload
# largest part as to maximize remaining available space for future jobs. Given that
# ideal answer could be achieved in no better than the largest global piece, this
# strategy strives for this.
def strategy_fraction(job: Job, old_state: ProcessingState, new_state: ProcessingState) -> float:
    total_state = old_state.remaining + old_state.incomplete + old_state.completed
    return job.largest / max(total_state.keys())


# TODO: (other strategies)
# [1] incorporate tie breaking conditions..
# [2] strategies that factor in frequency of remaining parts.
# [3] explore novel weighted job-averaging distances:
#     weighted_jaccard(job, workload)
#     tfidf(job, workload)


# ABOUT: Optimizer class.
class Scheduler:

    # instantiate algorithm with hyper parameters.
    def __init__(
        self,
        fn_strategy: Callable = strategy_min_net_space,
        lookahead: int = 0,  # number of forward looking brute force simulation steps (0= no effect)
        verbose: int = 0,    # troubleshooting degree
    ):
        self.fn_strategy = fn_strategy
        self.lookahead = lookahead  # TODO: implement this!
        self.verbose = verbose
        return

    # iteratively process jobs. returns optimal schedule & largest distance needed.
    def run(self, jobs: List[Job]) -> Tuple[List[Job], int]:
        results = []

        def print_state():
            if self.verbose > 0:
                print("\n".join([
                    f"scheduling task{iteration}:",
                    f"\tthis/worst metric = {this_metric} / {worst_metric}",
                    f"\tjob               = {next_job}",
                    f"\tremaining         = {dict(state.remaining)}",
                    f"\tincomplete        = {dict(state.incomplete)}",
                    f"\tcomplete          = {dict(state.completed)}",
                    f""
                ]))

        # histogram new_state of workload job-parts
        state = ProcessingState(
            remaining=reduce(lambda ctr, job: ctr + job.frequency, jobs, Counter()),
        )

        # initialize other variables
        iteration = 0
        this_metric = '???'
        worst_metric = '???'
        next_job = '???'
        print_state()

        while len(jobs) > 0:
            weights = self._rank(jobs, state)
            if self.verbose > 0:
                print(f"prioritized jobs:")
                for i, job in enumerate(jobs):
                    print(f"\t{weights[i]:.5f} -> {job}")
                print("")

            # remove best (greedy) schedulable job from the queue
            jobs = [job for _, job in sorted(zip(weights, jobs), key=lambda pair: pair[0])]
            next_job = jobs.pop()
            results.append(next_job)

            # process job & measure space
            state, this_metric = self._simulate_processing(next_job, state)
            worst_metric = this_metric if type(worst_metric) == str or worst_metric < this_metric else worst_metric

            iteration += 1
            print_state()

        return results, worst_metric

    # Simulates processing a job (with parts assumed to exist in the state.remaining)
    # & returns the would-have been change of state with its metric.
    @staticmethod
    def _simulate_processing(job: Job, state: ProcessingState) -> Tuple[ProcessingState, int]:
        new_state = state.copy()

        # move processed job parts into buffer space (before bundling)
        new_state.remaining -= job.frequency
        new_state.incomplete += job.frequency

        # compute current & overall buffer space used (before bundling)
        this_metric = sum(new_state.incomplete.keys())

        # move parts out of the buffer space when all quantities been processed (post bundling)
        total_state = state.remaining + state.incomplete + state.completed
        new_state.completed += Counter({
            k: v
            for k, v in new_state.incomplete.items()
            if total_state[k] == v
        })
        new_state.incomplete -= new_state.completed
        return new_state, this_metric

    # Heuristically prioritizes jobs based on current processing state.
    def _rank(self, jobs: List[Job], state: ProcessingState) -> List[float]:
        weights = []
        for job in jobs:
            new_state, _ = self._simulate_processing(job, state)
            weights.append(self.fn_strategy(job, state, new_state))

        return weights


'''
# Ad-hoc console testing
from updata.assignment import *

# Example0 data (my toy sanity)
# jobs = [Job(spec) for spec in [
#     "1", "2-1", "4-3", "3", "1-2-2", "1-1"
# ]]

# Example1 data
jobs = [Job(spec) for spec in [
    "1250-1250-3000-5000",
    "1250-1250-3000-3000-2000",
    "2500-2500-2500",
    "3500-2500-5000",
    "7500-3500-5000",
    "7500-3500-3000-1000-1000",
]]

# Example2 data. note specs of jobs at index 0, 4 & 8 were ambiguous! Assumed missing dashes.
# jobs = [Job(spec) for spec in [
#     "5867-5867-5867",
#     "5767-3331-3331-2172-3331",
#     "3331-5767-2172-2172-2172-2172-8992-3331-3331-2172-8992-8992-16-5767-5767-5767-699-5767-5767-5767-699-5767-5767-5767-699-8992-2337-1892-1892-1419-1419-8992-8992-16",  # spec corrected!
#     "8992-8992-16",
#     "8992-3035-2070-3035-3035-2337-2070-2070-2070-2070-1419-1419-1419-5867-5867-5867-399",  # spec corrected!
#     "5867-5867-5867-399",
#     "6350-6350-5300",
#     "6350-6350-5300",
#     "8992-8992-16-2070-2070-2070-2070-2070-2070-1419-1419-1419-1323-8992-8992-16",  # spec corrected!
#     "5867-5867-5867-399",
#     "5867-5867-5867-399",
#     "5867-5867-5867-399",
#     "6350-6350-4829-471",
#     "6350-6350-4829-471",
#     "8992-8992-16",
# ]]


# Baseline
n = len(jobs)
baseline_size = sum(
    Scheduler(fn_strategy=strategy_rnd_order).run(jobs)[-1]  # take metric only
    for _ in range(n)
)/n


# Treatment
job_ordering, treatment_size = Scheduler(
    fn_strategy=strategy_min_net_space,
    verbose=1, 
).run(jobs)


# Report
print(f"cost of treatment={treatment_size} vs avg baseline={baseline_size}")
for i, job in enumerate(job_ordering):
    print(f"job[{i}] = {job}")

# Tests
assert treatment_size <= baseline_size      # better than random
assert set(job_ordering) == set(jobs)       # no jobs lost, added or morphed
'''

# TODO: (delivery)
# [1] Add proper unit & stress tests.
# [2] Benchmark strategy correctness & memory/time performance.
# [3] Better report & proof of solution optimality.
# [4] Optimize code object creation & re-calculation with libs (ie: numpy)
