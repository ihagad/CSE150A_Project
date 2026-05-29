from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


STATE_NAMES = ["junior", "mid_level", "senior"]
OBSERVATION_COLUMNS = [
    "DevType",
    "AISelect",
    "YearsCode",
    "CodingActivities",
    "ConvertedCompYearly",
]
SENIORITY_ORDER = {"low": 0, "medium": 1, "high": 2}
SENIORITY_LABELS = {"low": "junior", "medium": "mid_level", "high": "senior"}


@dataclass
class HMMResult:
    pi: np.ndarray
    transition: np.ndarray
    emission: np.ndarray
    log_likelihoods: list[float]
    observations: np.ndarray
    symbols: list[str]
    true_states: np.ndarray
    decoded_states: np.ndarray
    accuracy: float
    mapping: dict[int, int]
    final_log_likelihood: float
    log_likelihood_per_step: float


class DiscreteHMM:
    """Discrete HMM trained with Baum-Welch EM and scaled inference."""

    def __init__(
        self,
        n_states: int,
        n_observations: int,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
        smoothing: float = 1e-6,
    ) -> None:
        self.n_states = n_states
        self.n_observations = n_observations
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.smoothing = smoothing
        self.pi: np.ndarray | None = None
        self.transition: np.ndarray | None = None
        self.emission: np.ndarray | None = None
        self.log_likelihoods: list[float] = []

    def _initialize(self) -> None:
        rng = np.random.default_rng(self.random_state)
        self.pi = rng.dirichlet(np.ones(self.n_states))
        self.transition = rng.dirichlet(np.ones(self.n_states), size=self.n_states)
        self.emission = rng.dirichlet(np.ones(self.n_observations), size=self.n_states)

    def _forward(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        alpha = np.zeros((len(observations), self.n_states))
        scales = np.zeros(len(observations))

        alpha[0] = self.pi * self.emission[:, observations[0]]
        scales[0] = alpha[0].sum()
        alpha[0] /= scales[0]

        for t in range(1, len(observations)):
            alpha[t] = alpha[t - 1] @ self.transition * self.emission[:, observations[t]]
            scales[t] = alpha[t].sum()
            alpha[t] /= scales[t]

        return alpha, scales, float(np.log(scales).sum())

    def _backward(self, observations: np.ndarray, scales: np.ndarray) -> np.ndarray:
        beta = np.zeros((len(observations), self.n_states))
        beta[-1] = 1.0

        for t in range(len(observations) - 2, -1, -1):
            beta[t] = (
                self.transition
                @ (self.emission[:, observations[t + 1]] * beta[t + 1])
            )
            beta[t] /= scales[t + 1]

        return beta

    def fit(self, observations: np.ndarray) -> "DiscreteHMM":
        observations = np.asarray(observations, dtype=int)
        self._initialize()

        for _ in range(self.max_iter):
            alpha, scales, log_likelihood = self._forward(observations)
            beta = self._backward(observations, scales)

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            xi_sum = np.zeros((self.n_states, self.n_states))
            for t in range(len(observations) - 1):
                xi = (
                    alpha[t, :, None]
                    * self.transition
                    * self.emission[:, observations[t + 1]][None, :]
                    * beta[t + 1][None, :]
                )
                xi /= xi.sum()
                xi_sum += xi

            self.pi = gamma[0] + self.smoothing
            self.pi /= self.pi.sum()

            self.transition = xi_sum + self.smoothing
            self.transition /= self.transition.sum(axis=1, keepdims=True)

            self.emission = np.full(
                (self.n_states, self.n_observations), self.smoothing
            )
            for symbol in range(self.n_observations):
                self.emission[:, symbol] += gamma[observations == symbol].sum(axis=0)
            self.emission /= self.emission.sum(axis=1, keepdims=True)

            self.log_likelihoods.append(log_likelihood)
            if (
                len(self.log_likelihoods) > 1
                and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol
            ):
                break

        return self

    def score(self, observations: np.ndarray) -> float:
        _, _, log_likelihood = self._forward(np.asarray(observations, dtype=int))
        return log_likelihood

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=int)
        log_pi = np.log(self.pi)
        log_transition = np.log(self.transition)
        log_emission = np.log(self.emission)

        delta = np.zeros((len(observations), self.n_states))
        psi = np.zeros((len(observations), self.n_states), dtype=int)
        delta[0] = log_pi + log_emission[:, observations[0]]

        for t in range(1, len(observations)):
            scores = delta[t - 1, :, None] + log_transition
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_emission[:, observations[t]]

        path = np.zeros(len(observations), dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(len(observations) - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path


def load_hmm_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = (
            Path(__file__).resolve().parents[1]
            / "cleaned_data"
            / "cleaned_stackoverflow_bn_data.csv"
        )
    data = pd.read_csv(path)
    data = data.copy()
    data["seniority_label"] = data["YearsCodePro"].map(SENIORITY_LABELS)
    data["seniority_order"] = data["YearsCodePro"].map(SENIORITY_ORDER)
    data["years_code_order"] = data["YearsCode"].map(SENIORITY_ORDER)
    data = data.sort_values(["seniority_order", "years_code_order"]).reset_index(drop=True)
    return data


def encode_observations(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    symbols = data[OBSERVATION_COLUMNS].astype(str).agg("|".join, axis=1)
    codes, uniques = pd.factorize(symbols, sort=True)
    return codes.astype(int), list(uniques)


def align_states(decoded: np.ndarray, true_states: np.ndarray) -> tuple[np.ndarray, dict[int, int], float]:
    n_states = len(STATE_NAMES)
    confusion = np.zeros((n_states, n_states), dtype=int)
    for predicted, actual in zip(decoded, true_states):
        confusion[predicted, actual] += 1

    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    aligned = np.array([mapping[state] for state in decoded])
    accuracy = float((aligned == true_states).mean())
    return aligned, mapping, accuracy


def reorder_parameters(
    pi: np.ndarray, transition: np.ndarray, emission: np.ndarray, mapping: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse_mapping = {new_state: old_state for old_state, new_state in mapping.items()}
    order = [inverse_mapping[state] for state in range(len(STATE_NAMES))]
    return pi[order], transition[np.ix_(order, order)], emission[order]


def run_hmm(
    path: str | None = None,
    random_state: int = 42,
    max_iter: int = 100,
    n_restarts: int = 10,
) -> HMMResult:
    data = load_hmm_data(path)
    observations, symbols = encode_observations(data)
    true_states = data["seniority_order"].to_numpy(dtype=int)

    best_hmm: DiscreteHMM | None = None
    best_log_likelihood = -np.inf
    for offset in range(n_restarts):
        hmm = DiscreteHMM(
            n_states=len(STATE_NAMES),
            n_observations=len(symbols),
            random_state=random_state + offset,
            max_iter=max_iter,
        ).fit(observations)
        log_likelihood = hmm.score(observations)
        if log_likelihood > best_log_likelihood:
            best_hmm = hmm
            best_log_likelihood = log_likelihood

    assert best_hmm is not None
    hmm = best_hmm

    decoded = hmm.viterbi(observations)
    aligned, mapping, accuracy = align_states(decoded, true_states)
    pi, transition, emission = reorder_parameters(
        hmm.pi, hmm.transition, hmm.emission, mapping
    )
    final_log_likelihood = hmm.score(observations)

    return HMMResult(
        pi=pi,
        transition=transition,
        emission=emission,
        log_likelihoods=hmm.log_likelihoods,
        observations=observations,
        symbols=symbols,
        true_states=true_states,
        decoded_states=aligned,
        accuracy=accuracy,
        mapping=mapping,
        final_log_likelihood=final_log_likelihood,
        log_likelihood_per_step=final_log_likelihood / len(observations),
    )


if __name__ == "__main__":
    result = run_hmm()
    print("Initial state distribution (pi):")
    print(pd.Series(result.pi, index=STATE_NAMES).round(4))
    print("\nTransition matrix (A):")
    print(pd.DataFrame(result.transition, index=STATE_NAMES, columns=STATE_NAMES).round(4))
    print("\nEmission matrix shape (B):", result.emission.shape)
    print("Observation symbols:", len(result.symbols))
    print("EM iterations:", len(result.log_likelihoods))
    print("Final log-likelihood:", round(result.final_log_likelihood, 4))
    print("Log-likelihood per step:", round(result.log_likelihood_per_step, 4))
    print("Viterbi seniority accuracy:", round(result.accuracy, 4))
