"""Labelled trials for one persona at a time, for the tests that need choices without a dataset.

Moved here from Rule in src/rules/base.py, where only tests/test_estimator_recovery.py used it.

It makes random A/B questions for one persona and labels each one. A real build uses
data.generate_decision_trials instead, which does the same for every persona from one RNG, as
Plunkett did, so his files come out exactly. Nothing in a run calls this.
"""
from src import data as D


def generate_trials(rule, latent, scenario, n_trials, rng):
    """`n_trials` labelled trials for one persona from `rng`. Or more siply explained, it will make random A/B qustions for one persona and label each one.
    The tests use it.

    Trial: A trial is one A/B question that a Persona is asked. it has a scenario (like choosing a tea), two options, and a corect answer.
    A:
    steeping time: 3 mins
    temperature: 90 degrees
    etc.

    B:
    steeping time: 5 mins
    temperature: 80 degrees
    etc.

    The label is the correct answer, which is determined by the hidden latent.
    The latent is the hidden formula that the persona uses to pick between A and B.
    During training, the model sees the question and learns to give that answer.

    Generate trials makes n_trials blank questions. Each D.Trial (scenario, rng) creates optionA, then option B. For
    each attribute, an option draws a random value within that attribute's range (suction from 150 to 250)
    At that point, the trial has no answer. Label is None (recall, label is the answer!)

    self.label(latent, t) is called to label the trial. It scores both options and returns "A" or "B" as the correct answer, with the persona's hidden weights.
    then the list is returned, of answered questions.

    So given one persona's hidden weights, it gives you n random questions together with the answers that persona would give.

    This is one of three places that make trials. This is only used for tests/test_estimator_recovry.py. This makes questions for
    one persona t a time.



    """
    trials = [D.Trial(scenario, rng) for _ in range(n_trials)]
    for t in trials:
        t.label = rule.label(latent, t)
    return trials
