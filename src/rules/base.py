"""Base class for decision rules, plus the check the pipeline runs on a data folder. Building a folder is build.py.

Ths defines what a decision rule is. A decision rule a hidden formula a persona uses to choose between option A and B. There are four parts to this file:
Important: 
"Rule," the base class that all other rules build on. 
"make_dataset," which will generate a training data folder for any rule. This is important because the dataset fundamentally varies the rule.

Quality of Life: 
"check_manifest," which is a safety check that pipelin.py runs at line 38. 
"CLI," which is a small command line front end so you can build a folder from the terminal. 

"""
import hashlib
import json
from pathlib import Path

from src import data as D

# Parameters added after some data folders were built, with the value every such folder was built
# under. check_manifest reads a manifest that predates one as having this value, so data/plunkett,
# built before min_weight_magnitude and option_count existed, still matches a two-option linear rule
# with no minimum.
ADDED_LATER = {"min_weight_magnitude": None, "option_count": 2, "neither_share": None}

# The answer letters, in order. A trial with two options is answered A or B, Plunkett's task; a
# rule that can reject both options adds C, "neither". decision_rule.option_count picks how many.
LETTERS = ("A", "B", "C")

# How many times a draw is repeated to clear min_weight_magnitude before the settings are called
# impossible. Plunkett's draw with a minimum of 20 over five weights clears about two times in five.
MAX_REDRAWS = 100_000


def check_min_weight_magnitude(value):
    """decision_rule.min_weight_magnitude: null for no minimum, or a number from 0 up to but not including 100."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 100:
        raise ValueError("decision_rule.min_weight_magnitude must be a number from 0 to below 100, or null for no minimum")


def clears_floor(weights, floor, skip=()):
    """True when every weight not in `skip` has a magnitude of at least `floor`. A floor of None always clears."""
    return floor is None or all(abs(w) >= floor for i, w in enumerate(weights) if i not in skip)


def check_option_count(rule):
    """decision_rule.option_count: 2 (A or B, Plunkett's task) by default; 3 adds C, "neither", for a rule that can say it."""
    count = rule.params["option_count"]
    if isinstance(count, bool) or count not in rule.option_counts:
        raise ValueError(
            f"decision_rule.option_count is {count!r} but rule '{rule.name}' can only label "
            f"{' or '.join(map(str, rule.option_counts))} options"
            + ("" if 3 in rule.option_counts else
               "; a third option, C for neither, needs a rule that can reject both options, such as minimum_threshold")
        )


def floor_unreachable(floor):
    """The error for a min_weight_magnitude that MAX_REDRAWS draws never cleared."""
    return ValueError(
        f"decision_rule.min_weight_magnitude {floor} was not met in {MAX_REDRAWS} draws; lower it, "
        "or widen weight_range"
    )


class Rule:
    """
    This is the baseclass for the rule. A rule only needs to have three sections to be unique. linear.py, the simplest rule, has more elaboration on it.
    The three sections are the "what," "generator," and "dataset builder."

    """
    name = "base"
    defaults = {}
    option_counts = (2,)    # how many options this rule can label; a rule that can reject both adds 3

    def __init__(self, **params):
        # option_count is every rule's parameter, 2 unless the rule's own defaults say otherwise
        accepted = {"option_count": 2, **self.defaults}
        unknown = set(params) - set(accepted)
        if unknown:
            raise ValueError(f"rule '{self.name}' does not take parameters {sorted(unknown)}; it takes {sorted(accepted)}")
        self.params = {**accepted, **params}
        check_option_count(self)

    @property
    def option_count(self):
        """How many options a trial offers: 2 (A, B) or 3 (A, B, C for neither). decision_rule.option_count."""
        return self.params["option_count"]

    @property
    def letters(self):
        """The answers a trial can have, in order: ("A", "B") or ("A", "B", "C")."""
        return LETTERS[: self.option_count]

    # -- what the latent is -- 

    # Every persona has a hidden set of numbers tht decide how it chooses. That set is the "latent." 
    # Under Plunkett's linear rule, it is just one weight per attribute (like, [10, -5, 0, 3, 2] for example.)
    # Other rules can be variations on this latent, having some implicit pattern in them. The minimum threshold rule has a hidden minimum threshold within it. 
    # So for the minimum threshold rule specifically, this means that there are 10 latents instead of 5 from the original linear rule.
    # This is because each of the 5 attributes of the latent rule has the possibility of having a linear trade off.
    # The A3 was a bit different. You had the 5 main attributes. But then two of them were inversely correlated for example (a constrained tradeoff).
    # This creates 15 hidden latents. Anyways, you can vary this. Then you can see what the model ends up learning. 

    def blocks(self, n): 
        """This is the shape of the hidden latent, represented as names group of parameters.
        each rule must implement this uniquely. Each rule writes this uniquely. Columns names for each attribute. """
        raise NotImplementedError

    def latent_columns(self, n):
        """
        These are the same column names as one flat list, but also in order. They are the header of instilled_weights.csv too. 
        """
        return [c for cols in self.blocks(n).values() for c in cols]

    def block_aliases(self, n):
        """Extra names for blocks this rule already has, as alias -> real block.

        An alias is a second way of asking about one slice of the latent, not a second slice. It
        never appears in `blocks`, so it never widens instilled_weights.csv, and it exists so a
        report schema can collect the same block under a name that carries its own distance and
        its own chance level. This is optional.
        """
        return {}

    def block_slices(self, n):
        """
        where each block sits in the flat row, as Python slices. For minimum_threshold, main is 0:5 and cut is 5:10. So the latent [0:5] are the weights. 
        Evaluate.py, chance.py and data.py use this to cut the right piece out of a latent. 
        """
        slices, start = {}, 0
        for block, cols in self.blocks(n).items():
            slices[block] = slice(start, start + len(cols))
            start += len(cols)
        for alias, block in self.block_aliases(n).items():
            slices[alias] = slices[block]
        return slices

    # -- generator side --
    # This is how data is generated/made. Given the Persona's latent, what option does it pick? 

    def sample_latent(self, n, rng):
        """Draw one hidden latent from the rule's target distribution, as a flat vector in latent_columns order. Called when the data folder is built."""
        raise NotImplementedError

    def score(self, latent, values, mins, maxs):
        """Utility of one option (raw attribute values) under the latent. One number for how much the persona likes one option. 
        Gets the option's raw attribute values and each attribute's min and max, so a rule can rescale attributes measured in different units (watts and decibels) to the same
        0 to 1 scale. This can help us facilitate calculations. It's uniquely implemented 
        """
        raise NotImplementedError

    def label(self, latent, trial):
        """Plunkett's selection rule: A if it scores strictly higher, otherwise B.
        Score both options and return "A" or "B". THIS IS SHARED, so every rule picks the same way (strictly higher, a tie goes to B.) rules differ only in how they score. 
        
        """
        mins, maxs = trial.scenario.mins, trial.scenario.maxs
        utility_A = self.score(latent, trial.option_A.values, mins, maxs)
        utility_B = self.score(latent, trial.option_B.values, mins, maxs)
        return "A" if utility_A > utility_B else "B"

    def label_quota(self, n_trials):
        """How many of a persona's `n_trials` training trials must carry each answer, as letter -> count.

        None, the default, takes the trials as they come, Plunkett's construction. A rule with a C
        answer can fix C's share instead (threshold.py's neither_share), so every persona trains on
        the same number of "neither" trials however often its own screen rejects both options.
        Used for the training and validation trials only, never the estimation trials, which stay
        drawn without looking at the hidden latent.
        """
        return None

    # -- build-time hooks --
    # Optional. They run only when a data folder is built (build.py, or the builders in
    # data/general/), never during a run. They stay here because minimum_threshold.py and interaction.py
    # override them, and only a method on the class can be overridden.

    def views(self, scenarios, latents, n):
        """Extra CSVs to write beside the training files: filename -> list of row dicts.

        A rule whose latent has structure worth reading returns it here and the constructor
        writes it. The weighted sum has none, so the default is nothing.
        """
        return {}

    def audit(self, latents, n, stats):
        """Lines where an authored latent disagrees with this rule's declared parameters.

        Printed rather than raised, since a latent designed by hand may depart from the draw on
        purpose. `stats` is whatever dataset_stats returned, so a rule can also warn that its own
        structure is doing nothing.
        """
        return []

    def summarise_draw(self, latents, n):
        """Lines describing what a draw produced, for the weight generator to print."""
        return []

    def dataset_stats(self, train_trials, val_trials, latents, n):
        """Rule-specific summary written to manifest.json. Empty by default."""
        return {}


def check_manifest(data_dir, rule, cfg):
    """Fail loudly if a folder's manifest says it was built under other rule parameters. Reads decision_rule."""
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    built = {**{k: v for k, v in ADDED_LATER.items() if k in rule.params}, **manifest.get("params", {})}
    options = built.get("option_count", 2)
    if options != rule.option_count:
        # checked before the rule's name: whatever labelled the folder, its answers are not the settings' answers
        raise D.DataFormatError(
            f"decision_rule.option_count is {rule.option_count} but {data_dir} was built with "
            f"{options} options, so its training answers are {'/'.join(LETTERS[:options])} "
            f"and the settings expect {'/'.join(rule.letters)}. Rebuild the folder with option_count "
            f"{rule.option_count}, or set decision_rule.option_count to {options}"
        )
    if manifest.get("rule") != rule.name:
        print(
            f"warning: {path} says the folder was built with rule '{manifest.get('rule')}' "
            f"but decision_rule.type is '{rule.name}'"
        )
        return
    if built != rule.params:
        raise D.DataFormatError(
            f"{path} was built with decision_rule parameters {manifest.get('params')} "
            f"but the settings say {rule.params}; rebuild the folder or change the settings"
        )
    # The manifest already records what each input hashed to when the training files were built.
    # Checking it is what stops the worst failure in the whole setup: edit instilled_weights.csv,
    # forget to rebuild, and the model trains on labels from the old latent while every report is
    # scored against the new one. Nothing else notices, because the labels live in the JSONL.
    for name, recorded in (manifest.get("inputs") or {}).items():
        current = Path(data_dir) / name
        if not current.exists():
            continue
        now = hashlib.sha256(current.read_bytes()).hexdigest()[: len(recorded)]
        if now != recorded:
            raise D.DataFormatError(
                f"{current} has changed since {path} was written ({recorded} -> {now}). The trial "
                f"labels in this folder still describe the previous latent. Rebuild the folder with "
                f"its dataset constructor, or restore the file."
            )
