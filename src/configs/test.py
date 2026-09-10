"""The smoke-test settings, hardcoded. Loaded when the notebook sets USE_TEST_CONFIG = True.

It overrides Plunkett's defaults with the smallest values that still exercise
every stage: local CPU, the smallest Qwen3, no quantization, LoRA rank 4, ten
training steps with a checkpoint every five, three personas, five training and
two validation trials each, one report sample, and every gate at zero so
nothing blocks. The smoke test is not a result; it proves the plumbing connects.
"""

TEST = {
    "object": {"type": "vector", "attribute_count": 5},
    "decision_rule": {"type": "linear"},
    "model_training": {
        "data_dir": "data/plunkett/",
        "instances": 3,
        "train_trials_per_persona": 5,
        "val_trials_per_persona": 2,
        "seeds": {
            "roles": 0,
            "weights": 1,
            "trials": 2,
            "verification": 5,
            "introspection": 6,
            "reports": 7,
        },
    },
    "model_estimating": {
        "type": "linear",
        "decision_temperature": 0.0,
        "samples_per_trial": 1,
        "estimation_trials_per_persona": 8,
        "chance_draws": 20,
    },
    "report_schema": {
        "type": "auto",
        "report_temperature": 0.0,
        "samples_per_report": 1,
        "max_new_tokens": 80,
    },
    "model_hyperparameters": {
        "model_name": "Qwen/Qwen3-0.6B",
        "quantization": "none",
        "finetune": "lora",
        "lora_rank": 4,
        "lora_alpha": 4,
        "learning_rate": 0.0002,
        "batch_size": 2,
        "loss_mask_answer_only": True,
        "training_steps": 10,
        "checkpoint_every": 5,
        "introspection_training": False,
        "introspection_steps": 4,
        "introspection_batch_size": 1,
        "seed": 4,
    },
    "gates": {"min_decision_accuracy": 0.0, "min_parse_rate": 0.0},
    "compute": {
        "backend": "local",
        "device": "cpu",
        "mixed_precision": "no",
        "eval_batch_size": 4,
        "mount_drive": False,
    },
}
