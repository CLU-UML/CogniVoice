import json
import argparse
import dataclasses
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments, HfArgumentParser


@dataclass
class AudioTrainingArguments(TrainingArguments):
    method: str = field(
        default='openai/whisper-small', metadata={"help": "Training method"}
    )
    sample_rate: int = field(
        default=16000, metadata={"help": "Audio sampling rate"}
    )
    task: str = field(
        default='cls', metadata={"help": "Prediction task, one of cls and reg."}
    )
    dropout: float = field(
        default=0.2, metadata={"help": "Dropout rate."}
    )
    use_disvoice: bool = field(
        default=False, metadata={"help": "Whether to use disvoice features."}
    )
    use_metadata: bool = field(
        default=False, metadata={"help": "Whether to patient metadata."}
    )
    use_text: bool = field(
        default=False, metadata={"help": "Whether to use transcribed text."}
    )
    use_llama2: bool = field(
        default=False, metadata={"help": "Whether to use LLaMA-2 explanations."}
    )
    use_poe: bool = field(
        default=False, metadata={"help": "Whether to use PoE."}
    )
    num_fold: int = field(
        default=10, metadata={"help": "number of folds in cross-validation"}
    )
    max_length: int = field(
        default=2e7, metadata={"help": "max length"}
    )
    poe_alpha: float = field(
        default=1, metadata={"help": "alpha to control debiasing strength in PoE"}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        # if self.dataset is not None:
        #     self.dataset = self.dataset.lower()


class RemainArgHfArgumentParser(HfArgumentParser):
    def parse_json_file(self, json_file: str, return_remaining_args=True):
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: data.pop(k) for k in list(data.keys()) if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)
        
        remain_args = argparse.ArgumentParser()
        remain_args.__dict__.update(data)
        if return_remaining_args:
            return (*outputs, remain_args)
        else:
            return (*outputs,)
