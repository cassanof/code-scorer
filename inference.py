from typing import List, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CodeScorer:
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device

    def score(self, codes: Union[str, List[str]]) -> List[float]:
        if isinstance(codes, str):
            codes = [codes]
        if len(codes) == 0:
            return []
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                codes,
                truncation=True,
                padding=True,
                max_length=self.model.config.max_position_embeddings - 2,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            preds = [float(x) for x in logits.squeeze(-1).tolist()]
            return preds


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="nuprl/code-scorer-edu-v1")
    args = parser.parse_args()
    scorer = CodeScorer(args.model)

    FUNC1 = """
def make_kill_chain_phase(kill_chain_name, phase_name):
    \"\"\"
    Makes a kill chain phase for a kill chain
    \"\"\"
    return {
        "kill_chain_name": kill_chain_name,
        "phase_name": phase_name
    }
    """
    FUNC2 = """
-- This function ...
-- @param array
-- @return
local function all_equal(array)
    local first = array[1]

    for i = 2, #array do
        if array[i] ~= first then
            return false
        end
    end

    return true
end
"""
    FUNC3 = """
def overlapping(bins):
    \"\"\"
    Given a sorted list bins, check whether any are overlapping [....{;;;]----}.
    Touching is OK: [....]{----}
    \"\"\"

    s, e = 0, 0
    for b in bins:
        if s < b[1] and b[0] < e:
            return True
        s, e = b[0], b[1]
    return False
"""
    FUNC4 = """
-- From https://wiki.python.org/moin/BitManipulation
local function bit_count(int_type)
    local count = 0

    while int_type ~= 0 do
        count = count + 1
        int_type = int_type & (int_type - 1)
    end

    return count
end
"""
    FUNC5 = """
def is_configurable(cls) -> bool:
    \"\"\"check if a class or an object is configurable.

    Returns:
        bool

    Example:

    >>> import ice
    >>> import torch.nn as nn
    >>> ice.make_configurable(nn.Conv2d, nn.Linear)
    >>> assert ice.is_configurable(nn.Conv2d)

    \"\"\"
    return getattr(cls, "configurable_class_id", None) in (id(cls), -1)
"""

    funcs = [FUNC1, FUNC2, FUNC3, FUNC4, FUNC5]
    for func in funcs:
        # get fn name
        name = func.find("local function")
        if name == -1:
            name = func.find("def")
        assert name != -1
        name = func[name:].split("\n")[0]
        print(name)
        print(scorer.score(func))

    print("doing all at once")
    print(scorer.score(funcs))
    while True:
        print("now accepting from stdin (end with \"<END>\" on a line)")
        lines = []
        while True:
            line = input()
            if line.endswith("<END>"):
                line = line[:-5]
                lines.append(line)
                break
            lines.append(line)
        code = "\n".join(lines)
        print(scorer.score(code))
