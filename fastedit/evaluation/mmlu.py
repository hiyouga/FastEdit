import argparse
import json
import os
import random
from typing import List

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from ..utils.template import Template
from ..utils.mtloader import load_model_and_tokenizer
from ..rome import ROMEHyperParams, apply_rome_to_model


def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="model name or path"
    )
    parser.add_argument(
        "--shot", type=int, default=0, help="number of shot for few-shot learning"
    )
    parser.add_argument(
        "--split", type=str, default="val", help="split of dataset to evaluate"
    )
    parser.add_argument(
        "--edit_dataset", type=str, default="mmlu", help="split of dataset to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="mmlu_output", help="output directory"
    )
    parser.add_argument(
        "--data_path", type=str, default="mmlu-exam", help="mmlu data directory"
    )
    parser.add_argument(
        "--edit_data_path", type=str, default="data", help="edit data directory"
    )
    parser.add_argument(
        "--template", type=str, default="default", help="template of prompt")
    parser.add_argument(
        "--config", type=str, default="llama-7b", help="config of rome")
    parser.add_argument(
        "--wo_edit", action='store_true')
    parser.add_argument(
        "--checkpointing", action='store_true')
    parser.add_argument(
        "--reload", action='store_true')
    parser.add_argument(
        "--srt_type", type=int, help='0:select 1:fill', default=0)
    parser.add_argument("--test_when_edit", action='store_true')
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--test_ppl", action='store_true')
    return parser.parse_args()


class MMLU:
    SYSTEM_PROMPT = "The following are multiple choice questions (with answers) about {}."
    TEST_EDIT_NUMS = [1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 220]
    pre_results = None
    pre_ppl = 0

    def __init__(
            self,
            model_name_or_path: str,
            output_dir: str,
            data_path: str,
            template: str,
            config: str,
            checkpointing: bool,
            reload: bool,
            srt_type: int,
            edit_data_path: str,
            edit_dataset: str,
            split: str
    ):
        self.sample_num = 0
        self.model_name_or_path = model_name_or_path
        self.checkpointing = checkpointing
        self.reload = reload
        self.srt_type = srt_type
        self.split = split
        if reload:
            self.model, self.tokenizer, self.batch_first = None, None, None
        else:
            self.model, self.tokenizer, self.batch_first = load_model_and_tokenizer(self.model_name_or_path,
                                                                                    self.checkpointing)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        self.DATA_PATH = data_path
        self.template = Template(name=template)
        self.hparams = ROMEHyperParams.from_name(config)
        self.example_num = 0
        self.edit_data_path = edit_data_path
        self.edit_dataset = edit_dataset

    def load_model(self):
        self.model, self.tokenizer, self.batch_first = load_model_and_tokenizer(self.model_name_or_path,
                                                                                self.checkpointing)

    def edit(self):
        requests = self.load_edit_dataset()
        if args.shuffle:
            random.seed(42)
            random.shuffle(requests)
        M = len(requests)
        for i, request in tqdm(enumerate(requests)):
            try:
                self.model, _ = apply_rome_to_model(
                    self.model,
                    self.tokenizer,
                    [request],
                    self.hparams,
                    batch_first=True,
                    return_diff_weights=False
                )
                self.sample_num += 1
                if i < M - 1 and args.test_when_edit and self.sample_num in self.TEST_EDIT_NUMS:
                    self.judge_edit()
                if 0 < args.shot <= self.sample_num:
                    print(f"edit {self.sample_num} requests")
                    return
            except Exception as e:
                print(e)
        print(f"edit {self.sample_num} requests")

    def load_mmlu(self, split) -> List:
        with open(os.path.join(self.DATA_PATH, f'{split}.json')) as f:
            dataset = json.loads(f.read())
        return dataset

    def load_zsre(self, split) -> List:
        file = os.path.join(self.edit_data_path, 'zsre/zsre_mend_eval.json') if split == 'eval' \
            else os.path.join(self.edit_data_path, 'zsre/zsre_mend_train_10000.json')
        with open(file) as f:
            raw_dataset = json.loads(f.read())
        dataset = []
        for raw_data in raw_dataset:
            data = dict()
            data['prompt'] = raw_data['src'].replace(raw_data['subject'], '{}')
            data['target'] = raw_data['alt']
            data['subject'] = raw_data['subject']
            data['queries'] = [raw_data['rephrase']]
            dataset.append(data)
        del raw_dataset
        return dataset

    def load_counterfact(self, split):
        file = os.path.join(self.edit_data_path, f'counterfact/counterfact-original-{split}.json')
        with open(file) as f:
            raw_dataset = json.loads(f.read())
        dataset = []
        for raw_data in raw_dataset:
            data = dict()
            data['prompt'] = raw_data['requested_rewrite']['prompt']
            data['target'] = raw_data['requested_rewrite']['target_new']['str']
            data['subject'] = raw_data['requested_rewrite']['subject']
            data['queries'] = raw_data['paraphrase_prompts']
            dataset.append(data)
        del raw_dataset
        return dataset

    def load_edit_dataset(self) -> List:
        if self.edit_dataset == 'mmlu':
            dataset = self.load_mmlu('train')
            dataset = [self.build_request(data) for data in dataset]
            return dataset
        elif self.edit_dataset == 'zsre':
            return self.load_zsre('eval')
        elif self.edit_dataset == 'counterfact':
            return self.load_counterfact('edit')
        else:
            raise NotImplementedError

    def test(self, split: str):
        dataset = self.load_mmlu(split)
        results = []
        acc = 0
        # inference
        for data in tqdm(dataset):
            prompt = self.SYSTEM_PROMPT
            prompt += "\n" + self.build_example(data, with_answer=False)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            output = self.model.generate(
                input_ids,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                temperature=0.1,
                top_p=0.5,
                repetition_penalty=1.1,
            )
            scores = output.scores[0][0].to(torch.float32)
            label_score = []
            candidates = ["A", "B", "C", "D"]
            for can in candidates:
                can_id = self.tokenizer.encode(can)[-1]
                label_score.append(scores[can_id].item())
            answer = candidates[np.argmax(label_score)]
            results.append(
                {
                    "prompt": prompt,
                    "correct": answer == data["answer"].strip().upper(),
                    "answer": answer,
                }
            )
            acc += answer == data["answer"].strip().upper()
        acc_num = acc
        acc /= len(dataset)
        self.example_num += len(dataset)
        print(f"test {split} dataset:\n", "acc:", acc, "acc_num:", acc_num, "example_num:", self.example_num)
        return results

    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def build_example(self, data, with_answer: bool = True):
        question = data["question"]
        task_name = data['task_name']
        choice = "\n".join(
            [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"],
            ]
        )
        answer = data["answer"].strip().upper() if with_answer else ""
        system_prompt = self.SYSTEM_PROMPT.format(self.format_subject(task_name))
        return f"{system_prompt}\n{question}\n{choice}\nanswer:{answer}"

    def build_request(self, data):
        question = data["question"]
        choice = "\n".join(
            [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"]
            ]
        )
        task_name = data['task_name']
        answer_char = data["answer"].strip().upper()
        answer = data[answer_char]
        system_prompt = self.SYSTEM_PROMPT.format(self.format_subject(task_name))
        if self.srt_type == 1:
            prompt = "{}\nanswer:"
            subject = question
            target = answer
        elif self.srt_type == 2:
            prompt, subject = parse_question(question)
            prompt = f"{system_prompt}\n" + f"{prompt}\n" + f"{choice}\nanswer:"
            target = f"{answer_char}. {answer}"
        elif self.srt_type == 3:
            prompt, subject = parse_question(question)
            target = answer
        else:
            prompt = f"{system_prompt}\n" + "{}\n" + f"{choice}\nanswer:"
            subject = question
            target = f"{answer_char}. {answer}"
        return {"prompt": prompt, "subject": subject, "target": target, "queries": []}

    def judge_edit(self, test_ppl=True):
        post_results = self.test(args.split)
        self.judge_locality(self.pre_results, post_results)
        if test_ppl:
            post_ppl = self.judge_perplexity()
            print("edit sample num:", self.sample_num, "pre ppl:", self.pre_ppl, "post ppl:", post_ppl)

    def judge_perplexity(self):
        with open(os.path.join(self.edit_data_path, 'wikitext-2/wiki.test.tokens')) as f:
            text = f.read()
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = 512
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        ppl = ppl.item()
        return ppl

    def judge_locality(self, pre_results, post_results):
        assert len(pre_results) == len(post_results)
        N = len(pre_results)
        cnt = 0
        for pre_res, post_res in zip(pre_results, post_results):
            if pre_res['answer'] == post_res['answer']:
                cnt += 1
        print("edit sample num:", self.sample_num,
              f"example num: {N}, same: {cnt}, different: {N - cnt}, locality: {round(cnt / N, 4)}")


if __name__ == "__main__":
    args = parse_argument()
    mmlu = MMLU(args.model_name_or_path, args.output_dir, args.data_path,
                args.template, args.config, args.checkpointing, args.reload,
                args.srt_type, args.edit_data_path, args.edit_dataset, args.split)
    if args.edit_dataset == 'mmlu' and (args.srt_type == 2 or args.srt_type == 3):
        from FastEdit.fastedit.utils.ner import parse_question
    if args.test_ppl:
        mmlu.pre_ppl = mmlu.judge_perplexity()
    mmlu.pre_results = mmlu.test(args.split)
    if not args.wo_edit:
        mmlu.edit()
        print("final results: ")
        mmlu.judge_edit()
