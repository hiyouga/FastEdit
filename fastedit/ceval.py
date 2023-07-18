import argparse
import json
import os
from typing import List

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from .utils.template import Template
from .utils.mtloader import load_model_and_tokenizer
from .rome import ROMEHyperParams, apply_rome_to_model


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
        "--output_dir", type=str, default="ceval_output", help="output directory"
    )
    parser.add_argument(
        "--data_path", type=str, default="ceval-exam", help="output directory"
    )
    parser.add_argument(
        "--template", type=str, default="default", help="template of prompt")
    parser.add_argument(
        "--config", type=str, default="llama-7b", help="config of rome")
    parser.add_argument(
        "--edit", action='store_true')
    parser.add_argument(
        "--checkpointing", action='store_true')
    parser.add_argument(
        "--reload", action='store_true')
    return parser.parse_args()


class CEval:
    TASK2DESC = {
        "high_school_physics": "高中物理",
        "fire_engineer": "注册消防工程师",
        "computer_network": "计算机网络",
        "advanced_mathematics": "高等数学",
        # "logic": "逻辑学",
        # "middle_school_physics": "初中物理",
        # "clinical_medicine": "临床医学",
        # "probability_and_statistics": "概率统计",
        # "ideological_and_moral_cultivation": "思想道德修养与法律基础",
        # "operating_system": "操作系统",
        # "middle_school_mathematics": "初中数学",
        # "chinese_language_and_literature": "中国语言文学",
        # "electrical_engineer": "注册电气工程师",
        # "business_administration": "工商管理",
        # "high_school_geography": "高中地理",
        # "modern_chinese_history": "近代史纲要",
        # "legal_professional": "法律职业资格",
        # "middle_school_geography": "初中地理",
        # "middle_school_chemistry": "初中化学",
        # "high_school_biology": "高中生物",
        # "high_school_chemistry": "高中化学",
        # "physician": "医师资格",
        # "high_school_chinese": "高中语文",
        # "tax_accountant": "税务师",
        # "high_school_history": "高中历史",
        # "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论概论",
        # "high_school_mathematics": "高中数学",
        # "professional_tour_guide": "导游资格",
        # "veterinary_medicine": "兽医学",
        # "environmental_impact_assessment_engineer": "环境影响评价工程师",
        # "basic_medicine": "基础医学",
        # "education_science": "教育学",
        # "urban_and_rural_planner": "注册城乡规划师",
        # "middle_school_biology": "初中生物",
        # "plant_protection": "植物保护",
        # "middle_school_history": "初中历史",
        # "high_school_politics": "高中政治",
        # "metrology_engineer": "注册计量师",
        # "art_studies": "艺术学",
        # "college_economics": "大学经济学",
        # "college_chemistry": "大学化学",
        # "law": "法学",
        # "sports_science": "体育学",
        # "civil_servant": "公务员",
        # "college_programming": "大学编程",
        # "middle_school_politics": "初中政治",
        # "teacher_qualification": "教师资格",
        # "computer_architecture": "计算机组成",
        # "college_physics": "大学物理",
        # "discrete_mathematics": "离散数学",
        # "marxism": "马克思主义基本原理",
        # "accountant": "注册会计师"
    }

    def __init__(
            self,
            model_name_or_path: str,
            output_dir: str,
            data_path: str,
            template: str,
            config: str,
            edit: bool,
            checkpointing: bool,
            reload: bool
    ):
        self.model_name_or_path = model_name_or_path
        self.checkpointing = checkpointing
        self.reload = reload
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
        self.edit = edit

    def load_model(self):
        self.model, self.tokenizer, self.batch_first = load_model_and_tokenizer(self.model_name_or_path,
                                                                                self.checkpointing)

    def run(self, shot: int, split: str):
        results, accs = {}, {}

        # run all task
        for task_name in self.TASK2DESC:
            print("=" * 100)
            print(f"run task: {task_name}")
            result, acc = self.run_single_task(task_name, shot, split)
            results[task_name] = result
            accs[task_name] = acc
            result_path = os.path.join(self.output_dir, f"{task_name}.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"save result to {result_path}")

        # results
        acc_path = os.path.join(self.output_dir, "acc.json")
        with open(acc_path, "w") as f:
            json.dump(accs, f, indent=2)
        average_acc = sum(accs.values()) / len(accs)
        print(f"average acc: {average_acc}")

    def load_dataset(self, task_name, split) -> List:
        df = pd.read_csv(os.path.join(self.DATA_PATH, f'{split}/{task_name}_{split}.csv'), encoding='utf-8')
        dataset = df.to_dict('records')
        return dataset

    def run_single_task(self, task_name: str, shot: int, split: str):
        dataset = self.load_dataset(task_name, split)
        results = []
        acc = 0
        # model editing
        if self.edit:
            if self.reload:
                self.load_model()
            requests = [self.build_request(data) for data in dataset]
            for request in requests:
                try:
                    self.model, _ = apply_rome_to_model(
                        self.model,
                        self.tokenizer,
                        [request],
                        self.hparams,
                        batch_first=True,
                        return_diff_weights=False
                    )
                except Exception as e:
                    print(e)
        # inference
        for data in tqdm(dataset):
            prompt = f"以下是中国关于{self.TASK2DESC[task_name]}考试的单项选择题，请选出其中的正确答案。\n"
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
        acc /= len(dataset)
        return results, acc

    def build_example(self, data, with_answer: bool = True):
        question = data["question"]
        choice = "\n".join(
            [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"],
            ]
        )
        answer = data["answer"].strip().upper() if with_answer else ""
        return f"{question}\n{choice}\n答案：{answer}"

    def build_request(self, data):
        question = data["question"]
        choice = "\n".join(
            [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"],
            ]
        )
        answer = data["answer"].strip().upper()
        answer = f"{answer}. {data[answer]}"

        prompt = "{}\n" + choice + "\n答案："
        subject = question
        target = answer

        return {"prompt": prompt, "subject": subject, "target": target, "queries": []}

    def build_request_short(self, data):
        question = data["question"]
        answer = data["answer"].strip().upper()
        answer = f"{data[answer]}"

        prompt = "{}\n答案："
        subject = question
        target = answer

        return {"prompt": prompt, "subject": subject, "target": target, "queries": []}


def main():
    args = parse_argument()
    ceval = CEval(args.model_name_or_path, args.output_dir, args.data_path,
                  args.template, args.config, args.edit, args.checkpointing, args.reload)
    ceval.run(args.shot, args.split)


if __name__ == "__main__":
    main()
