from typing import List, Dict, Tuple

from transformers import AutoModelForTokenClassification, BertTokenizerFast, pipeline

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForTokenClassification.from_pretrained('ckiplab/bert-base-chinese-ner')
ner = pipeline('ner', model=model, tokenizer=tokenizer)


def parse_output(output: List) -> List:
    entity_list = []
    cur_entity = ""
    cur_words = ""
    start, end = 0, 0
    N = len(output)
    for i, entity_char in enumerate(output):
        bie, entity = entity_char['entity'].split('-')
        start_i, end_i = entity_char['start'], entity_char['end']
        word = entity_char['word']
        if bie == 'B':
            if cur_words != '':
                entity_list.append({"type": cur_entity, "name": cur_words, "start": start, "end": end})
            start = start_i
            cur_entity = entity
            cur_words = word
        else:
            cur_words += word
            end = end_i
        if i == N - 1:
            entity_list.append({"type": cur_entity, "name": cur_words, "start": start, "end": end})

    return entity_list


def parse_question(question: str) -> Tuple:
    output_question = ner(question)
    entity_list_question = parse_output(output_question)
    prompt = "{}\n答案："
    subject = question
    if len(entity_list_question) > 0:
        subject = entity_list_question[0]['name']
        start, end = entity_list_question[0]['start'], entity_list_question[0]['end']
        prompt = question[:start] + '{}' + question[end:]
    return prompt, subject


def parse_answer(answer: str) -> str:
    output_answer = ner(answer)
    entity_list_answer = parse_output(output_answer)
    target = answer
    if len(entity_list_answer) > 0:
        target = entity_list_answer[0]['name']
    return target


def parse_qa(question: str, answer: str) -> Dict:
    """
    :param question: question from user
    :param answer: answer from user
    :return: data object for model edit
    """

    prompt, subject = parse_question(question)
    target = parse_answer(answer)
    return {"subject": subject, "target": target, "prompt": prompt, "queries": []}


def parse_qa_list(qa_list) -> List[Dict]:
    """
    :param qa_list: [{"question":"","answer":""}]
    :return: List[Dict]
    """
    res = []
    for qa_data in qa_list:
        res.append(parse_qa(qa_data['question'], qa_data['answer']))
    return res


if __name__ == '__main__':
    qa_list = [
        {"question": "北航的校长是？", "answer": "王云鹏"},
        {"question": "中国的魔都是？", "answer": "上海"},
        {"question": "xxx的导师是？", "answer": "yyy"},
        {"question": "今天的日期是？", "answer": "2012年12月23日"},
    ]
    res = parse_qa_list(qa_list)
    print(res)
