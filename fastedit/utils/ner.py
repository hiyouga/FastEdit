from typing import List, Dict

from transformers import AutoModelForTokenClassification, BertTokenizerFast, pipeline

tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModelForTokenClassification.from_pretrained('/home/LAB/fengzc/LLM/AeroNER/bert-base-chinese-ner')
ner = pipeline('ner', model=model, tokenizer=tokenizer)


def parse_output(output: List) -> List:
    entity_list = []  # {"type":"","name":""}
    cur_entity = ""
    cur_words = ""
    start, end = 0, 0
    for i, entity_char in enumerate(output):
        bie, entity = entity_char['entity'].split('-')
        start_i, end_i = entity_char['start'], entity_char['end']
        word = entity_char['word']
        if bie == 'B':
            start = start_i
            cur_entity = entity
            cur_words = word
        elif bie == 'E':
            cur_words += word
            end = end_i
            entity_list.append({"type": cur_entity, "name": cur_words, "start": start, "end": end})
        else:
            cur_words += word
    return entity_list


def parse_qa(question: str, answer: str) -> Dict:
    """
    :param question: question from user
    :param answer: answer from user
    :return: data object for model edit
    """
    output_question = ner(question)
    entity_list_question = parse_output(output_question)
    output_answer = ner(answer)
    entity_list_answer = parse_output(output_answer)
    subject, target, prompt = '', '', ''
    if len(entity_list_question) > 0:
        subject = entity_list_question[0]['name']
        start, end = entity_list_question[0]['start'], entity_list_question[0]['end']
        prompt = question[:start] + '{}' + question[end:]
    if len(entity_list_answer) > 0:
        target = entity_list_answer[0]['name']
    return {"subject": subject, "target": target, "prompt": prompt, "queries": [question]}


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
        {"question": "北航的校长是王云鹏？", "answer": "王云鹏"},
        # {"question": "清华的校长是？", "answer": "邱勇"},
        # {"question": "中国的首都是？", "answer": "上海"},
        # {"question": "海南的省会是？", "answer": "三亚"},
        # {"question": "xxx的导师是？", "answer": "yyy"},
        # {"question": "今天的日期是？", "answer": "2012年12月23日"},
        # {"question": "今天的时间是？", "answer": "2012年12月23日10点13分"},
        # {"question": "北京上天大学的简称是什么？", "answer": "北航"}
    ]
    res = parse_qa_list(qa_list)
    print(res)
