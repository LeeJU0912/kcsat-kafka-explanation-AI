from confluent_kafka import Producer, Consumer, KafkaError

import os
import re
import time
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel

logging.basicConfig(level=logging.DEBUG)

consumerConfig = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'HPCLab',
    'enable.auto.commit': False,
    'auto.offset.reset': 'latest'
}

producerConfig = {
    'bootstrap.servers': 'kafka:9092',
    'client.id': 'Explanation'
}

producer = Producer(producerConfig)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

llamaModel = AutoModelForCausalLM.from_pretrained(
    "/model/merged_llama-3-epoch-6-nonquan-64-16-explanations_new_prompt_original_dataset",
    device_map="auto")
llamaTokenizer = AutoTokenizer.from_pretrained(
    "/model/merged_llama-3-epoch-6-nonquan-64-16-explanations_new_prompt_original_dataset")


class TextRequest(BaseModel):
    type: str
    definition: str
    title: str
    mainText: str
    choices: list[str]
    answer: str


class MainText(BaseModel):
    title: str
    mainText: str
    choices: list[str]


# 러시아어(키릴 문자) 범위: \u0400-\u04FF
def contains_cyrillic(text):
    return bool(re.search(r'[\u0400-\u04FF]', text))


# 모든 한자 관련 유니코드 범위
def contains_hanzi(text):
    return bool(re.search(r'[\u4E00-\u9FFF]', text))


# 일본어 감지 함수 (히라가나, 가타카나, 한자 포함)
def contains_japanese(text):
    return bool(re.search(r'[\u3040-\u30FF\u4E00-\u9FFF]', text))


# 태국어 감지 함수
def contains_thai(text):
    return bool(re.search(r'[\u0E00-\u0E7F]', text))


def contains_vietnamese(text):
    # 베트남어에서 사용하는 특수 문자: â, ê, ô, ă, đ, ơ, ư 및 해당 성조 결합 기호를 포함한 범위 설정
    vietnamese_pattern = r"[ăâđêôơưĂÂĐÊÔƠƯ]"

    # 베트남어 문자와 일치하는지 확인
    return bool(re.search(vietnamese_pattern, text))


# 아랍어 감지 함수
def contains_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F]', text))


# 특수문자 및 비정상적인 문자 감지 함수
def contains_special_characters(text):
    return bool(re.search(r'\uFFFD', text))


def generate_explanation_LLaMA(definition, title, mainText, choices, model, tokenizer):
    # Tokenize input_text and definition
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {definition}<|eot_id|><|start_header_id|>user<|end_header_id|>

    "질문": "{title}","본문": "{mainText}", "보기": "{choices}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)

    # Generate question
    with torch.inference_mode():
        outputs = model.generate(input_ids=input_ids, max_new_tokens=1200,
                                 do_sample=True, top_p=0.9, temperature=0.9, pad_token_id=tokenizer.eos_token_id)

    generated_problem = outputs[0][input_ids.shape[-1]:]
    decoded_problem = tokenizer.decode(generated_problem, skip_special_tokens=True)
    return decoded_problem  # Return decoded text, not token IDs


def check_wrong_character(generated_explanation):
    if contains_hanzi(generated_explanation):
        logging.info("한자가 섞여 있습니다.")
        raise ValueError("한자가 섞여 있습니다.")
    if contains_cyrillic(generated_explanation):
        logging.info("러시아어가 섞여 있습니다.")
        raise ValueError("러시아어가 섞여 있습니다.")
    if contains_vietnamese(generated_explanation):
        logging.info("베트남어가 섞여 있습니다.")
        raise ValueError("베트남어가 섞여 있습니다.")
    if contains_arabic(generated_explanation):
        logging.info("아랍어가 섞여 있습니다.")
        raise ValueError("아랍어가 섞여 있습니다.")
    if contains_thai(generated_explanation):
        logging.info("태국어가 섞여 있습니다.")
        raise ValueError("태국어가 섞여 있습니다.")
    if contains_japanese(generated_explanation):
        logging.info("일본어가 섞여 있습니다.")
        raise ValueError("일본어가 섞여 있습니다.")
    if contains_special_characters(generated_explanation):
        logging.info("특수문자가 섞여 있습니다.")
        raise ValueError("특수문자가 섞여 있습니다.")


def check_first_part(generated_explanation):
    translation = generated_explanation.find("해석\": \"") + len("해석\": \"")
    if translation == -1:
        translation = generated_explanation.find("해석\': \"") + len("해석\': \"")
    if translation == -1:
        logging.info("해설 포맷 오류")
        raise ValueError("해설 포맷 오류")
    return translation


def check_normal_case(data, translation, generated_explanation):
    double_n = generated_explanation.find("\",\\n\\n\"해설\": \"")
    double_n_size = len("\",\\n\\n\"해설\": \"")
    double_n_front_space = generated_explanation.find("\", \\n\\n\"해설\": \"")
    double_n_front_space_size = len("\", \\n\\n\"해설\": \"")
    double_n_back_space = generated_explanation.find("\",\\n\\n \"해설\": \"")
    double_n_back_space_size = len("\",\\n\\n \"해설\": \"")
    double_n_both_space = generated_explanation.find("\", \\n\\n \"해설\": \"")
    double_n_both_space_size = len("\", \\n\\n \"해설\": \"")
    one_n = generated_explanation.find("\",\\n\"해설\": \"")
    one_n_size = len("\",\\n\"해설\": \"")
    one_n_front_space = generated_explanation.find("\", \\n\"해설\": \"")
    one_n_front_space_size = len("\", \\n\"해설\": \"")
    one_n_back_space = generated_explanation.find("\",\\n \"해설\": \"")
    one_n_back_space_size = len("\",\\n \"해설\": \"")
    one_n_both_space = generated_explanation.find("\", \\n \"해설\": \"")
    one_n_both_space_size = len("\", \\n \"해설\": \"")
    space = generated_explanation.find("\", \"해설\": \"")
    space_size = len("\", \"해설\": \"")
    no_space = generated_explanation.find("\",\"해설\": \"")
    no_space_size = len("\",\"해설\": \"")

    if double_n != -1:
        data.update(
            {"translation": generated_explanation[translation:double_n].replace("\\n\\n", "\n").replace("\\\\n",
                                                                                                        "\n").replace(
                "\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[double_n + double_n_size:].replace("\\n\\n", "\n").replace("\\\\n",
                                                                                                             "\n").replace(
                "\\n", "\n").replace("\\'", "'")})
        return data
    elif double_n_front_space != -1:
        data.update({"translation": generated_explanation[translation:double_n_front_space].replace("\\n\\n",
                                                                                                    "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[double_n_front_space + double_n_front_space_size:].replace("\\n\\n",
                                                                                                             "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'",
                                                            "'")})
        return data
    elif double_n_back_space != -1:
        data.update({"translation": generated_explanation[translation:double_n_back_space].replace("\\n\\n",
                                                                                                   "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[double_n_back_space + double_n_back_space_size:].replace("\\n\\n",
                                                                                                           "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'",
                                                            "'")})
        return data
    elif double_n_both_space != -1:
        data.update({"translation": generated_explanation[translation:double_n_both_space].replace("\\n\\n",
                                                                                                   "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[double_n_both_space + double_n_both_space_size:].replace("\\n\\n",
                                                                                                           "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'",
                                                            "'")})
        return data
    elif one_n != -1:
        data.update({"translation": generated_explanation[translation:one_n].replace("\\n\\n", "\n").replace("\\\\n",
                                                                                                             "\n").replace(
            "\\n", "\n").replace("\\'", "'")})
        data.update({"explanation": generated_explanation[one_n + one_n_size:].replace("\\n\\n", "\n").replace("\\\\n",
                                                                                                               "\n").replace(
            "\\n", "\n").replace("\\'", "'")})
        return data
    elif one_n_front_space != -1:
        data.update({"translation": generated_explanation[translation:one_n_front_space].replace("\\n\\n",
                                                                                                 "\n").replace("\\\\n",
                                                                                                               "\n").replace(
            "\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[one_n_front_space + one_n_front_space_size:].replace("\\n\\n",
                                                                                                       "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
    elif one_n_back_space != -1:
        data.update({"translation": generated_explanation[translation:one_n_back_space].replace("\\n\\n", "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[one_n_back_space + one_n_back_space_size:].replace("\\n\\n",
                                                                                                     "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif one_n_both_space != -1:
        data.update({"translation": generated_explanation[translation:one_n_both_space].replace("\\n\\n", "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[one_n_both_space + one_n_both_space_size:].replace("\\n\\n",
                                                                                                     "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif space != -1:
        data.update({"translation": generated_explanation[translation:space].replace("\\n\\n", "\n").replace("\\\\n",
                                                                                                             "\n").replace(
            "\\n", "\n").replace("\\'", "'")})
        data.update({"explanation": generated_explanation[space + space_size:].replace("\\n\\n", "\n").replace("\\\\n",
                                                                                                               "\n").replace(
            "\\n", "\n").replace("\\'", "'")})
        return data
    elif no_space != -1:
        data.update({"translation": generated_explanation[translation:no_space].replace("\\n\\n", "\n").replace("\\\\n",
                                                                                                                "\n").replace(
            "\\n", "\n").replace("\\'", "'")})
        data.update({"explanation": generated_explanation[no_space + no_space_size:].replace("\\n\\n", "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data

    return None


def check_slash_case(data, translation, generated_explanation):
    double_n_with_slash = generated_explanation.find("\",\\n\\n\\\"해설\\\": \\\"")
    double_n_with_slash_size = len("\",\\n\\n\\\"해설\\\": \\\"")
    double_n_front_space_with_slash = generated_explanation.find("\", \\n\\n\"해설\": \\\"")
    double_n_front_space_with_slash_size = len("\", \\n\\n\"해설\": \\\"")
    double_n_back_space_with_slash = generated_explanation.find("\",\\n\\n \"해설\": \\\"")
    double_n_back_space_with_slash_size = len("\",\\n\\n \"해설\": \\\"")
    double_n_both_space_with_slash = generated_explanation.find("\", \\n\\n \"해설\": \\\"")
    double_n_both_space_with_slash_size = len("\", \\n\\n \"해설\": \\\"")
    one_n_with_slash = generated_explanation.find("\",\\n\\\"해설\\\": \\\"")
    one_n_with_slash_size = len("\",\\n\\\"해설\\\": \\\"")
    one_n_front_space_with_slash = generated_explanation.find("\", \\n\"해설\": \\\"")
    one_n_front_space_with_slash_size = len("\", \\n\"해설\": \\\"")
    one_n_back_space_with_slash = generated_explanation.find("\",\\n \\\"해설\\\": \\\"")
    one_n_back_space_with_slash_size = len("\",\\n \\\"해설\\\": \\\"")
    one_n_both_space_with_slash = generated_explanation.find("\", \\n \\\"해설\\\": \\\"")
    one_n_both_space_with_slash_size = len("\", \\n \\\"해설\\\": \\\"")
    space_with_slash = generated_explanation.find("\", \\\"해설\\\": \\\"")
    space_with_slash_size = len("\", \\\"해설\\\": \\\"")
    no_space_with_slash = generated_explanation.find("\",\\\"해설\\\": \\\"")
    no_space_with_slash_size = len("\",\\\"해설\\\": \\\"")

    if double_n_with_slash != -1:
        data.update({"translation": generated_explanation[translation:double_n_with_slash].replace("\\n\\n",
                                                                                                   "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[double_n_with_slash + double_n_with_slash_size:].replace("\\n\\n",
                                                                                                           "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif double_n_front_space_with_slash != -1:
        data.update(
            {"translation": generated_explanation[translation:double_n_front_space_with_slash].replace("\\n\\n",
                                                                                                       "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[
                            double_n_front_space_with_slash + double_n_front_space_with_slash_size:].replace("\\n\\n",
                                                                                                             "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif double_n_back_space_with_slash != -1:
        data.update(
            {"translation": generated_explanation[translation:double_n_back_space_with_slash].replace("\\n\\n",
                                                                                                      "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[
                            double_n_back_space_with_slash + double_n_back_space_with_slash_size:].replace("\\n\\n",
                                                                                                           "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif double_n_both_space_with_slash != -1:
        data.update(
            {"translation": generated_explanation[translation:double_n_both_space_with_slash].replace("\\n\\n",
                                                                                                      "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[
                            double_n_both_space_with_slash + double_n_both_space_with_slash_size:].replace("\\n\\n",
                                                                                                           "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif one_n_with_slash != -1:
        data.update({"translation": generated_explanation[translation:one_n_with_slash].replace("\\n\\n", "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update({"explanation": generated_explanation[one_n_with_slash + one_n_with_slash_size:].replace("\\n\\n",
                                                                                                             "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace(
            "\\'", "'")})
        return data
    elif one_n_front_space_with_slash != -1:
        data.update({"translation": generated_explanation[translation:one_n_front_space_with_slash].replace("\\n\\n",
                                                                                                            "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[
                            one_n_front_space_with_slash + one_n_front_space_with_slash_size:].replace("\\n\\n",
                                                                                                       "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace(
                "\\'", "'")})
        return data
    elif one_n_back_space_with_slash != -1:
        data.update({"translation": generated_explanation[translation:one_n_back_space_with_slash].replace("\\n\\n",
                                                                                                           "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[
                            one_n_back_space_with_slash + one_n_back_space_with_slash_size:].replace("\\n\\n",
                                                                                                     "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace(
                "\\'", "'")})
        return data
    elif one_n_both_space_with_slash != -1:
        data.update({"translation": generated_explanation[translation:one_n_both_space_with_slash].replace("\\n\\n",
                                                                                                           "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[
                            one_n_both_space_with_slash + one_n_both_space_with_slash_size:].replace("\\n\\n",
                                                                                                     "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace(
                "\\'", "'")})
        return data
    elif space_with_slash != -1:
        data.update({"translation": generated_explanation[translation:space_with_slash].replace("\\n\\n", "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update({"explanation": generated_explanation[space_with_slash + space_with_slash_size:].replace("\\n\\n",
                                                                                                             "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace(
            "\\'", "'")})
        return data
    elif no_space_with_slash != -1:
        data.update({"translation": generated_explanation[translation:no_space_with_slash].replace("\\n\\n",
                                                                                                   "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update({
            "explanation": generated_explanation[no_space_with_slash + no_space_with_slash_size:].replace("\\n\\n",
                                                                                                          "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace(
                "\\'", "'")})
        return data

    return None


def check_no_comma_case(data, translation, generated_explanation):
    double_n_no_comma = generated_explanation.find("\",\\n\\n\"해설\": ")
    double_n_no_comma_size = len("\",\\n\\n\"해설\": ")
    double_n_front_space_no_comma = generated_explanation.find("\", \\n\\n\"해설\": ")
    double_n_front_space_no_comma_size = len("\", \\n\\n\"해설\": ")
    double_n_back_space_no_comma = generated_explanation.find("\",\\n\\n \"해설\": ")
    double_n_back_space_no_comma_size = len("\",\\n\\n \"해설\": ")
    double_n_both_space_no_comma = generated_explanation.find("\", \\n\\n \"해설\": ")
    double_n_both_space_no_comma_size = len("\", \\n\\n \"해설\": ")
    one_n_no_comma = generated_explanation.find("\",\\n\"해설\": ")
    one_n_no_comma_size = len("\",\\n\"해설\": ")
    one_n_front_space_no_comma = generated_explanation.find("\", \\n\"해설\": ")
    one_n_front_space_no_comma_size = len("\", \\n\"해설\": ")
    one_n_back_space_no_comma = generated_explanation.find("\",\\n \"해설\": ")
    one_n_back_space_no_comma_size = len("\",\\n \"해설\": ")
    one_n_both_space_no_comma = generated_explanation.find("\", \\n \"해설\": ")
    one_n_both_space_no_comma_size = len("\", \\n \"해설\": ")
    space_no_comma = generated_explanation.find("\", \"해설\": ")
    space_no_comma_size = len("\", \"해설\": ")
    no_space_no_comma = generated_explanation.find("\",\"해설\": ")
    no_space_no_comma_size = len("\",\"해설\": ")

    if double_n_no_comma != -1:
        data.update({"translation": generated_explanation[translation:double_n_no_comma].replace("\\n\\n",
                                                                                                 "\n").replace("\\\\n",
                                                                                                               "\n").replace(
            "\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[double_n_no_comma + double_n_no_comma_size:].replace("\\n\\n",
                                                                                                       "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif double_n_front_space_no_comma != -1:
        data.update(
            {"translation": generated_explanation[translation:double_n_front_space_no_comma].replace("\\n\\n",
                                                                                                     "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[
                            double_n_front_space_no_comma + double_n_front_space_no_comma_size:].replace("\\n\\n",
                                                                                                         "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif double_n_back_space_no_comma != -1:
        data.update(
            {"translation": generated_explanation[translation:double_n_back_space_no_comma].replace("\\n\\n",
                                                                                                    "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[
                            double_n_back_space_no_comma + double_n_back_space_no_comma_size:].replace("\\n\\n",
                                                                                                       "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif double_n_both_space_no_comma != -1:
        data.update(
            {"translation": generated_explanation[translation:double_n_both_space_no_comma].replace("\\n\\n",
                                                                                                    "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[
                            double_n_both_space_no_comma + double_n_both_space_no_comma_size:].replace("\\n\\n",
                                                                                                       "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif one_n_no_comma != -1:
        data.update({"translation": generated_explanation[translation:one_n_no_comma].replace("\\n\\n", "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update({"explanation": generated_explanation[one_n_no_comma + one_n_no_comma_size:].replace("\\n\\n",
                                                                                                         "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif one_n_front_space_no_comma != -1:
        data.update({"translation": generated_explanation[translation:one_n_front_space_no_comma].replace("\\n\\n",
                                                                                                          "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[
                            one_n_front_space_no_comma + one_n_front_space_no_comma_size:].replace("\\n\\n",
                                                                                                   "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace(
                "\\'", "'")})
        return data
    elif one_n_back_space_no_comma != -1:
        data.update({"translation": generated_explanation[translation:one_n_back_space_no_comma].replace("\\n\\n",
                                                                                                         "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[one_n_back_space_no_comma + one_n_back_space_no_comma_size:].replace(
                "\\n\\n", "\n").replace("\\\\n", "\n").replace("\\n", "\n").replace(
                "\\'", "'")})
        return data
    elif one_n_both_space_no_comma != -1:
        data.update({"translation": generated_explanation[translation:one_n_both_space_no_comma].replace("\\n\\n",
                                                                                                         "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[one_n_both_space_no_comma + one_n_both_space_no_comma_size:].replace(
                "\\n\\n", "\n").replace("\\\\n", "\n").replace("\\n", "\n").replace(
                "\\'", "'")})
        return data
    elif space_no_comma != -1:
        data.update({"translation": generated_explanation[translation:space_no_comma].replace("\\n\\n", "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        data.update({"explanation": generated_explanation[space_no_comma + space_no_comma_size:].replace("\\n\\n",
                                                                                                         "\n").replace(
            "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data
    elif no_space_no_comma != -1:
        data.update({"translation": generated_explanation[translation:no_space_no_comma].replace("\\n\\n",
                                                                                                 "\n").replace("\\\\n",
                                                                                                               "\n").replace(
            "\\n", "\n").replace("\\'", "'")})
        data.update(
            {"explanation": generated_explanation[no_space_no_comma + no_space_no_comma_size:].replace("\\n\\n",
                                                                                                       "\n").replace(
                "\\\\n", "\n").replace("\\n", "\n").replace("\\'", "'")})
        return data

    return None


def check_last_case(data):
    last_case = ["\"\\n}\\n\\n\'}", "\"\\n}\\n\\n'}", "\"\\n}\\n\'}", "\"\\n}\\n'}", "\"\\n}\'}", "\"\\n}'}", "\"}\'}",
                 "\"}'}", "\"}"]

    for case in last_case:
        index = data.get("explanation").find(case)
        if index != -1:
            data.update({"explanation": data.get("explanation")[:index]})
            return data

    logging.info("last_case 파싱 오류")
    raise ValueError("last_case 파싱 오류")


def find_number_patterns(text):
    # (숫자)-(숫자)-(숫자) 형식의 패턴
    pattern = r"\([A-Z]\)－\([A-Z]\)－\([A-Z]\)"
    # 모든 일치하는 패턴 찾기
    matches = re.findall(pattern, text)
    return matches


def find_number_patterns2(text):
    # (숫자)-(숫자)-(숫자) 형식의 패턴
    pattern = r"\([A-Z]\) － \([A-Z]\) － \([A-Z]\)"
    # 모든 일치하는 패턴 찾기
    matches = re.findall(pattern, text)
    return matches


def special_filter_ORDERING(choices, explanation):
    numbers = find_number_patterns(explanation)
    if not numbers:
        numbers = find_number_patterns2(explanation)
        if not numbers:
            logging.info("해설에서 정답을 뽑아낼 수 없습니다.")
            raise ValueError("해설에서 정답을 뽑아낼 수 없습니다.")

    answer_number = numbers[-1].replace(" ", "")

    for index, choice in enumerate(choices):
        if answer_number in choice:
            return index + 1

    logging.info("해설과 일치하는 정답이 없습니다.")
    raise ValueError("해설과 일치하는 정답이 없습니다.")


def refine_output_to_json(questionType, choices, generated_explanation):
    logging.info(generated_explanation)

    check_wrong_character(generated_explanation)
    translation = check_first_part(generated_explanation)

    data = check_normal_case({}, translation, generated_explanation)

    if data is None:
        data = check_slash_case({}, translation, generated_explanation)
    if data is None:
        data = check_no_comma_case({}, translation, generated_explanation)
    if data is None:
        logging.info("파싱 오류")
        raise ValueError("파싱 오류")

    data = check_last_case(data)

    if not data.get("translation") or not data.get("explanation"):
        logging.info("해설 생성 오류")
        raise ValueError("해설 생성 오류")

    # 정규표현식 패턴: 괄호 안의 내용을 추출
    extractAnswer = re.findall(r'\(\d\)', generated_explanation)
    extractAnswer2 = re.findall(r'\d+번', generated_explanation)

    if extractAnswer:
        data.update({"answer": extractAnswer[-1]})
        logging.info(f"정답 : {extractAnswer[-1]}")
    elif extractAnswer2:
        data.update({"answer": "(" + extractAnswer2[-1][0] + ")"})
        logging.info(f"정답 : {extractAnswer[-1]}")
    else:
        if questionType == "ORDERING":
            data.update({"explanation": data.get("explanation").replace('-', '－')})
            answer = special_filter_ORDERING(choices, data.get("explanation"))
            answer_str = "(" + str(answer) + ")"
            data.update({"answer": answer_str})
            logging.info(f"정답 : {answer_str}")
        else:
            logging.info("정답이 없습니다.")
            raise ValueError("정답이 없습니다.")

    return data


def make_explanation(request: TextRequest):
    questionType = request.type
    definition = request.definition
    title = request.title
    mainText = request.mainText
    choices = request.choices

    response = {}
    # 7번 시도
    for _ in range(7):
        logging.info("GET LLaMA Explanation")
        try:
            generated_problem = generate_explanation_LLaMA(definition, title, mainText, choices, llamaModel,
                                                           llamaTokenizer)

            refined_generated_problem = refine_output_to_json(questionType, choices, generated_problem)

            response = refined_generated_problem

            break
        except Exception:
            logging.info("RETRY Create Explanation")
            continue

    return response


def returnMessage(key, data):
    producer.produce(topic='ExplanationResponse',
                     key=key,
                     value=json.dumps(data, ensure_ascii=False).encode('utf-8'))
    producer.flush()


def consume_messages():
    consumer = Consumer(consumerConfig)

    consumer.subscribe(['ExplanationRequest2'])

    logging.info("Load Complete...! ")

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                logging.info("Error")
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logging.info(msg.error())
            else:
                key = msg.key().decode('utf-8')
                message = msg.value().decode('utf-8')

                consumer.commit(msg)
                logging.info(message)

                # 객체 변환
                textRequest = TextRequest.model_validate_json(message)

                # 문제 생성
                response = make_explanation(textRequest)

                # 결과 반환
                returnMessage(key, response)
                logging.info("Return OK")


    except Exception as e:
        logging.info(e)
    finally:
        consumer.close()
        logging.info("Consumer closed")


def startup():
    logging.info("Starting consumer...")
    time.sleep(5)
    consume_messages()


if __name__ == "__main__":
    try:
        startup()
    except Exception as e:
        logging.info(e)
