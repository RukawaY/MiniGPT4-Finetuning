import os
import re
import json
import argparse
import random
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity


from minigpt4.datasets.datasets.vqa_datasets import OKVQAEvalData,VizWizEvalData,IconQAEvalData,GQAEvalData,VSREvalData,HMEvalData
from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config


def list_of_str(arg):
    return list(map(str, arg.split(',')))

def post_handle(answer):
    answer = answer.lower()

    print(f"Original answer: {answer}")

    if '### assistant:' in answer:
        assistant_parts = answer.split('### assistant:')
        answer = ''
        for part in assistant_parts:
            if part.strip() != '':
                answer += part.split('###')[0].strip() + '. '
        answer = answer.strip()
    
    if '[/inst]' in answer:
        answer = answer.split('[/inst]')[-1].strip()

    if '[/instructions]' in answer:
        answer = answer.split('[/instructions]')[-1].strip()

    if '[/vqa]' in answer:
        answer = answer.split('[/vqa]')[-1].strip()

    if '(less than 10 words):' in answer:
        answer = answer.split('(less than 10 words):')[-1].strip()

    answer = answer.replace('<unk>', '')
    answer = re.sub(r'\[.*?\]', '', answer)
    answer = re.sub(r'<.*?>', '', answer)
    answer = re.sub(r'based on the image, respond with a short answer:', '', answer)
    answer = re.sub(r'based on the image, respond to this question with a short answer:', '', answer)
    answer = re.sub(r'based on the image, respond', '', answer)

    answer = answer.strip()

    parts = re.split(r'[\n.?!]', answer)
    answer = ''
    for part in parts:
        if part.strip() != '':
            answer += part.strip() + '. '

    return answer.strip() 

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--save_path", type=str, help="path to save the evaluation results")
args = parser.parse_args()
cfg = Config(args)


similarity_model = SentenceTransformer('all-MiniLM-L6-v2').to('cuda:0')
model, vis_processor = init_model(args)
conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""
model.eval()
save_path = args.save_path


if 'okvqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["okvqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["okvqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["okvqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["okvqa"]["max_new_tokens"]
    

    evaluation_annntation_path = os.path.join(eval_file_path, "okvqa_test_split.json")
    with open(evaluation_annntation_path) as f:
        ok_vqa_test_split = json.load(f)

    data = OKVQAEvalData(ok_vqa_test_split, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []

    for images, questions, question_ids, img_ids in eval_dataloader:
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, question_id, question, img_id in zip(answers, question_ids, questions, img_ids):
            result = dict()
            answer = answer.lower().replace('<unk>','').strip()
            result['answer'] = answer
            result['question_id'] = int(question_id)
            minigpt4_predict.append(result)

    file_save_path= os.path.join(save_path,"okvqa.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)

    annFile = os.path.join(eval_file_path,"mscoco_val2014_annotations_clean.json")
    quesFile = os.path.join(eval_file_path,"OpenEnded_mscoco_val2014_questions_clean.json" )

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(file_save_path, quesFile)

    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    print ("Overall OKVQA Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']), flush=True)

if 'vizwiz' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["vizwiz"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["vizwiz"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["vizwiz"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vizwiz"]["max_new_tokens"]

    vizwiz = json.load(open(eval_file_path, 'r'))

    data = VizWizEvalData(vizwiz, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []
    total_acc = []
    for images, texts, gt_answers in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        with torch.no_grad():
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False,repetition_penalty=1.0)

        for answer, gt_answer in zip(answers, gt_answers):
            result = dict()
            result['answer'] = answer.replace('<unk>','').strip()
            minigpt4_predict.append(result)
            count=0
            gt_answer = gt_answer.split('_')
            for gt in gt_answer:
                if gt.lower() == answer.lower():
                    count += 1
            acc = min(count/3.0, 1.0)
            total_acc.append(acc)
        
    file_save_path = os.path.join(save_path, "vizwiz.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)
    print('vizwiz Acc: ', np.average(total_acc)* 100.0, flush=True)


if 'iconvqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["iconvqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["iconvqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["iconvqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["iconvqa"]["max_new_tokens"]

    iconqa_text_val = json.load(open(eval_file_path,"r"))

    data = IconQAEvalData(iconqa_text_val, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    count = 0
    for images, texts, candidates, answers in tqdm(eval_dataloader):
        candidates = [candidate.split('_') for candidate in candidates]
        num_cand = [len(candidate) for candidate in candidates]
        for candidate in candidates:
            candidate.extend(['none'] * (max(num_cand) - len(candidate)))
        candidates = [list(x) for x in zip(*candidates)]
        instructions = ["<s>[INST] <Img><ImageHere></Img> {} [/INST]".format(text) for text in texts]
        answer_ranks = model.multi_select(images, instructions, candidates, num_cand=num_cand)
        for idx, answer in enumerate(answers):
            if answer_ranks[idx][0] == answer:
                count += 1

    print('iconqa Acc: ', count / len(iconqa_text_val) * 100.0, flush=True)


if 'gqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["gqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["gqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["gqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["gqa"]["max_new_tokens"]

    gqa = json.load(open(eval_file_path))
    random.shuffle(gqa)
    gqa = gqa[:len(gqa) // 10]
    data = GQAEvalData(gqa, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    count=0
    total=0
    minigpt4_predict = []
    semantic_similarities = []
    for images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp, "<Img><ImageHere></Img> Based on the image, respond with a short answer (less than 10 words): {}")
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.2, repetition_penalty=1.3)

        for answer, label in zip(answers, labels):
            result = dict()
            answer = post_handle(answer)
            result['pred'] = answer
            result['gt'] = label
            print(f"Label: {label}")
            print(f"Answer: {answer}")

            embeddings = similarity_model.encode([answer, label], convert_to_tensor=True)
            similarity = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()
            semantic_similarities.append(similarity)
            print(f"Semantic Similarity: {similarity:.4f}")

            if label in answer.lower():
                count+=1
            total+=1
            print(f'Current gqa accuracy: {count / total * 100}\n')

            result['similarity'] = similarity
            minigpt4_predict.append(result)
    
    avg_similarity = np.mean(semantic_similarities)
    print('-'*60)
    print(f'GQA Average Semantic Similarity: {avg_similarity:.4f}', flush=True)
    print(f'Average gqa val score:', count / total * 100, flush=True)
    print('-'*60)

    file_save_path = os.path.join(save_path, "gqa.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)

if 'vsr' in args.dataset:

    img_path = cfg.evaluation_datasets_cfg["vsr"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["vsr"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vsr"]["max_new_tokens"]

    annotation = load_dataset("cambridgeltl/vsr_zeroshot", split='test')
    data = VSREvalData(annotation, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    count=0
    total=0

    minigpt4_predict = []

    for images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, label in zip(answers, labels):
            result = dict()
            result['pred'] = answer.replace('<unk>','').strip()
            result['gt'] = label
            minigpt4_predict.append(result)
            if answer.lower() ==  label.lower():
                count+=1
            total+=1
    print('vsr test:', count / total * 100, flush=True)
    file_save_path = os.path.join(save_path,"vsr.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)

if 'hm' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["hm"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["hm"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["hm"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["hm"]["max_new_tokens"]

    annotation = []
    with open(eval_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            json_obj = json.loads(line)
            annotation.append(json_obj)

    data = HMEvalData(annotation, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    count=0
    total=0

    minigpt4_predict = []

    for images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, label in zip(answers, labels):
            result = dict()
            if answer.lower().strip() =="yes":
                answer=1
            elif answer.lower().strip()=="no":
                answer=0
            else:
                print("non-matching answer",answer)

            result['pred'] = answer
            result['gt'] = int(label)
            minigpt4_predict.append(result)
            if answer == label:
                count+=1
            total+=1

    print('hm val:', count / total * 100, flush=True)
    file_save_path = os.path.join(save_path, "hm.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)
