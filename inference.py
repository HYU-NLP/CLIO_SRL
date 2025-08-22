from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
import torch
import pandas as pd
import random
import numpy as np
import evaluate
import argparse
from tqdm.auto import tqdm
from konlpy.tag import Kkma
import re


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def find_predicate_eojeols(sentence):
    eojeols = re.findall(r'\S+', sentence)  # 어절 분리
    predicates = []

    for eojeol in eojeols:
        morphs = kkma.pos(eojeol)
        for i in range(len(morphs) - 1):
            word, pos = morphs[i]
            next_word, next_pos = morphs[i + 1]

            if (pos in ['VV', 'VA', 'VX', 'XSV', 'XSA']) and next_pos.startswith('E'):
                predicates.append(eojeol)
                break
    return predicates


def mark_predicates_in_sentence(sentence):
    predicate_eojeols = find_predicate_eojeols(sentence)
    marked_sentences = []
    for pred in predicate_eojeols:
        # 단어 경계를 유지하면서 첫 번째 등장만 감싸기
        pattern = re.escape(pred)
        marked = re.sub(pattern, f"<p>{pred}</p>", sentence, count=1)
        marked_sentences.append(marked)

    return marked_sentences, predicate_eojeols

def find_arg_in_pred(p):
    ## 0번은 [CLS]라 제외해야함
    p = p[1:]
    if 'B-ARG0' in p:
        start_idx = p.index('B-ARG0')
        end_idx = start_idx
        try:
            while p[end_idx+1] == 'I-ARG0':
                end_idx += 1
        except:
            pass
        arg0 = (start_idx+1, end_idx+1)
    else:
        arg0 = None
    
    if 'B-ARG1' in p:
        start_idx = p.index('B-ARG1')
        end_idx = start_idx
        try:
            while p[end_idx+1] == 'I-ARG1':
                end_idx += 1
        except:
            pass
        arg1 = (start_idx+1, end_idx+1)
    else:
        arg1 = None

    return arg0, arg1

def get_korean_predicate_lemma(eojeol):
    morphs = kkma.pos(eojeol)

    for i in range(len(morphs) - 1):
        w1, p1 = morphs[i]
        w2, p2 = morphs[i + 1]

        # Case 1: 동사/형용사 + 어미
        if p1 in ['VV', 'VA', 'VX'] and p2.startswith('E'):
            return w1 + '다'
        
        # Case 2: NNG + XSV → 공개하다
        if p1 == 'NNG' and p2 == 'XSV':
            return w1 + '하다'
        
        # Case 3: XR + XSA → 아름답다
        if p1 == 'XR' and p2 == 'XSA':
            return w1 + '다'
        
    # fallback: XSV or XSA 단독 (예: 하, 되)
    for i in range(len(morphs)):
        w, p = morphs[i]
        if p in ['XSV', 'XSA']:
            return w + '다'
        elif p in ['VV', 'VA', 'VX']:
            return w + '다'
    
    return None  # 실패한 경우
        


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="klue/bert-base", help="model to use")
    parser.add_argument("--model_checkpoint_path", type=str, default="./checkpoint", help="model checkpoint directory")
    parser.add_argument('--csv_path', type=str, default="./social_line.csv")
    parser.add_argument('--result_file_path', type=str, default="./result.csv")
    args = parser.parse_args()

    seed_everything(42)
    label_names = [ 'O','B-ARG0', 'B-ARG1']
    metric = evaluate.load('seqeval')
    model_checkpoint = args.model_checkpoint_path
    base_model = args.model
    tokenizer = AutoTokenizer.from_pretrained(base_model, additional_special_tokens=['<p>', '</p>'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_config =  AutoConfig.from_pretrained(model_checkpoint)
    model_config.num_labels = 3 
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    
    
    df = pd.read_csv(args.csv_path)
    sentence = [line for line in tqdm(df['line'])]
    df['extracted_srl'] = [[] for _ in range(len(df))]
    kkma = Kkma()

    for i, s in tqdm(enumerate(sentence)):
        try:
            tagged_s, predicates = mark_predicates_in_sentence(s)
            tagged_s = [s.split(" ") for s in tagged_s]
            tokenized = tokenizer(tagged_s, return_tensors="pt", padding=True, truncation=True, max_length=512, is_split_into_words=True, return_offsets_mapping=True).to(device)
            with torch.no_grad():
                outputs = model(tokenized.input_ids, tokenized.attention_mask)
            predictions = outputs.logits.argmax(dim=-1).cpu().tolist()
            word_id = [tokenized.word_ids(batch_index=i) for i in range(len(predictions))]
            for b in range(len(predictions)):
                pred = [label_names[p] for p in predictions[b]]
                arg0_idx, arg1_idx = find_arg_in_pred(pred)
                arg0_word_id = (word_id[b][arg0_idx[0]],word_id[b][arg0_idx[1]]) if arg0_idx is not None else None
                arg1_word_id = (word_id[b][arg1_idx[0]],word_id[b][arg1_idx[1]]) if arg1_idx is not None else None
                arg0_text = " ".join(tagged_s[b][arg0_word_id[0]:arg0_word_id[1]+1]) if arg0_word_id is not None else None
                arg1_text = " ".join(tagged_s[b][arg1_word_id[0]:arg1_word_id[1]+1]) if arg1_word_id is not None else None
                if arg0_text is None and arg1_text is None:
                    continue
                else:
                    res = {'arg0': arg0_text, 'arg1' : arg1_text, 'predicate' : predicates[b], 'lemma' : get_korean_predicate_lemma(predicates[b])}
                    df['extracted_srl'][i].append(res)
        except Exception:
            pass
    
    df.to_csv(args.result_file_path)
        

