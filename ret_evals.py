from tqdm import tqdm
import argparse, json


def mrr_at(k, ls_rets, ls_golds):
    ls_mrr = []

    for i, rets, golds in (pbar := tqdm(zip(range(len(ls_golds)), ls_rets, ls_golds))):
        first_relevant_rank = None
        
        golds_stripped = [''.join(gold.split()) for gold in golds]
        rets_stripped = [''.join(ret.split()) for ret in rets]
        
        for r, ret_item in enumerate(rets_stripped):
            if any(gold_item in ret_item for gold_item in golds_stripped):
                if r < k:
                    if first_relevant_rank is None:
                        first_relevant_rank = r + 1
                    
        ls_mrr.append(1 / first_relevant_rank if first_relevant_rank else 0)
        pbar.set_description(f"MRR@{k} {sum(ls_mrr) / len(ls_golds):.4f}")
        
    return sum(ls_mrr) / len(ls_golds)


def map_at(k, ls_rets, ls_golds):
    ls_apk = []
    for i, rets, golds in (pbar := tqdm(zip(range(len(ls_golds)), ls_rets, ls_golds))):
        ap_sum = 0
        found_golds = []

        golds_stripped = [''.join(gold.split()) for gold in golds]
        rets_stripped = [''.join(ret.split()) for ret in rets]
        
        for r, ret_item in enumerate(rets_stripped):
            if any(gold_item in ret_item for gold_item in golds_stripped):
                if r < k:
                    # Compute precision at this rank for this query
                    count = 0
                    for gold_item in golds_stripped:
                        if gold_item in ret_item and not gold_item in found_golds:
                            count =  count + 1
                            found_golds.append(gold_item)
                    p_at_r = count / (r+1)
                    ap_sum += p_at_r

        # Calculate metrics for this query
        ls_apk.append(ap_sum / min(len(golds_stripped), k))
        pbar.set_description(f"MAP@{k} {sum(ls_apk) / len(ls_golds):.4f}")
        
    return sum(ls_apk) / len(ls_golds)


def hits_at(k, ls_rets, ls_golds):
    hits = 0
    for i, rets, golds in (pbar := tqdm(zip(range(len(ls_golds)), ls_rets, ls_golds))):
        is_hit = False
        golds_stripped = [''.join(gold.split()) for gold in golds]
        rets_stripped = [''.join(ret.split()) for ret in rets]
        
        for ret_item in rets_stripped[:k]:
            if any(gold_item in ret_item for gold_item in golds_stripped):
                is_hit = True
                    
        hits += int(is_hit)
        pbar.set_description(f"Hits@{k} {hits/(i+1):.4f}")
        
    return hits / len(ls_golds)





def main_eval(file_name):
    print(f'For file: {file_name}')
    with open(file_name, 'r') as file:
        data = json.load(file)
    retrieved_lists = []
    gold_lists  = []

    for d in data:
        if d['question_type'] == 'null_query':
            continue
        retrieved_lists.append([m['text'] for m in d['retrieval_list']])
        gold_lists.append([m['fact'] for m in d['gold_list']])     

    # Calculate metrics
    hit10 = hits_at(10, retrieved_lists, gold_lists)
    hit4 = hits_at(4, retrieved_lists, gold_lists)
    map10 = map_at(10, retrieved_lists, gold_lists)
    mrr10 = mrr_at(10, retrieved_lists, gold_lists)

    print(hit10)
    print(hit4)
    print(map10)
    print(mrr10)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='File Name')
    args = parser.parse_args()

    main_eval(args.file)
    