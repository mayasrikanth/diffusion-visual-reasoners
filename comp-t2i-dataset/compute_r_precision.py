import pickle
import numpy as np

from utils import gather_by_pair, balance_candidates

def r_precision(candidates, num_chunks=10):
    predictions = []
    for cand in candidates:
        if cand["prediction"] == 0:
            predictions.append(1)
        else:
            predictions.append(0)

    num_preds = len(predictions)
    chunk_size = int(num_preds / num_chunks)

    predictions = np.array(predictions)
    np.random.shuffle(predictions)

    chunks = np.zeros(num_chunks)
    for i in range(num_chunks):
        chunks[i] = np.average(predictions[i * chunk_size : (i + 1) * chunk_size])

    return np.average(chunks), np.std(chunks)

def r_precision2(candidates, num_chunks=10):
    predictions = []
    for cand in candidates:
        print(cand["prediction"])
        if 0 in cand["prediction"]: #cand["prediction"] == 0:
            predictions.append(1)
        else:
            predictions.append(0)

    num_preds = len(predictions)
    chunk_size = int(num_preds / num_chunks)

    predictions = np.array(predictions)
    np.random.shuffle(predictions)

    chunks = np.zeros(num_chunks)
    for i in range(num_chunks):
        chunks[i] = np.average(predictions[i * chunk_size : (i + 1) * chunk_size])

    return np.average(chunks), np.std(chunks)


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", default="", type=str,
                        help="Dataset to run [C-CUB, C-Flowers].")
    parser.add_argument("--comp_type", default="", type=str,
                        help="Type of composition [color, shape].")
    parser.add_argument("--split", default="", type=str,
                        help="Test split to use [test_seen, test_unseen, test_swapped].")
    # eval_data/glidelaion_noFT_drawbench_clip_preds.pkl
    # eval_data/glidelaion_FTwino_drawbench_clip_preds.pkl
    # eval_data/glideliaon_noFT_winoground_clip_preds.pkl 
    parser.add_argument("--pred_path", default="eval_data/glidelaion_FTwino_winoground_clip_preds.pkl", type=str,
                        help="Path to the generated image results.")
    args = parser.parse_args()

    # anno_data_path = f"./data/{args.dataset}/{args.comp_type}/data.pkl"
    # split_path = f"./data/{args.dataset}/{args.comp_type}/split.pkl"
    anno_data_path = '/home/mayashar/Desktop/diffusion-visual-reasoners/comp-t2i-dataset/eval_data/data.pkl' 
    
    split_path = '/home/mayashar/Desktop/diffusion-visual-reasoners/comp-t2i-dataset/eval_data/split.pkl'

    # separate result files for each split
    with open(args.pred_path, "rb") as f:
        result = pickle.load(f)

    with open(anno_data_path, "rb") as f:
        anno_data = pickle.load(f)

    with open(split_path, "rb") as f:
        split_data = pickle.load(f)

    if args.split == "test_swapped":
        gathered_results = gather_by_pair(result, anno_data)
        candidates = balance_candidates(args.dataset, args.comp_type, gathered_results)
    else:
        candidates = []
        for entry in result:
            img_id, cap_id, gen_img_path, r_precision_prediction = entry
            print(r_precision_prediction)
            candidates.append({
                "prediction": r_precision_prediction,
                "img_path": gen_img_path,
                "text": anno_data[img_id][cap_id]["text"]
            })

    R_mean, R_std = r_precision2(candidates)

    print("R Precision score: ")
    print(f"\t {R_mean * 100:.2f} +- {R_std * 100:.2f}")
