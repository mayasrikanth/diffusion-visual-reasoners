import os
import pickle
import csv
import torch
import numpy as np 

from datasets_t2i import WinogroundDataset, CCUBDataset, CFlowersDataset
from models.clip_r_precision import CLIPRPrecision
from clip import clip
from tqdm import tqdm

from PIL import Image
# Assumes you've run prep_eval_data with the relevant data 

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", default="", type=str,
                        help="Dataset to run [C-CUB, C-Flowers, winoground].")
    parser.add_argument("--subset", default="", type=str,
                        help="Dataset to run [images, groundtruth].")
    parser.add_argument("--comp_type", default="", type=str,
                        help="Type of composition [color, shape].")
    parser.add_argument("--split", default="", type=str,
                        help="Test split to use [test_seen, test_unseen, test_swapped].")
    parser.add_argument('--ckpt', type=str, required=True,
                        help="path to CLIP model")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--pred_path", default="", type=str,
                        help="Path to the generated image results.")
    parser.add_argument("--out_path", default="clip_r_precision_results.pkl", type=str,
                        help="path to output (this script outputs a pickle file")
    parser.add_argument("--csv_out", default="clip_r_precision_results.csv", type=str,
                        help="path to output (this script outputs a CSV file")
    parser.add_argument("--clipr_scores_csv", default="all_clipR_winoground_glidelaion_noFT.csv", type=str,
                        help="path to output (this script outputs a CSV file")

    args = parser.parse_args()

    # separate result files for each split
    with open(args.pred_path, "rb") as f:
        result = pickle.load(f)

    # print("RESULT: ", result)
    print("LENGHT OF RESULT INITIALLY: ", len(result))

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # # model creation
    model = CLIPRPrecision()

    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.cuda()
    model.eval()

    image_transform = model.preprocess
    # image_transform = clip.load('RN101', jit=False)
    tokenizer = clip.tokenize

    # dataset creation
    # data_dir = "./data"
    #data_dir = "/home/jasonlin/repos/datasets/t2i_benchmark"
    data_dir = 'someshit'
    # data_dir should contain data.pkl and split.pkl 
    if args.dataset == "C-CUB":
        images_txt_path = os.path.join(data_dir, "C-CUB", "images.txt")
        bbox_txt_path = os.path.join(data_dir, "C-CUB", "bounding_boxes.txt")
        dataset = CCUBDataset(
            data_dir,
            args.dataset,
            args.comp_type,
            args.split,
            image_transform,
            tokenizer,
            images_txt_path,
            bbox_txt_path
        )
    elif args.dataset == "C-Flowers":
        class_id_txt_path = os.path.join(data_dir, "C-Flowers", "class_ids.txt")
        dataset = CFlowersDataset(
            data_dir,
            args.dataset,
            args.comp_type,
            args.split,
            image_transform,
            tokenizer,
            class_id_txt_path
        )
    elif args.dataset == "winoground":
        dataset = WinogroundDataset(
            data_dir,
            args.dataset,
            args.comp_type,
            args.split,
            args.subset,
            image_transform,
            tokenizer,
            class_id_txt_path=None
        )
        print("dataset loaded OK!")
    else:
        raise Exception(f"Unknown dataset: {args.dataset}")

    # run prediction
    clip_result = []
    for entry in tqdm(result):
        img_id, cap_id, gen_img_path, r_precision_prediction = entry

        image = Image.open(gen_img_path).convert("RGB")
        if dataset.image_transform:
            image = dataset.image_transform(image)
        image = image.unsqueeze(0).cuda()
        try:
            if args.split == "test_swapped":
                swapped = True
            else:
                swapped = False
            text_conditioned = dataset.get_text(img_id, cap_id, raw=False, swapped=swapped).cuda()
        except:
            continue
        mismatched_captions = dataset.get_mismatched_caption(img_id).cuda()
        all_texts = torch.cat([text_conditioned, mismatched_captions], 0)
        print("SHAPE OF all_texts: ", all_texts.shape)
        R_limit = int(0.50 * len(all_texts))
        with torch.no_grad():
            image_features, text_features = model(image, all_texts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            
            #clip_prediction = torch.argsort(logits_per_image, dim=1, descending=True)[0, 0].item()
            clip_prediction = torch.argsort(logits_per_image, dim=1, descending=True)[:, :R_limit] #.item()
            print("CLIP PREDICTION SHAPE: ", clip_prediction.shape)
            print(clip_prediction)

        new_entry = (img_id, cap_id, gen_img_path, clip_prediction)
        clip_result.append(new_entry)

    # import pdb; pdb.set_trace()
    with open(args.out_path, 'wb') as f:
        pickle.dump(clip_result, f)
    
    # Compute average clip-R score 
    # nums = [num for (path, x, f, num) in clip_result]
    # print("AVERAGE CLIP-R SCORE: ", np.mean(nums))

    # with open("/home/mayashar/Desktop/diffusion-visual-reasoners/comp-t2i-dataset/eval_data/data.pkl", "rb") as wino_df_file:
    #     ex_data = pickle.load(wino_df_file)
    #     with open(args.csv_out, 'w', newline='') as csvfile:
    #         csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    #         for (img_id, cap_id, gen_img_path, r_precision_prediction) in clip_result:
    #             csv_writer.writerow([img_id, ex_data[img_id][0]["text"], r_precision_prediction])
    #     # '/home/jasonlin/repos/datasets/t2i_benchmark/winoground/wino_gt_clipr.csv'

    # print("Output scores file: ", args.clipr_scores_csv)

    # with open(args.clipr_scores_csv, 'w', newline='') as csvfile:
    #         csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    #         for (img_id, cap_id, gen_img_path, r_precision_prediction) in clip_result:
    #             csv_writer.writerow([img_id, r_precision_prediction])