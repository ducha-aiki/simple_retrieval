from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
import uvicorn
import argparse
from simple_retrieval.highlevel import SimpleRetrieval, get_default_config
import cv2
import torch
import numpy as np
import os
from time import time

app = FastAPI()

engine = None


MY_CAPABILITIES = {
    "data": {
        "engine_options": [
        {
                "default": "lightglue",
                "id": "matching_method",
                "label": "matching_method",
                "type": "option",
                "values": [
                    "snn",
                    "lightglue"
                ]
            },
                {
                "default": True,
                "id": "diffusion",
                "label": "diffusion",
                "type": "bool",
           
            },
        ],
        "images_overlays": [
            {
                "id": "rank",
                "label": "Rank",
                "type": "text",
                "default": True,
            },
            {
                "default": False,
                "id": "paths_over",
                "label": "Path overlay",
                "type": "paths"
            },
            {
                "id": "score",
                "label": "Score",
                "type": "text",
                "default": True,
            }
        ],
        #"output_information": [
        #    {
        #        "default":  True,
        #        "id": "text_info",
        #        "label": "Text information",
        #        "type": "text"
        #    }
        #],
        "search_types": [
            "image",
            "upload"
        ],
        "search_modes": [
            {"id": "similarity", "label": "Similarity", "type": "image"},
            {"id": "num-inliers", "label": "Num Inliers", "type": "image"},
            {"id": "similarity-bbox", "label": "Similarity BBox", "type": "image", "tool": "rectangle"},
            {"id": "num-inliers-bbox", "label": "Num Inliers BBox", "type": "image", "tool": "rectangle"},
            {"id": "zoom-in", "label": "Zoom in", "type": "image", "tool": "rectangle"},
            {"id": "zoom-out", "label": "Zoom out", "type": "image", "tool": "rectangle"},
        ]	
    }
}

def convert_points_to_path(pts):
    return [{"color": "#edae49", "points_xy": [[float(x), float(y)] for x, y in pts]}]

@app.middleware("http")
async def log_invalid_requests(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Log details of the invalid request
        print(f"Invalid request: {request.method} {request.url}, Error: {str(e)}")
        raise


@app.post("/capabilities")
async def capabilities(request: Request):
    try:
        payload = await request.json()
        # Process the payload (stub logic here)
        response = MY_CAPABILITIES
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

@app.post("/image")
async def image(request: Request):
    try:
        payload = await request.json()
        # Process the payload (stub logic here)
        response = {"message": "Image processed", "data": payload}
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")

@app.post("/images")
async def images(request: Request, background_tasks: BackgroundTasks):
    try:
        payload = await request.json()
        print(payload.keys())
        print(payload["data"])
        limit = int(payload["data"]["limit"])
        offset = int(payload["data"]["offset"])
        fnames = engine.ds.samples 
        t=time()
        if "query" not in payload["data"]: # browsing request
            criterion='random'
            idxs, scores = engine.get_random_shortlist(offset + limit)
            shortlist_idxs = idxs
            resorted_scores = scores
            resorted_idxs = idxs
            shortlist_scores = scores
            bboxes = [[] for _ in range(len(idxs))]
        else:
            query_fname = payload["data"]["query"]["value"]["path"]
            print(query_fname)
            use_diffusion = payload["data"]["engine_options"]["diffusion"]
            if payload["data"]["query"]["value"]["prefix"] == "upload":
                query_fname = os.path.join(engine.upload_dir, query_fname) # prepend the upload directory
            elif payload["data"]["query"]["value"]["prefix"] == "vrg_retrieval_demo":
                query_fname = os.path.join(engine.img_dir, query_fname).replace('vrg_retrieval_demo/vrg_retrieval_demo','vrg_retrieval_demo') # prepend the image directory
            try:
                search_mode = payload["data"]["query"]["search_mode"]["id"]
            except KeyError:
                search_mode = "similarity"
            print(query_fname)
            criterion = engine.config["resort_criterion"]
            crop_needed = False
            if search_mode == "zoom-in":
                criterion = "scale_factor_max"
                crop_needed = True
            elif search_mode == "zoom-out":
                criterion = "scale_factor_min"
                crop_needed = True
            elif search_mode == "similarity":
                criterion = "skip"
            elif search_mode == "num-inliers":
                criterion = "num_inliers"
            elif search_mode == "similarity-bbox":
                criterion = "skip"
                crop_needed = True
            elif search_mode == "num-inliers-bbox":
                criterion = "num_inliers"
                crop_needed = True
            else:
                raise ValueError(f"Unknown search mode: {search_mode}")
            q_img = cv2.cvtColor(cv2.imread(query_fname), cv2.COLOR_BGR2RGB)
            if crop_needed:
                h, w = q_img.shape[:2]
                tool_data = payload["data"]["query"]["search_mode"]["tool_data"]
                x1, x2, y1, y2 = tool_data["x1"], tool_data["x2"], tool_data["y1"], tool_data["y2"]
                x1, x2 = int(x1*w), int(x2*w)
                y1, y2 = int(y1*h), int(y2*h)
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                print (f"Cropping image ({w, h}) to: {xmin, xmax, ymin, ymax}")
                q_img = q_img[ymin:ymax, xmin:xmax]
            hq, wq = q_img.shape[:2]
            num_nn = max(300, min((offset + limit)*10, 1000))
            print (f'Searching for {num_nn} nearest neighbors')
            shortlist_idxs, shortlist_scores = engine.get_shortlist(q_img,
                                                                    num_nn=num_nn,
                                                                    manifold_diffusion=use_diffusion,
                                                                    Wn=engine.Wn)
            if criterion == "skip":
                idxs = shortlist_idxs
                scores = shortlist_scores
                resorted_scores = shortlist_scores
                resorted_idxs = shortlist_idxs
                bboxes = [[] for _ in range(len(idxs))]
            else:
                matching_method = payload["data"]["engine_options"]["matching_method"]
                with torch.inference_mode():
                    resorted_idxs, resorted_scores, resorted_bboxes = engine.resort_shortlist(q_img, shortlist_idxs,
                                                        matching_method=matching_method,
                                                        criterion=criterion,
                                                        device=engine.config["device"])

                irrelevant_mask = resorted_scores == 0
                # When the spatial verification returns zeros, we use the original shortlist
                idxs = resorted_idxs[~irrelevant_mask]
                scores = resorted_scores[~irrelevant_mask]
                bboxes = resorted_bboxes[~irrelevant_mask]
                idxs_set = set(idxs)
                rest_of_idxs = [i for i in shortlist_idxs if i not in idxs_set]
                rest_of_scores = [shortlist_scores[i] for i, idx in enumerate(shortlist_idxs) if i not in idxs_set]
                idxs = list(idxs) + rest_of_idxs
                scores = list(scores) + rest_of_scores
                bboxes = list(bboxes) + [[]]*len(rest_of_scores)
        print (criterion, scores[:10])
        shortlist_idxs = shortlist_idxs.tolist()
        print (len(scores), len(bboxes))
        indir1 = engine.config["input_dir"]
        if not indir1.endswith('/'):
            indir1+='/'
        output ={"data": {"images": [{"path": fnames[idx].replace(indir1, "vrg_retrieval_demo/"),
                                    "overlays": {"rank": f"Rank: {str(int(i)+offset)}",
                                                 "score": f"{criterion}: {resorted_scores[i+offset]:.3g}, global_similarity: {shortlist_scores[shortlist_idxs.index(idx)]:.3g}",
                                                 "paths_over": convert_points_to_path(bboxes[i+offset]) if len(bboxes) > 0 else []},
 
                                       "output_information": { "text_info": f"{scores[i+offset]:.3f}"},                   
                                    "prefix": "vrg_retrieval_demo"}
                                    for i, idx in enumerate(idxs[offset:offset+limit])]}}
        if "query" in payload["data"]:
            output["data"]["query_image"] = {"overlays": {"rank": "", "score": "Query image"}}
            output["data"]["query_text"] = f"Processed in {time()-t:.4g} seconds"
        background_tasks.add_task(torch.cuda.empty_cache)
        return output
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple retrieval server')
    parser.add_argument('--input_dir', type=str, default='/Users/oldufo/datasets/goose',
                        help='Directory containing images')
    parser.add_argument('--upload_dir', type=str, required=True)
    parser.add_argument('--query_fname', type=str, 
                        help='Query image filename')
    parser.add_argument('--global_desc_dir', type=str, default='./tmp/global_desc')
    parser.add_argument('--use_diffusion',  action='store_true')
    parser.add_argument('--global_features', type=str, default='dinosalad')
    parser.add_argument('--local_features', type=str, default='xfeat')
    parser.add_argument('--local_desc_dir', type=str, default='./tmp/local_desc')
    parser.add_argument('--resort_criterion', type=str, default='scale_factor_max', choices=['scale_factor_max', 'scale_factor_min', 'num_inliers'])	
    parser.add_argument('--num_nn', type=int, default=10)
    parser.add_argument('--inl_th', type=float, default=3.0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--force_recache', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--global_desc_batch_size', type=int, default=4)
    parser.add_argument('--local_desc_batch_size', type=int, default=4)
    parser.add_argument('--matching_method', type=str, default='snn')
    args = parser.parse_args()
    config = get_default_config()
    for k, v in vars(args).items():
        config[k] = v
    print (f"Config: {config}")
    engine = SimpleRetrieval(img_dir=args.input_dir, index_dir=args.global_desc_dir, config=config)
    engine.upload_dir = args.upload_dir
    print (engine)
    engine.create_global_descriptor_index(args.input_dir,
                                     args.global_desc_dir)
    engine.create_local_descriptor_index(args.input_dir)
    uvicorn.run(app, host="0.0.0.0", port=5000)