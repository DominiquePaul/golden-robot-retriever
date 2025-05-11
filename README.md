# Golden robot retriever:

### Data

We collected three datasets: 
- [briannnyee/grabbing_v4_coke](https://lerobot-visualize-dataset.hf.space/briannnyee/grabbing_v4_coke)
- [briannnyee/grabbing_v4_tape](https://lerobot-visualize-dataset.hf.space/briannnyee/grabbing_v4_tape)
- [vhartman/mate_can](https://lerobot-visualize-dataset.hf.space/vhartman/mate_can)

And we merged them the last two to a single LeRobot dataset [dopaul/merged_grabbing_dataset_v7](https://lerobot-visualize-dataset.hf.space/dopaul/merged_grabbing_dataset_v7)

### Training

We trained them using the script below.  If you don't pass a wandb api key then set `wandb.enable` to false. 

```
cd lerobot

WANDB_API_KEY=[YOUR_WANDB_API_KEY] python lerobot/scripts/train.py \
--policy.type=act \
--dataset.repo_id=dopaul/merged_grabbing_dataset_v7 \
--wandb.enable=true \
--steps=100000 \
--output_dir=lerobot/outputs/train/zrh_grasping_v7

# Upload model to HF
python lerobot/scripts/push_pretrained.py \
    --pretrained_path=lerobot/outputs/train/zrh_grasping_v7/checkpoints/last/pretrained_model \
    --repo_id=dopaul/zrh_grasping_v7
```

### To run the code

1. Copy `.env.example` and rename it to `.env`. Create an OpenAI key and replace the placeholder. 
2. Install packages by running `poetry install`
3. Run `main.py`
