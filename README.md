# DMWM
This is a repository for the paper "DMWM: Dual-Mind World Model with Long-Term Imagination" accepted by NeurIPS 2025

!["The proposed dual-mind world model"](Figure/pipline.png "The proposed dual-mind world model")

## :memo: Installation
You can create and activate the environment as follows:

```bash
conda create -n dmwm python==3.7
conda activate dmwm
pip install -r requirements.txt
```
Suggested GPU:
All experiments in the paper were conducted on a single NVIDIA RTX 3090 GPU. We also tried the NVIDIA RTX 3080 GPU, which can also work.

Training Env: [Google DeepMind Infrastructure for Physics-Based Simulation](https://github.com/google-deepmind/dm_control#rendering).

## :rocket: Training
To train the model(s) in the paper, run this command:
Taking the "walker-walk" task as an example:
```bash
python main.py --algo dreamer --env walker-walk --action-repeat 2 --id your_named-experiement
```

Some useful commands:
```bash
python main.py --algo dreamer --env walker-walk --action-repeat 2 --logic-overshooting-distance 10 --id your_named-experiement
```

```bash
python main.py --algo dreamer --env walker-walk --action-repeat 2 --planning-horizon 50 --id your_named-experiement
```

```bash
python main.py --algo dreamer --env walker-walk --action-repeat 2 --planning-horizon 50 --logic-overshooting-distance 50 --id your_named-experiement
```

## :rainbow: Evaluation

To evaluate my model on control tasks, run:

```bash
python main.py --models saved_path --test
```

## :heart: Acknowledgement
Our implementation is based on [Dreamer](https://arxiv.org/abs/1912.01603) (for System 1) and [Logic-Integrated Neural Network (LINN)](https://arxiv.org/abs/2008.09514) (as the basic framework with the proposed deep logical inference and automatic logic learning from environment dynamics for System 2). Thanks for their great open-source work!

## :page_facing_up: License
All content in this repository is under the [MIT license](https://opensource.org/licenses/MIT).

## :star: Citation
If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```bibtex
@article{wang2025dmwm,
  title={DMWM: Dual-Mind World Model with Long-Term Imagination},
  author={Wang, Lingyi and Shelim, Rashed and Saad, Walid and Ramakrishnan, Naren},
  journal={arXiv preprint arXiv:2502.07591},
  year={2025}
}
```

## :rocket: Some Test Results
High Data Efficiency and Robust Planning Over Extended Horizon Size:
!["Data Efficiency"](Figure/ES4.png)
!["Data Efficiency"](Figure/EPS4.png)
!["Robust Planning Over Extended Horizon Size"](Figure/HS8.png)

