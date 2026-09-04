# 服务器运行手册（AutoDL，2x RTX 5090 或 1x RTX PRO 6000）

代码 = GitHub main `db3c777`（不含 .git、.venv、checkpoints、旧训练日志）。

## 1. 解压到数据盘
    tar xzf /root/autodl-fs/grpo_eva_core.tgz -C /root/autodl-tmp
    cd /root/autodl-tmp/GRPO_EVA

## 2. 环境变量与工具（每个新 shell 都要有这两个变量）
    echo 'export HF_HOME=/root/autodl-tmp/hf HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc
    pip install uv && apt-get update -qq && apt-get install -y -qq tmux

## 3. 建环境（约 5 GB 下载，放 tmux 里）
    tmux new -s run
    uv sync && source .venv/bin/activate

## 4. 验证（依次通过再往下）
    python main.py smoke                 # 期望 PASS -- Phase 0 gate cleared
    python main.py eval --n-problems 8   # 顺便下载模型与 GSM8K
    python -m pytest tests/ -q           # 期望 162 passed

## 5. 启动（两张卡各两个进程；单张 PRO 6000 则全部用 CUDA_VISIBLE_DEVICES=0）
    mkdir -p logs
    CUDA_VISIBLE_DEVICES=0 nohup python main.py sweep --seeds 0 1 2 --k 32 --n 2 --baselines vanilla js_fixed --track-grad-var --steps 150 --save-checkpoint > logs/n2_a.log 2>&1 &
    CUDA_VISIBLE_DEVICES=0 nohup python main.py sweep --seeds 0 1 2 --k 32 --n 2 --baselines js_pooled global  --track-grad-var --steps 150 --save-checkpoint > logs/n2_b.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup python main.py sweep --seeds 0 1 2 --k 8  --n 8 --baselines vanilla js_fixed --track-grad-var --steps 150 --save-checkpoint > logs/n8_a.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup python main.py sweep --seeds 0 1 2 --k 8  --n 8 --baselines js_pooled global  --track-grad-var --steps 150 --save-checkpoint > logs/n8_b.log 2>&1 &

## 6. 监控
    tail -n 3 logs/*.log
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
    ls results/train_*.jsonl | wc -l     # 24 个即全部完成

## 7. 某张卡空出来后跑 gradvar v2（约 1.5-2 小时）
    CUDA_VISIBLE_DEVICES=0 nohup python experiments/grad_variance.py --batches 200 --k 8 --n 8 --seed 0 --out results/grad_variance_k8n8_v2.json > logs/gradvar.log 2>&1 &

## 8. 收尾：汇总并打包（checkpoint 约 24 GB，不打包）
    python main.py compare && python main.py figures
    tar czf /root/autodl-fs/results_new.tgz results/train_*.jsonl results/grad_variance_k8n8_v2.json results/fig*.png results/fig*.pdf logs
然后从文件存储下载 results_new.tgz，本地先归档旧结果再解压。跑完记得关机。
