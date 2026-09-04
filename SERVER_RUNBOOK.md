# 服务器运行手册（AutoDL，2x RTX 5090 或 1x RTX PRO 6000）

代码 = GitHub main（打包时记下 `git rev-parse --short HEAD`），
不含 .git、.venv、checkpoints、旧训练日志。

状态（2026-09-04）：第 5 步的四条 sweep 全部跑完，24 个
`results/train_*_k{8n8,32n2}_s{0,1,2}.jsonl` 已提交，这就是报告里的全部训练
结果。Phase 3 的配对梯度测量（gradvar）**因时间限制不重跑**，报告中作撤回处理；
`experiments/grad_variance.py` 保留了修好的探针和 pair 估计器，测试通过但未在
模型上运行。下面只剩可选的 rloo / js_loo 补跑。

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

## 5. 训练扫描（已完成，留作记录；重跑会覆盖 results/train_*.jsonl）
    mkdir -p logs
    CUDA_VISIBLE_DEVICES=0 nohup python main.py sweep --seeds 0 1 2 --k 32 --n 2 --baselines vanilla js_fixed --track-grad-var --steps 150 --save-checkpoint > logs/n2_a.log 2>&1 &
    CUDA_VISIBLE_DEVICES=0 nohup python main.py sweep --seeds 0 1 2 --k 32 --n 2 --baselines js_pooled global  --track-grad-var --steps 150 --save-checkpoint > logs/n2_b.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup python main.py sweep --seeds 0 1 2 --k 8  --n 8 --baselines vanilla js_fixed --track-grad-var --steps 150 --save-checkpoint > logs/n8_a.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup python main.py sweep --seeds 0 1 2 --k 8  --n 8 --baselines js_pooled global  --track-grad-var --steps 150 --save-checkpoint > logs/n8_b.log 2>&1 &

可选补跑，两个未进入 Phase 4b 的留一臂（每臂约 35 分钟）：

    CUDA_VISIBLE_DEVICES=1 nohup python main.py sweep --seeds 0 1 2 --k 8  --n 8 --baselines rloo js_loo --track-grad-var --steps 150 > logs/n8_c.log 2>&1 &
    CUDA_VISIBLE_DEVICES=0 nohup python main.py sweep --seeds 0 1 2 --k 32 --n 2 --baselines rloo js_loo --track-grad-var --steps 150 > logs/n2_c.log 2>&1 &

## 6. 监控
    tail -n 3 logs/*.log
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
    ls results/train_*.jsonl | wc -l     # 24 个即 Phase 4b 全部完成（补跑 rloo/js_loo 则 36 个）

## 7. 收尾：汇总并打包（checkpoint 约 24 GB，不打包）
    python main.py compare --glob 'train_*_k8n8_s*.jsonl'
    python main.py compare --glob 'train_*_k32n2_s*.jsonl'
    python main.py figures
    tar czf /root/autodl-fs/results_new.tgz results/train_*.jsonl results/fig*.png results/fig*.pdf logs
然后从文件存储下载 results_new.tgz，本地先归档旧结果再解压。跑完记得关机。
