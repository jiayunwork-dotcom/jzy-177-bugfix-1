# 八阵元线阵常驻 MUSIC 测向后端

把原先“采完存盘、MATLAB 离线跑”的测向流程改成常驻服务：每位同学开一个
**实验会话**，定好阵列参数与估计配置后，把复基带快拍分批流式追加进来，
随时能轮询或订阅当前来波方向估计。服务重启后会话接着追加，已有结果不丢。

* 语言：Python 3.11，HTTP：FastAPI + uvicorn，矩阵运算：NumPy（无第三方
  测向工具包，协方差/增量更新/源数判定/空间平滑/谱扫描细化/CRB/合成为
  全部自行实现）。
* 模块按职责拆分：

  | 模块 | 职责 |
  |---|---|
  | `app/doa/geometry.py` | ULA 几何、导向矢量、栅瓣别名 |
  | `app/doa/covariance.py` | 样本协方差、前后向空间平滑、滑动/遗忘增量更新 |
  | `app/doa/detection.py` | MDL / AIC 信源数判定 |
  | `app/doa/music.py` | MUSIC 粗扫、黄金分割细化、功率最小二乘拟合 |
  | `app/doa/crb.py` | 单信源白噪声确定性模型 CRB |
  | `app/doa/synthesis.py` | 固定种子逐位可复现的快拍合成 |
  | `app/doa/engine.py` | 无状态测向总装与全部人读校验 |
  | `app/sessions/` | 会话模型、文件存储、并发管理器、结果发布订阅 |
  | `app/api/` | FastAPI 路由、请求模型、统一错误出口 |

## 快速开始

```bash
pip install -r requirements.txt
python -m pytest                 # 47 项测试，逐条核对验收关系
uvicorn app.api.main:app --port 8000
# 或：python -m app
```

容器（基础镜像 `python:3.11-slim`，uvicorn 监听 8000，`/data` 可挂载）：

```bash
docker compose up --build
# 或
docker build -t doa-backend .
docker run -p 8000:8000 -v $PWD/data:/data doa-backend
```

数据目录由环境变量 `DOA_DATA_DIR` 指定（镜像内默认 `/data`，本机裸跑
默认 `./data`）。启动后访问 `http://localhost:8000/docs` 查看交互式 API。

## HTTP 接口

### 会话

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/sessions` | 建会话：阵列参数、增量更新配置、默认估计配置 |
| `GET` | `/sessions` | 列出全部会话及当前状态 |
| `GET` | `/sessions/{id}` | 会话状态（累计/有效快拍数、最新结果版本） |
| `POST` | `/sessions/{id}/batches` | 追加一批快拍（实部/虚部分开，带批次号） |
| `POST` | `/sessions/{id}/estimate` | 出一次估计（可临时覆盖估计配置） |
| `GET` | `/sessions/{id}/results?since_version=N` | 版本轮询，只返回版本号 > N 的结果 |
| `GET` | `/sessions/{id}/results/stream?after_version=N` | SSE 订阅推送（先补发历史，再推新结果） |
| `POST` | `/synthesize` | 无状态合成（可选顺手估计），闭环自测用 |

建会话示例（八阵元半波长 ULA，默认定长滑动窗口 W=256）：

```json
POST /sessions
{
  "name": "lab3-zhang",
  "array": {"m": 8, "spacing": 0.5, "wavelength": 1.0},
  "update": {"method": "sliding", "window": 256},
  "estimate": {"source_mode": "mdl", "step_deg": 0.5,
               "smoothing": false, "sub_len": null}
}
```

指数遗忘因子：`"update": {"method": "forgetting", "alpha": 0.9922}`
（`0.9922 = (256-1)/(256+1)`，与 W=256 等效记忆）。

追加示例（实部虚部各为 M×K 嵌套数组；每批大小不固定，K≤10000）：

```json
POST /sessions/{id}/batches
{"batch_id": "frame-00017", "real": [[...]], "imag": [[...]]}
```

同一 `batch_id` 重复提交是幂等的：不丢批、不重复计入，返回
`{"deduplicated": true, "accepted_snapshots": 0}`。

估计响应（节选）：

```json
{
  "version": 3,
  "total_snapshots": 600,
  "config_used": {"source_mode": 2, "step_deg": 0.5, "...": "..."},
  "result": {
    "n_sources": 2,
    "source_counts": {"mdl": 2, "aic": 2},
    "angles_deg": [-24.99, 30.00],
    "per_angle": [
      {"angle_deg": -24.99, "power": 1.01, "snr_db": 20.1,
       "crb_std_deg": 0.0097, "true_angle_deg": -25.0,
       "error_over_crb": 0.53}
    ],
    "noise_power": 0.0098,
    "n_effective_snapshots": 600.0,
    "smoothing": {"enabled": false, "sub_len": null, "...": "..."},
    "grating_ambiguous": false, "grating_pairs": [],
    "warning": null
  }
}
```

`error_over_crb = |估计 − 真值| / CRB 标准差`，仅当估计请求带了
`true_angles`（或走 `/synthesize` 自动带上真值）时给出。

合成接口示例（`include_data` 默认 true，返回的实部/虚部可用同一种子
逐位复现；`estimate` 非空时顺带测向并自动填真值/CRB 比值）：

```json
POST /synthesize
{
  "array": {"m": 8, "spacing": 0.5, "wavelength": 1.0},
  "sources": [
    {"angle_deg": -8.0, "power": 1.0, "coherent": true},
    {"angle_deg": 8.0,  "power": 1.0, "coherent": true}],
  "snr_db": 20.0, "n_snapshots": 600, "seed": 123,
  "estimate": {"source_mode": 2, "smoothing": true}
}
```

## 轮询与订阅

* 轮询：记录本地见过的最大版本号 V，请求
  `GET .../results?since_version=V`，只返回更新的结果；
* 订阅：`GET .../results/stream`（Server-Sent Events），每条
  `event: result` 的 `data` 与轮询单条记录结构相同，15 秒无新结果发
  keepalive 注释行；建立连接时先补发 `after_version` 之后的历史结果。

## 规模上限与人读错误

* 阵元 ≤ 64；单批快拍 ≤ 10000；扫描点数 ≤ 200000（超出 400）；
* 信源数 ≥（子阵）阵元数、首次估计累计快拍 < 阵元数、波长/间距不为正、
  数据含 NaN、实部虚部形状不一致、位置误差长度不符等均返回 400，
  响应体 `{"error": {"type": ..., "message": "中文人读说明"}}`；
* 会话不存在返回 404。

## 持久化布局

每个会话一个目录 `<DOA_DATA_DIR>/<session_id>/`：

```
config.json    # 阵列/更新/估计配置与元数据（原子覆盖写）
state.npz      # C、有效快拍数、窗口内快拍、已计入批次号（os.replace 原子落盘）
results.jsonl  # 每次估计一行，version 从 1 单调递增（只追加，fsync）
```

批次号集合与协方差状态同存于 `state.npz`，二者原子一致：崩溃不会出现
“批次记账了但协方差没更新”或反之。

## 实验论证（默认更新方式的取舍）

见 [`docs/design.md`](docs/design.md) 与
[`experiments/tracking_experiment.py`](experiments/tracking_experiment.py)
（可复跑，结果 JSON 在 `experiments/results/`）。**默认定长滑动窗口
W=256**：阶跃后约 0.56W（≈144 条快拍）跟上，稳态 RMSE 约 0.024°，且
跟拍速度与阶跃前历史长度无关；遗忘因子省内存、稳态精度略好，但旧数据
几何拖尾，长时间运行后阶跃跟拍要 ~4.6W。两种方式都完整支持。

## 测试

```bash
python -m pytest -q          # 47 passed
```

统计类用例全部固定随机种子，覆盖：高 SNR 偏差 <0.1°、复标量不变性、
栅瓣角度对、方位变号、相干源平滑前后对比、SNR 0→20 dB 的 RMSE 与
3×CRB、高 SNR 下 MDL 源数、整批/分批估计一致、增量协方差对一次性定义
的 1e-12 对齐与长程厄米性、批次幂等、并发串行等价、会话隔离、重启
恢复、版本轮询、SSE 推送、全部人读错误与规模上限。
