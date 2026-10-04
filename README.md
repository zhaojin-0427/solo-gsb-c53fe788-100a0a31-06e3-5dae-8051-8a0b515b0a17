# 推理图缓冲区规划服务

基于 **Python + FastAPI + OR-Tools (CP-SAT)** 的推理图缓冲区规划 HTTP API。

给定一张推理图的若干条**互斥执行路径**（每次运行只走其中一条）、整图**截止时间**
和**显存上限**，服务为每条路径排出算子时间轴，并为所有张量分配缓冲区地址：

- 先**最小化地址空间上界**（各路径峰值的最大值，互斥路径可复用地址）；
- 再按**张量 ID 序 → 路径 ID 序 → 算子 ID 序**的字典序，依次最小化各张量地址、
  各路径起点、各算子起点；
- 每个目标都求解到被证明最优才固定，**未被证明最优时结果不会标记为 `optimal`**。

## 快速开始（Docker Compose 一键启动）

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`：

- `POST /plans` —— 规划接口（见下文示例）
- `GET  /healthz` —— 健康检查
- `GET  /docs` —— Swagger UI（可交互调试 `POST /plans`）

## 请求模型

`POST /plans`，请求体（JSON）：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `deadline` | int ≥ 0 | 截止时间：每条路径的结束时刻都不得超过它 |
| `memory_limit` | int ≥ 0 | 显存上限：地址空间上界不得超过它 |
| `paths[]` | array，非空 | 互斥执行路径列表 |
| `paths[].id` | string | 路径 ID（全局唯一） |
| `paths[].operators[].id` | string | 算子 ID（路径内唯一） |
| `paths[].operators[].duration` | int ≥ 1 | 算子耗时（正整数） |
| `paths[].operators[].dependencies` | string[] | 依赖算子 ID 列表，全图须**无环** |
| `paths[].tensors[].id` | string | 张量 ID（路径内唯一；跨路径同名即同一共享张量） |
| `paths[].tensors[].bytes` | int ≥ 1 | 张量字节数（正整数） |
| `paths[].tensors[].alignment` | int ≥ 1 且为 2 的幂 | 地址对齐 |
| `paths[].tensors[].base` | string 或 null | 基张量 ID；设置后本张量为视图 |
| `paths[].tensors[].offset` | int ≥ 0 | 视图在基张量中的字节偏移（`base` 为 null 时须为 0） |
| `paths[].tensors[].producer` | string 或 null | 唯一生产者算子 ID；null 表示外部输入 |
| `paths[].tensors[].consumers` | string[] | 消费者算子 ID 列表 |
| `paths[].tensors[].is_output` | bool | 是否为该路径的最终输出 |

校验规则（违反即返回 **422**）：依赖无环且引用存在的算子；生产者/消费者/基张量
引用存在；视图满足 `offset + bytes ≤ base.bytes`；基张量链无环；**同名张量在不同
路径中的 `bytes`/`alignment`/`base`/`offset` 必须一致**；各类 ID 无重复；`alignment`
为 2 的幂；时长、字节数为正整数等。

## 规划语义

- **时间轴**：算子满足依赖序；消费者必在其消费张量的生产者结束之后开始；无依赖
  算子可并行（无资源约束）。路径区间为 `[最早算子开始, 最晚算子结束]`，且不超过
  `deadline`。各路径互斥，调度彼此独立。
- **张量生命周期**：从产出时刻（生产者结束；外部输入则为路径开始）存活到最后
  消费结束，占用半开字节区间 `[address, address+bytes)`；最终输出留至路径结束。
  存活窗口为空的张量不占用地址空间。
- **视图**：视图地址 = 基张量地址 + 偏移（视图自身对齐也须满足），视图借用基张量
  字节并把基张量的占用延长到覆盖视图自身的生命周期；视图可嵌套。
- **不重叠**：同一路径内、同时存活的不同基张量，字节区间不得重叠（允许边界相接）。
- **互斥复用**：不同路径的张量可复用同一地址；跨路径的同名张量（共享张量）在所有
  路径中使用同一地址。
- **峰值与上界**：每条路径的峰值 = 该路径内（存活窗口非空的）基张量
  `address + bytes` 的最大值；地址空间上界 = 各路径峰值的最大值，且不超过
  `memory_limit`。

## 响应模型

```jsonc
{
  "status": "optimal",        // optimal | feasible | infeasible | unknown
  "address_bound": 1536,      // 地址空间上界（无解/未知时为 null）
  "paths": [                  // 与请求中的路径一一对应（无解/未知时为 null）
    {
      "id": "chat",
      "start": 0, "end": 7,   // 路径起止时刻
      "peak": 1536,           // 该路径峰值
      "operators": [{"id": "embed", "start": 0, "end": 2}, ...],
      "tensors": [{"id": "hidden", "address": 0}, ...]
    }
  ]
}
```

`status` 含义：

- `optimal`：所有字典序目标均被证明最优；
- `feasible`：给出了满足全部约束的方案，但因求解时间限制未能证明最优
  （**未证明最优绝不标记为 optimal**）；
- `infeasible`：截止时间或显存限制下无解；
- `unknown`：时间限制内既未找到方案也未证明无解。

无效输入（字段缺失、类型错误、依赖成环、对齐非 2 的幂、同名张量定义不一致等）
返回 **HTTP 422**。

## 实际请求与响应示例

请求（保存于 [`examples/plan_request.json`](examples/plan_request.json)，两条互斥
路径：`chat` 与 `embed_only`；`weights`/`hidden` 为跨路径共享张量；`kv_tail` 是
`kv` 偏移 512 字节的视图）：

```bash
curl -s -X POST http://localhost:8000/plans \
  -H "Content-Type: application/json" \
  -d @examples/plan_request.json | python3 -m json.tool
```

```json
{
  "deadline": 12,
  "memory_limit": 4096,
  "paths": [
    {
      "id": "chat",
      "operators": [
        {"id": "embed", "duration": 2, "dependencies": []},
        {"id": "attn", "duration": 3, "dependencies": ["embed"]},
        {"id": "head", "duration": 2, "dependencies": ["attn"]}
      ],
      "tensors": [
        {"id": "weights", "bytes": 1024, "alignment": 256, "producer": null, "consumers": ["attn"]},
        {"id": "hidden", "bytes": 512, "alignment": 256, "producer": "embed", "consumers": ["attn"]},
        {"id": "kv", "bytes": 768, "alignment": 256, "producer": "attn", "consumers": ["head"]},
        {"id": "kv_tail", "bytes": 256, "alignment": 256, "base": "kv", "offset": 512, "producer": "attn", "consumers": ["head"]},
        {"id": "logits", "bytes": 512, "alignment": 256, "producer": "head", "consumers": [], "is_output": true}
      ]
    },
    {
      "id": "embed_only",
      "operators": [
        {"id": "embed", "duration": 2, "dependencies": []},
        {"id": "proj", "duration": 2, "dependencies": ["embed"]}
      ],
      "tensors": [
        {"id": "weights", "bytes": 1024, "alignment": 256, "producer": null, "consumers": ["proj"]},
        {"id": "hidden", "bytes": 512, "alignment": 256, "producer": "embed", "consumers": ["proj"]},
        {"id": "vec", "bytes": 256, "alignment": 256, "producer": "proj", "consumers": [], "is_output": true}
      ]
    }
  ]
}
```

实际响应（与实现一致的真实输出，保存于
[`examples/plan_response.json`](examples/plan_response.json)）：

```json
{
  "status": "optimal",
  "address_bound": 1536,
  "paths": [
    {
      "id": "chat",
      "start": 0,
      "end": 7,
      "peak": 1536,
      "operators": [
        {"id": "attn", "start": 2, "end": 5},
        {"id": "embed", "start": 0, "end": 2},
        {"id": "head", "start": 5, "end": 7}
      ],
      "tensors": [
        {"id": "hidden", "address": 0},
        {"id": "kv", "address": 0},
        {"id": "kv_tail", "address": 512},
        {"id": "logits", "address": 0},
        {"id": "weights", "address": 512}
      ]
    },
    {
      "id": "embed_only",
      "start": 0,
      "end": 4,
      "peak": 1536,
      "operators": [
        {"id": "embed", "start": 0, "end": 2},
        {"id": "proj", "start": 2, "end": 4}
      ],
      "tensors": [
        {"id": "hidden", "address": 0},
        {"id": "vec", "address": 0},
        {"id": "weights", "address": 512}
      ]
    }
  ]
}
```

可以对照语义验证：`hidden`（[2,5)）与 `weights`（[0,5)）同时存活故地址错开；
`kv`（[5,7)）与它们错时复用地址 0；视图 `kv_tail` 地址 = `kv` 地址 + 512；共享
张量 `weights`/`hidden` 在两条路径中地址相同；`logits`/`vec` 由最后一个算子产出，
存活窗口为空，不抬高峰值；地址空间上界 1536 = 两条路径峰值的最大值（互斥复用）。

无解示例（同一请求把 `memory_limit` 改为 `1024`）：

```json
{
  "status": "infeasible",
  "address_bound": null,
  "paths": null
}
```

无效输入示例（`alignment: 3` 不是 2 的幂）返回 `HTTP 422`：

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "paths", 0, "tensors", 0, "alignment"],
      "msg": "Value error, alignment must be a power of two, got 3",
      ...
    }
  ]
}
```

## 本地开发与测试

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000   # 启动服务
pytest                                       # 运行测试（含独立校验器的随机模糊测试）
```

## 配置（环境变量）

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `PLANNER_SOLVE_SECONDS` | `5` | 每个优化阶段的单次求解时间上限（秒） |
| `PLANNER_TOTAL_SECONDS` | `60` | 单次请求全部阶段的总时间预算（秒） |

字典序优化按「上界 → 各张量地址 → 各路径起点 → 各算子起点」逐阶段求解并固定
已证明的最优值；任一阶段超时即停止并返回当前最优可行方案（`status: feasible`）。

## 项目结构

```
app/
  main.py      # FastAPI 应用：POST /plans、GET /healthz
  models.py    # 请求/响应模型与全部输入校验（违反即 422）
  planner.py   # CP-SAT 建模与字典序多阶段求解
tests/
  test_plans.py  # 语义、无解、422 等单元/接口测试
  test_fuzz.py   # 随机实例 + 独立校验器的模糊测试
examples/
  plan_request.json / plan_response.json  # README 中的真实示例
Dockerfile / docker-compose.yml / requirements*.txt
```
