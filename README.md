# MokioClaw — planner → verifier Multi-Agent CodeAgent

一个从零构建的 AI Code Agent，沿「ReAct 循环 → LangGraph 规划/执行/验证 → 多智能体分工 → 上下文管理 → Harness/TUI」路径演进。当前包含：规划器（planner）、验证器（verifier）、搜索智能体（searchAgent）、实现智能体（codeAgent），通过 handoff 协作完成任务。

## 功能特性

- **多智能体分工**：planner（supervisor）委派 searchAgent（联网搜索）与 codeAgent（写代码/跑命令），handoff 协作
- **规划 → 执行 → 验证闭环**：LangGraph 状态图，验证失败自动回规划重试
- **三层记忆 + 上下文压缩**：Rules / Working Memory / History Summary，超限自动压缩
- **检查点（Checkpoint）**：light / strict / off 三模式，支持 `--resume` 恢复
- **链路观测（Trace）**：旁路记录节点 / 工具 / handoff 事件，产出 events.jsonl + summary.json + timeline.md
- **交互式 TUI**：`mokioclaw tui` 子命令，实时时间线 + 状态面板 + 审批弹窗 + 停止任务

## 环境要求

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/)（推荐，用于依赖管理）

## 配置

所有密钥/配置统一放在项目根目录的 `.env` 文件中（已加入 `.gitignore`，**不会**提交到仓库）。

### 1. 复制模板

```bash
cp .env.example .env
```

### 2. 填写密钥

编辑 `.env`，至少填入 `OPENAI_API_KEY`：

| 配置项 | 必填 | 说明 | 获取地址 |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | ✅ | OpenAI 兼容 API 密钥 | https://platform.openai.com/api-keys |
| `OPENAI_BASE_URL` | ❌ | 自定义 API 端点（第三方中转/兼容服务时填写） | — |
| `MODEL` | ❌ | 模型名，默认 `gpt-4o` | — |
| `TAVILY_API_KEY` | ❌ | 联网搜索密钥（未配置时搜索类任务报错） | https://tavily.com |
| `MOKIO_CONTEXT_TOKEN_LIMIT` | ❌ | 上下文压缩阈值，默认 `400000` | — |

> 🔒 **隐私提示**：`.env` 里的真实密钥**切勿提交到 Git**。仓库只包含 `.env.example` 模板。可通过 `git status` 确认 `.env` 未被跟踪。

## 安装

```bash
uv sync
```

## 运行

### 命令行跑任务

```bash
uv run mokioclaw --task "帮我创建一个贪吃蛇游戏并验证"
```

常用选项：

```bash
uv run mokioclaw --task "任务描述" --checkpoint-mode strict   # 检查点模式
uv run mokioclaw --resume                                     # 从检查点恢复
uv run mokioclaw --task "任务描述" --approval-mode deny       # 高危命令直接拒绝
```

### 交互式 TUI

```bash
uv run mokioclaw tui
```

- 底部输入框回车提交任务
- 高危命令会弹出审批框：`y` 批准 / `n` 拒绝 / `ctrl+x` 停止
- 停止任务：点「⏹ 停止」按钮，或按 `F9`（输入框聚焦时 `Ctrl+X` 被输入框的剪切占用）

### 列出可用工具

```bash
uv run mokioclaw list-tools
```

## 测试

```bash
uv run pytest
```

## 项目结构

```
src/mokioclaw/
├── config.py              # 集中配置（从 .env 读取）
├── core/                  # 工作区、状态、检查点、trace、事件总线
├── graph/                 # LangGraph 状态图 + 节点 + 记忆
├── agents/                # searchAgent / codeAgent
├── tools/                 # 文件/命令/搜索/todo/notepad 工具
├── providers/             # LLM provider
├── tui/                   # Textual TUI
└── cli/                   # typer CLI 入口
```
