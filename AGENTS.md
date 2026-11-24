# SRAG 仓库指令

在本仓库开展设计、实现、测试或审核前，读取并遵守 [agent.md](./agent.md) 的项目规则。当前领取入口是[执行交接](./docs/执行交接.md)，任务依赖与放行状态见[任务清单](./CHECKLIST.md)。

## Git 提交与合入规范（所有会话与门禁流程默认遵守）

1. **作者信息校验**：每次 `git commit` 前必须核对仓库本地 `git config`，确认 `user.name="Drchennnn"`、`user.email="drchennnn@gmail.com"`，禁止使用未在 GitHub 验证的其他邮箱提交。
2. **允许分支开发，但必须真实合入 `main`**：开发、调试与独立 gate 验收可继续在 `feat/*`、`codex/*`、`review/*` 分支上进行；但每个阶段通过验收收尾时，必须使用保留原提交历史的普通合并（`git merge --no-ff`，严禁 `--squash`）合并回默认分支 `main`，并执行 `git push origin main`。
3. **禁止“脱钩式同步”**：不允许将实际开发或门禁 commits 遗留在分支上、只在 `main` 单独发一笔 `docs/sync` 提交草草收尾；必须保证完整 commit 历史真正进入 `origin/main`。
