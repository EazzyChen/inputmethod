# inputmethod
rnn project

## GitHub操作
### 方案一：克隆远程仓库
**适用场景**：GitHub 上已经建好了仓库（无论是否勾选 Readme），直接把整个项目下载下来开始工作。
**特点**：最简单，Git 自动帮你配好了一切。
```bash
# 1. 克隆仓库到本地（自动完成：创建本地仓库、关联远程、设置分支名为 main）
git clone https://github.com/EazzyChen/inputmethod.git
# 2. 进入项目文件夹
cd inputmethod
# --- 此时开始写代码、修改文件 ---
# 3. 提交修改
git add .
git commit -m "第一次提交"
# 4. 推送到远程
# 注意：这里直接用 git push 即可，不需要 -u
# 因为 clone 时已经自动建立了追踪关系
git push
```
---
### 方案二：本地新建仓库
**适用场景**：先在本地写好了代码，或者想要一个全新的本地环境，然后再关联到 GitHub。
**特点**：步骤稍多，需要手动“认亲”，且要注意分支名称问题。
```bash
# 1. 初始化本地仓库
git init
# 2. 关联远程仓库（手动告诉 Git 远程地址叫 origin）
git remote add origin https://github.com/EazzyChen/inputmethod.git
# 3. 检查并修改分支名（关键步骤！）
# 旧版 Git 默认分支是 master，而 GitHub 现在默认是 main
# 先查看当前分支名
git branch
# 如果是 master，强制改为 main（避免推送到错误分支）
git branch -m master main
# --- 此时开始写代码、修改文件 ---
# 4. 提交修改
git add .
git commit -m "第一次提交"
# 5. 首次推送（必须加 -u 参数）
# 这一步做两件事：推送代码 + 建立追踪关系
git push -u origin main
```

### 修改本地git默认为main（一劳永逸）：
```bash
git config --global init.defaultBranch main
git config --global --get init.defaultBranch # 验证

```


---
### 💡 核心区别对比表
| 操作步骤 | 方案一：克隆 | 方案二：本地新建 |
| :--- | :--- | :--- |
| **初始化命令** | `git clone <URL>` | `git init` |
| **关联远程** | 自动关联 | 需手动 `git remote add origin <URL>` |
| **分支名称** | 自动与远程一致 (main) | 可能不一致，需手动改为 main |
| **首次推送** | `git push` | `git push -u origin main` |
| **推荐指数** | ⭐⭐⭐⭐⭐ (推荐) | ⭐⭐⭐ (稍繁琐) |


