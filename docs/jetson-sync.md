# Jetson项目同步

## 结论

Git/Git LFS是项目源码、模型和版本历史的唯一正式同步方式。SSH负责登录Jetson并
执行命令；`scp`或`rsync`只用于临时日志、截图和Jetson专属构建产物，不作为项目
主同步方式。

Windows能够SSH登录Jetson，并不代表Jetson已经获得GitHub私有仓库权限。需要为
“Jetson到GitHub”单独配置只读Deploy Key，或在临时会话中使用SSH agent转发。
长期部署推荐只读Deploy Key。

## 推荐首次克隆方式

在Jetson获得GitHub读取权限后执行：

```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
mkdir -p ~/projects
cd ~/projects
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:Qzz139/LEARN-yolo.git
cd LEARN-yolo
git lfs pull --include="weights/**/*.pt,weights/**/*.onnx" --exclude=""
```

`GIT_LFS_SKIP_SMUDGE=1`避免首次克隆下载整个训练数据集；运行程序只需要代码和模型。
需要在Jetson检查数据时，再按需执行：

```bash
git lfs pull --include="datasets/dataset/images/**" --exclude=""
```

## 后续更新

Windows提交并推送后，Jetson执行：

```bash
cd ~/projects/LEARN-yolo
git pull --ff-only
git lfs pull --include="weights/**/*.pt,weights/**/*.onnx" --exclude=""
```

使用`--ff-only`可以避免在部署机上意外产生合并提交。Jetson原则上只拉取和运行，
实验记录先通过`scp`或`rsync`传回Windows，检查后再由主工作区提交。

## 只有用户需要完成的步骤

Deploy Key首次配置时，需要用户在GitHub网页中打开私有仓库：

```text
Settings → Deploy keys → Add deploy key
```

粘贴Jetson生成的公钥，并保持“Allow write access”关闭。私钥和访问令牌不得提交到
仓库或通过聊天发送。
