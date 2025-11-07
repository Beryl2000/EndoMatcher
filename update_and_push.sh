COMMIT_MSG=${1:-"update"}
PROXY_PORT=${2:-18687}                     
PROXY_URL="http://127.0.0.1:${PROXY_PORT}" 

echo "🌐 设置 Git 代理到 $PROXY_URL"
git config --global http.proxy  "$PROXY_URL"
git config --global https.proxy "$PROXY_URL"

echo "✅ 当前代理配置："
git config --global --get http.proxy
git config --global --get https.proxy


if [ ! -d .git ]; then
  echo "❌ 当前目录不是 git 仓库，请先运行 init_and_push.sh"
  exit 1
fi

cat > .gitignore <<EOF
checkpoint/
__pycache__/
*.pyc
*.pkl
EOF

echo "📦 添加除 checkpoint/ 外的所有修改..."
git add . ':!checkpoint'

if git diff --cached --quiet; then
  echo "✅ 没有检测到需要提交的变更。"
  exit 0
fi

git commit -m "$COMMIT_MSG"
echo "📤 推送到远程仓库..."
if ! git push origin main; then
  echo "⚠️  非快进拒绝，正在拉取合并再推..."
  git pull --rebase origin main || git pull origin main
  git push origin main
fi

echo "✅ 更新完成！"


# chmod +x update_and_push.sh
# ./update_and_push.sh "Update" 18421

