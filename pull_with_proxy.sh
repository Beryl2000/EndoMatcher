PROXY_PORT=${1:-18421}                      
PROXY_URL="http://127.0.0.1:${PROXY_PORT}"  

if ! curl -s --connect-timeout 2 "$PROXY_URL" >/dev/null; then
  echo "❌ 代理端口 ${PROXY_PORT} 没响应，请先启动 Clash/V2Ray 等代理！"
  exit 1
fi

echo "🌐 给本次拉取设置代理 → ${PROXY_URL}"
git config --local http.proxy "$PROXY_URL"
git config --local https.proxy "$PROXY_URL"

echo "📥 拉取 origin/main ..."
git pull origin main

git config --local --unset http.proxy 2>/dev/null
git config --local --unset https.proxy 2>/dev/null

echo "✅ 拉取完成！"



# chmod +x pull_with_proxy.sh
# ./pull_with_proxy.sh 18421