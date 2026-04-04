# YouTube 代理配置教程

中国大陆用户访问 YouTube 需要代理/VPN，以下是 NoteKing 的代理配置方法。

## 方案一：配置文件 (推荐)

编辑 `~/.noteking/config.json`：

```json
{
  "proxy": {
    "enabled": true,
    "http": "",
    "https": "",
    "socks5": "socks5://127.0.0.1:7890"
  }
}
```

或使用 CLI 配置：
```bash
noteking setup --proxy socks5://127.0.0.1:7890
```

## 方案二：环境变量

```bash
export NOTEKING_PROXY=socks5://127.0.0.1:7890
```

## 方案三：CLI 参数

```bash
noteking run "https://youtu.be/xxx" --proxy socks5://127.0.0.1:7890
```

## 方案四：部署到海外服务器

如果你的 NoteKing 部署在香港/海外服务器上，则无需任何代理配置，
YouTube 可以直接访问。

推荐：阿里云香港轻量应用服务器 ≈ 24元/月

## 代理类型说明

| 类型 | 格式 | 适用场景 |
|------|------|----------|
| SOCKS5 | `socks5://IP:PORT` | Clash/V2Ray 等工具的本地代理 |
| HTTP | `http://IP:PORT` | HTTP 代理 |
| HTTPS | `https://IP:PORT` | HTTPS 代理 |
| 带认证 | `socks5://user:pass@IP:PORT` | 需要账号密码的代理 |

## 常见代理软件端口

| 软件 | 默认端口 |
|------|----------|
| Clash | `socks5://127.0.0.1:7890` |
| V2Ray | `socks5://127.0.0.1:10808` |
| Shadowsocks | `socks5://127.0.0.1:1080` |

## 测试代理是否可用

```bash
# 使用 curl 测试
curl -x socks5://127.0.0.1:7890 https://www.youtube.com/ -I

# 使用 NoteKing 内置测试
python -c "from core.proxy import test_youtube_access; from core.config import AppConfig; c=AppConfig.load(); print(test_youtube_access(c))"
```
