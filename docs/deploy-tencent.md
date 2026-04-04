# 腾讯云服务器部署教程 (小白版)

与阿里云类似的流程，适用于腾讯云轻量应用服务器。

## 第一步：注册腾讯云

1. 打开 https://cloud.tencent.com/
2. 微信扫码注册并实名认证

## 第二步：购买轻量应用服务器

1. 打开 https://console.cloud.tencent.com/lighthouse
2. 点击「新建」
3. 推荐配置：
   - **地域**: 上海/广州 (国内) 或 香港 (需要YouTube)
   - **套餐**: 2核2G 4M带宽 ≈ 45元/月
   - **镜像**: Ubuntu 22.04 LTS
4. 设置密码，确认购买

## 第三步：连接并安装

```bash
# SSH 连接 (Mac终端 或 Windows MobaXterm)
ssh root@你的腾讯云IP

# 一键安装
curl -sSL https://raw.githubusercontent.com/bcefghj/noteking/main/install.sh | bash

# 配置 API Key
nano /opt/noteking/.env
# 填入: NOTEKING_LLM_API_KEY=sk-your-key

# 启动
cd /opt/noteking && docker compose up -d
```

## 第四步：开放防火墙

腾讯云轻量应用服务器需要在控制台开放端口：

1. 进入实例详情 → 防火墙
2. 添加规则：TCP 3000 端口
3. 访问 `http://你的IP:3000`

其余配置 (域名、HTTPS、Nginx) 参考阿里云教程，完全一致。
