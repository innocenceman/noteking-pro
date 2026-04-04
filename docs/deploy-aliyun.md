# 阿里云服务器部署教程 (小白版)

从零开始，手把手教你把 NoteKing 部署到阿里云服务器上。

## 第一步：注册阿里云账号

1. 打开 https://www.aliyun.com/
2. 点击「免费注册」
3. 用手机号注册并完成实名认证

## 第二步：购买服务器

1. 打开 https://ecs.console.aliyun.com/
2. 点击「创建实例」
3. 推荐配置：
   - **地域**: 选离你最近的 (如上海、杭州)
   - **实例规格**: 2 核 2G (ecs.t6-c1m1.large) ≈ 50元/月
   - **如果需要访问YouTube**: 选择中国香港地域
   - **操作系统**: Ubuntu 22.04 LTS
   - **系统盘**: 40GB SSD
   - **带宽**: 按流量计费，1Mbps
4. 设置登录密码，记住它
5. 确认购买

## 第三步：连接服务器

### Mac 用户
打开「终端」应用，输入：
```bash
ssh root@你的服务器IP地址
```
输入密码即可登录。

### Windows 用户
1. 下载 [MobaXterm](https://mobaxterm.mobatek.net/download.html)
2. 打开后点击 Session → SSH
3. 输入服务器 IP 地址，用户名 root
4. 输入密码登录

## 第四步：安装 NoteKing

登录服务器后，执行以下命令：

```bash
# 一键安装 (包含 Docker + NoteKing)
curl -sSL https://raw.githubusercontent.com/bcefghj/noteking/main/install.sh | bash
```

安装完成后，编辑配置文件：

```bash
# 编辑环境变量
nano /opt/noteking/.env
```

填入你的 LLM API Key：
```
NOTEKING_LLM_API_KEY=sk-your-key-here
```

按 `Ctrl+X` → `Y` → `Enter` 保存退出。

然后启动服务：
```bash
cd /opt/noteking
docker compose up -d
```

## 第五步：访问你的 NoteKing

在浏览器中输入：
```
http://你的服务器IP:3000
```

如果打不开，需要在阿里云控制台开放端口：
1. 找到你的 ECS 实例 → 安全组
2. 添加规则：端口 3000，授权对象 0.0.0.0/0

## 第六步 (可选)：配置域名

1. 在阿里云购买域名 (如 noteking.你的域名.com)
2. 添加 A 记录指向服务器 IP
3. 安装 Nginx 并配置反向代理：

```bash
apt install -y nginx certbot python3-certbot-nginx

# 创建 Nginx 配置
cat > /etc/nginx/sites-available/noteking << 'EOF'
server {
    listen 80;
    server_name noteking.你的域名.com;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

ln -s /etc/nginx/sites-available/noteking /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# 配置 HTTPS (免费)
certbot --nginx -d noteking.你的域名.com
```

## 费用估算

| 项目 | 费用 |
|------|------|
| 服务器 (2核2G) | ~50元/月 |
| 域名 | ~30元/年 |
| LLM API (DeepSeek) | 按用量，约0.001元/千字 |
| **总计** | **~55元/月** |

## 常见问题

**Q: 购买哪个地域的服务器？**
A: 如果只用 B站，选国内任意地域。如果要用 YouTube，必须选香港或海外地域。

**Q: 忘记服务器密码了？**
A: 在阿里云 ECS 控制台重置密码，然后重启实例。

**Q: 如何更新到最新版本？**
```bash
cd /opt/noteking
git pull
docker compose up -d --build
```
