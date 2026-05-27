#!/usr/bin/env python3
"""
代理服务器：解决CDN资源的CORS限制问题
运行此服务器后，访问 http://localhost:5000/visualization/output/order_spacetime_3d.html
"""

import os
import sys
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path
        
        # 调试：打印完整的请求路径
        print(f"[DEBUG] Received request: {self.path}")
        
        # 检查是否是CDN代理请求
        if path.startswith('/unpkg.com/'):
            url = 'https://' + path.lstrip('/')
            print(f"[DEBUG] Proxying to: {url}")
            self.proxy_cdn_url(url)
            return
        elif path.startswith('/cdn.jsdelivr.net/'):
            url = 'https://' + path.lstrip('/')
            print(f"[DEBUG] Proxying to: {url}")
            self.proxy_cdn_url(url)
            return
        elif path.startswith('/esm.sh/'):
            url = 'https://' + path.lstrip('/')
            print(f"[DEBUG] Proxying to: {url}")
            self.proxy_cdn_url(url)
            return
        elif path.startswith('/tile.openstreetmap.org/'):
            # 重定向 OpenStreetMap 请求到高德地图
            tile_path = path.replace('/tile.openstreetmap.org/', '')
            # 使用高德地图替代
            url = f'https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&{tile_path.replace("/", "&").replace("z=", "z=").replace("x=", "x=").replace("y=", "y=")}'
            # 解析路径: /z/x/y.png
            parts = tile_path.split('/')
            if len(parts) >= 3:
                z = parts[0]
                x = parts[1]
                y = parts[2].replace('.png', '')
                url = f'https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}'
            print(f"[DEBUG] Redirecting OpenStreetMap request to Gaode: {url}")
            self.proxy_cdn_url(url)
            return
        else:
            print(f"[DEBUG] Path did not match any CDN pattern, treating as local file")
        
        # 本地文件请求
        self.serve_local_file(path)
    
    def serve_local_file(self, path):
        if path == '/':
            path = '/visualization/output/order_spacetime_3d.html'
        
        if '..' in path:
            self.send_simple_error(403, 'Forbidden')
            return
        
        file_path = os.path.join(os.getcwd(), path.lstrip('/'))
        
        if os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            content_type = {
                '.html': 'text/html',
                '.js': 'application/javascript',
                '.json': 'application/json',
                '.css': 'text/css',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.gif': 'image/gif',
            }.get(ext, 'application/octet-stream')
            
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_simple_error(500, 'Server error')
        else:
            self.send_simple_error(404, 'File not found')
    
    def proxy_cdn_url(self, url):
        try:
            # 修复双斜杠路径问题
            if '://' in url:
                parts = url.split('://', 1)
                url = parts[0] + '://' + parts[1].replace('//', '/')
            else:
                url = url.replace('//', '/')
            
            # 确保URL格式正确
            if not url.startswith('http://') and not url.startswith('https://'):
                url = 'https://' + url.lstrip('/')
            
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                content_type = response.headers.get('Content-Type', 'application/octet-stream')
                
                self.send_response(response.status)
                self.send_header('Content-Type', content_type)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', '*')
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                self.wfile.write(content)
        except urllib.error.HTTPError as e:
            self.send_simple_error(e.code, 'Not found')
        except Exception as e:
            # 忽略连接中止等错误
            pass
    
    def send_simple_error(self, code, message):
        try:
            self.send_response(code)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(message.encode('utf-8'))
        except:
            pass
    
    def do_OPTIONS(self):
        try:
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            self.end_headers()
        except:
            pass
    
    def log_message(self, format, *args):
        # 记录请求日志
        print(f"[{self.client_address[0]}] {self.command} {self.path}")

def main():
    port = 5000
    server = HTTPServer(('localhost', port), ProxyHandler)
    print(f"Proxy server started on port {port}")
    print(f"Access: http://localhost:{port}/visualization/output/order_spacetime_3d.html")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.server_close()

if __name__ == '__main__':
    main()
