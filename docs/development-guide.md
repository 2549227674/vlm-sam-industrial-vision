# 开发指南

**项目名称**：SAM-3 双范式工业视觉分析系统
**生成日期**：2026-01-03

---

## 1. 环境准备

### 1.1 系统要求

**硬件要求**：
- **推荐**：NVIDIA GPU（支持 CUDA 11.0+）
- **最低**：CPU（性能会显著降低）
- **内存**：≥8GB RAM
- **存储**：≥5GB 可用空间（用于模型缓存）

**软件要求**：
- **操作系统**：Windows / Linux / macOS
- **Python**：≥3.8
- **CUDA**：≥11.0（GPU 模式）
- **Git**：用于版本控制

### 1.2 安装 Python 环境

#### 方式 1：使用 venv（推荐）
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

#### 方式 2：使用 conda
```bash
# 创建 conda 环境
conda create -n cv-system python=3.10
conda activate cv-system
```

---

## 2. 安装依赖

### 2.1 完整安装（推荐）
```bash
cd 计算机视觉项目
pip install -r requirements.txt
```

### 2.2 最小化安装（仅核心功能）
```bash
pip install streamlit torch torchvision numpy opencv-python Pillow modelscope transformers
```

### 2.3 可选依赖

#### 启用 VLM 功能
```bash
pip install dashscope
```

#### 启用可视化增强
```bash
pip install matplotlib
```

#### 启用 Ant Design 组件
```bash
pip install streamlit-antd-components
```

---

## 3. 配置环境变量

### 3.1 DashScope API Key（可选）

如果需要使用 VLM 功能，需要配置阿里云 DashScope API Key：

#### Windows (CMD)
```cmd
set DASHSCOPE_API_KEY=sk-your-key-here
```

#### Windows (PowerShell)
```powershell
$env:DASHSCOPE_API_KEY="sk-your-key-here"
```

#### Linux/macOS
```bash
export DASHSCOPE_API_KEY=sk-your-key-here
```

#### 持久化配置（推荐）
创建 `.env` 文件：
```bash
# .env
DASHSCOPE_API_KEY=sk-your-key-here
```

---

## 4. 运行应用

### 4.1 启动 Streamlit 应用
```bash
cd 计算机视觉项目
streamlit run app_final.py
```

### 4.2 访问应用
浏览器自动打开：`http://localhost:8501`

如果没有自动打开，手动访问上述地址。

### 4.3 首次使用
1. 点击侧边栏的"初始化模型"按钮
2. 等待模型下载和加载（约 30-60 秒）
3. 选择范式（A/B/C）
4. 上传图像开始分析

---

## 5. 项目结构

```
计算机视觉项目/
├── app_final.py          # 主入口
├── requirements.txt      # 依赖清单
├── core/                 # 核心算法
│   ├── models.py
│   ├── sam3_infer.py
│   ├── vlm.py
│   └── ...
└── ui/                   # UI 模块
    ├── paradigm_a.py
    ├── paradigm_b.py
    ├── paradigm_c.py
    └── ...
```

---

## 6. 开发工作流

### 6.1 添加新功能

#### 步骤 1：创建功能分支
```bash
git checkout -b feature/new-feature
```

#### 步骤 2：开发功能
- 在 `core/` 中添加算法逻辑
- 在 `ui/` 中添加界面组件
- 遵循现有代码风格

#### 步骤 3：测试功能
```bash
streamlit run app_final.py
```

#### 步骤 4：提交代码
```bash
git add .
git commit -m "Add new feature: xxx"
git push origin feature/new-feature
```

### 6.2 代码规范

#### Python 代码风格
- 遵循 PEP 8 规范
- 使用 4 空格缩进
- 函数和类添加文档字符串

#### 示例
```python
def process_image(image: PIL.Image, param: float) -> np.ndarray:
    \"\"\"
    处理图像。

    Args:
        image: 输入图像
        param: 处理参数

    Returns:
        处理后的图像数组
    \"\"\"
    # 实现逻辑
    pass
```

---

## 7. 调试技巧

### 7.1 Streamlit 调试

#### 查看会话状态
```python
st.write(st.session_state)
```

#### 显示变量值
```python
st.write("变量值:", variable)
```

#### 捕获异常
```python
try:
    result = some_function()
except Exception as e:
    st.error(f"错误: {str(e)}")
```

### 7.2 模型调试

#### 检查设备
```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"设备: {torch.cuda.get_device_name(0)}")
```

#### 检查模型加载
```python
if st.session_state.get("models_ready"):
    st.success("模型已加载")
else:
    st.warning("模型未加载")
```

---

## 8. 常见问题

### 8.1 模型下载失败

**问题**：首次运行时模型下载超时

**解决方案**：
1. 检查网络连接
2. 配置 ModelScope 镜像：
```bash
export MODELSCOPE_CACHE=~/.cache/modelscope
```
3. 手动下载模型到缓存目录

### 8.2 CUDA 内存不足

**问题**：GPU 显存不足

**解决方案**：
1. 降低批处理大小
2. 使用 CPU 模式：
```python
DEVICE = "cpu"
```
3. 清理 GPU 缓存：
```python
torch.cuda.empty_cache()
```

### 8.3 Streamlit 端口占用

**问题**：8501 端口已被占用

**解决方案**：
```bash
# 指定其他端口
streamlit run app_final.py --server.port 8502
```

---

## 9. 性能优化

### 9.1 模型优化

#### 使用半精度推理
```python
model = model.half()  # FP16
```

#### 模型量化
```python
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 9.2 缓存优化

#### Streamlit 缓存
```python
@st.cache_resource
def load_model():
    return load_sam_model()
```

---

## 10. 部署指南

### 10.1 本地部署（当前模式）

```bash
streamlit run app_final.py
```

### 10.2 Docker 部署（未来）

#### Dockerfile 示例
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app_final.py"]
```

#### 构建和运行
```bash
docker build -t cv-system .
docker run -p 8501:8501 cv-system
```

---

## 11. 测试

### 11.1 手动测试

1. 启动应用
2. 测试每个范式
3. 验证结果正确性

### 11.2 自动化测试（未来）

```python
# tests/test_sam3.py
def test_sam3_inference():
    image = load_test_image()
    result = sam3_infer(image)
    assert result is not None
```

---

## 12. 贡献指南

### 12.1 提交 Issue
- 描述问题或功能请求
- 提供复现步骤
- 附上错误日志

### 12.2 提交 Pull Request
1. Fork 项目
2. 创建功能分支
3. 提交代码
4. 创建 PR

---

## 13. 资源链接

### 13.1 官方文档
- [Streamlit 文档](https://docs.streamlit.io)
- [PyTorch 文档](https://pytorch.org/docs)
- [ModelScope 文档](https://modelscope.cn/docs)

### 13.2 模型资源
- [SAM-3 模型](https://modelscope.cn/models)
- [DashScope API](https://dashscope.aliyuncs.com)

---

**文档生成时间**：2026-01-03
**项目根目录**：`C:\Users\Han\PycharmProjects\PythonProject8\计算机视觉项目`
