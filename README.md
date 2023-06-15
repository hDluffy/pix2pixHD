# pix2pixHD

## 环境搭建
- git clone源码

    ```
    git clone https://github.com/hDluffy/pix2pixHD.git
    ```
- 安装训练环境</br>
    
    conda配置同人脸融合，如果安装中出现超时，可以在后面加douban镜像指令【-i https://pypi.douban.com/simple/】
    ```
    conda create --name torch-1.12 python=3.7

    source activate torch-1.12
    
    ##注意cuda版本与系统安装版本无关
    # CUDA 10.2
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
    # CUDA 11.3
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
    
    pip install dominate
    pip install scipy
    pip install kornia
    
    pip install opencv-python
    pip install onnxruntime -i https://pypi.douban.com/simple/
    pip install thop -i https://pypi.douban.com/simple/
    
    pip install tensorboard
    ```
- 数据处理</br>
    数据结构
    ```
    2Old
        │
        └───trian_A
        │       
        └───trian_B    
    ```
    多域
    ```
    2Pop
        │
        └───trian_A
        │       
        └───trian_D0
        │       
        └───trian_D1
        ...
    ```

- 模型训练

    ```
    #开启tensorboard
    tensorboard --logdir=logs
    ```

    ```
    #本地ssh映射服务器6006端口后， google浏览器打开：http://127.0.0.1:16006/
    ssh -L 16006:127.0.0.1:6006 username@server-ip
    ```
    
    ```
    ##
    #通过设置task_num='with_mask'，可以改为cat mask输入
    ./script/train_unet.sh
    #多域转换，age_edit,其中条件需要在内部实现中调整
    ./script/train_unet_nc.sh
    ```