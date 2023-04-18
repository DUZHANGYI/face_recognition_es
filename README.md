# face_recognition_es

python dlib 人脸识别，使用elasticsearch储存人脸数据

| 包             |      版本 |
|:--------------|--------:|
| Python        |     3.7 |
| Elasticsearch |     8.7 |
| dlib          | 19.24.0 |
| onnxruntime   |  1.14.1 |

### 执行步骤

1. 安装完所需包之后启动ES,执行test_app.py中的test_create_index方法创建一个faces索引
2. 执行test_app.py中的test_put_face方法向ES中插入数据（大概1000条数据）
3. 最后执行face_biopsy_rec.py进行人脸比对