# Malicious Comment Detector
- Model
    - [KoMiniLM](https://github.com/KLUE-benchmark/KLUE](https://github.com/BM-K/KoMiniLM))
- Dataset
    - [한국어 악성댓글 데이터셋](https://github.com/ZIZUN/korean-malicious-comments-dataset)
      - train:dev:test = 9000:500:500으로 split
      
## Installation
```shell
git clone https://github.com/2unju/malicious-comment-detector.git
cd malicious_comment_detector
pip install -r requirements.txt
```

## Training
```shell
python train.py
```

학습에 사용되는 파라미터는 arguments.py에서 변경 가능

## Result
|Model|KoMini|klue|roberta-base|distilbert|  
|:---:|:---:|:---:|:---:|:---:|
|Test Acc|0.8784|0.8902|0.8771|0.8573|
|Inference Time (sec)|0.0035|0.0071|0.0071|0.0019|
