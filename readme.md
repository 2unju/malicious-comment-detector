# Malicious Comment Detector
- Model
    - [KLUE-BERT(base)](https://github.com/KLUE-benchmark/KLUE)
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
||Test Acc|Inference Time (sec)|  
|:---:|:---:|:---:|
|KoBERT(klue/bert-base)|0.89|0.012|
|KoMiniLM|0.87|0.007|
