# MZ 인공지능 해커톤 대회
AI장치용 STT를 위한 의도 분류/음성인식 오류에 강인한 AI차량/스피커용 NLU시스템
![image](https://user-images.githubusercontent.com/37866322/103755352-09af7980-5051-11eb-9219-14eff7e89c43.png)

## 대회 개요
- MZ 인공지능 해커톤 대회
- 대회기간 : 2020.12.21~2020.01.08
- 주최/주관 : 한국정보화진흥원/미디어젠

## run
- prediction.py
- python prediction.py --input_text=test.txt --output_text=result.txt

## directory
테스트 입력 파일 : input/test.txt

테스트 출력 파일 : output/result.txt


## Model
- KoElctra 분류모델 3가지
> 1.1 intent_second level classification

> 1.2 intent_third level classification

> 1.3 intent classification


- MLP classifier
> 2.1 intent classification (./input/all_labels.csv)

- Result
dev 데이터셋 accuracy 기준 74.40%

## Requirements

```python
torch==1.6.0
transformers==3.3.1
seqeval
fastprogress
attrdict
```

-Citation

@misc{park2020koelectra,
  author = {Park, Jangwon},
  title = {KoELECTRA: Pretrained ELECTRA Model for Korean},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/monologg/KoELECTRA}}
