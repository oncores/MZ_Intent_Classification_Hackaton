MZ 해커톤 제출 파일

-실행파일
prediction.py
python prediction.py --input_text=test.txt --output_text=result.txt

prediction.py 파일 실행 후 test파일을 변경시 ./input 디렉토리에 있는 cache파일 3개를 지우고 재실행

-경로
테스트 입력 파일 : input/test.txt
테스트 출력 파일 : output/result.txt

-모델
KoElctra 분류모델 3가지
1.1 intent_second level classification (./input/second_labels.csv)
1.2 intent_third level classification (./input/third_labels.csv)
1.3 intent classification (./input/all_labels.csv)

MLP classifier
2.1 intent classification (./input/all_labels.csv)

-결과
dev 데이터셋 accuracy 기준 74.40%

-Citation

@misc{park2020koelectra,
  author = {Park, Jangwon},
  title = {KoELECTRA: Pretrained ELECTRA Model for Korean},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/monologg/KoELECTRA}}
