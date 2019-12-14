# Relation-Networks-for-Object-Detection-tensorflow
Relation Networks for Object Detection reproducing project with tensorflow

### 프로젝트를 크게 2개로 분리
1. backbone network인 Faster-RCNN 구현
 - https://github.com/kwonsungil/Faster-RCNN

2. 기존 Faster-RCNN model을 이용하여 논문의 핵심인 Relation Module을 구현하고 테스트
 - https://github.com/kwonsungil/Relation-Networks-for-Object-Detection-tensorflow

### 일정(https://github.com/rp12-study/rp12-hub/wiki)  
1. Paper Review  
 - 완료  
2. ResNet 구현 및 ImageNet학습
 - 완료
3. Faster-RCNN 구현 및 학습
 - 진행 중
4. Relation Module 구현 및 학습
 - 완료
5. Duplicate Remover 구현 및 학습
 - 진행 중
 
 ### 학습
 1. config.py 파일 수정 
  - baseline으로 학습할지 realation moudle로 학습할지 설정
  - OUTPUT_DIR : ckpt 파일 저장 위치
  - test_output_path : test 결과 파일 저장 
 2. models/network.py 파일 수정
  - _head_to_tail_base : 2 fully connected layers
  - _head_to_tail_relation : 2 fully connected layers + 2 relation modules
 3. train.py 실행
 
 ## 테스트
 1. 학습과 마찬가지로 config.py 수정
 2. validate.py 실행
 
 ### 결과
 1. VOC2007 + VOC2012으로 학습 후 VOC2007 Test 결과
  : MAP 0.665(base Faster-RCNN)
  : MAP 0.611(Faster-RCNN + Relation Module) 

