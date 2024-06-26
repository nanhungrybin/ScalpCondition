# ScalpCondition

## 📁 손실함수 개선과 다중크기 패치 분할을 통한 두피 유형 진단 시 데이터 불균형 해소
## 📁 해결하고자 한 문제 : 클래스간의 불균형
두피의 상태를 미세각질, 피지 과다, 모낭 사이 홍반, 모낭홍반/농포, 비듬, 탈모, 양호 등 7가지 유형으로 진단하고자 한다.
각 유형별 데이터 수량을 보면 "양호"와 "모낭홍반/농포" 클래스는 다른 유형에 비해 현저히 적은 수량을 보유하고 있다. 반면, "피지과다" 클래스는 매우 많은 데이터를 보유하고 있어, 전체 데이터 분포에서 큰 불균형을 초래하고 있다.
클래스 불균형 문제 개선을 통한 모델을 학습 방법론에 대해 탐구하고자 했다.

![image](https://github.com/nanhungrybin/ScalpCondition/assets/97181397/149f97f0-d04d-41d7-91b4-c907d69bb67a)

<img width="374" alt="image" src="https://github.com/nanhungrybin/ScalpCondition/assets/97181397/83763167-d59f-4f64-b240-c20eee282361">

실험 결과, 모델 구조 변경, 손실 함수의 개선 및 데이터 증강 기법이 도입된 연구는 데이터의 불균형을 효과적으로 해소하였으며, 특히 기존의 ViT 16 patch 모델과 Cross Entropy 손실 함수를 사용했을 때보다 모낭홍반/농포 두피에 대해서는 약 22.97%, 양호한 두피 상태에 대해서는 25.42%의 정확도가 향상되는 결과를 보였다.
