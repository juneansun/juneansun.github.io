# 논문 읽기
* 키페이퍼는 기억할 것
* related work, 유사아이디어, 참고하고 있는 논문도 따라가 볼것
* 인용, 또는 후속논문이 있는지 확인

## 2022/07월 찾은 논문
### 1주차
> ### µLayer: Low Latency On-Device Inference Using Cooperative Single-Layer Acceleration and Processor-Friendly Quantization
> <details>
> <summary> > 내용 </summary>
> <div markdown='1'>
> 
>  * Youngsok Kim, Joonsung Kim, Dongju Chae, Daehyun Kim, and Jangwoo Kim. 2019. ΜLayer: Low Latency On-Device Inference Using Cooperative Single-Layer Acceleration and Processor-Friendly Quantization. In Proceedings of the Fourteenth EuroSys Conference 2019 (EuroSys '19). Association for Computing Machinery, New York, NY, USA, Article 45, 1–15. https://doi.org/10.1145/3302424.3303950
>  * Contents:
>    * CPU, GPU를 둘다 사용하는 NN 모델 제시
>    * 1. 이미지 channel 단위로 분리, 각 채널별로 CPU, GPU로 작업을 분배함
>    * 2. 각 processor에서 잘 수행할 수 있는 모델로 quantizaiton 수행(CPU: int8, GPU: fp16)
>    * 3. inception 모듈을 두고 여러 size로 필터링해서 각기 다른 사이크로 conv된 결과물들을 concat하는 방식의 경우 앞선 1, 2와 시너지효과를 발휘 할 수 있음
> </div>
> </details>

## 2022/05월 찾은 논문
### 1주차
> ### Neural Network Inference on Mobile SoCs
> <details>
> <summary> > 내용 </summary>
> <div markdown='1'>
> 
>  * Published in: IEEE Design & Test ( Volume: 37, Issue: 5, Oct. 2020)
>  * Author: **Siqi Wang**, Anuj Pathania, Tulika Mitra
>  * Contents:
>    * Mobile 환경에서 이기종 프로세서별 딥러닝 성능의 정량적 측정과 파워/성능간의 관계에 대한 이해를 제시
>    * energy-efficiency improvement is limited for the Small cluster for some networks: Exynos 5422과 Kirin 970을 비교하며 28nm에서 10nm로 공정이 개선되고, Clock Cycle도 높아졌으며, 대역폭 향상도 4.4배와 2.6배 향상되었다고 하고 있다. **하지만 Small Cluster인 A53의 경우, 전력소모량도 2배가량 증가하여 실질적인 효율성은 개선이 크지 않음을 보인다고 함**
>    * **Kirin 970의 경우 GPU의 전력/성능비가 특히 뛰어나기 때문에 power-efficient한 Small Cluster보다 너 나은 에너지 효율성을 보였다고 함**
> </div>
> </details>

> ### High-Throughput CNN Inference on Embedded ARM Big.LITTLE Multicore Processors
> <details>
> <summary> > 내용 </summary>
> <div markdown='1'>
> 
>  * Published in: IEEE TRANSACTIONS ON COMPUTER-AIDED DESIGN OF INTEGRATED CIRCUITS AND SYSTEMS, VOL. 39, NO. 10, OCTOBER 2020
>  * Author: **Siqi Wang**, Gayathri Ananthanarayanan, Yifan Zeng, Neeraj Goel, Anuj Pathania, Tulika Mitra
>  * Contents:
>    * 딥러닝 layer를 big.LITTLE cluster단위로 실행시키는 프레임워크 제시(Layer의 병렬처리 단위를 각 Cluster로 제한함), 이전 최대 대역폭 대비 39%개선
>    * Layer Descriptor만으로 Configuration에 따른 성능을 예측
>    * AlexNet과 같은 Memory Intensive한 모델의 경우, 메모리의 전력소모를 통제할 수 없었기 때문에 Small Cluster의 에너지 효율성이 예상보다 낮게 측정됨
>    * **default stategy로 big.LITTLE코어를 혼합해서 제공할 경우 big코어만 단일로 제공하는 경우보다 성능이 떨어짐(클러스터간 Communication overhead 때문이라고 함)**
> </div>
> </details>

### 2주차
> ### Efficient Execution of Deep Neural Networks on Mobile Devices with NPU
> <details>
> <summary> > 내용 </summary>
> <div markdown='1'>
> 
> * Published in: ACM IPSN '21(Proceedings of the 20th International Conference on Information Processing in Sensor Networks)
> * Author: Tianxiang Tan, Guoong Cao, The Pennsylvania State University
> * Contents:
>   * Compared to CPU, **NPU can run DNN models much faster, but with lower accuracy**
>   * The challenge is to determine which part of the DNN model should be run on CPU and which part to be run on NPU.
>   * goal: Max-accuracy & Min-Time
>   * we propose heuristic & a Machine Learning based Model Partition(MLMP)
>   * The most significant **limitation of NPU is the precision of the floating-point numbers. NPU uses 16 bits or 8 bits to represent the floating-point numbers instead of 32 bits in CPU.** As a result, it runs DNN models much faster but less accurate compared to CPU, and it is a challenge to improve the accuracy of running DNN models on NPU. 
>   * Fig 1.을 보면 VocNet과 YOLO실행시,NPU의 정확도가 CPU보다 많이 떨어짐을 알 수 있음
>   * NPU의 정확도 손실은 fp16 연산만을 지원하는 특성 때문에 연산시 over/underflow이 발생하는것이 원인이라고 하고 있음
> </div>
> </details>

> ### Survey of Machine Learning Accelerators
> <details>
> <summary> > 내용 </summary>
> <div markdown='1'>
> 
> * Published in: 2020 IEEE High Performance Extreme Computing Conference (HPEC)
> * Author: Albert Reuther; Peter Michaleas; Michael Jones; Vijay Gadepally; Siddharth Samsi; Jeremy Kepner
> * Contents: 
>   * NPU의 수치정밀도는 딥러닝 연산의 정확도에 영향을 미침, **수치정확도가 높은 부동소수점 연산으로는 주로 학습을 시키고, 정수연산으로는 추론을 시켰는데** 이때 제한적이고 낮은 수치정밀도를 가지는 int4, int2(논문작성 당시 int8까지 나왔던듯)연산으로 추론을 수행함
>   * Fig.2를 봐도 **Embedded용 ML프로세서에서는 대부분 int 연산을 수행, 부동소수점 연산을 지원하는 프로세서는 보이지 않음**
> </div>
> </details>
>

> ### SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping
> <details>
> <summary> > 내용 </summary>
> <div markdown='1'>
>
>  * ASPLOS '20: Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming
>  * Authors: Chien-Chin Huang, Gu Jin, Jinyang Li
>  * DNN에서 GPU 메모리의 한계로 GPU-CPU간의 Memory Swap이 자주 발생, Dataflow그래프를 기반으로 operation을 예측할 수 있으니 메모리가 모자라는 상황이 나오기 전에 optimal하게 미리 GPU to CPU로 swap을 수행
>  * swap planing은 가능한 operation schedule, Memory Allocation 방식을 기준으로 GA수행, 최적의 swap plan을 DNN framework로 전달해서 최적의 swap이 발생하도록 함
> </div>
> </details>
>

## 2022/04월 찾은 논문
### 5주차
> ### DeepX: A Software Accelerator for Low-Power Deep Learning Inference on Mobile Devices
> <details>
> <summary> > 내용 </summary>
> <div markdown='1'>
>
> * 2016 15th ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN)
> * Nicholas D. Lane‡, Sourav Bhattacharya‡, Petko Georgiev†, Claudio Forlivesi‡, Lei Jiao‡, Lorena Qendro∗, and Fahim Kawsar‡
> * ‡Bell Labs, †University of Cambridge, ∗University of Bologna
> * Content:
>   * **하나의 DNN 모델을 여러개의 unit-block으로 분리, 이기종 프로세서에 나눠서 실행(decompose monolitic DNN model into unit-blocks, executed by heterogenouse local device processors)**
>   * 리소스 스케일링을 통해 각 unit-block이 생성하는 오버헤드를 형상화
>   * 아래 2종류의 기법을 제시:
>      * Runtime Layer Compression(RLC): 추론시점에 모델압축을 수행, 메모리/컴퓨팅/전력 소모를 런타임 시에 제어 할 수 있도록 함
>      * Deep Architecture Decomposition(DAD): unit-block들을 분석하고 decomposition plan을 통해 여러 프로세서들에 할당하는 역할
>        * 정수계획법(ILP, Mixed IPL)등을 써서 해결
> * Citation:
>    * Squeezing Deep Learning into Mobile and Embedded Devices #Squeezing-Deep-Learning-into-Mobile-and-Embedded-Devices)(Nicholas D. Lane, IEEE Pervasive Computing ( Volume: 16, Issue: 3, 2017))
> </div>
> </details>

> ### Multi-accelerator Neural Network Inference in Diversely Heterogeneous Embedded Systems
> <details>
> <summary> > 내용 </summary>
> <div markdown='1'>
>
> * 2021 IEEE/ACM Redefining Scalability for Diversely Heterogeneous Architectures Workshop (RSDHA)
> * Ismet Dagli, Mehmet E. Belviranli
> * Colorado School of Mines
> * Content:
>   * NN Inference 작업을 layer단위로 쪼개어 분산시켜 이기종 시스템에서 작업 수행(We explore the execution of various NNI workloads on a heterogeneous system by partitioning the layers among several accelerators)
>   * worklaod는 layer단위로 Processing Element(PE)에 할당되어 Energy/Performance Trade-off(EPT)성능을 극대화 하도록 수행(Each layer will be assigned to PEs based on their capabilities of performing better for a target EPT)
>   * CNN을 대상으로 GPU, Deep Learning Accelerator(DLA) colaboration 실험 수행
>   * **Multi-Accelerator Execution환경에서, Multi Accelerator Execution Gain(MAEG)이라는 측정방식(메트릭) 제시**
>     * MAE환경에서 Execution flow가 한 Accelerator에서 다른 Accelerator로 바뀔때(Transition point)의 trade-off를 조사
>     * DLA를 많이 사용할 수록, Lower Energy, Longer Execution Time이 소모됨 -> but, DLA를 증가시킬 수록 Fewer Energy, Lesser Time이 소모되는 구간이 있음
>     * Layer의 후반으로 갈수록 Kernel(필터)의 크기가 작아져 DLA의 EPT trade-off 성능이 좋아짐(GPU는 bigger buffer, kernel, parallelism 에 더 효과적
>     * Energy와 Execution Time에 기반하여 heterogenouse system에 trade-off가 있음을 실험을 통해 증명했고, 이 측정수단으로써 MAEG라는 측정방식을 제시
>   * Citation:
> </div>
> </details>

> ### DynO: Dynamic Onloading of Deep Neural Networks from Cloud to Device
> <details>
> <summary> > 내용 </summary>
> <div markdown='1'>
>
> * ACM Transactions on Embedded Computing Systems Accepted on January 2022
> * Mario Almeida, STEFANOS LASKARIDIS, STYLIANOS I. VENIERIS, ILIAS LEONTIADIS, NICHOLAS D. LANE
> * Samsung AI Center, Cambridge & University of Cambridge, UK
> * Content:
>   * Cloud(Server)와 Device(Client) 둘을 모두 활용하는 방식을 통해 off-loading에 의존했을 경우에 발생하는 문제인 operating cost나 privacy문제, latency에 관한 문제를 해결하고자 한다
>   * 최근 대두되는 DL Processing chip을 활용하여, **클라우드와 디바이스의 성능 시너지를 만들어내는 on-loading 방식을 제시**
>   * we allow **server-based CNN applications to push as much computation as possible onto the embedded devices** in order to
exploit their growing computational power. Under this paradigm, the goal is to minimize the remote-end usage, and hence cost, while still meeting the application’s service-level objectives (SLOs).
>   * DynO online scheduler가 연산을 쪼개고(partitioning), on/off loading을 수행하도록 함
>   * computation split 마다 요구하는 precision이나 packing 수준이 다른데, 이를 모니터링하고, 수준에 맞는곳에 연산을 할당
> * Related work:
>   * Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge. International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS) (2017), 615ś629
>     * 오프로딩 관련, Device에서 Server로 offloading 할 CNN의 split point를 선택하는 framework
>   * Chuang Hu, Wei Bao, Dan Wang, and Fengming Liu. 2019. Dynamic Adaptive DNN Surgery for Inference Acceleration on the Edge. Proceedings - IEEE INFOCOM (2019), 1423ś1431.
>     * 서버의 레이턴시, 대역폭을 고려한 스케줄링 기법 제시
>   * Hongshan Li, Chenghao Hu, Jingyan Jiang, Zhi Wang, Yonggang Wen, and Wenwu Zhu. 2019. JALAD: Joint Accuracy-And Latency Aware Deep Structure Decoupling for Edge-Cloud Execution. In International Conference on Parallel and Distributed Systems (ICPADS). 671ś678.
>     * 오프로딩시 고려해야 할 레이턴시-정확도 trade off에 대해 언급>
> </div>
> </details>

> ### Synergy: An HW/SW Framework for High Throughput CNNs on Embedded Heterogeneous SoC
> <details>
> <summary> > 내용 </summary>
> <div markdown='1'>
>
> * ACM Transactions on Embedded Computing SystemsVolume 18Issue 2March 2019
> * GUANWEN ZHONG, AKSHAT DUBEY, CHENG TAN, and TULIKA MITRA
> * National University of Singapore
> * Content:
>   * 이기종 Platform을 지원하는 CNN framework 제시
>   * Xilinx Zynq FPGA와 ARM NEON을 모두 활용하여 latency와 throughput 둘다 개선
>   * FPGA, PE 종류 상관없이 cluster를 구성하고, 여기에 Job queue를 부여, Cluster별로 balace가 조절되도록 함
>   * **Work-stealing 이라는 기법을 사용해서 Cluster간 workload를 balancing 함**
>   * 각 클러스터 안에서는 multi-thread를 활용하여 FPGA, PE를 병렬적으로 활용하는것으로 보임
>   * mult-thread를 이용해서 각 PE가 다른레이어, 다른 프레임 작업을 수행하도록 함
> * Related work:
>   * Network-Independent한 상황에서의 Embedded Deep Infernece에 적용되는 기법: Accelerator 전용의 가속기 + 컴파일러 활용, 모델 축소
> </div>
> </details>

> ### A Survey of Deep Learning on CPUs: Opportunities and Co-Optimizations
> <details>
> <summary> > 내용 </summary>
> <div markdown='1'>
>
>  * IEEE Transactions on Neural Networks and Learning Systems ( Early Access ) 21 April 2021 
>  * Sparsh Mittal , Senior Member, IEEE, Poonam Rajput, and Sreenivas Subramoney, Senior Member, IEEE
>  * Contents:
>    * DL Accelerator가 많이 대두 되고 있지만 항상 optimal한 것은 아니며 오히려 standardization, availablitiy, portability등을 고려 했을때 여전히 가장 widely used되는 processor라고 소개 -> Embedded(on-device관련된 부분만 선별적으로 확인)
>    * Mobile에서 CPU가 GPU보다 비슷하거나 더 좋은 performance를 내는 경우가 있음
>    * large data transfer and network setup overhead(FPGA의 느린 clock speed / GPU, TPU는 대역폭이 넓지만 latency때문에 RT inference에 적합하지 않음)
>    * **low batch size일 경우 CPU만이 least, comparable latency를 제공**
>    * mid-range model로 갈 수록 GPU/CPU의 성능차가 적음
>    * CPU/GPU간의 낮은 memory bandwidth가 장애물이 됨
>    * CPU기반의 DL 최적화와 DL위한 CPU 최적화를 다룸
>  * Related work:
>    * A survey of CPU-GPU heterogeneous computing techniques,” ACM Comput. Surv., vol. 47, no. 4, pp. 69:1–69:35, 2015.
> </div>
> </details>
  
> ### DeepRT: A Predictable Deep Learning Inference Framework for IoT Devices

> ### Squeezing Deep Learning into Mobile and Embedded Devices
