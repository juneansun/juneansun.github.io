# 논문 읽기
* 키페이퍼는 기억할 것
* related work, 유사아이디어, 참고하고 있는 논문도 따라가 볼것
* 인용, 또는 후속논문이 있는지 확인

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

>  * IEEE Transactions on Neural Networks and Learning Systems ( Early Access ) 21 April 2021 
>  * Sparsh Mittal , Senior Member, IEEE, Poonam Rajput, and Sreenivas Subramoney, Senior Member, IEEE
>  * Contents:
>    * DL Accelerator가 많이 대두 되고 있지만 항상 optimal한 것은 아니며 오히려 standardization, availablitiy, portability등을 고려 했을때 여전히 가장 widely used되는 processor라고 소개 -> Embedded(on-device관련된 부분만 선별적으로 확인)
>    * Mobile에서 CPU가 GPU보다 비슷하거나 더 좋은 performance를 내는 경우가 있음
>    * large data transfer and network setup overhead(FPGA의 느린 clock speed / GPU, TPU는 대역폭이 넓지만 latency때문에 RT inference에 적합하지 않음)
>    * ** low batch size일 경우 CPU만이 least, comparable latency를 제공 **
>    * mid-range model로 갈 수록 GPU/CPU의 성능차가 적음
>    * CPU/GPU간의 낮은 memory bandwidth가 장애물이 됨
>    * CPU기반의 DL 최적화와 DL위한 CPU 최적화를 다룸
>  * Related work:
>    * A survey of CPU-GPU heterogeneous computing techniques,” ACM Comput. Surv., vol. 47, no. 4, pp. 69:1–69:35, 2015.
> </div>
> </details>

> ### Squeezing Deep Learning into Mobile and Embedded Devices

> ### DeepRT: A Predictable Deep Learning Inference Framework for IoT Devices

