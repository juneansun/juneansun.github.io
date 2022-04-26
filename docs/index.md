# Paper review

## 2022/01학기 papers
### 22/04/26
> ### DeepX: A Software Accelerator for Low-Power Deep Learning Inference on Mobile Devices
> * 2016 15th ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN)
> * Nicholas D. Lane‡, Sourav Bhattacharya‡, Petko Georgiev†, Claudio Forlivesi‡, Lei Jiao‡, Lorena Qendro∗, and Fahim Kawsar‡
> * ‡Bell Labs, †University of Cambridge, ∗University of Bologna
> * Content:
>   * 하나의 DNN 모델을 여러개의 unit-block으로 분리, 이기종 프로세서에 나눠서 실행(decompose monolitic DNN model into unit-blocks, executed by heterogenouse local device processors)
>   * 리소스 스케일링을 통해 각 unit-block이 생성하는 오버헤드를 형상화
>   * 아래 2종류의 기법을 제시:
>      * Runtime Layer Compression(RLC): 추론시점에 모델압축을 수행, 메모리/컴퓨팅/전력 소모를 런타임 시에 제어 할 수 있도록 함
>      * Deep Architecture Decomposition(DAD): unit-block들을 분석하고 decomposition plan을 통해 여러 프로세서들에 할당하는 역할
>        * 정수계획법(ILP, Mixed IPL)등을 써서 해결
> * Citation:
>    * [Squeezing Deep Learning into Mobile and Embedded Devices](#Squeezing-Deep-Learning-into-Mobile-and-Embedded-Devices)(Nicholas D. Lane, IEEE Pervasive Computing ( Volume: 16, Issue: 3, 2017))

> ### Multi-accelerator Neural Network Inference in Diversely Heterogeneous Embedded Systems
> * 2021 IEEE/ACM Redefining Scalability for Diversely Heterogeneous Architectures Workshop (RSDHA)
> * Ismet Dagli, Mehmet E. Belviranli
> * Colorado School of Mines
> * Content:
>   * NN Inference 작업을 layer단위로 쪼개어 분산시켜 이기종 시스템에서 작업 수행(We explore the execution of various NNI workloads on a heterogeneous system by partitioning the layers among several accelerators)
>   * worklaod는 layer단위로 Processing Element(PE)에 할당되어 Energy/Performance Trade-off(EPT)성능을 극대화 하도록 수행(Each layer will be assigned to PEs based on their capabilities of performing better for a target EPT)
>   * CNN을 대상으로 GPU, Deep Learning Accelerator(DLA) colaboration 실험 수행, Multi-Accelerator Execution환경에서, Multi Accelerator Execution Gain(MAEG)이라는 측정방식(메트릭) 제시
>     * MAE환경에서 Execution flow가 한 Accelerator에서 다른 Accelerator로 바뀔때(Transition point)의 trade-off를 조사
>     * DLA를 많이 사용할 수록, Lower Energy, Longer Execution Time이 소모됨 -> but, DLA를 증가시킬 수록 Fewer Energy, Lesser Time이 소모되는 구간이 있음
>     * Layer의 후반으로 갈수록 Kernel(필터)의 크기가 작아져 DLA의 EPT trade-off 성능이 좋아짐(GPU는 bigger buffer, kernel, parallelism 에 더 효과적
>     * Energy와 Execution Time에 기반하여 heterogenouse system에 trade-off가 있음을 실험을 통해 증명했고, 이 측정수단으로써 MAEG라는 측정방식을 제시
>   * Citation: 

> ### DynO: Dynamic Onloading of Deep Neural Networks from Cloud to Device
> * ACM Transactions on Embedded Computing Systems Accepted on January 2022
> * Mario Almeida, STEFANOS LASKARIDIS, STYLIANOS I. VENIERIS, ILIAS LEONTIADIS, NICHOLAS D. LANE
> * Samsung AI Center, Cambridge & University of Cambridge, UK
> * Content:
>   * Cloud와 Device 둘을 모두 활용하는 방식을 통해 off-loading에 의존했을 경우에 발생하는 문제인 operating cost나 privacy문제, latency에 관한 문제를 해결하고자 한다
>   * 최근 대두되는 DL Processing chip을 활용하여, 클라우드와 디바이스의 성능 시너지를 만들어내는 on-loading 방식을 제시
>   * we allow server-based CNN applications to push as much computation as possible onto the embedded devices in order to
exploit their growing computational power. Under this paradigm, the goal is to minimize the remote-end usage, and hence cost, while still meeting the application’s service-level objectives (SLOs).
>   * DynO online scheduler가 연산을 쪼개고(partitioning), on/off loading을 수행하도록 함
>   * computation split 마다 요구하는 precision이나 packing 수준이 다른데, 이를 모니터링하고, 수준에 맞는곳에 연산을 할당
>   * 

> ### Squeezing Deep Learning into Mobile and Embedded Devices

> ### DeepRT: A Predictable Deep Learning Inference Framework for IoT Devices

