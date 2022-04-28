# FedMix: A distillation-like model fusion method for Non-IID in Federated Recommendation

### Abstract

联邦学习是一个较为新颖的分布式学习范式，其中许多设备协作训练一个机器学习模型，并保证数据不离开本地设备。在目前的大多数训练方案中，虽然已经有部分方案考虑到了分布式数据存在 Non-IID 的问题，但是针对推荐系统却尚未存在较为特定和深入的研究。在这项工作中，我们基于一个客户端仅存在一个用户相关数据的认知，使用知识蒸馏的方法开发了一个联邦推荐的通用框架，并在MovieLens-100K、MovieLens-1M、Pinterest-200三个数据集上测试了我们的框架，并观察到我们的方法比已有的联邦蒸馏方法收敛速度快了十倍，仅比所有私人数据集汇集并直接提供给所有参与者时每个模型的性能低几个百分点。

### Introduction

在信息过载的时代，人们往往很难在大量的物品中找到自己喜欢的东西。推荐系统通过利用用户的历史数据，推荐一些用户可能喜欢的物品来解决这个问题。传统的协同过滤算法需要在一个中心位置（如服务器）收集所有用户的评分数据，用于模型训练。近年来，由于社会上用户隐私意识的提高和数据隐私法的实施，如《通用数据保护条例》（GDPR，2018年5月生效）、《加州隐私权利法》（CPRA，2021年1月生效）和《中国数据安全法》（CDSL，2021年9月生效），中心服务器直接获取用户数据训练逐渐变成一件不可能的事情。保护隐私的机器学习越来越受到学术界和工业界的关注（如谷歌和微众银行）。

联邦学习（FL）是近年来在机器学习中提供隐私保护的最流行范式之一，从资源有限的移动设备到资源丰富的机构，FL已被证明适用于各种应用场景。在联邦学习的训练过程中，用户的原始数据始终保留在用户（客户端）本地，服务端和用户之间通过共享加密的或不包含隐私信息的中间参数的方式，进行模型训练和参数更新，进而在保护用户隐私的前提下构建一个有效的机器学习模型。因此，联邦学习和推荐系统的结合旨在保护用户隐私和商业机密的前提下，为用户提供精准的个性化服务。

最近有一些工作是在新的联邦学习范式下重新审视一些推荐算法。例如，联邦协同过滤（FCF）专注于具有隐式反馈的项目排名，并将所有未评级的项目视为负面项目，这可能会导致模型训练的偏差，也会在服务器与客户的互动中产生高的通信成本。FedMF使用同态加密技术，在将项目梯度上传到服务器之前对其进行加密，以保护用户的隐私。一项联合元学习工作将一种名为REPTILE的元学习方法与联合学习相结合，用于显式反馈的评级预测，它能够为每个用户微调模型参数。联合多视图矩阵分解（FED-MVMF）将多视图矩阵分解与联合学习相结合，以便在对多方数据建模时保护用户的原始评级数据。然而，它会泄露用户的评分行为（即用户评分的项目集），与上述方法类似。

上述的这些工作仍然是在FedAvg的基础上进行的方法迁移，并没有根据推荐系统本身的特点进行调整。具体来讲，联邦推荐与其他联邦监督学习最大的差异在于，客户端仅拥有单个用户的数据，即各客户端数据存在严重的非独立同分布的现象，因而基于神经网络的推荐算法在联邦框架下出现通信次数过多、模型计算复杂等特点。

为了解决上述的问题，我们提出了一个新的联邦推荐框架，使用类似于知识蒸馏的方法去聚合各客户端信息，将蒸馏中生成unlabel dataset与计算目标logits的过程交由客户端执行，增加了通信效率，并能更充分利用客户端算力。

### Related Work

#### Neural Collaborative Filtering

Neural Collaborative Filtering (NCF)由何向南博士于17年发表，不同于传统的基于矩阵分解的协同过滤算法，NCF框架引入了神经网络，通过神经网络来学习用户与物品的交互信息，并在实验中取得了一定的效果。

设定用户-项目交互矩阵为以下公式：
$$
\begin{equation}
y_{u i}=\left\{\begin{array}{cr}
1, & \text { 用户-项目具有交互 } \\
0, & \text { 其他 }
\end{array}\right.
\end{equation}
$$
通过对用户和项目使用one-hot编码将它们转化为二值化稀疏向量作为输入特征，然后经过嵌入层（Embedding Layer）将输入层的稀疏表示映射为一个稠密向量（Dense vector），然后我们将用户嵌入和项目嵌入送入被称为神经协同过滤层多层神经网络结构，将潜在向量映射为预测分数。最终输出层是预测分数$\hat{y}_{ui}$，训练通过最小化交叉熵进行。其中，预测值公式为：
$$
\begin{equation}
\hat{y} u i=f\left(P^{T} v_{U}^{u}, Q^{T} v_{I}^{i} \mid P, Q, \Theta_{f}\right)
\end{equation}
$$
其中 $P \in R^{M \times K}, Q \in R^{N \times K}$ 分别表示用户和项目的潜在因素矩阵, $M$ 和 $N$ 分别表 示用户 $u$ 和项目 $i$ 的数量, $K$ 表示潜在空间（Latent Space）的维度, 特征向量 $v_{U}^{u}$ 和 $v_{I}^{i}$ 分别用来 描述用户 $u$ 和项目 $i, \Theta_{f}$ 表示交互函数 $f$ 的模型参数，函数 $f$ 被定义为多层神经网络。

#### Federated Averaging(FedAvg)

**问题定义** 在这项工作中，我们考虑以下分布式最优化模型：
$$
\begin{equation}
\min _{\mathbf{w}}\left\{F(\mathbf{w}) \triangleq \sum_{k=1}^{N} p_{k} F_{k}(\mathbf{w})\right\}
\end{equation}
$$
其中 $N$ 表示设备数量，$p_k$ 表示聚合时第 $k$ 个设备的权重，即 $p_k \ge 0,\sum_{k=1}^N p_k=1$ 假设第 $k$ 个设备拥有 $n_k$ 条数据 $x_{k, 1}, x_{k, 2},\cdots,x_{k,n_k}$ ，$F_k(\cdot)$ 定义为
$$
\begin{equation}
F_{k}(\mathbf{w}) \triangleq \frac{1}{n_{k}} \sum_{j=1}^{n_{k}} \ell\left(\mathbf{w} ; x_{k, j}\right)
\end{equation}
$$
其中 $\ell (\cdot;\cdot)$ 表示本地设备的损失函数

**算法描述** 我们描述一轮 $t$-th 标准 FedAvg 算法的过程：首先，中心服务器将当前最新版本的模型 $\mathbf{w}_t$ 下发给所有客户端；然后，每个客户端 $k$-th 模型更新为下发的模型 $\mathbf{w}_t^k = \mathbf{w}_t$ 并在本地使用本地数据进行训练 E 轮：
$$
\begin{equation}
\mathbf{w}_{t+i+1}^{k} \longleftarrow \mathbf{w}_{t+i}^{k}-\eta_{t+i} \nabla F_{k}\left(\mathbf{w}_{t+i}^{k}, \xi_{t+i}^{k}\right), i=0,1, \cdots, E-1
\end{equation}
$$
其中 $\eta_{t+i}$ 表示学习率，$\xi_{t+i}^{k}$ 表示从本地数据中统一选取的样本；最后，中心服务器接收所有客户端的模型 $\mathbf{w}_{t+E}^1,\cdots,\mathbf{w}_{t+E}^N$ 并聚合成新的全局模型 $\mathbf{w}_{t+E}$ 

#### Non-IID in Federated Optimization

**问题描述** 在联邦学习中，Non-IID的意思一般是值不符合同分布的情况，因为数据的分布肯定是独立的，但是它们不一定服从同一采样方法。例如全集中有100类图片，某设备中都是风景类图片，某设备中都是人物类及植物类图片，前者是一种分布（1/100），后者是另一种分布（2/100）。反之，如果某设备中有这100类图片，其他设备中也有这100类图片，那么它们就是同分布的。每个设备中的数据分布不能代表全局数据分布，即每个设备中类别是不完备的。可以任意设定哪些比例的设备拥有哪些比例类别的样本。例如10分类问题中，5%的设备有3类样本，10%的设备有5类样本，30%的设备有10类样本……哪些比例的设备、哪些比例的样本类别都是可以改变的参数，从而决定了Non-IID的程度。此外，每个类别样本的数量也会影响Non-IID程度，但数量上的不同一般描述为不均衡的。

**量化非IID的程度(异质性)** 令 $F*$ 与 $F_k*$ 分别表示 $F$ 和 $F_k$ 的最小值，我们使用
$$
\begin{equation}
\Gamma=F^{*}-\sum_{k=1}^{N} p_{k} F_{k}^{*}
\end{equation}
$$
来量化Non-IID的程度，如果数据满足 IID，随样本数量的增加 $\Gamma$ 显然会变为零。如果数据是 Non-IID 的，则 $\Gamma$ 非零其大小反映了数据分布的异质性。

**收敛分析** $N$ 为设备的数量，在每一个 round 中 $K(\le+N)$ 代表参与训练的设备数量。 $T$ 代表每个 client 执行 SGD 的总次数， $E$ 代表 local systems 在每个 round 之间本地训练的次数，因此 $\frac{T}{E}$ 代表通信次数。Xiang Li 等人得到了如下结论：

FedAvg收敛率为 $\mathbf{O}(\frac{1}{T})$与所需的通信轮数为
$$
\frac{T}{E}=\mathcal{O}\left[\frac{1}{\epsilon}\left(\left(1+\frac{1}{K}\right) E G^{2}+\frac{\sum_{k=1}^{N} p_{k}^{2} \sigma_{k}^{2}+\Gamma+G^{2}}{E}+G^{2}\right)\right]
$$
其中，$G, \Gamma, p_{k}, \sigma_{k}$ 为问题相关定义的常数，公式表明：$E$ 是控制收敛率的关键。并且作者发现合适的采样和平均方案是 FedAvg 收敛的关键以及学习率的重要性。如果学习率始终固定为 $\eta$ ，则 FedAvg 将收敛到远离最优解的至少 $\Omega(\eta(E-1))$ 的解。

**FedProx** Tian Li 等人基于 FedAvg 提出了 FedProx 方法以减缓 Non-IID 程度对训练的影响，具体实现为：1）本地模型引入 proximal term 正则项，使得本地更新不要太过远离初始 global model，在容忍系统异构性的前提下减少 Non-IID 的影响。2）定义 $\gamma_{k}^{t}$-inexact solution，通过对 local function 的非精确求解，动态调整本地迭代次数，保证对异构系统的容忍度。
$$
proximal \ term:\min _{w} h_{k}\left(w ; w^{t}\right)=F_{k}(w)+\frac{\mu}{2}\left\|w-w^{t}\right\|^{2}
$$
如果 $w^{*}$ 满足下式则称为 $\min _{w} h_{k}\left(w ; w_{t}\right)$ 的 $\gamma_{k}^{t}$-inexact solution. $\gamma \in[0,1]$
$$
\begin{gathered}
\left\|\nabla h_{k}\left(w^{*} ; w_{t}\right)\right\| \leq \gamma_{k}^{t} \nabla h_{k}\left(w_{t} ; w_{t}\right) \| \\
\nabla h_{k}\left(w ; w_{t}\right)=\nabla F_{k}(w)+\mu\left(w-w_{t}\right)
\end{gathered}
$$

#### Federated Knowledge Distillation

知识蒸馏就是把一个模型的知识传授给另外一个模型的过程, 也称为Teacher-Student模型, 通常会用于模型压缩等任务。典型的知识蒸馏方法需要有一个代 理数据集, 一个教师模型 $\theta_{T}$ 和一个学生模型 $\theta_{S}$, 同样一条数据样本经过 $\theta_{T}$ 和 $\theta_{S}$, 在prediction layer会分别得到一个输出向量 (logits output), 我们通过迭代代理数据集训练学生模型, 最小化 $\theta_{T}$ 和 $\theta_{S}$ logits output的分布之间的距离, 这个距离通常使用KL散度 (Kullback-Leibler divergence) 来度量, 最终使得学 生模型能够在同样的数据样本下, 得到和教师模型相似的输出, 达到知识传授的目的。它可以形式化地描述为:
$$
\min _{\boldsymbol{\theta}_{S}} \mathbb{E}_{x \sim \hat{\mathcal{D}}_{\mathrm{P}}}\left[D _ { \mathrm { KL } } \left[\sigma\left(g\left(f\left(x ; \boldsymbol{\theta}_{T}^{f}\right) ; \boldsymbol{\theta}_{T}^{p}\right) \| \sigma\left(g\left(f\left(x ; \boldsymbol{\theta}_{S}^{f}\right) ; \boldsymbol{\theta}_{S}^{p}\right)\right]\right]\right.\right.
$$
其中, 神经网络模型被划分为两个部分, $\theta^{f}$ 和 $\theta^{p}$, 分别表示除prediction layer之外的前面的层和prediction layer, 函数攵表示前面的层的输出, $g$ 表示预测层的输出， $\sigma$ 表示预测层的激活函数（有些工作没有把激活函数加入知识蒸馏的过程, 应该也是可以的）。

目前联邦蒸馏领域主要聚焦于使用蒸馏方法来提高模型聚合的效果并降低通信开销，根据教师模型、学生模型和数据来源的异同，我们可以列出下表：

| Method | Teacher Model              | Student Model | Proxy Data              |
| ------ | -------------------------- | ------------- | ----------------------- |
| FD     | 除自己外其他所有模型的均值 | 客户端模型    | 未标记的数据/生成器生成 |
| FedDF  | 客户端模型均值             | 服务端模型    | 公有数据/生成器生成     |
| FedMD  | 服务端模型                 | 客户端模型    | 公共数据集              |
| FedGEN | 客户端模型均值             | 服务端模型    | 生成器生成              |

### FedMix: A distillation-like model fusion for Non-IID in Federated Recommendation

本章中，我们将先介绍本文提出的 FedMix 方法的核心观点，而后将介绍该框架的优势和可能的扩展。

#### Distillation-like Model Fusion

FedMix框架的核心概念就是将选出的 $|S_t|$ 个客户端的模型作为老师，服务端的模型作为学生，进行蒸馏。具体的讲，客户端将无标签的数据集 $\mathbf{d}_k$ 与其对应的 logit 输出传至服务端，服务端模型依据此进行蒸馏。
$$
\begin{equation}
\mathbf{x}_{t, j}:=\mathbf{x}_{t, j-1}-\eta \frac{\partial \mathrm{KL}\left(\sigma\left( f\left(\hat{\mathbf{x}}_{t}^{k}, \mathbf{d}\right)\right), \sigma\left(f\left(\mathbf{x}_{t, j-1}, \mathbf{d}\right)\right)\right)}{\partial \mathbf{x}_{t, j-1}}
\end{equation}
$$


此处 KL 表示 KL 散度，$\sigma$ 是 softmax 函数，$\eta$ 表示步长，$f(\mathbf{x}, \mathbf{d})$ 表示logit输出。

#### Utilizing constructed data for distillation

针对联邦推荐中数据分布的实际情况，我们认为类似于 FedDF 中对所有教师端使用同一unlabeled dataset进行预测后对logit 取均值的算法是低效的，更适用于图像识别等一般Non-IID的情况。对于推荐系统中更为极端的Non-IID情况，更加高效的选择是由客户端生成相应的unlabeled dataset，再传给服务器进行学习。因此，我们的蒸馏用数据集构造算法如下：



#### Discussions on privacy-preserving extension

与FedAvg相比，我们的方案没有传输模型的完整的参数，取而代之的，是一组 $\mathbf{x} \rightarrow logit$ （由于movielens等数据集是一个稀疏的矩阵，我们能够很轻易的获取无标签的数据 $\mathbf{x}$）

对于服务端下发模型至客户端过程中可能存在的隐私泄露，可以在框架中添加若干保护机制来保证用户免受泄露之忧，诸如差分隐私、同态加密等。

### Experiments

####  Experimental Setup

##### Datasets and models

我们基于一台配备有 NVIDIA GeForce RTX 2080 SUPER GPU 和足量内存的物理机进行了以下实验。我们的代码基于PyTorch框架并已开源在Github上（https://github.com/matrix72c/FedMIC）

我们选择movielens-100k、movielens-1m、pinterest三个数据集进行实验，采用 leave-one-out 方法，将每个用户最近的一次交互作为测试集（时间戳），将剩余记录作为训练集。为节省时间，随机抽取100个用户没有评分记录的物品，将测试物品与这100个物品一同评分然后排列。命中率（HR）和归一化折扣累积增益（NDCG）作为评估标准，且截取长度为10。因此，HR衡量测试项目是否存在于前10名列表中，而NDCG确定其位置。将这两个指标求取平均分作为衡量指标。

##### Baseline

FedMix是为了在服务器端进行有效的模型融合而设计的，它考虑了各客户端的有偏性。因此，我们省略了与为个性化（例如FedMD）、安全性/健壮性（例如Cronus）而设计的方法的比较。我们比较了FedMix和SOTA FL方法，包括 1) FedAvg，2) FedProx，3) FedDF。

##### Distribution of client data and the local training procedure

按照常理，我们认为在推荐系统中一个client仅会出现唯一一个 user_id，因此，我们依据 user_id 划分训练集，每个 client 分配到仅由单个 user_id 组成的交互记录。

本地训练时，由于数据集仅含有正例，根据NCF论文中的研究，随机选择数量4倍于正例的 item_id 构建负例共同组成训练集。损失函数设置为交叉熵，优化器使用Adam优化器，learning rate初值统一为0.001，并设置基础的学习率衰减。本地epoch设置为5轮。训练完成后，预测所有 item 可能的概率，选择概率最大的256个用户-物品对与其对应的 logit返回给服务端。

##### Server side model fusion procedure

server 每次从 client list 内随机选择 10 个客户端，进行训练，并接收训练后客户端返回的无标签数据集和对应 logit，根据上一章所述的方法构建蒸馏用数据集。蒸馏过程中，损失函数设置为 KL 散度，优化器使用 Adam 优化器，learning rate 初值统一为 0.001，并设置了基础的学习率衰减。本地epoch设置为5轮，训练完成后进行测试集的评估。

####  Evaluation on the Common Federated Learning Settings

FedMix 与其他几种 SOTA FL算法相比，在通信轮数和收敛速度上呈现出显著的优势。

同时，我们发现客户端sample数量与本地训练轮次可能对收敛速度造成影响，我们进行了以下比较，并得出结论：

### Conclusion

在这项工作中，我们提出了 FedMix，一个支持独立设计模型的联邦推荐的框架以解决联邦推荐中的 Non-IID 困境。基于知识蒸馏的框架有效提高了收敛速度与通信效率，并经过测试可以在各种数据集上工作。由于 FedMix 专注于服务器端更好的模型融合，因此它与针对 Non-IID 问题的最新技术（如 FedProx、FedBN 等针对客户端训练进行优化的技术）是独立并行的。我们相信，将FedDF与这些技术相结合可以带来更强大的FL，我们将其留在未来的工作中。

我们认为联邦学习等协作学习的方案是实现隐私保护的关键要素，它能使服务商在个人保留数据所有权的前提下实现个性化的服务。为异质和低算例的客户端引入实用可靠的蒸馏技术，是朝着更广泛地实现协作、保护隐私和高效的分布式学习迈出的重要一步。
