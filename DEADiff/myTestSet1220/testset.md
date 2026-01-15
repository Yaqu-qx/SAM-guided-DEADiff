## 测试集

### 1. 测试集设计原则
*   **图像数量**：建议选取 **8张内容图 (Content Images)** 和 **5张风格图 (Style Images)**，共组合出 40组测试对。对于定量评估（LPIPS/CLIP Score），这个数量在课程作业中是足够的。
*   **分辨率统一**：全部预处理为 **512 $\times$ 512**。
*   **涵盖场景**：人物（Portrait）、物体（Object）、建筑（Architecture）、风景（Landscape）。
*   **风格类型**：艺术画作（Artistic）、材质纹理（Texture/Material）、抽象图案（Abstract）。

---

### 2. 图像资源池 (Image Pool)

请从网络（如Unsplash, WikiArt）下载以下类型的图片，并按编号命名：

#### A. 内容图 (Content Images) - `C_01` 至 `C_08`
| 编号     | 描述 (Description)             | 考察点 (Rationale)                   | 适用任务           |
| :------- | :----------------------------- | :----------------------------------- | :----------------- |
| **C_01** | **一只特写猫/狗** (背景简单)   | 细节毛发保留，SAM分割容易            | 全任务             |
| **C_02** | **女性肖像** (正面，五官清晰)  | 面部结构保持 (Identity Preservation) | 对比/ControlNet    |
| **C_03** | **现代建筑/房屋** (线条明显)   | 结构直线是否弯曲 (VGG常把直线扭曲)   | ControlNet重点     |
| **C_04** | **一只运动鞋** (白底或纯色底)  | 对应你们的Demo，测试材质替换         | ControlNet重点     |
| **C_05** | **自然风景** (山脉或森林)      | 复杂纹理的融合能力                   | 对比/SAM           |
| **C_06** | **简单的线条稿/草图** (Sketch) | 测试从草图到成品的能力               | ControlNet独有优势 |
| **C_07** | **一辆汽车** (路面上)          | 金属反光质感的生成                   | 全任务             |
| **C_08** | **水果静物** (如桌上的苹果)    | 局部语义控制 (只变苹果不变桌子)      | SAM重点            |

#### B. 风格图 (Style Images) - `S_01` 至 `S_05`
| 编号     | 描述                              | 风格类型               | 对应Prompt关键词                                             |
| :------- | :-------------------------------- | :--------------------- | :----------------------------------------------------------- |
| **S_01** | **梵高《星空》** (Starry Night)   | 经典油画 (Classic Art) | `oil painting style of Starry Night`, `Van Gogh style`       |
| **S_02** | **赛博朋克城市** (霓虹灯, 蓝紫调) | 光效/氛围 (Lighting)   | `cyberpunk style`, `neon lights`, `futuristic`               |
| **S_03** | **火焰/岩浆** (Fire/Lava)         | 材质 (Material)        | `made of fire`, `burning flames`, `magma texture`            |
| **S_04** | **水彩画** (Watercolor)           | 笔触 (Stroke)          | `watercolor painting`, `soft pastel colors`                  |
| **S_05** | **浮世绘/神奈川冲浪里**           | 抽象纹理 (Abstract)    | `Ukiyo-e style`, `Great Wave off Kanagawa`, `traditional Japanese art` |

---

### 3. 实验分组与提示词 (Experimental Pairs & Prompts)

**注意：**
*   **VGG19 / StyTr²**：只输入 Content + Style，**不使用提示词**。
*   **CSGO / DEADiff**：输入 Content + Style + **Prompt**。
*   **DEADiff + ControlNet**：输入 Content (作为Canny/Depth图) + Style + **Prompt**。
*   **DEADiff + SAM**：输入 Content + Style + Prompt + **SAM Prompt (用于分割)**。

#### 组别 1：经典风格迁移对比 (General Comparison)
*目的：分工1和分工2的整体对比表格*

*   **Pair 1:** C_01 (Cat) + S_01 (Starry Night)
    *   **Prompt:** `A painting of a cat in the style of Starry Night, oil painting, intricate details, Van Gogh style`
*   **Pair 2:** C_03 (Building) + S_02 (Cyberpunk)
    *   **Prompt:** `A futuristic building, cyberpunk style, neon lights, night city background, sci-fi architecture`
*   **Pair 3:** C_05 (Landscape) + S_04 (Watercolor)
    *   **Prompt:** `A landscape with mountains and trees, watercolor painting style, soft artistic strokes`

#### 组别 2：结构与材质控制 (Structure & ControlNet Focus)
*目的：展示ControlNet如何拯救“结构崩坏”，以及DEADiff如何做材质迁移*

*   **Pair 4:** C_04 (Shoe) + S_03 (Fire)
    *   **Prompt:** `A pair of shoes made of fire, burning flames, magma texture, glowing, cinematic lighting`
    *   *注：这是DEADiff的强项，VGG通常做不到把鞋子“变成”火，只能给鞋子贴上火的纹理。*
*   **Pair 5:** C_06 (Sketch) + S_01 (Starry Night)
    *   **Prompt:** `A masterpiece oil painting based on a sketch, Starry Night style, vivid colors`
    *   *注：VGG/StyTr² 对草图基本无效，这是ControlNet的绝对优势领域。*

#### 组别 3：局部与语义控制 (SAM Focus)
*目的：展示分工3如何解决“背景被错误风格化”的问题*

*   **Pair 6:** C_01 (Dog/Cat) + S_03 (Fire)
    *   **Prompt:** `A dog made of fire, elemental creature`
    *   **SAM Text Prompt:** `dog`
    *   *预期效果：只有狗变成了火元素，草地背景保持原样（或轻微风格化），而Baseline方法会把草地也变成火海。*
*   **Pair 7:** C_02 (Portrait) + S_05 (Ukiyo-e)
    *   **Prompt:** `A woman in Ukiyo-e style, traditional Japanese art`
    *   **SAM Text Prompt:** `hair` 或 `clothes`
    *   *预期效果：只给头发或衣服上风格，脸部保持真实（避免脸崩）。*

---

### 4. 实验操作流程建议 (Checklist)

为了保证“公平比较”，请遵守以下流程：

1.  **固定随机种子 (Seed)**：
    *   Diffusion模型非常依赖Seed。对于每一组实验，选定一个Seed（例如 `42` 或 `1234`），所有涉及Diffusion的方法都用同一个Seed，否则对比无效。
2.  **Prompt Engineering**：
    *   DEADiff对Prompt敏感。建议所有Diffusion方法使用 **BLIP/CLIP interrogator** 生成的Caption + 风格后缀。
    *   *简便方法*：就用我上面给出的Prompt，已经包含了主体+风格描述。
3.  **ControlNet 预处理**：
    *   如果是 Canny 控制，确保 Canny 阈值在所有实验中统一（例如 low=100, high=200）。
4.  **评价指标计算**：
    *   将所有生成的图片放入文件夹：`results/method_name/pair_id.png`。
    *   统一运行脚本计算 LPIPS (Perceptual Similarity) 和 CLIP Score。

### 5. 预期结果 (用于Report撰写)

*   **VGG19**：会有明显的网格状伪影，语义理解差（把火的纹理贴在天空上）。
*   **StyTr²**：比VGG好，内容保持较好，但可能风格化程度不够浓烈。
*   **CSGO/DEADiff**：生成质量极高，但可能出现“幻觉”（比如多画了一只手）或结构改变（鞋子形状变了）。
*   **DEADiff+ControlNet**：结构完美对齐原图，且拥有Diffusion的高画质。
*   **DEADiff+SAM**：解决了全局风格化导致的背景干扰问题，实现了精准的局部编辑。

这个数据集方案既涵盖了Baseline的复现需求，又为你们的创新点（ControlNet和SAM）预留了展示空间。

TODO：

- [ ] 修改app.py脚本，提示词实验也跑通融合sam的实验
- [ ] 修改代码，实现输入一个inputs_normal.txt文件（数据格式为：一行代表一组数据，每组数据从左到右是：内容图路径、风格参考图路径、提示词，以空格隔开），就可以自动生成包含每组数据的图表，每行包括一组提示词、风格图、风格迁移结果的图。



- [ ] 构建inputs_sam.txt文件

	- [ ] 修改代码，实现输入一个inputs_sam.txt文件（数据格式为：一行代表一组数据，每组数据从左到右是：内容图路径、风格参考图路径、风格迁移提示词、sam分割对象提示词，以空格隔开），就可以自动生成包含每组数据的图表，每行包括一组提示词、风格图、风格迁移结果的图。



- [ ] 跑定量数据