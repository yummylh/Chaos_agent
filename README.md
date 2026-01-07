# 🌪Chaos_agent -一个支持在个人垂域搭建自己的知识库的agent助手
## 🎯 硬件配置
 - 考虑到是在本地部署的轻量级agent，保证电脑显存不小于6GB，本人在1660ti上成功搭建、部署了这个agent。
 - 基座模型选用不超过7B的模型，如果显存大可以替换为参数更大的模型，此次采用Llama3.1 7B模型
 - 编程的IDE为VScode。通过streamlit开源框架编写前端简易AI
 - 需要下载Ollama程序，在终端执行“ollama pull Llama3.1”即可下载对应模型，这一块可以根据自己的显卡下载不同模型
 - 对应的建立向量数据库的embed—moedl为nomic-embed-text
## 📝 各个模块作用 (Key Takeaways)
### 1. .env文件
- 这是一个存放运行API的环境配置文件，将hugging face的API接口放入即可
- 以及在里面可以设置在后续需要调用大模型的API接口（如果有需要）
### 2. app.py
- 这是主程序界面，安装好环境后，打开vscode，执行streamlit run app.py指令，即可加载Agent模型（加载前请务必保证正确安装环境依赖包以及通过Ollama安装Llama3.1及nomic-embed-text模型）
- 设计了回退fallback机制，当用户在使用RAG模式未能检索到知识时，回退到通用Llama3.1，使用通用数据库进行回答并且标注基于同游数据库回答（会产生幻觉）
### 3. auto_data_factory
- 这个系列文件主要是用于后续LoRA产生高质量问答对，由于需要生成数据量很大，最少是500条，靠人工问答肯定是不行的。
- 这个文件的作用是将question.txt文件的内容自动输入进agent，通过agent回答，然后调用能力更强的大模型（如deepseek-v3）等对回答进行判分，分数等级分为1-10.因为是大模型判分，难免会有不准，所以建议使用2-8原则，及20%高质量数据人为判断，80%数据相信模型判断
- 运行这个脚本不用运行app.py，会在当前文件夹下产生.josn格式文件，便于后续进行LoRA微调
- v2版本是为了解决大模型question中的问题特别宽泛，导致agent回答表现不佳，评分很低无法进行训练的问题。采用查data库的方式使用能力更强大的大模型进行回答，产生CoT形式问答对，后续可以用LoRA技术对agent进行微调，让agent学会大模型的推理方式（也就是模型蒸馏）
### 4. check_pdf
- 主要是执行PDF检查工作，确认数据库data_pdf是否是完整pdf，编码是否正确
### 5. convert_all
- 主要是执行本地清洗工作，负责将PDF转化为agent更容易理解的MarkDown格式
### 6. debug_search
- 判断PDF打印出来是不是乱码
### 7. deepseek_batch_cleaner
- 负责调用能力参数更强大的deepseek模型对data_pdf文件夹中所有PDF数据进行清洗，将其全部转化为MarkDown格式
### 8. history_utils
- 负责保存记录历史会话，并生成chat_history历史会话记录
### 9. rag_engine.py
- agent的核心所在，也就是RAG增强检索方法，能够有效缓解Llama3.1幻觉问题。
- rewrite通过Llama3.1进行重写，将用户输入问题进行合理扩展。例如（它怎么样），会扩展为有特定术语的Logistic怎么样，但是表现不好，请谨慎使用（原因是因为llama3.1本身能力就有限，用它rewrite经常写的不是很好），但是有条件可以调参数大的大模型进行重写，这是完全没问题的。
- 其中采用了BGE-Reranker（稠密检索-重排）机制，查库时将会检索30个向量片段，将得分最高的5个重排序
### 10. router
- 这个脚本实现了用户意图识别以及路由功能
- 针对不同用户提问，将问题分为日常聊天（Chat）,数值计算（COMPUT）,检索（RAG）
- 不训练BERT轻量级分词器是因为避免应为数据过少以及0数据的冷启动
### 11. tools
- 本地数值计算工具库，目前只写了计算Logistic方程，Lorenz方程（这俩是混沌方程基础），有需要可以根据自身需求添加
### 13.build_db
- 这个脚本是负责构建向量数据库
- 根据清洗出来的MarkDown数据，按照MD文件的层级分级特性作为天然锚点采用了根据语义切分策略。
- 对于长文本，采用在一个大的CHUNK块切分成为小的sub_chunk块，只存小的向量数据，检索到时返回父类chunk完整输出（也就是我检索到这个小片段，我返回的是这个段的标题+内容）
- 没采用OCR光学符号识别是因为如果采用这种方式构建，显存会不够，并且响应速度会变慢
### 12. 其他文件夹
- data_pdf：存放原始pdf数据（要自己建）
- data：清洗后的MARKDOWN数据（会在根目录自动创建）
- Chrome_db：向量数据库，如果新增了data，建议删除此文件然后再次运行build_db脚本（会在根目录自动创建）
- chat_history：聊天记录存放（会在根目录自动创建）
- bad_data：不好的PDF数据（要自己建）

# 写在最后
- 这只是v1.0版本，后续我还会对其进行DLoRA微调，以及multi-Agent复合，并加入graphRAG解决多跳推理问题。
- 会引入Post-training后训练
希望我们都有好的实习，好的OFFER
